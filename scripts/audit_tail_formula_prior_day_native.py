"""Quantify minute-versus-daily source differences on the fixed input probes."""
import json
import numpy as np
import pandas as pd

from trade_research import tail_formula_prior_day as study
from trade_research.corporate_cash import MINUTES, save_json, sha


def main():
    p = json.loads((study.ROOT/'feature_verification.json').read_text())
    assert p['passed'] and p['feature_report_sha256'] == sha(study.ROOT/'feature_report.json')
    keys = pd.DataFrame(p['fixed_native_cases'])[['date','code']]
    f = pd.read_parquet(study.ROOT/'features.parquet').merge(keys,on=['date','code'],validate='one_to_one')
    out = []
    for row in f.itertuples():
        file = MINUTES/row.code[:2].upper()/(row.code[3:]+'.parquet')
        d = pd.read_parquet(file,columns=['timestamp','high','low','close','volume'],
            filters=[('timestamp','>=',pd.Timestamp(row.yd_volume_first_date)),('timestamp','<',pd.Timestamp(row.date))]).sort_values('timestamp')
        d['date'] = d.timestamp.dt.strftime('%Y-%m-%d')
        a = d.groupby('date').agg(h=('high','max'),l=('low','min'),c=('close','last'),v=('volume','sum')).tail(20)
        assert len(a) == 20 and a.index[-1] == row.yd_source_date and a.index[-2] == row.yd_reference_date
        c1,c2 = round(float(a.c.iloc[-1]),2),round(float(a.c.iloc[-2]),2)
        hi,lo = round(float(a.h.iloc[-1]),2),round(float(a.l.iloc[-1]),2)
        native = np.array([100*(c1/c2-1)/row.V01,100*(c1-lo)/max(hi-lo,.01),100*(hi-lo)/c2/row.V01,20*a.v.iloc[-1]/sum(a.v)])
        stored = np.array([row.Y01,row.Y02,row.Y03,row.Y04])
        encode = lambda x: np.floor(np.clip(x*100+10000+.000001,0,999999)).astype(int)
        out.append(dict(date=row.date,code=row.code,maximum_absolute_input_difference=float(np.max(np.abs(native-stored))),
            encoded_differences=int((encode(native)!=encode(stored)).sum())))
    result = dict(passed=True,feature_report_sha256=sha(study.ROOT/'feature_report.json'),
        feature_verification_sha256=sha(study.ROOT/'feature_verification.json'),cases=out,
        maximum_absolute_input_difference=max(x['maximum_absolute_input_difference'] for x in out),
        encoded_differences=sum(x['encoded_differences'] for x in out),
        fixed_samples_only=True,client_source_parity_verified=False,new_2026_prices_read=False,outcomes_read=False)
    assert result['encoded_differences'] == 0
    save_json(study.ROOT/'native_input_probe.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
