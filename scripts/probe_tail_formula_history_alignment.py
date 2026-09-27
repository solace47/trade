"""Probe raw minute aggregation against daily context; not a client compilation test."""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from trade_research.corporate_cash import DAILY,MINUTES,save_json,sha

ROOT=Path('data/research/tail_formula_complete')
SOURCE=Path('data/research/tail_formula_1000')


def main():
    report=json.loads((ROOT/'feature_report.json').read_text())
    assert report['features_sha256']==sha(ROOT/'features.parquet')
    manifest=Path('data/research/economic_winner/input_manifest.json')
    hashes=json.loads(manifest.read_text())['source_sha256']
    daily_manifest=json.loads((SOURCE/'feature_report.json').read_text())['source_sha256']
    c=duckdb.connect();c.execute('SET threads=2')
    c.read_parquet(str(ROOT/'features.parquet')).create_view('features')
    keys=c.sql('''SELECT date,code,half FROM features WHERE formula_input_valid
        QUALIFY row_number() OVER(PARTITION BY half ORDER BY md5(date||code||'native-history-v1'))<=8
        ORDER BY code,date''').df()
    # Freeze the hash-based identities before reading their minute histories.
    metadata=dict(feature_report_sha256=sha(ROOT/'feature_report.json'),source_manifest_sha256=sha(manifest),
        sampled_keys=keys.to_dict('records'),outcomes_read=False,new_2026_prices_read=False)
    frozen=ROOT/'native_history_probe_keys.json'
    if frozen.exists():
        assert json.loads(frozen.read_text())==metadata
    else:
        save_json(frozen,metadata)
    actual=c.sql('SELECT date,code,A04,D01,D02,D03,D04,C01,C02,C03,C04,C05,C06,C07,C08 FROM features').df().set_index(['date','code'])
    rows=[]
    for k in keys.itertuples(index=False):
        dp=DAILY/(k.code.replace('.','_')+'.parquet')
        assert sha(dp)==daily_manifest[str(dp)]
        d=pd.read_parquet(dp,columns=['date','tradestatus'],filters=[('date','>=','2023-06-01'),('date','<',k.date)])
        past=d.loc[d.tradestatus.eq(1)].sort_values('date').tail(21)
        assert len(past)==21
        path=MINUTES/k.code[:2].upper()/(k.code[3:]+'.parquet')
        assert sha(path)==hashes[str(path)]
        raw=pq.read_table(path,filters=[('timestamp','>=',pd.Timestamp(past.date.min()).to_pydatetime()),
            ('timestamp','<=',pd.Timestamp(k.date+' 14:49').to_pydatetime())],
            columns=['timestamp','open','high','low','close','volume','turnover']).to_pandas().sort_values('timestamp')
        raw['date']=raw.timestamp.dt.strftime('%Y-%m-%d')
        raw[['open','high','low','close']]=raw[['open','high','low','close']].round(2).astype(float)
        agg=raw.groupby('date',sort=True).agg(op=('open','first'),hi=('high','max'),lo=('low','min'),cl=('close','last'),
            vol=('volume','sum'),amount=('turnover','sum'),bars=('timestamp','size'))
        before=agg.loc[agg.index<k.date].tail(21)
        aligned=before.index.tolist()==past.date.tolist()
        row=dict(date=k.date,code=k.code,prior_dates_match=aligned,minute_history_days=len(before),differences={})
        if len(before)==21 and k.date in agg.index:
            p=before.cl.iloc[::-1].reset_index(drop=True);today=agg.loc[k.date];price=actual.loc[(k.date,k.code),'A04']
            rebuilt=dict(D01=100*(p[0]/p[5]-1),D02=100*(p[0]/p[20]-1),D03=100*(price/p[:5].mean()-1),
                D04=100*(price/p[:20].mean()-1),C01=100*(price-today.lo)/max(today.hi-today.lo,.01),
                C02=100*(today.hi-today.lo)/p[0],C03=100*(today.hi-price)/p[0],
                C04=100*(min(today.op,price)-today.lo)/p[0],C05=today.vol/before.vol.tail(5).mean(),
                C06=today.amount/1e8,C07=100*(price/before.hi.tail(20).max()-1),C08=100*(price/before.lo.tail(20).min()-1))
            for field,value in rebuilt.items():
                original=float(actual.loc[(k.date,k.code),field])
                if not np.isclose(value,original,rtol=1e-9,atol=2e-6):
                    row['differences'][field]=dict(minute_aggregate=float(value),research_daily=original,difference=float(value-original))
        else:
            row['history_unavailable']=True
        rows.append(row)
    result=dict(keys_sha256=sha(frozen),sampled_windows=len(rows),date_alignment_failures=sum(not r['prior_dates_match'] for r in rows),
        windows_with_field_differences=sum(bool(r['differences']) for r in rows),details=rows,
        software_compilation_verified=False,probe_is_not_full_native_parity=True,outcomes_read=False,new_2026_prices_read=False)
    save_json(ROOT/'native_history_probe.json',result)
    return {k:v for k,v in result.items() if k!='details'}


if __name__=='__main__':
    print(json.dumps(main(),ensure_ascii=False,indent=2))
