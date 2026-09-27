"""Rebuild two transaction-average relations and replay frozen raw-prefix cases."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from trade_research import tail_formula_vwap as study
from trade_research.corporate_cash import MINUTES,save_json,sha


def main():
    old=study.checked_source();root=study.ROOT;r=json.loads((root/'feature_report.json').read_text())
    for key,path in [('protocol_sha256',study.PROTOCOL),('previous_feature_report_sha256',study.previous.ROOT/'feature_report.json'),
        ('daily_feature_report_sha256',study.base.SOURCE/'feature_report.json'),
        ('intraday_feature_report_sha256',study.INTRADAY/'feature_report.json'),('features_sha256',root/'features.parquet')]:
        assert r[key]==sha(path)
    f=pd.read_parquet(root/'features.parquet')
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')],old.drop(columns='formula_input_valid'),check_exact=True)
    assert f.prior_formula_input_valid.equals(old.formula_input_valid)
    assert list(r['native_expressions'].items())==list(study.EXPRESSIONS.items()) and r['native_header']==study.HEADER
    names=re.findall(r'\b([A-Z][A-Z0-9]*):=',study.HEADER)
    assert len(names)==len(set(names)) and len(study.EXPRESSIONS)==50
    c=study.base.conn();c.register('original',old)
    a=c.sql('''WITH ratios AS(SELECT *,amount_1449/volume_1449 AS vd,
        (amount_1449-a29)/(volume_1449-v29) AS ve,a29/v29 AS vt FROM original),
        quality AS(SELECT *,coalesce(isfinite(price_1449) AND isfinite(amount_1449) AND isfinite(volume_1449)
            AND isfinite(a29) AND isfinite(v29) AND isfinite(high_1449) AND isfinite(low_1449)
            AND isfinite(V01) AND isfinite(vd) AND isfinite(ve) AND isfinite(vt)
            AND amount_1449>a29 AND volume_1449>v29 AND a29>0 AND v29>0 AND V01>0 AND price_1449>0
            AND vd BETWEEN low_1449-.0101 AND high_1449+.0101
            AND ve BETWEEN low_1449-.0101 AND high_1449+.0101,false) AS good FROM ratios)
        SELECT date,code,vd AS vwap_day,ve AS vwap_early,vt AS vwap_late,good AS vwap_source_valid,
            CASE WHEN good THEN 100*(price_1449*volume_1449/amount_1449-1)/V01 END AS W01,
            CASE WHEN good THEN 100*((a29*(volume_1449-v29))/(v29*(amount_1449-a29))-1)/V01 END AS W02
        FROM quality ORDER BY date,code''').df()
    pd.testing.assert_frame_equal(f[a.columns],a,check_dtype=False,rtol=0,atol=2e-11)
    valid=old.formula_input_valid&a.vwap_source_valid&np.isfinite(f[list(study.EXPRESSIONS)]).all(axis=1)
    assert valid.equals(f.formula_input_valid)
    def encode(x):
        return np.floor(np.clip(x.to_numpy()*100+10000+.000001,0,999999)).astype('int32')
    np.testing.assert_array_equal(encode(a.loc[valid,['W01','W02']]),encode(f.loc[valid,['W01','W02']]))
    assert len(f)==r['rows'] and int(valid.sum())==r['valid']
    assert int((old.formula_input_valid&~valid).sum())==r['newly_invalid']
    assert int((~a.vwap_source_valid).sum())==r['invalid_vwap_sources']
    assert f.date.between('2024-01-01','2025-12-30').all()
    sample=f.loc[valid].copy()
    sample['key_hash']=[hashlib.sha256(('vwap-native/'+d+'/'+k).encode()).hexdigest() for d,k in zip(sample.date,sample.code)]
    sample=sample.sort_values('key_hash').groupby('half',sort=True).head(8)
    manifest=Path('data/research/economic_winner/input_manifest.json')
    assert sha(manifest)=='26ec21da15c414c1d725c8ff26208ef129eaa5f5873c61de56c915ae25ff6a70'
    hashes=json.loads(manifest.read_text())['source_sha256'];checked=set();cases=[]
    for row in sample.sort_values(['date','code']).itertuples():
        path=MINUTES/row.code[:2].upper()/(row.code[3:]+'.parquet')
        if path not in checked:
            assert sha(path)==hashes[str(path)];checked.add(path)
        raw=c.execute('''SELECT timestamp,open::DOUBLE AS open,high::DOUBLE AS high,low::DOUBLE AS low,
            close::DOUBLE AS close,volume::DOUBLE AS volume,turnover::DOUBLE AS amount
            FROM read_parquet(?) WHERE timestamp>=?::TIMESTAMP AND timestamp<=?::TIMESTAMP ORDER BY timestamp''',
            [str(path),row.date+' 09:30:00',row.date+' 14:49:00']).df()
        expected=list(pd.date_range(row.date+' 09:30:00',row.date+' 11:30:00',freq='min'))
        expected+=list(pd.date_range(row.date+' 13:01:00',row.date+' 14:49:00',freq='min'))
        assert raw.timestamp.tolist()==expected and len(raw)==230
        tail=raw.loc[raw.timestamp.dt.strftime('%H:%M').ge('14:21')];assert len(tail)==29
        sums=[raw.volume.sum(),raw.amount.sum(),tail.volume.sum(),tail.amount.sum()]
        np.testing.assert_allclose(sums,[row.volume_1449,row.amount_1449,row.v29,row.a29],rtol=2e-14,atol=.0001)
        np.testing.assert_allclose([raw.close.iloc[-1],raw.high.max(),raw.low.min()],
            [row.price_1449,row.high_1449,row.low_1449],rtol=0,atol=.0001)
        # Client V uses lots; reconstruct all three native ratios with that unit.
        dv,da,tv,ta=sums;dv/=100;tv/=100
        vw0=da/(100*dv);vwe=(da-ta)/(100*(dv-tv));vwt=ta/(100*tv)
        values=[100*(row.price_1449/vw0-1)/row.V01,100*(vwt/vwe-1)/row.V01]
        np.testing.assert_allclose(values,[row.W01,row.W02],rtol=0,atol=2e-11)
        cases.append(dict(date=row.date,code=row.code,bars=len(raw),tail_bars=len(tail),
            first_clock=raw.timestamp.iloc[0].strftime('%H:%M'),last_clock=raw.timestamp.iloc[-1].strftime('%H:%M'),
            day_vwap=vw0,early_vwap=vwe,late_vwap=vwt,W01=values[0],W02=values[1]))
    c.close()
    proof=dict(passed=True,feature_report_sha256=sha(root/'feature_report.json'),rows=len(f),valid=int(valid.sum()),
        all_previous_48_values_unchanged=True,all_quality_flags_features_and_encodings_rebuilt=True,
        raw_prefix_cases=len(cases),raw_prefix_minutes=sum(x['bars'] for x in cases),raw_cases=cases,
        native_lot_unit_arithmetic_rebuilt=True,native_client_values_verified=False,
        outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(root/'feature_verification.json',proof)
    print(json.dumps({k:v for k,v in proof.items() if k!='raw_cases'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
