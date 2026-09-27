"""Independently rebuild prior stock-day components and audit native input mapping."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trade_research import tail_formula_overnight as study
from trade_research.corporate_cash import MINUTES,save_json,sha


def main():
    root=study.ROOT;files=study.source_files();r=json.loads((root/'feature_report.json').read_text())
    for key,path in [('protocol_sha256',study.PROTOCOL),('previous_feature_report_sha256',study.previous.ROOT/'feature_report.json'),
        ('daily_feature_report_sha256',study.base.SOURCE/'feature_report.json'),('features_sha256',root/'features.parquet')]:
        assert r[key]==sha(path)
    assert list(r['native_expressions'].items())==list(study.EXPRESSIONS.items()) and r['native_header']==study.HEADER
    old=pd.read_parquet(study.previous.ROOT/'features.parquet');f=pd.read_parquet(root/'features.parquet')
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')],old.drop(columns='formula_input_valid'),check_exact=True)
    assert f.prior_formula_input_valid.equals(old.formula_input_valid)
    pieces=[]
    for path in files:
        d=pd.read_parquet(path,columns=['date','code','open','close','preclose','adjustflag','tradestatus'],
            filters=[('date','>=','2023-06-01'),('date','<=','2025-12-30')])
        d=d.loc[d.tradestatus.eq(1)].sort_values('date').reset_index(drop=True)
        assert not d.date.duplicated().any() and d.code.nunique()<=1
        pc=d.close.shift();prior_adjust=d.adjustflag.shift()
        good=(d.open.gt(0)&d.close.gt(0)&pc.gt(0)&d.adjustflag.eq(3)&prior_adjust.eq(3)
            &np.isfinite(d.open)&np.isfinite(d.close)&np.isfinite(pc))
        out=d[['date','code']].copy()
        out['history_rows']=pd.Series(np.ones(len(d))).rolling(20,min_periods=1).sum().shift()
        out['history_good']=good.astype(int).rolling(20,min_periods=1).sum().shift()
        out['history_start']=d.date.shift(20);out['history_end']=d.date.shift()
        out['history_reference_start']=d.date.shift(21)
        out['history_gap_mean']=((d.open-pc)/pc).where(good).rolling(20,min_periods=1).mean().shift()
        out['history_daytime_mean']=((d.close-d.open)/d.open).where(good).rolling(20,min_periods=1).mean().shift()
        out['history_reversal_count']=(d.open.gt(pc)&d.close.lt(d.open)).astype(float).where(good).rolling(20,min_periods=1).sum().shift()
        out['history_reference_breaks']=(d.preclose-pc).abs().gt(.005).astype(int).rolling(20,min_periods=1).sum().shift()
        pieces.append(out.loc[out.date.ge('2024-01-01')])
    a=old[['date','code']].merge(pd.concat(pieces,ignore_index=True),on=['date','code'],how='left',validate='one_to_one')
    full=a.history_rows.eq(20)
    # Partial initial windows have no usable new inputs; dates are checked on every complete history.
    for name in ['history_start','history_end','history_reference_start']:
        pd.testing.assert_series_equal(f.loc[full,name],a.loc[full,name],check_names=False)
    names=['history_rows','history_good','history_gap_mean','history_daytime_mean','history_reversal_count','history_reference_breaks']
    np.testing.assert_allclose(f[names],a[names],rtol=0,atol=3e-15,equal_nan=True)
    source_valid=(a.history_rows.eq(20)&a.history_good.eq(20)&a.history_end.lt(a.date)
        &a.history_reference_start.lt(a.history_start)&old.V01.gt(0)
        &np.isfinite(a[['history_gap_mean','history_daytime_mean','history_reversal_count']]).all(axis=1)&np.isfinite(old.V01))
    assert source_valid.equals(f.history_source_valid)
    expected=pd.DataFrame({'G01':100*a.history_gap_mean/old.V01,'G02':100*a.history_daytime_mean/old.V01,
        'G03':5*a.history_reversal_count}).where(source_valid,axis=0)
    np.testing.assert_allclose(f[list(expected)],expected,rtol=0,atol=1e-11,equal_nan=True)
    valid=old.formula_input_valid&source_valid&np.isfinite(f[list(study.EXPRESSIONS)]).all(axis=1)
    assert valid.equals(f.formula_input_valid)
    def encode(x):
        return np.floor(np.clip(x.to_numpy()*100+10000+.000001,0,999999)).astype('int32')
    np.testing.assert_array_equal(encode(expected.loc[valid]),encode(f.loc[valid,list(expected)]))
    assert len(f)==r['rows'] and int(valid.sum())==r['valid']
    assert int((old.formula_input_valid&~valid).sum())==r['newly_invalid']
    assert int((valid&f.history_reference_breaks.gt(0)).sum())==r['valid_with_historical_reference_break']
    assert f.date.between('2024-01-01','2025-12-30').all() and f.loc[source_valid,'history_end'].lt(f.loc[source_valid,'date']).all()
    sample=f.loc[valid,['date','code','half','history_start','history_reference_start','history_end']].copy()
    sample['key_hash']=[hashlib.sha256(('overnight-native/'+d+'/'+k).encode()).hexdigest() for d,k in zip(sample.date,sample.code)]
    sample=sample.sort_values('key_hash').groupby('half',sort=True).head(8)
    manifest=Path('data/research/economic_winner/input_manifest.json')
    assert sha(manifest)=='26ec21da15c414c1d725c8ff26208ef129eaa5f5873c61de56c915ae25ff6a70'
    hashes=json.loads(manifest.read_text())['source_sha256'];checked=set();cases=[]
    c=study.base.conn();c.read_parquet(files).create_view('daily')
    for row in sample.sort_values(['date','code']).itertuples():
        path=MINUTES/row.code[:2].upper()/(row.code[3:]+'.parquet')
        if path not in checked:
            assert sha(path)==hashes[str(path)];checked.add(path)
        raw=c.execute('''SELECT strftime(timestamp,'%Y-%m-%d') AS date,
            arg_min(open,timestamp)::DOUBLE AS first_open,arg_max(close,timestamp)::DOUBLE AS last_close,
            strftime(min(timestamp),'%H:%M') AS first_clock,strftime(max(timestamp),'%H:%M') AS last_clock,count(*) AS bars
            FROM read_parquet(?) WHERE timestamp>=?::TIMESTAMP AND timestamp<?::TIMESTAMP
            GROUP BY date ORDER BY date''',[str(path),row.history_reference_start+' 00:00:00',row.date+' 00:00:00']).df()
        daily=c.execute('''SELECT date,open::DOUBLE AS daily_open,close::DOUBLE AS daily_close FROM daily
            WHERE code=? AND date>=? AND date<? AND tradestatus=1 ORDER BY date''',
            [row.code,row.history_reference_start,row.date]).df()
        assert len(daily)==21
        joined=daily.merge(raw,on='date',how='left',validate='one_to_one')
        # Last 20 opens and all 21 closes are the values used by the native history expressions.
        opendiff=(joined.iloc[1:].first_open-joined.iloc[1:].daily_open).abs()
        closediff=(joined.last_close-joined.daily_close).abs()
        cases.append(dict(date=row.date,code=row.code,source_start=row.history_reference_start,
            source_end=row.history_end,expected_stock_days=21,missing_raw_days=int(joined.bars.isna().sum()),
            prior_20_open_mismatches_over_0001=int((opendiff.gt(.0001)|opendiff.isna()).sum()),
            prior_21_close_mismatches_over_0001=int((closediff.gt(.0001)|closediff.isna()).sum()),
            maximum_open_difference=float(opendiff.max()),maximum_close_difference=float(closediff.max()),
            raw_first_clocks=sorted(joined.first_clock.dropna().unique().tolist()),
            raw_last_clocks=sorted(joined.last_clock.dropna().unique().tolist())))
    c.close()
    proof=dict(passed=True,feature_report_sha256=sha(root/'feature_report.json'),rows=len(f),valid=int(valid.sum()),
        all_previous_48_values_unchanged=True,all_prior_history_and_new_integer_encodings_rebuilt=True,
        input_source_dates_strictly_prior=True,original_history_cases=len(cases),original_history_days=21*len(cases),
        raw_open_mismatches=sum(x['prior_20_open_mismatches_over_0001'] for x in cases),
        raw_close_mismatches=sum(x['prior_21_close_mismatches_over_0001'] for x in cases),original_cases=cases,
        native_client_values_verified=False,outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(root/'feature_verification.json',proof)
    print(json.dumps({k:v for k,v in proof.items() if k!='original_cases'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
