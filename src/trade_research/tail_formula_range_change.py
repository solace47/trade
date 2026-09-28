"""Change in strictly prior five-day versus twenty-day true range."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as previous
from . import tail_formula_volatility as old_atr
from .corporate_cash import DAILY, MINUTES, save_json, sha

STEM='tail_formula_range_change'
ROOT=Path('data/research')/STEM
PROTOCOL=Path('config')/(STEM+'_protocol.json')
DAILY_REPORT=Path('data/research/tail_formula_1000/feature_report.json')
EXPRESSIONS={**previous.EXPRESSIONS,'VR01':'20*('+ '+'.join(f'REF(TR0,B{i})' for i in range(5))+')/PAT'}
HEADER=previous.HEADER


def prior_ranges(high,low,close):
    """Current day is excluded from both averages."""
    high,low,close=map(pd.Series,[high,low,close])
    tr=pd.concat([high-low,(high-close.shift()).abs(),(low-close.shift()).abs()],axis=1).max(axis=1)
    return tr.rolling(5,min_periods=5).mean().shift(),tr.rolling(20,min_periods=20).mean().shift()


def checked_sources():
    p=json.loads(PROTOCOL.read_text());assert p['short_days']==5 and p['long_days']==20 and not p['new_2026_prices_allowed']
    for file,digest in p['source_hashes'].items():assert sha(Path(file))==digest
    for root in [previous.ROOT,old_atr.ROOT]:
        r=json.loads((root/'feature_report.json').read_text());v=json.loads((root/'feature_verification.json').read_text())
        assert v['passed'] and v['feature_report_sha256']==sha(root/'feature_report.json')
        assert r['features_sha256']==sha(root/'features.parquet')
    paths=json.loads(DAILY_REPORT.read_text())['source_sha256']
    for file,digest in paths.items():assert sha(Path(file))==digest
    return p,paths


def features():
    p,paths=checked_sources();assert not (ROOT/'feature_report.json').exists()
    c=base.conn();c.read_parquet(list(paths)).create_view('raw_daily')
    hist=c.sql('''WITH d AS(SELECT date,code,high::DOUBLE AS h,low::DOUBLE AS l,close::DOUBLE AS cl,
        preclose::DOUBLE AS pc,lag(close::DOUBLE) OVER(PARTITION BY code ORDER BY date) AS prior
        FROM raw_daily WHERE date BETWEEN '2023-06-01' AND '2025-12-30' AND tradestatus=1),
        t AS(SELECT *,greatest(h-l,abs(h-prior),abs(l-prior)) AS tr,
        coalesce(isfinite(h) AND isfinite(l) AND isfinite(prior) AND h>=l AND l>0 AND prior>0,false) AS good,
        coalesce(abs(pc-prior)>.005,false) AS ref_break FROM d),
        a AS(SELECT date,code,CASE WHEN count(tr) OVER s=5 THEN avg(tr) OVER s END AS rc_atr5,
        CASE WHEN count(tr) OVER w=20 THEN avg(tr) OVER w END AS rc_atr20,
        count(tr) OVER s AS rc_rows5,count(tr) OVER w AS rc_rows20,sum(good::INT) OVER w AS rc_good20,
        min(date) OVER w AS rc_first_date,max(date) OVER w AS rc_last_date,
        sum(ref_break::INT) OVER w AS rc_reference_breaks FROM t
        WINDOW s AS(PARTITION BY code ORDER BY date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING),
        w AS(PARTITION BY code ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING))
        SELECT * FROM a WHERE date>='2024-01-01' ORDER BY date,code''').df();c.close()
    f=pd.read_parquet(previous.ROOT/'features.parquet').merge(hist,on=['date','code'],how='left',validate='one_to_one').sort_values(['date','code']).reset_index(drop=True)
    valid=f.rc_rows5.eq(5)&f.rc_rows20.eq(20)&f.rc_good20.eq(20)&f.rc_last_date.lt(f.date)&f.rc_atr20.gt(0)&f.rc_atr5.ge(0)
    f['VR01']=(100*f.rc_atr5/f.rc_atr20).where(valid)
    f['range_change_valid']=valid;f['prior_formula_input_valid']=f.formula_input_valid
    f['formula_input_valid'] &= valid & np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
    ROOT.mkdir(parents=True,exist_ok=True);hist.to_parquet(ROOT/'history.parquet',index=False,compression='zstd')
    f.to_parquet(ROOT/'features.parquet',index=False,compression='zstd')
    r=dict(protocol_sha256=sha(PROTOCOL),source_hashes=p['source_hashes'],history_sha256=sha(ROOT/'history.parquet'),
        features_sha256=sha(ROOT/'features.parquet'),rows=len(f),valid=int(f.formula_input_valid.sum()),
        newly_invalid=int((f.prior_formula_input_valid&~f.formula_input_valid).sum()),
        valid_with_reference_breaks=int((f.formula_input_valid&f.rc_reference_breaks.gt(0)).sum()),
        first_history_date=f.rc_first_date.min(),last_history_date=f.rc_last_date.max(),expressions=EXPRESSIONS,native_header=HEADER,
        software_compilation_verified=False,native_source_parity_verified=False,new_group_outcomes_read=False,
        new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'feature_report.json',r);return {k:v for k,v in r.items() if k not in ['expressions','native_header']}


def verify_features():
    p,paths=checked_sources();r=json.loads((ROOT/'feature_report.json').read_text())
    assert r['protocol_sha256']==sha(PROTOCOL)
    for field in ['features','history']:assert r[field+'_sha256']==sha(ROOT/(field+'.parquet'))
    old=pd.read_parquet(previous.ROOT/'features.parquet');got=pd.read_parquet(ROOT/'features.parquet')
    pd.testing.assert_frame_equal(got[old.columns.drop('formula_input_valid')],old.drop(columns='formula_input_valid'),check_exact=True)
    rebuilt=[]
    for path in paths:
        d=pd.read_parquet(path,columns=['date','code','high','low','close','preclose','tradestatus'],filters=[('date','>=','2023-06-01'),('date','<=','2025-12-30')])
        d=d.loc[d.tradestatus.eq(1)].sort_values('date').reset_index(drop=True)
        if d.empty:continue
        assert not d.date.duplicated().any();out=d[['date','code']].copy()
        out['rc_atr5'],out['rc_atr20']=prior_ranges(d.high,d.low,d.close)
        prev=d.close.shift();good=np.isfinite(d[['high','low']]).all(axis=1)&np.isfinite(prev)&d.high.ge(d.low)&d.low.gt(0)&prev.gt(0)
        out['rc_rows5']=np.minimum(np.arange(len(d)),5);out['rc_rows20']=np.minimum(np.arange(len(d)),20)
        out['rc_good20']=good.astype(int).rolling(20,min_periods=1).sum().shift().fillna(0)
        out['rc_first_date']=d.date.shift(20).fillna(d.date.iloc[0]);out.loc[0,'rc_first_date']=None
        out['rc_last_date']=d.date.shift();out['rc_reference_breaks']=d.preclose.sub(prev).abs().gt(.005).astype(int).rolling(20,min_periods=1).sum().shift().fillna(0)
        rebuilt.append(out.loc[d.date.ge('2024-01-01')])
    hist=pd.concat(rebuilt,ignore_index=True).sort_values(['date','code']).reset_index(drop=True)
    pd.testing.assert_frame_equal(pd.read_parquet(ROOT/'history.parquet'),hist,check_dtype=False,rtol=0,atol=2e-12)
    f=old[['date','code','formula_input_valid']].merge(hist,on=['date','code'],how='left',validate='one_to_one')
    np.testing.assert_allclose(got.rc_atr20,old.atr20,rtol=0,atol=2e-12,equal_nan=True)
    valid=f.rc_rows5.eq(5)&f.rc_rows20.eq(20)&f.rc_good20.eq(20)&f.rc_last_date.lt(f.date)&f.rc_atr20.gt(0)&f.rc_atr5.ge(0)
    expected=(100*f.rc_atr5/f.rc_atr20).where(valid)
    np.testing.assert_array_equal(got.range_change_valid,valid)
    np.testing.assert_allclose(got.VR01,expected,rtol=0,atol=2e-9,equal_nan=True)
    final=old.formula_input_valid&valid&np.isfinite(expected);np.testing.assert_array_equal(got.formula_input_valid,final)
    enc=lambda a:np.floor(np.clip(100*a+10000+.000001,0,999999))
    np.testing.assert_array_equal(enc(got.loc[final,'VR01']),enc(expected[final]))
    names=re.findall(r'(?m)^([A-Z][A-Z0-9]*):=',HEADER)+list(EXPRESSIONS);assert len(names)==len(set(names))
    assert r['expressions']==EXPRESSIONS and r['native_header']==HEADER
    assert len(got)==r['rows'] and int(final.sum())==r['valid']
    proof=dict(passed=True,feature_report_sha256=sha(ROOT/'feature_report.json'),rows=len(got),
        every_prior_5_and_20_day_range_and_date_rebuilt=True,original_atr20_and_all48_inputs_unchanged=True,
        all_validity_flags_and_encodings_rebuilt=True,effective_input_intersection_unchanged=bool(final.equals(old.formula_input_valid)),
        new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'feature_verification.json',proof);return proof


def native():
    p,paths=checked_sources();v=json.loads((ROOT/'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256']==sha(ROOT/'feature_report.json')
    old=json.loads((old_atr.ROOT/'native_history_probe.json').read_text())
    assert old['date_alignment_failures']==old['unavailable_windows']==old['windows_with_field_differences']==0
    f=pd.read_parquet(ROOT/'features.parquet').set_index(['date','code'])
    hashes=json.loads(Path(p['minute_manifest']).read_text())['source_sha256'];cases=[];raw_count=0
    for sample in old['details']:
        day,code=sample['date'],sample['code'];row=f.loc[(day,code)]
        daily=pd.read_parquet(DAILY/(code.replace('.','_')+'.parquet'),columns=['date','tradestatus'],filters=[('date','>=','2023-06-01'),('date','<',day)])
        dates=daily.loc[daily.tradestatus.eq(1)].sort_values('date').date.tail(21).tolist();assert len(dates)==21
        path=MINUTES/code[:2].upper()/(code[3:]+'.parquet');assert sha(path)==hashes[str(path)]
        raw=pd.read_parquet(path,columns=['timestamp','high','low','close'],filters=[('timestamp','>=',pd.Timestamp(dates[0])),('timestamp','<',pd.Timestamp(day))]).sort_values('timestamp')
        raw['date']=raw.timestamp.dt.strftime('%Y-%m-%d');raw[['high','low','close']]=raw[['high','low','close']].round(2).astype(float)
        agg=raw.groupby('date').agg(high=('high','max'),low=('low','min'),close=('close','last'));assert agg.index.tolist()==dates
        tr=[max(agg.high.iloc[i]-agg.low.iloc[i],abs(agg.high.iloc[i]-agg.close.iloc[i-1]),abs(agg.low.iloc[i]-agg.close.iloc[i-1])) for i in range(1,21)]
        value=400*sum(tr[-5:])/sum(tr)
        np.testing.assert_allclose([sum(tr[-5:])/5,sum(tr)/20,value],[row.rc_atr5,row.rc_atr20,row.VR01],rtol=0,atol=2e-9)
        enc=lambda x:np.floor(np.clip(100*x+10000+.000001,0,999999));assert enc(value)==enc(row.VR01)
        cases.append(dict(date=day,code=code,source_sha256=sha(path),raw_minutes=len(raw),value=value));raw_count+=len(raw)
    assert len(cases)==32
    proof=dict(passed=True,protocol_sha256=sha(PROTOCOL),feature_report_sha256=sha(ROOT/'feature_report.json'),
        feature_verification_sha256=sha(ROOT/'feature_verification.json'),fixed_keys_reused_from=sha(old_atr.ROOT/'native_history_probe_keys.json'),
        samples=cases,fixed_samples=32,raw_minutes=raw_count,all_strictly_prior_ranges_native_expression_and_encodings_verified=True,
        software_compilation_verified=False,native_source_parity_verified=False,new_group_outcomes_read=False,
        new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'native_input_verification.json',proof);return {k:v for k,v in proof.items() if k!='samples'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['features','verify_features','native'])
    print(json.dumps(globals()[parser.parse_args().stage](),ensure_ascii=False,indent=2))
