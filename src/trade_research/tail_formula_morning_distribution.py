"""Fixed completed-morning volume-weighted quote center and minute range."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_morning_range as prior
from .corporate_cash import save_json,sha

STEM='tail_formula_morning_distribution'
ROOT=Path('data/research')/STEM
INPUTS=ROOT/'inputs'
PROTOCOL=Path('config')/(STEM+'_input_protocol.json')
INTENT=Path('config')/(STEM+'_intent.json')
META=prior.META
EXTRA_HEADER='''AMVPC:=VALUEWHEN(TIME=1130,SUM(ROUND(C*100)*V,121)/AMVV);
AMVRC:=VALUEWHEN(TIME=1130,SUM((ROUND(H*100)-ROUND(L*100))*V,121)/AMVV);
AMDVOK:=AMREADY AND AMVPC>0 AND AMVRC>=0;
'''
HEADER=prior.HEADER+EXTRA_HEADER
NEW_EXPRESSIONS={
    'AMCP':'IF(AMDVOK,100*(AMQC/AMVPC-1)/VP20,DRAWNULL)',
    'AMRP':'IF(AMDVOK,100*AMVRC/AMVPC/VP20,DRAWNULL)'}
ARMS={'control':prior.EXPRESSIONS,'distribution':{**prior.EXPRESSIONS,**NEW_EXPRESSIONS}}
EXPRESSIONS=ARMS['distribution']


def checked():
    p=json.loads(PROTOCOL.read_text());prior.checked()
    assert p['arms']==ARMS and p['native_header']==HEADER and p['intent_sha256']==sha(INTENT)
    assert p['no_new_raw_extraction'] and not p['new_2026_prices_allowed']
    for f,h in p['source_hashes'].items():assert sha(Path(f))==h,f
    gate=json.loads(Path(p['conditional_gate']).read_text())
    complete=json.loads(Path(p['conditional_completion']).read_text())
    assert gate['passed'] and not gate['supports_2024_extension'] and complete['passed']
    for f,h in complete['source_hashes'].items():assert sha(Path(f))==h,f
    fr=json.loads((prior.INPUTS/'feature_report.json').read_text())
    fv=json.loads((prior.INPUTS/'feature_verification.json').read_text())
    nv=json.loads((prior.INPUTS/'native_input_verification.json').read_text())
    assert fv['passed'] and nv['passed'] and fv['feature_report_sha256']==nv['feature_report_sha256']==sha(prior.INPUTS/'feature_report.json')
    assert fr['features_sha256']==sha(prior.INPUTS/'features.parquet')
    r=json.loads((prior.INPUTS/'raw_report.json').read_text())
    assert fr['raw_report_sha256']==sha(prior.INPUTS/'raw_report.json')
    for f,h in r['parts_sha256'].items():assert sha(Path(f))==h,f
    return p


def original():
    f=pd.read_parquet(prior.INPUTS/'features.parquet')
    assert len(f)==1258085 and f.date.ge('2024-01-01').all() and f.date.lt('2026-01-01').all()
    return f


def prepare():
    checked();assert not (INPUTS/'feature_report.json').exists();INPUTS.mkdir(parents=True,exist_ok=True)
    raw=json.loads((prior.INPUTS/'raw_report.json').read_text());pieces=[]
    for file in raw['parts_sha256']:
        d=pd.read_parquet(file,columns=['date','code','close','high','low','volume'])
        cc=np.floor(d.close.to_numpy(float)*100+.5)
        rc=np.floor(d.high.to_numpy(float)*100+.5)-np.floor(d.low.to_numpy(float)*100+.5)
        d['weighted_close']=cc*d.volume;d['weighted_range']=rc*d.volume
        a=d.groupby(['date','code'],sort=True)[['volume','weighted_close','weighted_range']].sum(min_count=1)
        a['AMVPC']=a.weighted_close/a.volume;a['AMVRC']=a.weighted_range/a.volume
        pieces.append(a.reset_index());print(json.dumps(dict(aggregated=len(pieces),parts=len(raw['parts_sha256']))),flush=True)
    a=pd.concat(pieces,ignore_index=True).sort_values(['date','code']).reset_index(drop=True)
    assert not a.duplicated(['date','code']).any()
    f=original()
    # Preserve every original key even if its already-invalid morning cache has
    # no rows; do not turn missing raw windows into dropped candidates.
    assert not len(a[['date','code']].merge(f[['date','code']],on=['date','code'],how='left',indicator=True).query('_merge == \"left_only\"'))
    a=f[['date','code']].merge(a,on=['date','code'],how='left',validate='one_to_one')
    pd.testing.assert_frame_equal(f[['date','code']],a[['date','code']],check_exact=True)
    good=(f.morning_input_valid & a.AMVPC.gt(0) & a.AMVRC.ge(0)).to_numpy(copy=True)
    good &= np.isfinite(a[['AMVPC','AMVRC']].to_numpy()).all(axis=1)
    f['distribution_prior_formula_input_valid']=f.formula_input_valid
    f['distribution_input_valid']=good;f['formula_input_valid'] &= good
    for name in ['AMVPC','AMVRC']:f[name]=a[name]
    with np.errstate(all='ignore'):
        f['AMCP']=100*(np.floor(f.A04.to_numpy()*100+.5)/a.AMVPC.to_numpy()-1)/f.V01
        f['AMRP']=100*a.AMVRC/a.AMVPC/f.V01
    f.loc[~good,['AMVPC','AMVRC',*NEW_EXPRESSIONS]]=np.nan
    a.to_parquet(INPUTS/'morning_distribution_aggregates.parquet',index=False,compression='zstd')
    f.to_parquet(INPUTS/'features.parquet',index=False,compression='zstd')
    r=dict(protocol_sha256=sha(PROTOCOL),features_sha256=sha(INPUTS/'features.parquet'),
        raw_report_sha256=sha(prior.INPUTS/'raw_report.json'),aggregates_sha256=sha(INPUTS/'morning_distribution_aggregates.parquet'),
        rows=len(f),valid=int(f.formula_input_valid.sum()),newly_invalid=int((f.distribution_prior_formula_input_valid&~f.formula_input_valid).sum()),
        expressions=EXPRESSIONS,native_header=HEADER,all_morning_caches_reused=True,no_new_raw_extraction=True,
        minute_close_weighted_center_not_transaction_vwap=True,new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True,
        software_compilation_verified=False,native_source_parity_verified=False)
    save_json(INPUTS/'feature_report.json',r)
    for file in ['full_labels.parquet','full_label_report.json','full_label_verification.json']:
        (INPUTS/file).symlink_to((prior.INPUTS/file).resolve())
    return {k:r[k] for k in ['rows','valid','newly_invalid']}


def verify():
    checked();r=json.loads((INPUTS/'feature_report.json').read_text())
    assert r['protocol_sha256']==sha(PROTOCOL) and r['features_sha256']==sha(INPUTS/'features.parquet')
    f=pd.read_parquet(INPUTS/'features.parquet');old=original()
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')],old.drop(columns='formula_input_valid'),check_exact=True)
    np.testing.assert_array_equal(f.distribution_prior_formula_input_valid,old.formula_input_valid)
    raw=json.loads((prior.INPUTS/'raw_report.json').read_text());c=base.conn();c.execute('SET threads=1');c.execute("SET memory_limit='4GB'")
    c.read_parquet(list(raw['parts_sha256'])).create_view('raw');c.register('original',old)
    expected=c.sql('''WITH a AS(SELECT date,code,count(*) AS n,count(DISTINCT timestamp) AS clocks,
        sum(volume) AS total,sum(round(close*100)*volume)/sum(volume) AS center,
        sum((round(high*100)-round(low*100))*volume)/sum(volume) AS span,
        max(round(high*100)) FILTER(WHERE volume>0) AS hi,min(round(low*100)) FILTER(WHERE volume>0) AS lo
        FROM raw GROUP BY date,code),b AS(SELECT *,coalesce(morning_input_valid AND n=121 AND clocks=121
        AND isfinite(center) AND center>0 AND isfinite(span) AND span>=0,false) AS valid
        FROM original LEFT JOIN a USING(date,code))
        SELECT date,code,valid,CASE WHEN valid THEN center END AS center,CASE WHEN valid THEN span END AS span,hi,lo,
        CASE WHEN valid THEN 100*(round(A04*100)/center-1)/V01 END AS AMCP,
        CASE WHEN valid THEN 100*span/center/V01 END AS AMRP FROM b ORDER BY date,code''').df();c.close()
    pd.testing.assert_frame_equal(f[['date','code']],expected[['date','code']],check_exact=True)
    np.testing.assert_array_equal(f.distribution_input_valid,expected.valid)
    np.testing.assert_allclose(f.high_cents,expected.hi,rtol=0,atol=0,equal_nan=True)
    np.testing.assert_allclose(f.low_cents,expected.lo,rtol=0,atol=0,equal_nan=True)
    np.testing.assert_allclose(f.AMVPC,expected.center,rtol=0,atol=2e-9,equal_nan=True)
    np.testing.assert_allclose(f.AMVRC,expected.span,rtol=0,atol=2e-9,equal_nan=True)
    encode=lambda a:np.floor(np.clip(100*a+10000+.000001,0,999999));diff=0.
    for name in NEW_EXPRESSIONS:
        np.testing.assert_allclose(f[name],expected[name],rtol=0,atol=2e-11,equal_nan=True)
        ok=expected.valid;np.testing.assert_array_equal(encode(f.loc[ok,name]),encode(expected.loc[ok,name]))
        diff=max(diff,float(np.max(np.abs(f.loc[ok,name]-expected.loc[ok,name]))))
    final=old.formula_input_valid & expected.valid;np.testing.assert_array_equal(f.formula_input_valid,final)
    names=re.findall(r'(?m)^([A-Z][A-Z0-9]*):=',HEADER)+list(EXPRESSIONS)
    assert len(names)==len({n.casefold() for n in names})
    out=dict(passed=True,feature_report_sha256=sha(INPUTS/'feature_report.json'),rows=len(f),valid=int(final.sum()),max_difference=diff,
        all_50_values_and_all_metadata_unchanged=True,all_complete_morning_keys_extrema_and_distribution_values_independently_verified=True,
        all_new_quality_and_encodings_verified=True,effective_input_intersection_unchanged=bool(final.equals(old.formula_input_valid)),
        same_quality_controls_required_before_fitting=not final.equals(old.formula_input_valid),no_new_raw_extraction=True,
        new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(INPUTS/'feature_verification.json',out);return out


def native_values(bars,quote,atr,outside=1000000.):
    """Literal same-day 11:30 saving and 14:49 evaluation; quote in yuan."""
    b=bars.sort_values('timestamp');times=b.timestamp.dt.hour*100+b.timestamp.dt.minute
    size=len(b);env=dict(Q=float(quote),VP20=float(atr),DRAWNULL=np.nan,
        DATE=np.r_[0,np.ones(size+2)],TIME=np.r_[1459,times.to_numpy(),1131,1449],
        B0=np.r_[241,np.arange(1,size+3)],ROUND=lambda a:np.floor(a+.5),ABS=np.abs,
        MAX=np.maximum,MIN=np.minimum,IF=np.where)
    for col,name in [('open','O'),('high','H'),('low','L'),('close','C'),('volume','V')]:
        env[name]=np.r_[outside,b[col].to_numpy(float),outside,outside]
    for name,method in [('SUM','sum'),('COUNT','sum'),('HHV','max'),('LLV','min')]:
        env[name]=lambda a,n,m=method:getattr(pd.Series(np.asarray(a,float)).rolling(int(n),min_periods=int(n)),m)().to_numpy()
    def valuewhen(mask,values):
        values=np.broadcast_to(values,len(mask));result=np.full(len(mask),np.nan)
        result[np.asarray(mask,bool)]=values[np.asarray(mask,bool)]
        return pd.Series(result).ffill().to_numpy()
    env['VALUEWHEN']=valuewhen
    for line in (prior.EXTRA_HEADER+EXTRA_HEADER).splitlines():
        name,expr=line.rstrip(';').split(':=');expr=re.sub(r'(?<![<>=!])=(?!=)','==',expr)
        if name=='AMCLK':expr="VALUEWHEN(TIME==1130,COUNT((TIME>=930)&(TIME<=1130),121))"
        elif name=='AMGD':
            cond=expr.split('COUNT(',1)[1].rsplit(',121)',1)[0]
            expr='VALUEWHEN(TIME==1130,COUNT('+' & '.join('('+s+')' for s in cond.split(' AND '))+',121))'
        elif name in ['AMREADY','AMDVOK']:expr=' & '.join('('+s+')' for s in expr.split(' AND '))
        env[name]=eval(expr,{'__builtins__':{}},env)
    with np.errstate(all='ignore'):
        return np.asarray([np.asarray(eval(e,{'__builtins__':{}},env))[-1] for e in NEW_EXPRESSIONS.values()])


def native():
    checked();fv=json.loads((INPUTS/'feature_verification.json').read_text())
    assert fv['passed'] and fv['feature_report_sha256']==sha(INPUTS/'feature_report.json')
    f=pd.read_parquet(INPUTS/'features.parquet');encode=lambda a:np.floor(np.clip(100*a+10000+.000001,0,999999))
    for start in range(0,len(f),50000):
        d=f.iloc[start:start+50000];env=dict(AMDVOK=d.distribution_input_valid.to_numpy(),AMQC=np.floor(d.A04.to_numpy()*100+.5),
            AMVPC=d.AMVPC.to_numpy(),AMVRC=d.AMVRC.to_numpy(),VP20=d.V01.to_numpy(),IF=np.where,DRAWNULL=np.nan)
        for name,e in NEW_EXPRESSIONS.items():
            with np.errstate(all='ignore'):actual=eval(e,{'__builtins__':{}},env)
            np.testing.assert_allclose(actual,d[name],rtol=0,atol=2e-11,equal_nan=True)
            good=d.distribution_input_valid.to_numpy();np.testing.assert_array_equal(encode(actual[good]),encode(d.loc[good,name]))
    source=prior.INPUTS/'native_input_verification.json';samples=json.loads(source.read_text())['source_samples']
    keys=pd.DataFrame(samples)[['date','code']]
    raw=json.loads((prior.INPUTS/'raw_report.json').read_text());c=base.conn();c.execute('SET threads=1');c.execute("SET memory_limit='4GB'")
    c.read_parquet(list(raw['parts_sha256'])).create_view('raw');c.register('keys',keys)
    observed=c.sql('SELECT r.* FROM raw r JOIN keys USING(date,code) ORDER BY date,code,timestamp').df();c.close()
    index=f.set_index(['date','code'])
    for sample in samples:
        bars=observed.loc[observed.date.eq(sample['date'])&observed.code.eq(sample['code'])].reset_index(drop=True)
        assert len(bars)==121;row=index.loc[(sample['date'],sample['code'])];got=native_values(bars,row.A04,row.V01)
        np.testing.assert_allclose(got,row[list(NEW_EXPRESSIONS)].to_numpy(float),rtol=0,atol=2e-11,equal_nan=True)
        np.testing.assert_allclose(got,native_values(bars,row.A04,row.V01,outside=0.),rtol=0,atol=2e-11,equal_nan=True)
        scaled=bars.copy();scaled[['open','high','low','close']]*=5;scaled.volume*=10
        np.testing.assert_allclose(got,native_values(scaled,row.A04*5,row.V01),rtol=0,atol=2e-11,equal_nan=True)
    flat=pd.DataFrame(dict(timestamp=pd.date_range('2025-01-02 09:30',periods=121,freq='min'),open=10.,high=10.,low=10.,close=10.,volume=1000.))
    np.testing.assert_allclose(native_values(flat,10.,2.),[0.,0.],rtol=0,atol=0)
    np.testing.assert_allclose(native_values(flat,10.2,2.),[1.,0.],rtol=0,atol=2e-11)
    varied=flat.copy();varied.loc[0,['high','low']]=[11.,9.];varied.loc[0,'volume']=0.
    np.testing.assert_allclose(native_values(varied,10.,2.),[0.,0.],rtol=0,atol=0)
    varied.loc[0,'volume']=1000.
    np.testing.assert_allclose(native_values(varied,10.,2.),[0.,10/121],rtol=0,atol=2e-11)
    with np.errstate(all='ignore'):
        flat.volume=0.;assert np.isnan(native_values(flat,10.,2.)).all()
        flat.volume=1000.;flat.loc[0,'high']=9.;assert np.isnan(native_values(flat,10.,2.)).all()
    out=dict(passed=True,feature_report_sha256=sha(INPUTS/'feature_report.json'),feature_verification_sha256=sha(INPUTS/'feature_verification.json'),
        all_new_literal_values_and_encodings_checked=2*len(f),source_sample_days=len(samples),source_sample_bars=len(observed),
        reused_50_native_proof_sha256=sha(source),all_50_values_and_complete_cache_hashes_unchanged_before_proof_reuse=True,
        literal_1130_saving_date_guard_integer_cents_scale_and_volume_unit_verified=True,
        flat_positive_quote_zero_volume_range_invalid_scale_and_outside_checks_passed=True,
        no_new_raw_extraction=True,software_compilation_verified=False,native_source_parity_verified=False,
        new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(INPUTS/'native_input_verification.json',out);return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','verify','native'])
    print(json.dumps(globals()[p.parse_args().stage](),ensure_ascii=False,indent=2))
