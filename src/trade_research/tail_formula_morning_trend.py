"""Fixed completed-morning log-price direction and deviation from a straight line."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_morning_range as prior
from .corporate_cash import save_json,sha

STEM='tail_formula_morning_trend'
ROOT=Path('data/research')/STEM
INPUTS=ROOT/'inputs'
PROTOCOL=Path('config')/(STEM+'_input_protocol.json')
INTENT=Path('config')/(STEM+'_intent.json')
META=prior.META
EXTRA_HEADER='''AMANCH:=VALUEWHEN(TIME=930,ROUND(C*100));
AMANCHD:=VALUEWHEN(TIME=930,DATE);
AMDR:=LN(MAX(ROUND(C*100),1)/MAX(AMANCH,1));
AMR1:=VALUEWHEN(TIME=1130,SUM(AMDR,121)/121);
AMR2:=VALUEWHEN(TIME=1130,SUM(AMDR*AMDR,121)/121);
AMSLO:=VALUEWHEN(TIME=1130,SUM((B0-61)*AMDR,121))/147620;
AMNOISE:=MAX(AMR2-AMR1*AMR1-AMSLO*AMSLO*147620/121,0);
AMTROK:=AMREADY AND AMANCHD=DATE AND AMANCH>0;
'''
HEADER=prior.HEADER+EXTRA_HEADER
NEW_EXPRESSIONS={
    'AMTS':'IF(AMTROK,12000*AMSLO/VP20,DRAWNULL)',
    'AMNR':'IF(AMTROK,100*SQRT(AMNOISE)/VP20,DRAWNULL)'}
ARMS={'control':prior.EXPRESSIONS,'trend':{**prior.EXPRESSIONS,**NEW_EXPRESSIONS}}
EXPRESSIONS=ARMS['trend']


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
        d=pd.read_parquet(file,columns=['date','code','timestamp','close']).sort_values(['date','code','timestamp'])
        d['cents']=np.floor(d.close.to_numpy(float)*100+.5)
        anchor=d.groupby(['date','code'],sort=False).cents.transform('first')
        with np.errstate(all='ignore'):d['r']=np.log(d.cents/anchor)
        d['r2']=d.r*d.r;d['xr']=(d.timestamp.dt.hour*60+d.timestamp.dt.minute-630)*d.r
        a=d.groupby(['date','code'],sort=True)[['r','r2','xr']].sum(min_count=1)
        a['AMSLO']=a.xr/147620
        a['AMNOISE']=np.maximum(a.r2/121-(a.r/121)**2-a.AMSLO**2*147620/121,0)
        pieces.append(a.reset_index());print(json.dumps(dict(aggregated=len(pieces),parts=len(raw['parts_sha256']))),flush=True)
    a=pd.concat(pieces,ignore_index=True).sort_values(['date','code']).reset_index(drop=True)
    assert not a.duplicated(['date','code']).any()
    f=original();extra=a[['date','code']].merge(f[['date','code']],on=['date','code'],how='left',indicator=True)
    assert not extra._merge.eq('left_only').any()
    a=f[['date','code']].merge(a,on=['date','code'],how='left',validate='one_to_one')
    pd.testing.assert_frame_equal(f[['date','code']],a[['date','code']],check_exact=True)
    good=f.morning_input_valid.to_numpy(copy=True)
    good &= np.isfinite(a[['AMSLO','AMNOISE']].to_numpy()).all(axis=1)&a.AMNOISE.ge(0).to_numpy()
    f['trend_prior_formula_input_valid']=f.formula_input_valid;f['trend_input_valid']=good;f['formula_input_valid'] &= good
    f['AMSLO']=a.AMSLO;f['AMNOISE']=a.AMNOISE
    f['AMTS']=12000*a.AMSLO/f.V01;f['AMNR']=100*np.sqrt(a.AMNOISE)/f.V01
    f.loc[~good,['AMSLO','AMNOISE',*NEW_EXPRESSIONS]]=np.nan
    a.to_parquet(INPUTS/'morning_trend_aggregates.parquet',index=False,compression='zstd')
    f.to_parquet(INPUTS/'features.parquet',index=False,compression='zstd')
    r=dict(protocol_sha256=sha(PROTOCOL),features_sha256=sha(INPUTS/'features.parquet'),
        raw_report_sha256=sha(prior.INPUTS/'raw_report.json'),aggregates_sha256=sha(INPUTS/'morning_trend_aggregates.parquet'),
        rows=len(f),valid=int(f.formula_input_valid.sum()),newly_invalid=int((f.trend_prior_formula_input_valid&~f.formula_input_valid).sum()),
        expressions=EXPRESSIONS,native_header=HEADER,all_morning_caches_reused=True,no_new_raw_extraction=True,
        new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True,
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
    np.testing.assert_array_equal(f.trend_prior_formula_input_valid,old.formula_input_valid)
    raw=json.loads((prior.INPUTS/'raw_report.json').read_text());c=base.conn();c.execute('SET threads=1');c.execute("SET memory_limit='4GB'")
    c.read_parquet(list(raw['parts_sha256'])).create_view('raw');c.register('original',old)
    expected=c.sql('''WITH z AS(SELECT date,code,timestamp,round(close*100) AS cents,
        extract(hour FROM timestamp)*60+extract(minute FROM timestamp)-630 AS x FROM raw),
        anchors AS(SELECT date,code,max(cents) FILTER(WHERE strftime(timestamp,'%H%M')='0930') AS anchor FROM z GROUP BY date,code),
        v AS(SELECT *,CASE WHEN isfinite(cents) AND cents>0 AND isfinite(anchor) AND anchor>0
            THEN ln(cents/anchor) END AS r FROM z JOIN anchors USING(date,code)),
        a AS(SELECT date,code,count(*) AS raw_n,count(DISTINCT timestamp) AS raw_clocks,
            sum(r) AS r1,sum(r*r) AS r2,sum(x*r)/147620. AS slope FROM v GROUP BY date,code),
        q AS(SELECT *,greatest(r2/121.-(r1/121.)*(r1/121.)-slope*slope*147620./121.,0.) AS noise FROM a),
        b AS(SELECT *,coalesce(morning_input_valid AND raw_n=121 AND raw_clocks=121
            AND isfinite(slope) AND isfinite(noise) AND noise>=0,false) AS valid
            FROM original LEFT JOIN q USING(date,code))
        SELECT date,code,valid,CASE WHEN valid THEN slope END AS slope,CASE WHEN valid THEN noise END AS noise,
            CASE WHEN valid THEN 12000.*slope/V01 END AS AMTS,
            CASE WHEN valid THEN 100.*sqrt(noise)/V01 END AS AMNR FROM b ORDER BY date,code''').df();c.close()
    pd.testing.assert_frame_equal(f[['date','code']],expected[['date','code']],check_exact=True)
    np.testing.assert_array_equal(f.trend_input_valid,expected.valid)
    np.testing.assert_allclose(f.AMSLO,expected.slope,rtol=0,atol=2e-12,equal_nan=True)
    np.testing.assert_allclose(f.AMNOISE,expected.noise,rtol=0,atol=2e-12,equal_nan=True)
    encode=lambda a:np.floor(np.clip(100*a+10000+.000001,0,999999));diff=0.
    for name in NEW_EXPRESSIONS:
        np.testing.assert_allclose(f[name],expected[name],rtol=0,atol=2e-10,equal_nan=True)
        ok=expected.valid;np.testing.assert_array_equal(encode(f.loc[ok,name]),encode(expected.loc[ok,name]))
        diff=max(diff,float(np.max(np.abs(f.loc[ok,name]-expected.loc[ok,name]))))
    final=old.formula_input_valid & expected.valid;np.testing.assert_array_equal(f.formula_input_valid,final)
    names=re.findall(r'(?m)^([A-Z][A-Z0-9]*):=',HEADER)+list(EXPRESSIONS)
    assert len(names)==len({n.casefold() for n in names})
    out=dict(passed=True,feature_report_sha256=sha(INPUTS/'feature_report.json'),rows=len(f),valid=int(final.sum()),max_difference=diff,
        all_50_values_and_all_metadata_unchanged=True,all_complete_morning_keys_log_price_moments_and_encodings_independently_verified=True,
        effective_input_intersection_unchanged=bool(final.equals(old.formula_input_valid)),
        same_quality_controls_required_before_fitting=not final.equals(old.formula_input_valid),no_new_raw_extraction=True,
        new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(INPUTS/'feature_verification.json',out);return out


def native_values(bars,quote,atr,outside=1000000.):
    """Literal same-day 11:30 saving and 14:49 evaluation; quote in yuan."""
    b=bars.sort_values('timestamp');times=b.timestamp.dt.hour*100+b.timestamp.dt.minute
    size=len(b);env=dict(Q=float(quote),VP20=float(atr),DRAWNULL=np.nan,
        DATE=np.r_[0,np.ones(size+2)],TIME=np.r_[1459,times.to_numpy(),1131,1449],
        B0=np.r_[241,np.arange(1,size+3)],ROUND=lambda a:np.floor(a+.5),ABS=np.abs,
        MAX=np.maximum,MIN=np.minimum,IF=np.where,LN=np.log,SQRT=np.sqrt)
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
        elif name in ['AMREADY','AMTROK']:expr=' & '.join('('+s+')' for s in expr.split(' AND '))
        env[name]=eval(expr,{'__builtins__':{}},env)
    with np.errstate(all='ignore'):
        return np.asarray([np.asarray(eval(e,{'__builtins__':{}},env))[-1] for e in NEW_EXPRESSIONS.values()])


def native():
    checked();fv=json.loads((INPUTS/'feature_verification.json').read_text())
    assert fv['passed'] and fv['feature_report_sha256']==sha(INPUTS/'feature_report.json')
    f=pd.read_parquet(INPUTS/'features.parquet');encode=lambda a:np.floor(np.clip(100*a+10000+.000001,0,999999))
    for start in range(0,len(f),50000):
        d=f.iloc[start:start+50000];env=dict(AMTROK=d.trend_input_valid.to_numpy(),AMQC=np.floor(d.A04.to_numpy()*100+.5),
            AMSLO=d.AMSLO.to_numpy(),AMNOISE=d.AMNOISE.to_numpy(),VP20=d.V01.to_numpy(),IF=np.where,SQRT=np.sqrt,DRAWNULL=np.nan)
        for name,e in NEW_EXPRESSIONS.items():
            with np.errstate(all='ignore'):actual=eval(e,{'__builtins__':{}},env)
            np.testing.assert_allclose(actual,d[name],rtol=0,atol=2e-10,equal_nan=True)
            good=d.trend_input_valid.to_numpy();np.testing.assert_array_equal(encode(actual[good]),encode(d.loc[good,name]))
    source=prior.INPUTS/'native_input_verification.json';samples=json.loads(source.read_text())['source_samples']
    keys=pd.DataFrame(samples)[['date','code']]
    raw=json.loads((prior.INPUTS/'raw_report.json').read_text());c=base.conn();c.execute('SET threads=1');c.execute("SET memory_limit='4GB'")
    c.read_parquet(list(raw['parts_sha256'])).create_view('raw');c.register('keys',keys)
    observed=c.sql('SELECT r.* FROM raw r JOIN keys USING(date,code) ORDER BY date,code,timestamp').df();c.close()
    index=f.set_index(['date','code'])
    for sample in samples:
        bars=observed.loc[observed.date.eq(sample['date'])&observed.code.eq(sample['code'])].reset_index(drop=True)
        assert len(bars)==121;row=index.loc[(sample['date'],sample['code'])];got=native_values(bars,row.A04,row.V01)
        np.testing.assert_allclose(got,row[list(NEW_EXPRESSIONS)].to_numpy(float),rtol=0,atol=2e-10,equal_nan=True)
        np.testing.assert_allclose(got,native_values(bars,row.A04,row.V01,outside=0.),rtol=0,atol=2e-10,equal_nan=True)
        scaled=bars.copy();scaled[['open','high','low','close']]*=5;scaled.volume*=10
        np.testing.assert_allclose(got,native_values(scaled,row.A04*5,row.V01),rtol=0,atol=2e-10,equal_nan=True)
    flat=pd.DataFrame(dict(timestamp=pd.date_range('2025-01-02 09:30',periods=121,freq='min'),open=10.,high=10.,low=10.,close=10.,volume=1000.))
    np.testing.assert_allclose(native_values(flat,10.,2.),[0.,0.],rtol=0,atol=0)
    np.testing.assert_allclose(native_values(flat,10.2,2.),[0.,0.],rtol=0,atol=2e-10)
    step=flat.copy();step.loc[61:,['open','high','low','close']]=10.01
    x=np.arange(121)-60;log_price=np.log(np.floor(step.close.to_numpy()*100+.5)/1000)
    coef=np.linalg.lstsq(np.c_[np.ones(121),x],log_price,rcond=None)[0]
    residual=log_price-coef[0]-coef[1]*x
    expected=np.array([12000*coef[1],100*np.sqrt(np.mean(residual*residual))])/2
    np.testing.assert_allclose(native_values(step,10.01,2.),expected,rtol=0,atol=2e-10)
    reverse=step.copy();reverse[['open','high','low','close']]=step[['open','high','low','close']].iloc[::-1].to_numpy()
    np.testing.assert_allclose(native_values(reverse,10.,2.),[-expected[0],expected[1]],rtol=0,atol=2e-10)
    varied=flat.copy();varied.loc[0,['high','low']]=[11.,9.];varied.loc[0,'volume']=0.
    np.testing.assert_allclose(native_values(varied,10.,2.),[0.,0.],rtol=0,atol=0)
    varied.loc[0,'volume']=1000.
    np.testing.assert_allclose(native_values(varied,10.,2.),[0.,0.],rtol=0,atol=2e-10)
    with np.errstate(all='ignore'):
        flat.volume=0.;assert np.isnan(native_values(flat,10.,2.)).all()
        flat.volume=1000.;flat.loc[0,'high']=9.;assert np.isnan(native_values(flat,10.,2.)).all()
    out=dict(passed=True,feature_report_sha256=sha(INPUTS/'feature_report.json'),feature_verification_sha256=sha(INPUTS/'feature_verification.json'),
        all_new_literal_values_and_encodings_checked=2*len(f),source_sample_days=len(samples),source_sample_bars=len(observed),
        reused_50_native_proof_sha256=sha(source),all_50_values_and_complete_cache_hashes_unchanged_before_proof_reuse=True,
        literal_1130_saving_date_guard_integer_cents_scale_and_volume_unit_verified=True,
        flat_step_independent_least_squares_reversal_zero_volume_invalid_scale_and_outside_checks_passed=True,
        no_new_raw_extraction=True,software_compilation_verified=False,native_source_parity_verified=False,
        new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(INPUTS/'native_input_verification.json',out);return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','verify','native'])
    print(json.dumps(globals()[p.parse_args().stage](),ensure_ascii=False,indent=2))
