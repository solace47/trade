"""Fixed completed-morning signed volume and volume-weighted quote changes."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_morning_range as prior
from .corporate_cash import save_json,sha

STEM='tail_formula_morning_flow'
ROOT=Path('data/research')/STEM
INPUTS=ROOT/'inputs'
PROTOCOL=Path('config')/(STEM+'_input_protocol.json')
INTENT=Path('config')/(STEM+'_intent.json')
META=prior.META
EXTRA_HEADER='AMVSG:=IF(ROUND(C*100)>ROUND(REF(C,1)*100),1,IF(ROUND(C*100)<ROUND(REF(C,1)*100),-1,0));\nAMVLG:=LN(MAX(ROUND(C*100),1)/MAX(ROUND(REF(C,1)*100),1));\nAMVTO:=VALUEWHEN(TIME=1130,SUM(V,120));\nAMVSD:=VALUEWHEN(TIME=1130,SUM(V*AMVSG,120));\nAMVLD:=VALUEWHEN(TIME=1130,SUM(V*AMVLG,120));\nAMFLWOK:=AMREADY AND AMVTO>0;\n'
HEADER=prior.HEADER+EXTRA_HEADER
NEW_EXPRESSIONS={
    'AMVS':'IF(AMFLWOK,100*AMVSD/AMVTO,DRAWNULL)',
    'AMVR':'IF(AMFLWOK,12000*AMVLD/AMVTO/VP20,DRAWNULL)'}
ARMS={'control':prior.EXPRESSIONS,'flow':{**prior.EXPRESSIONS,**NEW_EXPRESSIONS}}
EXPRESSIONS=ARMS['flow']


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
        d=pd.read_parquet(file,columns=['date','code','timestamp','close','volume']).sort_values(['date','code','timestamp'])
        cents=np.floor(d.close.to_numpy(float)*100+.5)
        d['cents']=cents;prev=d.groupby(['date','code'],sort=False).cents.shift(1)
        with np.errstate(all='ignore'):r=np.log(d.cents/prev)
        d['sv']=d.volume*np.sign(d.cents-prev);d['rv']=d.volume*r
        clock=d.timestamp.dt.hour*60+d.timestamp.dt.minute
        d=d.loc[clock.gt(570)&clock.le(690)]
        a=d.groupby(['date','code'],sort=True)[['volume','sv','rv']].sum(min_count=1).rename(columns={'volume':'AMVTO','sv':'AMVSD','rv':'AMVLD'})
        pieces.append(a.reset_index());print(json.dumps(dict(aggregated=len(pieces),parts=len(raw['parts_sha256']))),flush=True)
    a=pd.concat(pieces,ignore_index=True).sort_values(['date','code']).reset_index(drop=True)
    assert not a.duplicated(['date','code']).any()
    f=original();extra=a[['date','code']].merge(f[['date','code']],on=['date','code'],how='left',indicator=True)
    assert not extra._merge.eq('left_only').any()
    a=f[['date','code']].merge(a,on=['date','code'],how='left',validate='one_to_one')
    pd.testing.assert_frame_equal(f[['date','code']],a[['date','code']],check_exact=True)
    good=f.morning_input_valid.to_numpy(copy=True)
    good &= np.isfinite(a[['AMVTO','AMVSD','AMVLD']].to_numpy()).all(axis=1)&a.AMVTO.gt(0).to_numpy()
    f['flow_prior_formula_input_valid']=f.formula_input_valid;f['flow_input_valid']=good;f['formula_input_valid'] &= good
    for name in ['AMVTO','AMVSD','AMVLD']:f[name]=a[name]
    f['AMVS']=100*a.AMVSD/a.AMVTO;f['AMVR']=12000*a.AMVLD/a.AMVTO/f.V01
    f.loc[~good,['AMVTO','AMVSD','AMVLD',*NEW_EXPRESSIONS]]=np.nan
    a.to_parquet(INPUTS/'morning_flow_aggregates.parquet',index=False,compression='zstd')
    f.to_parquet(INPUTS/'features.parquet',index=False,compression='zstd')
    r=dict(protocol_sha256=sha(PROTOCOL),features_sha256=sha(INPUTS/'features.parquet'),
        raw_report_sha256=sha(prior.INPUTS/'raw_report.json'),aggregates_sha256=sha(INPUTS/'morning_flow_aggregates.parquet'),
        rows=len(f),valid=int(f.formula_input_valid.sum()),newly_invalid=int((f.flow_prior_formula_input_valid&~f.formula_input_valid).sum()),
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
    np.testing.assert_array_equal(f.flow_prior_formula_input_valid,old.formula_input_valid)
    raw=json.loads((prior.INPUTS/'raw_report.json').read_text());c=base.conn();c.execute('SET threads=1');c.execute("SET memory_limit='4GB'")
    c.register('original',old[['date','code','morning_input_valid','V01']]);pieces=[]
    for number,file in enumerate(raw['parts_sha256'],1):
        c.read_parquet(file).create_view('raw',replace=True)
        part=c.sql("""WITH z AS(SELECT date,code,timestamp,volume,round(close*100) AS cents FROM raw),
        v AS(SELECT *,lag(cents) OVER(PARTITION BY date,code ORDER BY timestamp) AS pc FROM z),
        a AS(SELECT date,code,count(*) AS raw_n,count(DISTINCT timestamp) AS raw_clocks,
            sum(volume) FILTER(WHERE strftime(timestamp,'%H%M')>'0930' AND strftime(timestamp,'%H%M')<='1130') AS vt,
            sum(volume*sign(cents-pc)) FILTER(WHERE strftime(timestamp,'%H%M')>'0930' AND strftime(timestamp,'%H%M')<='1130') AS sd,
            sum(CASE WHEN isfinite(cents) AND cents>0 AND isfinite(pc) AND pc>0 THEN volume*ln(cents/pc) END)
                FILTER(WHERE strftime(timestamp,'%H%M')>'0930' AND strftime(timestamp,'%H%M')<='1130') AS ld FROM v GROUP BY date,code),
        b AS(SELECT *,coalesce(morning_input_valid AND raw_n=121 AND raw_clocks=121
            AND isfinite(vt) AND vt>0 AND isfinite(sd) AND isfinite(ld),false) AS valid
            FROM original JOIN a USING(date,code))
        SELECT date,code,valid,CASE WHEN valid THEN vt END AS AMVTO,CASE WHEN valid THEN sd END AS AMVSD,
            CASE WHEN valid THEN ld END AS AMVLD,CASE WHEN valid THEN 100.*sd/vt END AS AMVS,
            CASE WHEN valid THEN 12000.*ld/vt/V01 END AS AMVR FROM b ORDER BY date,code""").df()
        pieces.append(part);print(json.dumps(dict(verified_parts=number,total_parts=len(raw['parts_sha256']))),flush=True)
    c.close();aggregate=pd.concat(pieces,ignore_index=True)
    assert not aggregate.duplicated(['date','code']).any()
    expected=f[['date','code']].merge(aggregate,on=['date','code'],how='left',validate='one_to_one')
    expected['valid']=expected.valid.fillna(False).astype(bool)
    pd.testing.assert_frame_equal(f[['date','code']],expected[['date','code']],check_exact=True)
    np.testing.assert_array_equal(f.flow_input_valid,expected.valid)
    for name in ['AMVTO','AMVSD']:np.testing.assert_array_equal(f[name],expected[name])
    np.testing.assert_allclose(f.AMVLD,expected.AMVLD,rtol=2e-12,atol=2e-8,equal_nan=True)
    encode=lambda a:np.floor(np.clip(100*a+10000+.000001,0,999999));diff=0.
    for name in NEW_EXPRESSIONS:
        np.testing.assert_allclose(f[name],expected[name],rtol=0,atol=2e-10,equal_nan=True)
        ok=expected.valid;np.testing.assert_array_equal(encode(f.loc[ok,name]),encode(expected.loc[ok,name]))
        diff=max(diff,float(np.max(np.abs(f.loc[ok,name]-expected.loc[ok,name]))))
    final=old.formula_input_valid & expected.valid;np.testing.assert_array_equal(f.formula_input_valid,final)
    names=re.findall(r'(?m)^([A-Z][A-Z0-9]*):=',HEADER)+list(EXPRESSIONS)
    assert len(names)==len({n.casefold() for n in names})
    out=dict(passed=True,feature_report_sha256=sha(INPUTS/'feature_report.json'),rows=len(f),valid=int(final.sum()),max_difference=diff,
        all_50_values_and_all_metadata_unchanged=True,all_complete_morning_keys_changes_volume_pairing_and_encodings_independently_verified=True,
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
    env['REF']=lambda a,n:np.r_[np.full(int(n),np.nan),np.asarray(a)[:-int(n)]]
    for line in (prior.EXTRA_HEADER+EXTRA_HEADER).splitlines():
        name,expr=line.rstrip(';').split(':=');expr=re.sub(r'(?<![<>=!])=(?!=)','==',expr)
        if name=='AMCLK':expr="VALUEWHEN(TIME==1130,COUNT((TIME>=930)&(TIME<=1130),121))"
        elif name=='AMGD':
            cond=expr.split('COUNT(',1)[1].rsplit(',121)',1)[0]
            expr='VALUEWHEN(TIME==1130,COUNT('+' & '.join('('+s+')' for s in cond.split(' AND '))+',121))'
        elif name in ['AMREADY','AMFLWOK']:expr=' & '.join('('+s+')' for s in expr.split(' AND '))
        env[name]=eval(expr,{'__builtins__':{}},env)
    with np.errstate(all='ignore'):
        return np.asarray([np.asarray(eval(e,{'__builtins__':{}},env))[-1] for e in NEW_EXPRESSIONS.values()])


def native():
    checked();fv=json.loads((INPUTS/'feature_verification.json').read_text())
    assert fv['passed'] and fv['feature_report_sha256']==sha(INPUTS/'feature_report.json')
    f=pd.read_parquet(INPUTS/'features.parquet');encode=lambda a:np.floor(np.clip(100*a+10000+.000001,0,999999))
    for start in range(0,len(f),50000):
        d=f.iloc[start:start+50000];env=dict(AMFLWOK=d.flow_input_valid.to_numpy(),
            AMVTO=d.AMVTO.to_numpy(),AMVSD=d.AMVSD.to_numpy(),AMVLD=d.AMVLD.to_numpy(),
            VP20=d.V01.to_numpy(),IF=np.where,DRAWNULL=np.nan)
        for name,e in NEW_EXPRESSIONS.items():
            with np.errstate(all='ignore'):actual=eval(e,{'__builtins__':{}},env)
            np.testing.assert_allclose(actual,d[name],rtol=0,atol=2e-10,equal_nan=True)
            good=d.flow_input_valid.to_numpy();np.testing.assert_array_equal(encode(actual[good]),encode(d.loc[good,name]))
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
    expected=np.array([100/120,12000*np.log(1001/1000)/120/2])
    np.testing.assert_allclose(native_values(step,10.01,2.),expected,rtol=0,atol=2e-10)
    reverse=step.copy();reverse[['open','high','low','close']]=step[['open','high','low','close']].iloc[::-1].to_numpy()
    np.testing.assert_allclose(native_values(reverse,10.,2.),-expected,rtol=0,atol=2e-10)
    quiet=step.copy();quiet.loc[61,'volume']=0.
    np.testing.assert_allclose(native_values(quiet,10.01,2.),[0.,0.],rtol=0,atol=2e-10)
    quiet.loc[62:,['open','high','low','close']]=10.
    np.testing.assert_allclose(native_values(quiet,10.,2.),[-100/119,12000*np.log(1000/1001)/119/2],rtol=0,atol=2e-10)
    varied=flat.copy();varied.loc[0,['high','low']]=[11.,9.];varied.loc[0,'volume']=0.
    np.testing.assert_allclose(native_values(varied,10.,2.),[0.,0.],rtol=0,atol=0)
    varied.loc[0,'volume']=1000.
    np.testing.assert_allclose(native_values(varied,10.,2.),[0.,0.],rtol=0,atol=2e-10)
    with np.errstate(all='ignore'):
        flat.volume=0.;assert np.isnan(native_values(flat,10.,2.)).all()
        flat.loc[0,'volume']=1000.;assert np.isnan(native_values(flat,10.,2.)).all()
        flat.volume=1000.;flat.loc[0,'high']=9.;assert np.isnan(native_values(flat,10.,2.)).all()
    out=dict(passed=True,feature_report_sha256=sha(INPUTS/'feature_report.json'),feature_verification_sha256=sha(INPUTS/'feature_verification.json'),
        all_new_literal_values_and_encodings_checked=2*len(f),source_sample_days=len(samples),source_sample_bars=len(observed),
        reused_50_native_proof_sha256=sha(source),all_50_values_and_complete_cache_hashes_unchanged_before_proof_reuse=True,
        literal_1130_saving_date_guard_integer_cents_scale_and_volume_unit_verified=True,
        flat_step_reversal_zero_volume_previous_quote_invalid_scale_and_outside_checks_passed=True,
        no_new_raw_extraction=True,software_compilation_verified=False,native_source_parity_verified=False,
        new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(INPUTS/'native_input_verification.json',out);return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','verify','native'])
    print(json.dumps(globals()[p.parse_args().stage](),ensure_ascii=False,indent=2))
