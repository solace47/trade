"""Fixed volume position and directed crossings of the completed morning high."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_morning_hold as prior
from .corporate_cash import save_json,sha

STEM='tail_formula_morning_volume'
ROOT=Path('data/research')/STEM
INPUTS=ROOT/'inputs'
PROTOCOL=Path('config')/(STEM+'_input_protocol.json')
INTENT=Path('config')/(STEM+'_intent.json')
VOLUME=Path('data/research/tail_formula_minute_pressure')
META=prior.META
PRICE_COLUMNS=prior.PRICE_COLUMNS
VOLUME_COLUMNS=[f'mp_v{i}' for i in range(21,50)]
MATCH_COLUMNS=[f'mp_c{i}' for i in range(21,50)]
EXTRA_HEADER='''MHFLAG:=IF(ROUND(C*100)>AMHC,1,0);
MVTOT:=VALUEWHEN(TIME=1449,SUM(V,29));
MVGOOD:=VALUEWHEN(TIME=1449,COUNT(V>=0,29));
MVREADY:=HOLDOK AND MVGOOD=29 AND MVTOT>0;
'''
HEADER=prior.HEADER+EXTRA_HEADER
NEW_EXPRESSIONS={
    'MHVA':'IF(MVREADY,100*VALUEWHEN(TIME=1449,SUM(MHFLAG*V,29))/MVTOT,DRAWNULL)',
    'MHVX':'IF(MVREADY,100*VALUEWHEN(TIME=1449,SUM((MHFLAG-REF(MHFLAG,1))*V,29))/MVTOT,DRAWNULL)'}
ARMS={'control':prior.EXPRESSIONS,'volume':{**prior.EXPRESSIONS,**NEW_EXPRESSIONS}}
EXPRESSIONS=ARMS['volume']


def checked():
    p=json.loads(PROTOCOL.read_text());prior.checked()
    assert p['arms']==ARMS and p['native_header']==HEADER and p['expected_keys']==1258085
    assert p['intent_sha256']==sha(INTENT) and p['no_new_raw_extraction'] and not p['new_2026_prices_allowed']
    for f,h in p['source_hashes'].items():assert sha(Path(f))==h,f
    gate=json.loads((prior.ROOT/'morning_hold_gate.json').read_text())
    complete=json.loads((prior.ROOT/'complete_results_manifest.json').read_text())
    assert gate['passed'] and not gate['supports_2024_extension'] and complete['passed']
    for f,h in complete['source_hashes'].items():assert sha(Path(f))==h,f
    for root in [prior.INPUTS,VOLUME]:
        fr=json.loads((root/'feature_report.json').read_text())
        fv=json.loads((root/'feature_verification.json').read_text())
        assert fv['passed'] and fv['feature_report_sha256']==sha(root/'feature_report.json')
        assert fr['features_sha256']==sha(root/'features.parquet')
    for path in [prior.INPUTS/'native_input_verification.json',
                 prior.INPUTS/'native_verification_execution_receipt.json',
                 Path('data/research/tail_formula_dense_flow/inputs/native_input_verification.json')]:
        assert json.loads(path.read_text())['passed']
    return p


def original_and_caches():
    f=pd.read_parquet(prior.INPUTS/'features.parquet')
    q=pd.read_parquet(prior.QUOTES,filters=[('date','>=','2024-01-01'),('date','<','2026-01-01')]).reset_index(drop=True)
    v=pd.read_parquet(VOLUME/'features.parquet',columns=['date','code','mp_bars','mp_clocks',*MATCH_COLUMNS,*VOLUME_COLUMNS])
    pd.testing.assert_frame_equal(f[['date','code']],q[['date','code']],check_exact=True)
    pd.testing.assert_frame_equal(f[['date','code']],v[['date','code']],check_exact=True)
    assert len(f)==1258085 and f.date.ge('2024-01-01').all() and f.date.lt('2026-01-01').all()
    # Full source association is required, including invalid rows, before new
    # values. A mismatch is investigated, never converted into a quality filter.
    np.testing.assert_array_equal(np.floor(q[PRICE_COLUMNS[1:]].to_numpy(float)*100+.5),
                                  np.floor(v[MATCH_COLUMNS].to_numpy(float)*100+.5))
    return f,q,v


def measure(cents,high_cents,volumes):
    above=(cents>np.asarray(high_cents)[:,None]).astype(float)
    total=volumes.sum(axis=1)
    with np.errstate(all='ignore'):
        return np.column_stack([100*(volumes*above[:,1:]).sum(axis=1)/total,
            100*(volumes*(above[:,1:]-above[:,:-1])).sum(axis=1)/total])


def prepare():
    checked();assert not (INPUTS/'feature_report.json').exists();INPUTS.mkdir(parents=True,exist_ok=True)
    f,q,v=original_and_caches();vv=v[VOLUME_COLUMNS].to_numpy(float)
    good=(f.hold_input_valid & v.mp_bars.eq(29) & v.mp_clocks.eq(29)).to_numpy(copy=True)
    good &= (np.isfinite(vv)&(vv>=0)).all(axis=1)&(vv.sum(axis=1)>0)
    got=measure(np.floor(q[PRICE_COLUMNS].to_numpy(float)*100+.5),f.high_cents.to_numpy(),vv)
    got[~good]=np.nan;f['volume_prior_formula_input_valid']=f.formula_input_valid
    f['volume_input_valid']=good;f['formula_input_valid'] &= good
    for i,name in enumerate(NEW_EXPRESSIONS):f[name]=got[:,i]
    f.to_parquet(INPUTS/'features.parquet',index=False,compression='zstd')
    out=dict(protocol_sha256=sha(PROTOCOL),features_sha256=sha(INPUTS/'features.parquet'),
        rows=len(f),valid=int(f.formula_input_valid.sum()),
        newly_invalid=int((f.volume_prior_formula_input_valid&~f.formula_input_valid).sum()),
        expressions=EXPRESSIONS,native_header=HEADER,
        all_29_cached_volume_price_clock_associations_exactly_matched=True,
        joint_time_representation_not_new_raw_information=True,no_new_raw_extraction=True,
        new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True,
        software_compilation_verified=False,native_source_parity_verified=False)
    save_json(INPUTS/'feature_report.json',out)
    for name in ['full_labels.parquet','full_label_report.json','full_label_verification.json']:
        (INPUTS/name).symlink_to((prior.INPUTS/name).resolve())
    return {k:out[k] for k in ['rows','valid','newly_invalid']}


def verify():
    checked();fr=json.loads((INPUTS/'feature_report.json').read_text())
    assert fr['protocol_sha256']==sha(PROTOCOL) and fr['features_sha256']==sha(INPUTS/'features.parquet')
    f=pd.read_parquet(INPUTS/'features.parquet');old,q,v=original_and_caches()
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')],old.drop(columns='formula_input_valid'),check_exact=True)
    np.testing.assert_array_equal(f.volume_prior_formula_input_valid,old.formula_input_valid)
    c=base.conn();c.execute('SET threads=1');c.execute("SET memory_limit='4GB'")
    c.register('original',old[['date','code','hold_input_valid','high_cents']]);c.register('prices',q[['date','code',*PRICE_COLUMNS]]);c.register('volumes',v)
    pc=','.join(PRICE_COLUMNS);vc=','.join(VOLUME_COLUMNS)
    good=' AND '.join(f'isfinite({n}) AND {n}>=0' for n in VOLUME_COLUMNS);total='+'.join(VOLUME_COLUMNS)
    expected=c.sql(f'''WITH b AS(SELECT *,[{pc}] AS p,[{vc}] AS v,
        coalesce(hold_input_valid AND mp_bars=29 AND mp_clocks=29 AND {good} AND ({total})>0,false) AS valid
        FROM original JOIN prices USING(date,code) JOIN volumes USING(date,code)),s AS(SELECT date,code,valid,
        list_extract(v,i) AS volume,CASE WHEN round(list_extract(p,i+1)*100)>high_cents THEN 1 ELSE 0 END AS above,
        CASE WHEN round(list_extract(p,i)*100)>high_cents THEN 1 ELSE 0 END AS previous_above
        FROM b CROSS JOIN range(1,30) t(i))
        SELECT date,code,bool_and(valid) AS valid,
        CASE WHEN bool_and(valid) THEN 100.*sum(volume*above)/sum(volume) END AS MHVA,
        CASE WHEN bool_and(valid) THEN 100.*sum(volume*(above-previous_above))/sum(volume) END AS MHVX
        FROM s GROUP BY date,code ORDER BY date,code''').df();c.close()
    pd.testing.assert_frame_equal(f[['date','code']],expected[['date','code']],check_exact=True)
    np.testing.assert_array_equal(f.volume_input_valid,expected.valid)
    encode=lambda x:np.floor(np.clip(100*x+10000+.000001,0,999999));diff=0.
    for n in NEW_EXPRESSIONS:
        np.testing.assert_allclose(f[n],expected[n],rtol=0,atol=2e-11,equal_nan=True)
        ok=expected.valid;np.testing.assert_array_equal(encode(f.loc[ok,n]),encode(expected.loc[ok,n]))
        diff=max(diff,float(np.max(np.abs(f.loc[ok,n]-expected.loc[ok,n]))))
    final=old.formula_input_valid & expected.valid;np.testing.assert_array_equal(f.formula_input_valid,final)
    names=re.findall(r'(?m)^([A-Z][A-Z0-9]*):=',HEADER)+list(EXPRESSIONS)
    assert len(names)==len({n.casefold() for n in names})
    out=dict(passed=True,feature_report_sha256=sha(INPUTS/'feature_report.json'),rows=len(f),valid=int(final.sum()),max_difference=diff,
        all_52_values_and_all_metadata_unchanged=True,all_29_price_volume_associations_verified=True,
        all_quality_volume_position_signed_crossings_and_encodings_independently_rebuilt=True,
        effective_input_intersection_unchanged=bool(final.equals(old.formula_input_valid)),
        same_quality_controls_required_before_fitting=not final.equals(old.formula_input_valid),
        no_new_raw_extraction=True,new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(INPUTS/'feature_verification.json',out);return out


def native_values(prices,high_cents,volumes,ready=True,outside_price=1000000.,outside_volume=1000000.):
    """Literal replay: prices in yuan, morning high explicitly in integer cents."""
    env=dict(C=np.r_[outside_price,prices,outside_price],V=np.r_[outside_volume,outside_volume,volumes,outside_volume],
        AMHC=float(high_cents),HOLDOK=bool(ready),TIME=np.r_[1419,np.arange(1420,1450),1450],
        ROUND=lambda a:np.floor(a+.5),IF=np.where,DRAWNULL=np.nan,
        REF=lambda a,n:pd.Series(a).shift(int(n)).to_numpy(),
        SUM=lambda a,n:pd.Series(np.asarray(a,float)).rolling(int(n),min_periods=int(n)).sum().to_numpy(),
        COUNT=lambda a,n:pd.Series(np.asarray(a,float)).rolling(int(n),min_periods=int(n)).sum().to_numpy(),
        VALUEWHEN=lambda mask,values:np.asarray(values)[np.flatnonzero(mask)[-1]])
    for line in EXTRA_HEADER.splitlines():
        n,e=line.rstrip(';').split(':=');e=re.sub(r'(?<![<>=!])=(?!=)','==',e).replace(' AND ',' and ')
        env[n]=eval(e,{'__builtins__':{}},env)
    return np.asarray([float(eval(re.sub(r'(?<![<>=!])=(?!=)','==',e),{'__builtins__':{}},env))
                       for e in NEW_EXPRESSIONS.values()])


def native():
    checked();fv=json.loads((INPUTS/'feature_verification.json').read_text())
    assert fv['passed'] and fv['feature_report_sha256']==sha(INPUTS/'feature_report.json')
    f=pd.read_parquet(INPUTS/'features.parquet');old,q,v=original_and_caches()
    encode=lambda x:np.floor(np.clip(100*x+10000+.000001,0,999999));checks=0
    for start in range(0,len(f),50000):
        ff=f.iloc[start:start+50000];cents=np.floor(q.iloc[start:start+50000][PRICE_COLUMNS].to_numpy(float)*100+.5)
        vv=v.iloc[start:start+50000][VOLUME_COLUMNS].to_numpy(float);above=(cents>ff.high_cents.to_numpy()[:,None]).astype(float)
        total=np.zeros(len(ff));positive=np.zeros(len(ff));cross=np.zeros(len(ff))
        for i in range(29):
            total+=vv[:,i];positive+=vv[:,i]*above[:,i+1];cross+=vv[:,i]*(above[:,i+1]-above[:,i])
        # Bind literal SUM products independently of the vectorized production.
        env=dict(MVREADY=ff.volume_input_valid.to_numpy(),MVTOT=total,IF=np.where,DRAWNULL=np.nan,
            VALUEWHEN=lambda mask,values:values,TIME=1449,SUM=lambda a,n:a,
            MHFLAG=positive,V=np.ones(len(ff)),REF=lambda a,n:positive-cross)
        for name,e in NEW_EXPRESSIONS.items():
            with np.errstate(all='ignore'):actual=eval(e.replace('TIME=1449','TIME==1449'),{'__builtins__':{}},env)
            expected=ff[name].to_numpy();np.testing.assert_allclose(actual,expected,rtol=0,atol=2e-11,equal_nan=True)
            ok=np.isfinite(expected);np.testing.assert_array_equal(encode(actual[ok]),encode(expected[ok]));checks+=len(ff)
    index=f.set_index(['date','code']);qs=q.set_index(['date','code']);vs=v.set_index(['date','code'])
    source=prior.prior.INPUTS/'native_input_verification.json';samples=json.loads(source.read_text())['source_samples']
    for sample in samples:
        key=sample['date'],sample['code'];row=index.loc[key];prices=qs.loc[key,PRICE_COLUMNS].to_numpy(float);vol=vs.loc[key,VOLUME_COLUMNS].to_numpy(float)
        actual=native_values(prices,row.high_cents,vol,ready=bool(row.volume_input_valid))
        np.testing.assert_allclose(actual,row[list(NEW_EXPRESSIONS)].to_numpy(float),rtol=0,atol=2e-11,equal_nan=True)
        np.testing.assert_allclose(actual,native_values(prices,row.high_cents,vol,ready=bool(row.volume_input_valid),outside_price=.01,outside_volume=0),rtol=0,atol=2e-11,equal_nan=True)
        np.testing.assert_allclose(actual,native_values(prices*5,row.high_cents*5,vol*10,ready=bool(row.volume_input_valid)),rtol=0,atol=2e-11,equal_nan=True)
    for prices,expected in [([10.]*30,[0.,0.]),([10.01]*30,[100.,0.]),
                            ([10.]+[10.01]*29,[100.,100/29]),([10.01]*29+[10.],[100*28/29,-100/29])]:
        np.testing.assert_allclose(native_values(prices,1000.,np.ones(29)),expected,rtol=0,atol=2e-11)
    with np.errstate(all='ignore'):
        assert np.isnan(native_values([10.01]*30,1000.,np.zeros(29))).all()
        assert np.isnan(native_values([10.01]*30,1000.,np.ones(29),ready=False)).all()
    helper=INPUTS/'morning_volume_inputs.tdx';helper.write_text(EXTRA_HEADER+'\n'.join(f'{n}:={e};' for n,e in NEW_EXPRESSIONS.items())+'\n')
    out=dict(passed=True,feature_report_sha256=sha(INPUTS/'feature_report.json'),feature_verification_sha256=sha(INPUTS/'feature_verification.json'),
        helper_sha256=sha(helper),scalar_checks=checks,sample_days=len(samples),
        reused_morning_source_proof_sha256=sha(source),
        reused_52_native_proof_sha256=sha(prior.INPUTS/'native_input_verification.json'),
        reused_29_volume_source_proof_sha256=sha(Path('data/research/tail_formula_dense_flow/inputs/native_input_verification.json')),
        all_52_values_and_complete_source_caches_unchanged_before_proof_reuse=True,
        all_literal_volume_position_and_signed_crossings_and_encodings_verified=True,
        high_binding_explicitly_integer_cents=True,flat_all_above_up_down_cross_zero_volume_invalid_scale_and_outside_checks_passed=True,
        no_new_raw_extraction=True,software_compilation_verified=False,native_source_parity_verified=False,
        new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(INPUTS/'native_input_verification.json',out);return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','verify','native'])
    print(json.dumps(globals()[p.parse_args().stage](),ensure_ascii=False,indent=2))
