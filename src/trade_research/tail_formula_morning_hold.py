"""Fixed joint duration above the completed morning high, using proven caches."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_morning_range as prior
from .corporate_cash import save_json, sha

STEM = 'tail_formula_morning_hold'
ROOT = Path('data/research')/STEM
INPUTS = ROOT/'inputs'
PROTOCOL = Path('config')/(STEM+'_input_protocol.json')
INTENT = Path('config')/(STEM+'_intent.json')
META = prior.META
QUOTES = prior.price.prior.INPUTS/'quotes.parquet'
PRICE_COLUMNS = [f'pv_c{i}' for i in range(20,50)]
EXTRA_HEADER = '''HOLDC:=VALUEWHEN(TIME=1449,COUNT(ROUND(C*100)>AMHC,30));
HOLDR:=VALUEWHEN(TIME=1449,IF(ROUND(C*100)>AMHC,IF(HOLDC=30,30,MIN(BARSLAST(ROUND(C*100)<=AMHC),30)),0));
HOLDOK:=AMREADY AND RTREADY;
'''
HEADER = prior.HEADER+EXTRA_HEADER
NEW_EXPRESSIONS = {'MHAC':'IF(HOLDOK,100*HOLDC/30,DRAWNULL)',
                   'MHRC':'IF(HOLDOK,100*HOLDR/30,DRAWNULL)'}
ARMS = {'control':prior.EXPRESSIONS,'hold':{**prior.EXPRESSIONS,**NEW_EXPRESSIONS}}
EXPRESSIONS = ARMS['hold']


def checked():
    p = json.loads(PROTOCOL.read_text()); prior.checked()
    assert p['arms']==ARMS and p['native_header']==HEADER and p['expected_keys']==1258085
    assert p['intent_sha256']==sha(INTENT) and p['no_new_raw_extraction']
    assert not p['new_2026_prices_allowed']
    for file,digest in p['source_hashes'].items(): assert sha(Path(file))==digest,file
    gate = json.loads((prior.ROOT/'morning_range_gate.json').read_text())
    complete = json.loads((prior.ROOT/'complete_results_manifest.json').read_text())
    assert gate['passed'] and not gate['supports_2024_extension'] and complete['passed']
    for file,digest in complete['source_hashes'].items(): assert sha(Path(file))==digest,file
    for root in [prior.INPUTS,prior.price.prior.INPUTS]:
        r = json.loads((root/'feature_report.json').read_text())
        for kind in ['feature','native_input']:
            v = json.loads((root/(kind+'_verification.json')).read_text())
            assert v['passed'] and v['feature_report_sha256']==sha(root/'feature_report.json')
        assert r['features_sha256']==sha(root/'features.parquet')
    r = json.loads((prior.INPUTS/'feature_report.json').read_text())
    assert r['morning_aggregates_sha256']==sha(prior.INPUTS/'morning_aggregates.parquet')
    q = json.loads((prior.price.prior.INPUTS/'feature_report.json').read_text())
    assert q['quotes_sha256']==sha(QUOTES)
    return p


def original_and_quotes():
    f = pd.read_parquet(prior.INPUTS/'features.parquet')
    q = pd.read_parquet(QUOTES,filters=[('date','>=','2024-01-01'),('date','<','2026-01-01')]).reset_index(drop=True)
    pd.testing.assert_frame_equal(f[['date','code']],q[['date','code']],check_exact=True)
    assert len(f)==1258085 and f.date.ge('2024-01-01').all() and f.date.lt('2026-01-01').all()
    return f,q


def measure(cents,high):
    above = cents>np.asarray(high)[:,None]
    count = above.sum(axis=1)
    last_not_above = np.where(~above,np.arange(30),-1).max(axis=1)
    run = 29-last_not_above
    return np.column_stack([100*count/30,100*run/30])


def prepare():
    checked(); assert not (INPUTS/'feature_report.json').exists(); INPUTS.mkdir(parents=True,exist_ok=True)
    f,q = original_and_quotes(); values = q[PRICE_COLUMNS].to_numpy(float)
    good = (f.morning_input_valid & q.pv_bars.eq(30) & q.pv_clocks.eq(30) & q.pv_good_bars.eq(30)
            & f.high_cents.gt(0) & np.isfinite(f.V01) & f.V01.gt(0)).to_numpy()
    good &= (np.isfinite(values)&(values>0)&(np.abs(values-np.floor(values*100+.5)/100)<=.0001)).all(axis=1)
    with np.errstate(all='ignore'): got = measure(np.floor(values*100+.5),f.high_cents.to_numpy())
    got[~good] = np.nan
    f['hold_prior_formula_input_valid'] = f.formula_input_valid
    f['hold_input_valid'] = good; f['formula_input_valid'] &= good
    for i,name in enumerate(NEW_EXPRESSIONS): f[name] = got[:,i]
    f.to_parquet(INPUTS/'features.parquet',index=False,compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL),features_sha256=sha(INPUTS/'features.parquet'),
        reused_morning_feature_report_sha256=sha(prior.INPUTS/'feature_report.json'),quotes_sha256=sha(QUOTES),
        rows=len(f),valid=int(f.formula_input_valid.sum()),
        newly_invalid=int((f.hold_prior_formula_input_valid & ~f.formula_input_valid).sum()),
        expressions=EXPRESSIONS,native_header=HEADER,joint_time_representation_not_new_raw_information=True,
        no_new_raw_extraction=True,no_new_group_outcomes_read=True,new_2026_prices_read=False,no_exit_rules=True,
        software_compilation_verified=False,native_source_parity_verified=False)
    save_json(INPUTS/'feature_report.json',r)
    for file in ['full_labels.parquet','full_label_report.json','full_label_verification.json']:
        (INPUTS/file).symlink_to((prior.INPUTS/file).resolve())
    return {k:r[k] for k in ['rows','valid','newly_invalid']}


def verify():
    checked(); r = json.loads((INPUTS/'feature_report.json').read_text())
    assert r['protocol_sha256']==sha(PROTOCOL) and r['features_sha256']==sha(INPUTS/'features.parquet')
    f = pd.read_parquet(INPUTS/'features.parquet'); old,q = original_and_quotes()
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')],old.drop(columns='formula_input_valid'),check_exact=True)
    np.testing.assert_array_equal(f.hold_prior_formula_input_valid,old.formula_input_valid)
    c = base.conn(); c.execute('SET threads=1'); c.execute("SET memory_limit='4GB'")
    c.register('original',old[['date','code','morning_input_valid','high_cents','V01']]); c.register('quotes',q)
    quality = ' AND '.join(f'isfinite(pv_c{i}) AND pv_c{i}>0 AND abs(pv_c{i}*100-round(pv_c{i}*100))<=.01000000001' for i in range(20,50))
    columns = ','.join(PRICE_COLUMNS)
    expected = c.sql(f'''WITH b AS(SELECT *,[{columns}] AS prices,
        coalesce(morning_input_valid AND high_cents>0 AND isfinite(V01) AND V01>0
        AND pv_bars=30 AND pv_clocks=30 AND pv_good_bars=30 AND {quality},false) AS valid
        FROM original JOIN quotes USING(date,code)), a AS(SELECT date,code,valid,i,
        round(list_extract(prices,i+1)*100)>high_cents AS above FROM b CROSS JOIN range(30) t(i))
        SELECT date,code,bool_and(valid) AS valid,
        CASE WHEN bool_and(valid) THEN 100.*count(*) FILTER(WHERE above)/30 END AS MHAC,
        CASE WHEN bool_and(valid) THEN 100.*(29-coalesce(max(i) FILTER(WHERE NOT above),-1))/30 END AS MHRC
        FROM a GROUP BY date,code ORDER BY date,code''').df(); c.close()
    pd.testing.assert_frame_equal(f[['date','code']],expected[['date','code']],check_exact=True)
    np.testing.assert_array_equal(f.hold_input_valid,expected.valid)
    encode = lambda x:np.floor(np.clip(100*x+10000+.000001,0,999999)); diff = 0.
    for name in NEW_EXPRESSIONS:
        np.testing.assert_allclose(f[name],expected[name],rtol=0,atol=2e-11,equal_nan=True)
        ok = expected.valid; np.testing.assert_array_equal(encode(f.loc[ok,name]),encode(expected.loc[ok,name]))
        diff = max(diff,float(np.max(np.abs(f.loc[ok,name]-expected.loc[ok,name]))))
    final = old.formula_input_valid & expected.valid; np.testing.assert_array_equal(f.formula_input_valid,final)
    names = re.findall(r'(?m)^([A-Z][A-Z0-9]*):=',HEADER)+list(EXPRESSIONS)
    assert len(names)==len({n.casefold() for n in names})
    out = dict(passed=True,feature_report_sha256=sha(INPUTS/'feature_report.json'),rows=len(f),valid=int(final.sum()),
        max_difference=diff,all_50_values_and_all_metadata_unchanged=True,
        all_quality_count_trailing_duration_and_encodings_independently_rebuilt=True,
        effective_input_intersection_unchanged=bool(final.equals(old.formula_input_valid)),
        same_quality_controls_required_before_fitting=not final.equals(old.formula_input_valid),
        original_full_window_sources_and_quality_proofs_reused=True,no_new_raw_extraction=True,
        no_unknown_zero_imputation=True,new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(INPUTS/'feature_verification.json',out); return out


def native_values(prices,high,outside=1000000.,previous_below=True,ready=True):
    prices = np.asarray(prices,float); size = len(prices)
    env = dict(C=np.r_[outside,prices,outside],AMHC=float(high),AMREADY=ready,RTREADY=ready,
        TIME=np.r_[1419,np.arange(1420,1420+size),1450],IF=np.where,MIN=np.minimum,
        ROUND=lambda a:np.floor(a+.5),DRAWNULL=np.nan,
        COUNT=lambda a,n:pd.Series(np.asarray(a,float)).rolling(int(n),min_periods=int(n)).sum().to_numpy(),
        VALUEWHEN=lambda mask,values:np.asarray(values)[np.flatnonzero(mask)[-1]])
    def barslast(mask):
        out = []; last = -1 if previous_below else None
        for i,value in enumerate(mask):
            if value:last=i
            out.append(i-last if last is not None else np.inf)
        return np.asarray(out,float)
    env['BARSLAST'] = barslast
    for line in EXTRA_HEADER.splitlines():
        name,expr = line.rstrip(';').split(':='); expr = re.sub(r'(?<![<>=!])=(?!=)','==',expr).replace(' AND ',' and ')
        env[name] = eval(expr,{'__builtins__':{}},env)
    return np.asarray([float(eval(expr,{'__builtins__':{}},env)) for expr in NEW_EXPRESSIONS.values()])


def native():
    checked(); fv = json.loads((INPUTS/'feature_verification.json').read_text())
    assert fv['passed'] and fv['feature_report_sha256']==sha(INPUTS/'feature_report.json')
    f = pd.read_parquet(INPUTS/'features.parquet'); old,q = original_and_quotes()
    morning_proof = prior.INPUTS/'native_input_verification.json'
    quote_proof = prior.price.prior.INPUTS/'native_input_verification.json'
    assert json.loads(morning_proof.read_text())['passed'] and json.loads(quote_proof.read_text())['passed']
    encode = lambda x:np.floor(np.clip(100*x+10000+.000001,0,999999)); checks = 0
    for start in range(0,len(f),50000):
        ff = f.iloc[start:start+50000]; cents = np.floor(q.iloc[start:start+50000][PRICE_COLUMNS].to_numpy(float)*100+.5)
        above = cents>ff.high_cents.to_numpy()[:,None]
        count = np.zeros(len(ff)); run = np.zeros(len(ff))
        for i in range(30):
            count += above[:,i]; run = np.where(above[:,i],run+1,0)
        env = dict(HOLDOK=ff.hold_input_valid.to_numpy(),HOLDC=count,HOLDR=run,IF=np.where,DRAWNULL=np.nan)
        for name,expr in NEW_EXPRESSIONS.items():
            actual = eval(expr,{'__builtins__':{}},env); expected = ff[name].to_numpy()
            np.testing.assert_allclose(actual,expected,rtol=0,atol=2e-11,equal_nan=True)
            ok = np.isfinite(expected); np.testing.assert_array_equal(encode(actual[ok]),encode(expected[ok])); checks += len(ff)
    indexed = f.set_index(['date','code']); quotes = q.set_index(['date','code'])
    samples = json.loads(morning_proof.read_text())['source_samples']
    for sample in samples:
        key = sample['date'],sample['code']; row = indexed.loc[key]; p = quotes.loc[key,PRICE_COLUMNS].to_numpy(float)
        actual = native_values(p,row.high_cents/100,ready=bool(row.hold_input_valid))
        np.testing.assert_allclose(actual,row[list(NEW_EXPRESSIONS)].to_numpy(float),rtol=0,atol=2e-11,equal_nan=True)
        np.testing.assert_allclose(actual,native_values(p,row.high_cents/100,outside=.01,previous_below=False,ready=bool(row.hold_input_valid)),rtol=0,atol=2e-11,equal_nan=True)
        np.testing.assert_allclose(actual,native_values(p*5,row.high_cents/20,ready=bool(row.hold_input_valid)),rtol=0,atol=2e-11,equal_nan=True)
    cases = [([10.]*30,[0.,0.]),([10.01]*30,[100.,100.]),([9.99]*30,[0.,0.]),
             ([10.01]*29+[10.],[100*29/30,0.]),([10.]+[10.01]*29,[100*29/30,100*29/30]),
             ([10.01]*15+[10.]*10+[10.01]*5,[100*20/30,100*5/30])]
    for prices,expected in cases:
        for state in [True,False]:np.testing.assert_allclose(native_values(prices,10.,previous_below=state),expected,rtol=0,atol=2e-11)
    assert np.isnan(native_values([10.01]*30,10.,ready=False)).all()
    helper = INPUTS/'morning_hold_inputs.tdx'; helper.write_text(EXTRA_HEADER+'\n'.join(f'{k}:={v};' for k,v in NEW_EXPRESSIONS.items())+'\n')
    out = dict(passed=True,feature_report_sha256=sha(INPUTS/'feature_report.json'),feature_verification_sha256=sha(INPUTS/'feature_verification.json'),
        helper_sha256=sha(helper),scalar_checks=checks,sample_days=len(samples),
        reused_morning_native_proof_sha256=sha(morning_proof),reused_full_quote_native_proof_sha256=sha(quote_proof),
        all_50_values_and_complete_source_caches_unchanged_before_proof_reuse=True,
        literal_count_and_capped_barslast_with_all_above_branch_verified=True,
        equality_last_break_all_above_no_above_scale_invalid_and_unseen_history_checks_passed=True,
        no_new_raw_extraction=True,software_compilation_verified=False,native_source_parity_verified=False,
        new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(INPUTS/'native_input_verification.json',out); return out


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('stage',choices=['prepare','verify','native'])
    print(json.dumps(globals()[parser.parse_args().stage](),ensure_ascii=False,indent=2))
