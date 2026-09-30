"""Finite visible-price exponential trends, with explicit native initialization.

These are two representations of an existing 30-quote cache, not additional
raw information or the conventional full-history MACD indicator.
"""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_dense_path as price
from .corporate_cash import save_json, sha

STEM = 'tail_formula_tail_ewtrend'
ROOT = Path('data/research')/STEM
INPUTS = ROOT/'inputs'
PROTOCOL = Path('config')/(STEM+'_protocol.json')
INTENT = Path('config/tail_formula_tail_ewtrend_intent.json')
META = price.META
PRIOR = price.prior


def coefficients():
    # Basis prices in chronological order. Initialization is a genuine first
    # observed quote, and neither recursion has an unseen historical state.
    basis = np.eye(30); fast = basis[0].copy(); slow = fast.copy(); signal = np.zeros(30)
    for i in range(1,30):
        fast += (basis[i]-fast)*(2/13)
        slow += (basis[i]-slow)*(2/27)
        signal += ((fast-slow)-signal)*.2
    return np.stack([fast-slow, fast-slow-signal])


WEIGHTS = coefficients()
NEW_EXPRESSIONS = {name: 'IF(RTREADY,100*('+ '+'.join(
    f'({float(w):.17g}*RTC{i:02d})' for i,w in enumerate(weights[::-1]))+
    ')/(RMC*VP20),DRAWNULL)' for name,weights in zip(['EWTD','EWTH'],WEIGHTS)}
ARMS = {'control': price.ARMS['control'], 'ew': {**price.ARMS['control'], **NEW_EXPRESSIONS}}
EXPRESSIONS = ARMS['ew']
HEADER = price.HEADER


def checked():
    p = json.loads(PROTOCOL.read_text()); price.checked()
    assert p['arms']==ARMS and p['native_header']==HEADER
    assert p['weights_chronological']==WEIGHTS.tolist() and p['expected_keys']==1815129
    assert p['intent_sha256']==sha(INTENT) and p['threshold']==.995
    assert list(p['folds'])==['2025h1','2025h2'] and not p['new_2026_prices_allowed']
    for file,digest in p['source_hashes'].items():
        assert sha(Path(file))==digest,file
    gate = json.loads(Path(p['conditional_gate']).read_text())
    complete = json.loads(Path(p['conditional_completion']).read_text())
    assert gate['passed'] and not gate['supports_followup'] and complete['passed']
    return p


def measure(prices,atr):
    values = np.asarray(prices,float); atr = np.asarray(atr,float)
    assert values.shape==(len(atr),30)
    cents = np.floor(values*100+.5); centered = cents-cents[:,-1:]
    with np.errstate(all='ignore'):
        return 100*(centered@WEIGHTS.T)/(cents[:,-1:]*atr[:,None])


def prepare():
    checked(); assert not (INPUTS/'feature_report.json').exists()
    INPUTS.mkdir(parents=True,exist_ok=True)
    f = pd.read_parquet(price.INPUTS/'features.parquet',columns=[*META,*ARMS['control'],'dense_path_valid'])
    q = pd.read_parquet(PRIOR.INPUTS/'quotes.parquet')
    pd.testing.assert_frame_equal(f[['date','code']],q[['date','code']],check_exact=True)
    values = measure(q[PRIOR.PRICE_COLUMNS].to_numpy(float),f.V01.to_numpy(float))
    valid = f.dense_path_valid.to_numpy() & np.isfinite(values).all(axis=1)
    values[~valid] = np.nan; f['prior_formula_input_valid'] = f.formula_input_valid
    for i,name in enumerate(NEW_EXPRESSIONS):
        f[name] = values[:,i]
    f['ewtrend_valid'] = valid; f['formula_input_valid'] &= valid
    f.to_parquet(INPUTS/'features.parquet',index=False,compression='zstd')
    report = dict(protocol_sha256=sha(PROTOCOL),features_sha256=sha(INPUTS/'features.parquet'),
        reused_quotes_sha256=sha(PRIOR.INPUTS/'quotes.parquet'),rows=len(f),valid=int(f.formula_input_valid.sum()),
        newly_invalid=int((f.prior_formula_input_valid&~f.formula_input_valid).sum()),
        expressions=EXPRESSIONS,native_header=HEADER,weights_chronological=WEIGHTS.tolist(),
        only_two_finite_representations_of_existing_30_quotes=True,no_new_raw_extraction=True,
        no_standard_macd_or_new_information_claim=True,new_group_outcomes_read=False,
        software_compilation_verified=False,native_source_parity_verified=False,
        new_2026_prices_read=False,no_exit_rules=True)
    save_json(INPUTS/'feature_report.json',report)
    for file in ['full_labels.parquet','full_label_report.json','full_label_verification.json']:
        (INPUTS/file).symlink_to((price.INPUTS/file).resolve())
    return {k:report[k] for k in ['rows','valid','newly_invalid']}


def verify():
    checked(); report = json.loads((INPUTS/'feature_report.json').read_text())
    assert report['protocol_sha256']==sha(PROTOCOL) and report['features_sha256']==sha(INPUTS/'features.parquet')
    f = pd.read_parquet(INPUTS/'features.parquet')
    original = pd.read_parquet(price.INPUTS/'features.parquet',columns=[*META,*ARMS['control'],'dense_path_valid'])
    pd.testing.assert_frame_equal(f[original.columns.drop('formula_input_valid')],original.drop(columns='formula_input_valid'),check_exact=True)
    np.testing.assert_array_equal(f.prior_formula_input_valid,original.formula_input_valid)
    q = pd.read_parquet(PRIOR.INPUTS/'quotes.parquet')
    pd.testing.assert_frame_equal(f[['date','code']],q[['date','code']],check_exact=True)
    # Independent recursion, rather than the production weighted inner product.
    cents = np.floor(q[PRIOR.PRICE_COLUMNS].to_numpy(float)*100+.5)
    fast = cents[:,0]-cents[:,-1]; slow = fast.copy(); signal = np.zeros(len(f))
    for i in range(1,30):
        fast = (11*fast+2*(cents[:,i]-cents[:,-1]))/13
        slow = (25*slow+2*(cents[:,i]-cents[:,-1]))/27
        signal = (4*signal+fast-slow)/5
    with np.errstate(all='ignore'):
        expected = 100*np.column_stack([fast-slow,fast-slow-signal])/(cents[:,-1:]*f.V01.to_numpy()[:,None])
    c = base.conn(); c.register('q',q); c.register('v',f[['date','code','V01']])
    good = ' AND '.join(f'isfinite(pv_c{i}) AND pv_c{i}>0' for i in range(20,50))
    valid = c.sql(f'''SELECT coalesce(pv_bars=30 AND pv_clocks=30 AND pv_good_bars=30
        AND isfinite(V01) AND V01>0 AND {good},false) AS valid FROM q JOIN v USING(date,code)
        ORDER BY date,code''').df().valid.to_numpy(); c.close()
    valid &= np.isfinite(expected).all(axis=1); expected[~valid] = np.nan
    np.testing.assert_array_equal(valid,f.ewtrend_valid)
    encode = lambda x:np.floor(np.clip(100*x+10000+.000001,0,999999))
    difference = 0.
    for i,name in enumerate(NEW_EXPRESSIONS):
        np.testing.assert_allclose(f[name],expected[:,i],rtol=0,atol=2e-11,equal_nan=True)
        ok = valid; np.testing.assert_array_equal(encode(f.loc[ok,name]),encode(expected[ok,i]))
        difference = max(difference,float(np.max(np.abs(f.loc[ok,name]-expected[ok,i]))))
    final = original.formula_input_valid.to_numpy() & valid
    np.testing.assert_array_equal(f.formula_input_valid,final)
    assert np.array_equal(final,original.formula_input_valid), 'Add same-quality controls before any fitting'
    variables = re.findall(r'(?m)^([A-Z][A-Z0-9]*):=',HEADER)+list(EXPRESSIONS)
    assert len(variables)==len({n.casefold() for n in variables})
    out = dict(passed=True,feature_report_sha256=sha(INPUTS/'feature_report.json'),rows=len(f),valid=int(final.sum()),
        all_48_values_and_metadata_unchanged=True,all_complete_recursive_values_and_encodings_rebuilt=True,
        max_difference=difference,effective_input_intersection_unchanged=True,
        explicit_initial_state_without_prior_or_future_prices=True,no_unknown_zero_imputation=True,
        new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(INPUTS/'feature_verification.json',out); return out


def native_values(prices,atr,outside=1000000.):
    env = dict(C=np.r_[outside,prices,outside,outside],VP20=float(atr),Q=float(prices[-1]),DRAWNULL=np.nan,
        TIME=np.r_[1419,np.arange(1420,1450),1450,1451],IF=np.where,ABS=np.abs,ROUND=lambda x:np.floor(x+.5),
        REF=lambda x,n:pd.Series(x).shift(int(n)).to_numpy(),
        COUNT=lambda x,n:pd.Series(np.asarray(x,float)).rolling(int(n),min_periods=int(n)).sum().to_numpy(),
        VALUEWHEN=lambda mask,values:np.asarray(values)[np.flatnonzero(mask)[-1]])
    for line in price.EXTRA_HEADER.splitlines():
        name,expr = line.rstrip(';').split(':='); expr = re.sub(r'(?<![<>=!])=(?!=)','==',expr)
        if name in ['RTCLK','RTREADY']:expr = expr.replace(' AND ',' and ')
        elif name=='RTGOOD':expr = 'VALUEWHEN(TIME==1449,COUNT((C>0) & (ABS(C*100-ROUND(C*100))<=0.01),30))'
        env[name] = eval(expr,{'__builtins__':{}},env)
    return np.asarray([float(eval(expr,{'__builtins__':{}},env)) for expr in NEW_EXPRESSIONS.values()])


def native():
    checked(); fv = json.loads((INPUTS/'feature_verification.json').read_text())
    assert fv['passed'] and fv['feature_report_sha256']==sha(INPUTS/'feature_report.json')
    old_proof = PRIOR.INPUTS/'native_input_verification.json'; prior = json.loads(old_proof.read_text()); assert prior['passed']
    f = pd.read_parquet(INPUTS/'features.parquet'); q = pd.read_parquet(PRIOR.INPUTS/'quotes.parquet')
    pd.testing.assert_frame_equal(f[['date','code']],q[['date','code']],check_exact=True)
    encode = lambda x:np.floor(np.clip(100*x+10000+.000001,0,999999)); checks = 0
    # Literal exported constants and integer anchor independently evaluated for
    # every row, before reusing the old raw-source probe receipts.
    for start in range(0,len(f),50000):
        ff = f.iloc[start:start+50000]; cents = np.floor(q.iloc[start:start+50000][PRIOR.PRICE_COLUMNS].to_numpy()[:,::-1]*100+.5)
        env = dict(RTREADY=ff.ewtrend_valid.to_numpy(),RMC=cents[:,0],VP20=ff.V01.to_numpy(),IF=np.where,DRAWNULL=np.nan)
        env.update({f'RTC{i:02d}':cents[:,i]-cents[:,0] for i in range(30)})
        for name,expr in NEW_EXPRESSIONS.items():
            with np.errstate(all='ignore'):got = eval(expr,{'__builtins__':{}},env)
            expected = ff[name].to_numpy(); np.testing.assert_allclose(got,expected,rtol=0,atol=2e-11,equal_nan=True)
            ok = np.isfinite(expected); np.testing.assert_array_equal(encode(got[ok]),encode(expected[ok])); checks += len(ff)
    samples = prior['samples']; indexed = f.set_index(['date','code']); quotes = q.set_index(['date','code'])
    for item in samples:
        key = (item['date'],item['code']); row = indexed.loc[key]
        p = quotes.loc[key,PRIOR.PRICE_COLUMNS].to_numpy(float)
        got = native_values(p,row.V01); np.testing.assert_allclose(got,row[list(NEW_EXPRESSIONS)],rtol=0,atol=2e-11)
        np.testing.assert_array_equal(encode(got),encode(native_values(p,row.V01,.01)))
        np.testing.assert_array_equal(encode(got),encode(native_values(p*10,row.V01)))
        np.testing.assert_allclose(native_values(np.repeat(p[-1],30),row.V01),[0,0],rtol=0,atol=0)
    helper = INPUTS/'finite_ewtrend_inputs.tdx'
    helper.write_text(price.EXTRA_HEADER+'\n'.join(f'{k}:={v};' for k,v in NEW_EXPRESSIONS.items())+'\n')
    out = dict(passed=True,feature_report_sha256=sha(INPUTS/'feature_report.json'),
        feature_verification_sha256=sha(INPUTS/'feature_verification.json'),helper_sha256=sha(helper),
        scalar_checks=checks,samples=samples,reused_raw_quote_verification_sha256=sha(old_proof),
        all_literal_native_constants_anchors_and_encodings_replayed=True,flat_and_price_scale_and_outside_perturbation_checked=True,
        old_raw_sources_reused_after_complete_cache_values_equal=True,no_new_raw_extraction=True,
        software_compilation_verified=False,native_source_parity_verified=False,
        new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(INPUTS/'native_input_verification.json',out); return {k:v for k,v in out.items() if k!='samples'}


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('stage',choices=['prepare','verify','native'])
    print(json.dumps(globals()[p.parse_args().stage](),ensure_ascii=False,indent=2))
