"""Use all 29 visible lag prices, rather than selected path summaries.

The same integer encoding still quantizes these coordinates. The complete
raw path is retained before that encoding; it is not a claim that models
recover every cent or that more coordinates improve next-morning returns.
"""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_tail_regression as prior
from .corporate_cash import MINUTES,save_json,sha

STEM = 'tail_formula_dense_path'
ROOT = Path('data/research')/STEM
INPUTS = ROOT/'inputs'
PROTOCOL = Path('config')/(STEM+'_protocol.json')
WINDOWS = prior.WINDOWS
META = prior.META
NEW_EXPRESSIONS = {f'DPC{i:02d}':f'IF(RTREADY,100*RTC{i:02d}/(RMC*VP20),DRAWNULL)' for i in range(1,30)}
ARMS = {'control':prior.ARMS['control'],'path':{**prior.ARMS['control'],**NEW_EXPRESSIONS}}
EXPRESSIONS = ARMS['path']
EXTRA_HEADER = prior.EXTRA_HEADER.split('RTS05:=')[0]
HEADER = prior.source.HEADER+EXTRA_HEADER


def checked():
    p = json.loads(PROTOCOL.read_text());prior.checked()
    assert p['arms'] == ARMS and p['native_header'] == HEADER and p['expected_keys'] == 1815129
    assert p['threshold'] == .995 and not p['new_2026_prices_allowed']
    for file,digest in p['source_hashes'].items():assert sha(Path(file)) == digest
    r = json.loads((prior.INPUTS/'feature_report.json').read_text())
    v = json.loads((prior.INPUTS/'feature_verification.json').read_text())
    nv = json.loads((prior.INPUTS/'native_input_verification.json').read_text())
    assert v['passed'] and nv['passed'] and v['effective_input_intersection_unchanged']
    assert v['feature_report_sha256'] == nv['feature_report_sha256'] == sha(prior.INPUTS/'feature_report.json')
    assert r['features_sha256'] == sha(prior.INPUTS/'features.parquet')
    assert r['quotes_sha256'] == sha(prior.INPUTS/'quotes.parquet')
    return p


def measure(prices,atr):
    price = np.asarray(prices,float);atr = np.asarray(atr,float)
    assert price.shape == (len(atr),30)
    cents = np.floor(price[:,::-1]*100+.5)
    with np.errstate(all='ignore'):
        return 100*(cents[:,1:]-cents[:,:1])/(cents[:,:1]*atr[:,None])


def prepare():
    checked();assert not (INPUTS/'feature_report.json').exists();INPUTS.mkdir(parents=True,exist_ok=True)
    f = pd.read_parquet(prior.INPUTS/'features.parquet',columns=[*META,*ARMS['control'],'tail_regression_valid'])
    h = pd.read_parquet(prior.INPUTS/'quotes.parquet')
    pd.testing.assert_frame_equal(f[['date','code']],h[['date','code']],check_exact=True)
    values = measure(h[prior.PRICE_COLUMNS].to_numpy(float),f.V01.to_numpy(float))
    good = f.tail_regression_valid.to_numpy() & np.isfinite(values).all(axis=1)
    values[~good] = np.nan
    f['prior_formula_input_valid'] = f.formula_input_valid
    for i,name in enumerate(NEW_EXPRESSIONS):f[name] = values[:,i]
    f['dense_path_valid'] = good;f['formula_input_valid'] &= good
    f.to_parquet(INPUTS/'features.parquet',index=False,compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL),features_sha256=sha(INPUTS/'features.parquet'),
        reused_quote_report_sha256=sha(prior.INPUTS/'feature_report.json'),rows=len(f),valid=int(f.formula_input_valid.sum()),
        newly_invalid=int((f.prior_formula_input_valid&~f.formula_input_valid).sum()),
        expressions=EXPRESSIONS,native_header=HEADER,all_29_coordinates_before_common_encoding=True,
        no_new_raw_window_extraction=True,no_regression_or_polynomial_summary_fields_used=True,
        new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True,
        software_compilation_verified=False,native_source_parity_verified=False)
    save_json(INPUTS/'feature_report.json',r)
    for file in ['full_labels.parquet','full_label_report.json','full_label_verification.json']:
        (INPUTS/file).symlink_to((prior.source.INPUTS/file).resolve())
    return {k:r[k] for k in ['rows','valid','newly_invalid']}


def verify():
    checked();r = json.loads((INPUTS/'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['features_sha256'] == sha(INPUTS/'features.parquet')
    f = pd.read_parquet(INPUTS/'features.parquet')
    old = pd.read_parquet(prior.source.INPUTS/'features.parquet',columns=[*META,*ARMS['control']])
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')],old.drop(columns='formula_input_valid'),check_exact=True)
    np.testing.assert_array_equal(f.prior_formula_input_valid,old.formula_input_valid)
    h = pd.read_parquet(prior.INPUTS/'quotes.parquet');c = base.conn()
    c.register('prices',h);c.register('old',old[['date','code','V01']])
    fields = ','.join(f'CASE WHEN valid THEN 100*(round(pv_c{49-i}*100)-round(pv_c49*100))/(round(pv_c49*100)*V01) END AS DPC{i:02d}' for i in range(1,30))
    good = ' AND '.join(f'isfinite(pv_c{i}) AND pv_c{i}>0' for i in range(20,50))
    expected = c.sql(f'''WITH a AS(SELECT *,coalesce(pv_bars=30 AND pv_clocks=30 AND pv_good_bars=30
        AND isfinite(V01) AND V01>0 AND {good},false) AS valid FROM prices JOIN old USING(date,code))
        SELECT date,code,valid,{fields} FROM a ORDER BY date,code''').df();c.close()
    pd.testing.assert_frame_equal(f[['date','code']],expected[['date','code']],check_exact=True)
    np.testing.assert_array_equal(f.dense_path_valid,expected.valid)
    for name in NEW_EXPRESSIONS:
        np.testing.assert_allclose(f[name],expected[name],rtol=0,atol=2e-11,equal_nan=True)
        good = np.isfinite(expected[name]);encode = lambda a:np.floor(np.clip(100*a+10000+.000001,0,999999))
        np.testing.assert_array_equal(encode(f.loc[good,name]),encode(expected.loc[good,name]))
    final = old.formula_input_valid & expected.valid
    np.testing.assert_array_equal(f.formula_input_valid,final)
    names = re.findall(r'(?m)^([A-Z][A-Z0-9]*):=',HEADER)+list(EXPRESSIONS)
    assert len(names) == len({n.casefold() for n in names})
    proof = dict(passed=True,feature_report_sha256=sha(INPUTS/'feature_report.json'),rows=len(f),valid=int(final.sum()),
        all_48_values_and_all_metadata_unchanged=True,all_29_direct_sql_coordinates_and_encodings_equal=True,
        exact_matched_control_validity=True,effective_input_intersection_unchanged=bool(final.equals(old.formula_input_valid)),
        original_30_clock_and_cent_quality_receipts_reused=True,
        new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(INPUTS/'feature_verification.json',proof);return proof


def native_value(prices,atr,outside=1000000.):
    env = dict(C=np.r_[outside,prices,outside,outside],VP20=float(atr),Q=float(prices[-1]),DRAWNULL=np.nan,
        TIME=np.r_[1419,np.arange(1420,1450),1450,1451],IF=np.where,ABS=np.abs,ROUND=lambda x:np.floor(x+.5),
        REF=lambda x,n:pd.Series(x).shift(int(n)).to_numpy(),
        COUNT=lambda x,n:pd.Series(np.asarray(x,float)).rolling(int(n),min_periods=int(n)).sum().to_numpy(),
        VALUEWHEN=lambda mask,values:np.asarray(values)[np.flatnonzero(mask)[-1]])
    for line in EXTRA_HEADER.splitlines():
        name,expr = line.rstrip(';').split(':=');expr = re.sub(r'(?<![<>=!])=(?!=)','==',expr)
        if name in ['RTCLK','RTREADY']:expr = expr.replace(' AND ',' and ')
        elif name == 'RTGOOD':expr = 'VALUEWHEN(TIME==1449,COUNT((C>0) & (ABS(C*100-ROUND(C*100))<=0.01),30))'
        env[name] = eval(expr,{'__builtins__':{}},env)
    return np.asarray([float(eval(expr,{'__builtins__':{}},env)) for expr in NEW_EXPRESSIONS.values()])


def native():
    checked();v = json.loads((INPUTS/'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(INPUTS/'feature_report.json')
    f = pd.read_parquet(INPUTS/'features.parquet');h = pd.read_parquet(prior.INPUTS/'quotes.parquet')
    pd.testing.assert_frame_equal(f[['date','code']],h[['date','code']],check_exact=True)
    checks = 0
    for start in range(0,len(f),50000):
        ff = f.iloc[start:start+50000];hh = h.iloc[start:start+50000]
        cents = np.floor(hh[prior.PRICE_COLUMNS].to_numpy(float)[:,::-1]*100+.5)
        env = dict(RMC=cents[:,0],VP20=ff.V01.to_numpy(),RTREADY=ff.dense_path_valid.to_numpy(),IF=np.where,DRAWNULL=np.nan)
        env.update({f'RTC{i:02d}':cents[:,i]-cents[:,0] for i in range(1,30)})
        for name,expr in NEW_EXPRESSIONS.items():
            with np.errstate(all='ignore'):got = eval(expr,{'__builtins__':{}},env)
            expected = ff[name].to_numpy(float);np.testing.assert_allclose(got,expected,rtol=0,atol=2e-11,equal_nan=True)
            good = np.isfinite(expected);encode = lambda a:np.floor(np.clip(100*a+10000+.000001,0,999999))
            np.testing.assert_array_equal(encode(got[good]),encode(expected[good]));checks += len(ff)
    sample = json.loads((prior.INPUTS/'native_input_verification.json').read_text())['samples']
    f = f.set_index(['date','code']);receipts = []
    for item in sample:
        date,code = item['date'],item['code'];row = f.loc[(date,code)]
        file = MINUTES/code[:2].upper()/(code[3:]+'.parquet');assert sha(file) == item['source_sha256']
        raw = pd.read_parquet(file,columns=['timestamp','close'],filters=[('timestamp','>=',pd.Timestamp(date+' 14:20')),
            ('timestamp','<=',pd.Timestamp(date+' 14:49'))]).sort_values('timestamp')
        assert raw.timestamp.dt.strftime('%H%M').tolist() == [f'14{i:02d}' for i in range(20,50)]
        got = native_value(raw.close.to_numpy(),row.V01);expected = row[list(NEW_EXPRESSIONS)].to_numpy(float)
        np.testing.assert_allclose(got,expected,rtol=0,atol=2e-11)
        np.testing.assert_array_equal(got,native_value(raw.close.to_numpy(),row.V01,.01))
        receipts.append(item)
    helper = INPUTS/'dense_path_inputs.tdx';helper.write_text(EXTRA_HEADER+'\n'.join(f'{k}:={v};' for k,v in NEW_EXPRESSIONS.items())+'\n')
    proof = dict(passed=True,protocol_sha256=sha(PROTOCOL),feature_report_sha256=sha(INPUTS/'feature_report.json'),
        feature_verification_sha256=sha(INPUTS/'feature_verification.json'),helper_sha256=sha(helper),
        scalar_checks=checks,samples=receipts,all_29_native_arithmetic_and_encodings_replayed=True,
        all_48_same_raw_probe_coordinates_rechecked=True,future_and_outside_window_prices_excluded=True,
        new_2026_prices_read=False,no_exit_rules=True,software_compilation_verified=False,native_source_parity_verified=False)
    save_json(INPUTS/'native_input_verification.json',proof);return {k:v for k,v in proof.items() if k!='samples'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','verify','native'])
    print(json.dumps(globals()[parser.parse_args().stage](),ensure_ascii=False,indent=2))
