"""All visible minute highs/lows as an incremental HLCV sequence study.

Runs only if the precommitted price-volume screen does not justify 2024
extension. No minute open data are included; this is not a full OHLCV path.
"""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_dense_flow as prior
from .corporate_cash import MINUTES, save_json, sha

STEM = 'tail_formula_dense_bars'
ROOT = Path('data/research') / STEM
INPUTS = ROOT / 'inputs'
PROTOCOL = Path('config') / (STEM + '_protocol.json')
INTENT = Path('config/tail_formula_dense_bars_intent.json')
META = prior.META
H_COLUMNS = [f'mp_h{i:02d}' for i in range(21,50)]
L_COLUMNS = [f'mp_l{i:02d}' for i in range(21,50)]
C_COLUMNS = [f'mp_c{i:02d}' for i in range(21,50)]
NEW_EXPRESSIONS = {f'D{k}{i:02d}': f'IF(HBREADY,100*(VALUEWHEN(TIME=1449,ROUND(REF({field},{i})*100))-RMC)/(RMC*VP20),DRAWNULL)'
                   for k,field in [('HI','H'),('LO','L')] for i in range(29)}
ARMS = {**prior.ARMS,'bars':{**prior.ARMS['flow'],**NEW_EXPRESSIONS}}
EXTRA_HEADER = ('HBGOOD:=VALUEWHEN(TIME=1449,COUNT(H>=C AND C>=L AND L>0 '
                'AND ABS(H*100-ROUND(H*100))<=0.01 AND ABS(L*100-ROUND(L*100))<=0.01,29));\n'
                'HBREADY:=VRREADY AND HBGOOD=29;\n')
HEADER = prior.HEADER + EXTRA_HEADER


def checked():
    p = json.loads(PROTOCOL.read_text()); prior.checked()
    assert p['arms']==ARMS and p['native_header']==HEADER and p['expected_keys']==1258085
    assert p['intent_sha256']==sha(INTENT) and list(p['folds'])==['2025h1','2025h2']
    assert p['threshold']==.995 and not p['new_2026_prices_allowed']
    for file,digest in p['source_hashes'].items():
        assert sha(Path(file))==digest
    gate = json.loads((prior.ROOT/'expansion_gate.json').read_text())
    assert gate['passed'] and not gate['expand_2024']
    fr = json.loads((prior.INPUTS/'feature_report.json').read_text())
    fv = json.loads((prior.INPUTS/'feature_verification.json').read_text())
    assert fv['passed'] and fv['feature_report_sha256']==sha(prior.INPUTS/'feature_report.json')
    assert fr['features_sha256']==sha(prior.INPUTS/'features.parquet')
    return p


def measure(high,low,last,atr):
    h,l,q,v = map(lambda x:np.asarray(x,float),[high,low,last,atr])
    assert h.shape==l.shape==(len(q),29) and v.shape==q.shape
    with np.errstate(all='ignore'):
        cents = np.floor(np.column_stack([h[:,::-1],l[:,::-1]])*100+.5)
        anchor = np.floor(q*100+.5)
        return 100*(cents-anchor[:,None])/(anchor[:,None]*v[:,None])


def range_inputs():
    return pd.read_parquet(prior.VOLUME/'features.parquet',
        columns=['date','code','mp_bars','mp_clocks',*H_COLUMNS,*L_COLUMNS,*C_COLUMNS])


def prepare():
    checked(); assert not (INPUTS/'feature_report.json').exists(); INPUTS.mkdir(parents=True,exist_ok=True)
    f = pd.read_parquet(prior.INPUTS/'features.parquet',columns=[*META,*ARMS['flow'],'dense_flow_valid'])
    h = range_inputs(); pd.testing.assert_frame_equal(f[['date','code']],h[['date','code']],check_exact=True)
    hh,ll,cc = [h[cols].to_numpy(float) for cols in [H_COLUMNS,L_COLUMNS,C_COLUMNS]]
    good = h.mp_bars.eq(29).to_numpy() & h.mp_clocks.eq(29).to_numpy() & f.dense_flow_valid.to_numpy()
    for x in [hh,ll,cc]:
        good &= np.isfinite(x).all(axis=1) & (x>0).all(axis=1) & (np.abs(x*100-np.floor(x*100+.5))<=.01).all(axis=1)
    good &= (hh>=cc).all(axis=1) & (cc>=ll).all(axis=1)
    values = measure(hh,ll,f.A04.to_numpy(),f.V01.to_numpy())
    good &= np.isfinite(values).all(axis=1); values[~good] = np.nan
    f['prior_formula_input_valid'] = f.formula_input_valid
    for i,name in enumerate(NEW_EXPRESSIONS):
        f[name] = values[:,i]
    f['dense_bars_valid'] = good; f['formula_input_valid'] &= good
    f.to_parquet(INPUTS/'features.parquet',index=False,compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL),features_sha256=sha(INPUTS/'features.parquet'),rows=len(f),
        valid=int(f.formula_input_valid.sum()),newly_invalid=int((f.prior_formula_input_valid&~f.formula_input_valid).sum()),
        expressions=ARMS['bars'],native_header=HEADER,all_58_ordered_high_low_coordinates_retained=True,
        no_minute_open_fields=True,no_new_raw_window_extraction=True,no_new_2023_window_extraction=True,
        new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True,
        software_compilation_verified=False,native_source_parity_verified=False)
    save_json(INPUTS/'feature_report.json',r)
    for file in ['full_labels.parquet','full_label_report.json','full_label_verification.json']:
        (INPUTS/file).symlink_to((prior.INPUTS/file).resolve())
    return {k:r[k] for k in ['rows','valid','newly_invalid']}


def verify():
    checked(); r = json.loads((INPUTS/'feature_report.json').read_text())
    assert r['protocol_sha256']==sha(PROTOCOL) and r['features_sha256']==sha(INPUTS/'features.parquet')
    f = pd.read_parquet(INPUTS/'features.parquet')
    old = pd.read_parquet(prior.INPUTS/'features.parquet',columns=[*META,*ARMS['flow']])
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')],old.drop(columns='formula_input_valid'),check_exact=True)
    np.testing.assert_array_equal(f.prior_formula_input_valid,old.formula_input_valid)
    c = base.conn(); c.register('ranges',range_inputs())
    c.read_parquet(str(prior.INPUTS/'features.parquet')).create_view('parent')
    good = ' AND '.join(f'isfinite({n}) AND {n}>0 AND abs({n}*100-round({n}*100))<=.01'
        for n in [*H_COLUMNS,*L_COLUMNS,*C_COLUMNS])
    good += ' AND '+' AND '.join(f'mp_h{i}>=mp_c{i} AND mp_c{i}>=mp_l{i}' for i in range(21,50))
    fields = ','.join(f'CASE WHEN valid THEN 100*(round(mp_{field}{49-i}*100)-round(A04*100))/(round(A04*100)*V01) END AS D{k}{i:02d}'
        for k,field in [('HI','h'),('LO','l')] for i in range(29))
    expected = c.sql(f'''WITH a AS(SELECT h.*,p.A04,p.V01,p.dense_flow_valid FROM ranges h JOIN parent p USING(date,code)),
        b AS(SELECT *,coalesce(dense_flow_valid AND mp_bars=29 AND mp_clocks=29 AND {good},false) AS valid FROM a)
        SELECT date,code,valid,{fields} FROM b ORDER BY date,code''').df(); c.close()
    pd.testing.assert_frame_equal(f[['date','code']],expected[['date','code']],check_exact=True)
    np.testing.assert_array_equal(f.dense_bars_valid,expected.valid)
    encode = lambda x:np.floor(np.clip(100*x+10000+.000001,0,999999))
    for name in NEW_EXPRESSIONS:
        np.testing.assert_allclose(f[name],expected[name],rtol=0,atol=2e-11,equal_nan=True)
        ok = np.isfinite(expected[name]); np.testing.assert_array_equal(encode(f.loc[ok,name]),encode(expected.loc[ok,name]))
    final = old.formula_input_valid & expected.valid
    np.testing.assert_array_equal(f.formula_input_valid,final)
    names = re.findall(r'(?m)^([A-Z][A-Z0-9]*):=',HEADER)+list(ARMS['bars'])
    assert len(names)==len({n.casefold() for n in names})
    proof = dict(passed=True,feature_report_sha256=sha(INPUTS/'feature_report.json'),rows=len(f),valid=int(final.sum()),
        all_106_parent_values_and_metadata_unchanged=True,all_58_direct_sql_coordinates_and_encodings_equal=True,
        effective_input_intersection_unchanged=bool(final.equals(old.formula_input_valid)),
        high_close_low_order_and_cent_quality_checked=True,new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(INPUTS/'feature_verification.json',proof); return proof


def native_value(high,low,close,last,atr,outside=1000000.):
    env = dict(H=np.r_[outside,high,outside,outside],L=np.r_[outside,low,outside,outside],
        C=np.r_[outside,close,outside,outside],RMC=np.floor(last*100+.5),VP20=atr,VRREADY=True,DRAWNULL=np.nan,
        TIME=np.r_[1420,np.arange(1421,1450),1450,1451],IF=np.where,ABS=np.abs,ROUND=lambda x:np.floor(x+.5),
        REF=lambda x,n:pd.Series(x).shift(int(n)).to_numpy(),
        COUNT=lambda x,n:pd.Series(np.asarray(x,float)).rolling(int(n),min_periods=int(n)).sum().to_numpy(),
        VALUEWHEN=lambda mask,values:np.asarray(values)[np.flatnonzero(mask)[-1]])
    guard = 'VALUEWHEN(TIME==1449,COUNT((H>=C)&(C>=L)&(L>0)&(ABS(H*100-ROUND(H*100))<=0.01)&(ABS(L*100-ROUND(L*100))<=0.01),29))'
    for line in EXTRA_HEADER.splitlines():
        name,expr = line.rstrip(';').split(':='); expr = guard if name=='HBGOOD' else expr.replace(' AND ',' and ')
        expr = re.sub(r'(?<![<>=!])=(?!=)','==',expr)
        env[name] = eval(expr,{'__builtins__':{}},env)
    return np.asarray([float(eval(re.sub(r'(?<![<>=!])=(?!=)','==',expr),{'__builtins__':{}},env)) for expr in NEW_EXPRESSIONS.values()])


def native():
    checked(); fv = json.loads((INPUTS/'feature_verification.json').read_text())
    assert fv['passed'] and fv['feature_report_sha256']==sha(INPUTS/'feature_report.json')
    f = pd.read_parquet(INPUTS/'features.parquet'); h = range_inputs(); checks = 0
    pd.testing.assert_frame_equal(f[['date','code']],h[['date','code']],check_exact=True)
    encode = lambda x:np.floor(np.clip(100*x+10000+.000001,0,999999))
    for start in range(0,len(f),50000):
        ff,hh = f.iloc[start:start+50000],h.iloc[start:start+50000]
        env = dict(HBREADY=ff.dense_bars_valid.to_numpy(),RMC=np.floor(ff.A04.to_numpy()*100+.5),VP20=ff.V01.to_numpy(),IF=np.where,DRAWNULL=np.nan)
        for k,field in [('HI','h'),('LO','l')]:
            for i in range(29):
                name = f'D{k}{i:02d}'; expr = NEW_EXPRESSIONS[name]
                literal = expr.replace(f'VALUEWHEN(TIME=1449,ROUND(REF({field.upper()},{i})*100))','LAGCENTS')
                env['LAGCENTS'] = np.floor(hh[f'mp_{field}{49-i}'].to_numpy()*100+.5)
                with np.errstate(all='ignore'):
                    got = eval(literal,{'__builtins__':{}},env)
                expected = ff[name].to_numpy(); np.testing.assert_allclose(got,expected,rtol=0,atol=2e-11,equal_nan=True)
                ok = np.isfinite(expected); np.testing.assert_array_equal(encode(got[ok]),encode(expected[ok])); checks += len(ff)
    sample = json.loads((prior.INPUTS/'native_input_verification.json').read_text())['samples']; assert len(sample)==32
    indexed = f.set_index(['date','code'])
    for item in sample:
        date,code = item['date'],item['code']; row = indexed.loc[(date,code)]
        file = MINUTES/code[:2].upper()/(code[3:]+'.parquet'); assert sha(file)==item['source_sha256']
        raw = pd.read_parquet(file,columns=['timestamp','high','low','close'],
            filters=[('timestamp','>=',pd.Timestamp(date+' 14:21')),('timestamp','<=',pd.Timestamp(date+' 14:49'))]).sort_values('timestamp')
        assert raw.timestamp.dt.strftime('%H%M').tolist()==[f'14{i:02d}' for i in range(21,50)]
        args = [raw[n].to_numpy(float) for n in ['high','low','close']]+[row.A04,row.V01]
        got = native_value(*args); expected = row[list(NEW_EXPRESSIONS)].to_numpy(float)
        np.testing.assert_allclose(got,expected,rtol=0,atol=2e-11)
        np.testing.assert_array_equal(got,native_value(*args,outside=.01))
    helper = INPUTS/'dense_bars_inputs.tdx'; helper.write_text(EXTRA_HEADER+'\n'.join(f'{k}:={v};' for k,v in NEW_EXPRESSIONS.items())+'\n')
    proof = dict(passed=True,protocol_sha256=sha(PROTOCOL),feature_report_sha256=sha(INPUTS/'feature_report.json'),
        feature_verification_sha256=sha(INPUTS/'feature_verification.json'),helper_sha256=sha(helper),scalar_checks=checks,samples=sample,
        all_58_native_arithmetic_and_encodings_replayed=True,all_32_raw_high_low_windows_replayed=True,
        future_and_outside_window_ranges_excluded=True,no_minute_open_fields=True,new_2026_prices_read=False,no_exit_rules=True,
        software_compilation_verified=False,native_source_parity_verified=False)
    save_json(INPUTS/'native_input_verification.json',proof); return {k:v for k,v in proof.items() if k!='samples'}


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('stage',choices=['prepare','verify','native'])
    print(json.dumps(globals()[parser.parse_args().stage](),ensure_ascii=False,indent=2))
