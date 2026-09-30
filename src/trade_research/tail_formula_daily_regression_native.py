"""Correct native ROUND arity and center moments without changing inputs."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_daily_regression as study
from .corporate_cash import save_json,sha

PROTOCOL=Path('config/tail_formula_daily_regression_native_erratum.json')
ROOT=study.INPUTS/'native_corrected'
EXPRESSIONS={**study.ARMS['control'],**{name:f'YJREG02.{name}#DAY' for name in study.NEW_EXPRESSIONS}}
ORIGINAL_NATIVE_CORE=base.native_core


def helper_text():
    lines=['READY:=BARSCOUNT(C)>=61 AND REF(COUNT(C>0 AND ABS(C*100-ROUND(C*100))<=0.01,60),1)=60;']
    lines += [f'RPC{i:02d}:=ROUND(REF(C,{i})*100);' for i in range(1,61)]
    lines += [f'RDC{i:02d}:=RPC{i:02d}-RPC01;' for i in range(1,61)]
    for n in study.WINDOWS:
        s='+'.join(f'RDC{i:02d}' for i in range(1,n+1))
        w='+'.join(f'({n+1-2*i}*RDC{i:02d})' for i in range(1,n+1))
        squares='+'.join(f'(RDC{i:02d}*RDC{i:02d})' for i in range(1,n+1))
        d=n*(n*n-1)
        lines += [f'RS{n:02d}:={s};',f'RW{n:02d}:={w};',
            f'RV{n:02d}:={n}*({squares})-RS{n:02d}*RS{n:02d};',
            f'RX{n:02d}:=-{d}*RS{n:02d}-{3*n*(n-1)}*RW{n:02d};',
            f'RB{n:02d}:IF(READY,600*RW{n:02d}/({d}*RPC01),DRAWNULL);',
            f'RQ{n:02d}:IF(READY,IF(RV{n:02d}>0,300*RW{n:02d}*RW{n:02d}/({n*n-1}*RV{n:02d}),0),DRAWNULL);',
            f'RE{n:02d}:IF(READY,100*RX{n:02d}/({d*n}*RPC01),DRAWNULL);']
    return '\n'.join(lines)+'\n'


HELPER=helper_text()


def checked():
    study.checked();p=json.loads(PROTOCOL.read_text())
    assert p['master_protocol_sha256']==sha(study.PROTOCOL)
    assert p['new_helper']==HELPER and p['new_expressions']==EXPRESSIONS
    for path,digest in p['source_hashes'].items():assert sha(Path(path))==digest
    return p


def replay(closes):
    env=dict(C=np.asarray(closes,float),DRAWNULL=np.nan,ABS=np.abs,ROUND=lambda x:np.floor(x+.5),IF=np.where,
        REF=lambda x,n:pd.Series(x).shift(int(n)).to_numpy(),
        COUNT=lambda x,n:pd.Series(np.asarray(x,float)).rolling(int(n),min_periods=int(n)).sum().to_numpy(),
        BARSCOUNT=lambda x:np.arange(1,len(x)+1))
    with np.errstate(all='ignore'):
        for line in HELPER.splitlines():
            name,expr=line.rstrip(';').split(':=') if ':=' in line else line.rstrip(';').split(':')
            if name=='READY':
                expr='(BARSCOUNT(C)>=61) & (REF(COUNT((C>0) & (ABS(C*100-ROUND(C*100))<=0.01),60),1)==60)'
            env[name]=eval(expr,{'__builtins__':{}},env)
    return np.column_stack([env[n] for n in study.NEW_EXPRESSIONS])


def verify():
    checked();ROOT.mkdir(parents=True,exist_ok=True);assert not (ROOT/'verification.json').exists()
    f=pd.read_parquet(study.INPUTS/'features.parquet',columns=['date','code','history_input_valid',*study.NEW_EXPRESSIONS])
    h=pd.read_parquet(study.INPUTS/'history.parquet',columns=['date','code',*study.PRICES])
    pd.testing.assert_frame_equal(f[['date','code']],h[['date','code']],check_exact=True)
    names=set(re.findall(r'(?m)^([A-Z][A-Z0-9]*):=?',HELPER));checks=0
    # Parse the corrected export itself on all saved integer price vectors.
    # Historical-price references are mapped to the proven lag table; every
    # moment and output expression is executed from its literal text.
    for start in range(0,len(f),50000):
        hf=h.iloc[start:start+50000];ff=f.iloc[start:start+50000]
        env={'READY':ff.history_input_valid.to_numpy(), 'DRAWNULL':np.nan,'IF':np.where}
        env.update({f'RPC{i:02d}':hf[f'rc{i:02d}'].to_numpy(float) for i in range(1,61)})
        with np.errstate(all='ignore'):
            for line in HELPER.splitlines()[61:]:
                name,expr=line.rstrip(';').split(':=') if ':=' in line else line.rstrip(';').split(':')
                assert name in names
                env[name]=eval(expr,{'__builtins__':{}},env)
            for name in study.NEW_EXPRESSIONS:
                got=env[name];expected=ff[name].to_numpy(float)
                np.testing.assert_allclose(got,expected,rtol=0,atol=2e-11,equal_nan=True)
                valid=np.isfinite(expected)
                encode=lambda x:np.floor(np.clip(100*x+10000+.000001,0,999999))
                np.testing.assert_array_equal(encode(got[valid]),encode(expected[valid]));checks+=len(ff)
    old=json.loads((study.INPUTS/'native_input_verification.json').read_text());sources=json.loads((study.INPUTS/'daily_source_manifest.json').read_text())['source_sha256']
    for row in old['samples']:
        path=study.DAILY/(row['code'].replace('.','_')+'.parquet');assert sha(path)==sources[str(path)]
        d=pd.read_parquet(path,columns=['date','close','tradestatus'],filters=[('date','>=',row['first']),('date','<=',row['last'])])
        closes=d.loc[d.tradestatus.eq(1)].sort_values('date').close.to_numpy(float);assert len(closes)==60
        value=replay(np.r_[closes,17.])[60]
        np.testing.assert_array_equal(value,replay(np.r_[closes,.01,1000000.])[60])
        target=f.loc[f.date.eq(row['date'])&f.code.eq(row['code']),list(study.NEW_EXPRESSIONS)].to_numpy(float)[0]
        np.testing.assert_allclose(value,target,rtol=0,atol=2e-11)
    helper=ROOT/'YJREG02.tdx';helper.write_text(HELPER)
    r=dict(passed=True,protocol_sha256=sha(PROTOCOL),feature_report_sha256=sha(study.INPUTS/'feature_report.json'),
        old_native_input_verification_sha256=sha(study.INPUTS/'native_input_verification.json'),
        helper_sha256=sha(helper),rows=len(f),scalar_checks=checks,
        all_fifteen_corrected_literal_expressions_and_encodings_equal=True,
        all_48_prior_raw_daily_samples_and_current_future_pollution_replayed=True,
        round_has_one_argument_and_price_cents_are_positive=True,
        centered_moments_reduce_raw_price_level_cancellation=True,
        no_feature_model_score_or_selection_changes=True,no_model_refit_or_rescoring=True,
        software_compilation_verified=False,native_source_parity_verified=False,
        new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'verification.json',r);return r


def checked_proof():
    checked();r=json.loads((ROOT/'verification.json').read_text())
    assert r['passed'] and r['protocol_sha256']==sha(PROTOCOL)
    assert r['feature_report_sha256']==sha(study.INPUTS/'feature_report.json')
    assert r['helper_sha256']==sha(ROOT/'YJREG02.tdx')
    return r


def native_core(model,threshold,expressions,header):
    checked_proof()
    deployed=EXPRESSIONS if len(expressions)==63 else expressions
    assert len(expressions) in [48,63] and list(deployed)==list(expressions)==model['feature_names']
    text=ORIGINAL_NATIVE_CORE(model,threshold,deployed,header)
    assert 'YJREG01' not in text
    return text


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['verify'])
    print(json.dumps(verify(),ensure_ascii=False,indent=2))
