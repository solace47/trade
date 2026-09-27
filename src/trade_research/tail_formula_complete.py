"""Complete the native feature set with the verified truncated daily context."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_relative as relative
from .tail_formula_joint import ROOT as JOINT,EXPRESSIONS as JOINT_EXPRESSIONS,HEADER as JOINT_HEADER
from .corporate_cash import save_json,sha

ROOT=Path('data/research/tail_formula_complete')
PROTOCOL=Path('config/tail_formula_complete_protocol.json')
DAILY_NAMES=['F04','F05','F06','F07','F08','F09','F15','F16']
HEADER=JOINT_HEADER+'''DH:=VALUEWHEN(TIME=1449,HHV(H,B0));
DL:=VALUEWHEN(TIME=1449,LLV(L,B0));
DV:=VALUEWHEN(TIME=1449,SUM(V,B0));
DA:=VALUEWHEN(TIME=1449,SUM(AMOUNT,B0));
PV5:=REF(SUM(V,B4),B0)/5;
PH20:=REF(HHV(H,B19),B0);
PL20:=REF(LLV(L,B19),B0);
'''
EXPRESSIONS={**JOINT_EXPRESSIONS,
    'C01':'100*(Q-DL)/MAX(DH-DL,0.01)',
    'C02':'100*(DH-DL)/DCP1',
    'C03':'100*(DH-Q)/DCP1',
    'C04':'100*(MIN(DYNAINFO(4),Q)-DL)/DCP1',
    'C05':'DV/PV5',
    'C06':'DA/100000000',
    'C07':'100*(Q/PH20-1)',
    'C08':'100*(Q/PL20-1)'}


def features():
    if (ROOT/'feature_report.json').exists():
        raise ValueError('Do not replace frozen complete formula inputs')
    ROOT.mkdir(parents=True,exist_ok=True)
    for source in [JOINT,base.SOURCE]:
        p=json.loads((source/'feature_verification.json').read_text())
        assert p['passed'] and p['feature_report_sha256']==sha(source/'feature_report.json')
    j=json.loads((JOINT/'feature_report.json').read_text());d=json.loads((base.SOURCE/'feature_report.json').read_text())
    assert j['features_sha256']==sha(JOINT/'features.parquet')
    assert d['output_sha256']['features.parquet']==sha(base.SOURCE/'features.parquet')
    f=pd.read_parquet(JOINT/'features.parquet')
    daily=pd.read_parquet(base.SOURCE/'features.parquet',columns=['date','code','formula_input_valid']+DAILY_NAMES)
    pd.testing.assert_frame_equal(f[['date','code']],daily[['date','code']],check_exact=True)
    for i,name in enumerate(DAILY_NAMES,1):
        f[f'C{i:02d}']=daily[name]
    f['formula_input_valid'] &= daily.formula_input_valid & np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
    f.to_parquet(ROOT/'features.parquet',index=False,compression='zstd')
    r=dict(protocol_sha256=sha(PROTOCOL),joint_report_sha256=sha(JOINT/'feature_report.json'),
        daily_report_sha256=sha(base.SOURCE/'feature_report.json'),features_sha256=sha(ROOT/'features.parquet'),
        rows=len(f),valid=int(f.formula_input_valid.sum()),expressions=EXPRESSIONS,native_header=HEADER,
        native_history_alignment_verified=False,software_compilation_verified=False,outcomes_read=False,
        new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'feature_report.json',r)
    return {k:v for k,v in r.items() if k not in ['expressions','native_header']}


def verify_features():
    r=json.loads((ROOT/'feature_report.json').read_text())
    assert r['protocol_sha256']==sha(PROTOCOL)
    assert r['joint_report_sha256']==sha(JOINT/'feature_report.json')
    assert r['daily_report_sha256']==sha(base.SOURCE/'feature_report.json')
    assert r['features_sha256']==sha(ROOT/'features.parquet')
    c=base.conn();c.read_parquet(str(base.SOURCE/'features.parquet')).create_view('daily')
    expected=c.sql('''SELECT date,code,100*(price_1449-low_1449)/greatest(high_1449-low_1449,.01) AS C01,
        100*(high_1449-low_1449)/p1 AS C02,100*(high_1449-price_1449)/p1 AS C03,
        100*(least(daily_open,price_1449)-low_1449)/p1 AS C04,volume_1449/v5 AS C05,amount_1449/1e8 AS C06,
        100*(price_1449/h20-1) AS C07,100*(price_1449/l20-1) AS C08 FROM daily ORDER BY date,code''').df()
    f=pd.read_parquet(ROOT/'features.parquet');j=pd.read_parquet(JOINT/'features.parquet')
    pd.testing.assert_frame_equal(f[j.columns],j,check_exact=True)
    pd.testing.assert_frame_equal(f[expected.columns],expected,check_dtype=False,rtol=0,atol=2e-12)
    assert len(f)==r['rows'] and int(f.formula_input_valid.sum())==r['valid']
    assert not f.code.str[3:].str.startswith(('92','688','300','301')).any()
    assert f.isST.eq(0).all() and f.tradestatus.eq(1).all()
    proof=dict(passed=True,feature_report_sha256=sha(ROOT/'feature_report.json'),rows=len(f),
        eight_new_fields_rebuilt_from_verified_daily_components=True,all_24_previous_fields_unchanged=True,
        native_history_alignment_verified=False,software_compilation_verified=False,outcomes_read=False,
        new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'feature_verification.json',proof)
    return proof


def setup(variant):
    base.ROOT=Path('data/research/tail_formula_complete_'+variant)
    base.PROTOCOL=PROTOCOL;relative.PROTOCOL=PROTOCOL
    base.FEATURES=ROOT;base.EXPRESSIONS=EXPRESSIONS;base.HEADER=HEADER


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['features','verify_features','model','verify_model','scores','calibrate','analyze','diagnose'])
    p.add_argument('--variant',choices=['relative','risk'],default='relative')
    a=p.parse_args()
    if a.stage in ['features','verify_features']:
        result=globals()[a.stage]()
    else:
        setup(a.variant)
        result=getattr(relative,a.stage)(a.variant) if a.stage in ['model','verify_model'] else getattr(base,a.stage)()
    print(json.dumps(result,ensure_ascii=False,indent=2))
