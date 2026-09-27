"""Combine verified afternoon paths with four strictly prior price contexts."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .corporate_cash import save_json, sha
from .tail_formula_intraday import ROOT as AFTERNOON, SOURCE, EXPRESSIONS as AFTERNOON_EXPRESSIONS, HEADER as AFTERNOON_HEADER, conn
from . import tail_formula_selective as study

ROOT=Path('data/research/tail_formula_joint')
PROTOCOL=Path('config/tail_formula_joint_protocol.json')
HEADER=AFTERNOON_HEADER+'DD:=DATE<>REF(DATE,1);\nB0:=BARSLAST(DD)+1;\n'
for i in range(1,21):
    HEADER+=f'B{i}:=B{i-1}+REF(BARSLAST(DD)+1,B{i-1});\n'
for i in range(1,22):
    HEADER+=f'DCP{i}:=REF(C,B{i-1});\n'
EXPRESSIONS={**AFTERNOON_EXPRESSIONS,
    'D01':'100*(DCP1/DCP6-1)',
    'D02':'100*(DCP1/DCP21-1)',
    'D03':'100*(Q/(('+ '+'.join(f'DCP{i}' for i in range(1,6))+')/5)-1)',
    'D04':'100*(Q/(('+ '+'.join(f'DCP{i}' for i in range(1,21))+')/20)-1)'}


def features():
    if (ROOT/'feature_report.json').exists():
        raise ValueError('Do not replace frozen joint inputs')
    ROOT.mkdir(parents=True,exist_ok=True)
    for source in [AFTERNOON,SOURCE]:
        proof=json.loads((source/'feature_verification.json').read_text())
        assert proof['passed'] and proof['feature_report_sha256']==sha(source/'feature_report.json')
    ar=json.loads((AFTERNOON/'feature_report.json').read_text())
    dr=json.loads((SOURCE/'feature_report.json').read_text())
    assert ar['features_sha256']==sha(AFTERNOON/'features.parquet')
    assert dr['output_sha256']['features.parquet']==sha(SOURCE/'features.parquet')
    f=pd.read_parquet(AFTERNOON/'features.parquet')
    daily=pd.read_parquet(SOURCE/'features.parquet',columns=['date','code','formula_input_valid','F11','F12','F13','F14'])
    pd.testing.assert_frame_equal(f[['date','code']],daily[['date','code']],check_exact=True)
    for dest,origin in zip(['D01','D02','D03','D04'],['F11','F12','F13','F14']):
        f[dest]=daily[origin]
    f['formula_input_valid'] &= daily.formula_input_valid&np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
    f.to_parquet(ROOT/'features.parquet',index=False,compression='zstd')
    report=dict(protocol_sha256=sha(PROTOCOL),afternoon_report_sha256=sha(AFTERNOON/'feature_report.json'),
        daily_report_sha256=sha(SOURCE/'feature_report.json'),features_sha256=sha(ROOT/'features.parquet'),
        rows=len(f),valid=int(f.formula_input_valid.sum()),expressions=EXPRESSIONS,native_header=HEADER,
        native_history_alignment_verified=False,software_compilation_verified=False,
        outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'feature_report.json',report)
    return {k:v for k,v in report.items() if k not in ['expressions','native_header']}


def verify_features():
    report=json.loads((ROOT/'feature_report.json').read_text())
    assert report['protocol_sha256']==sha(PROTOCOL)
    assert report['afternoon_report_sha256']==sha(AFTERNOON/'feature_report.json')
    assert report['daily_report_sha256']==sha(SOURCE/'feature_report.json')
    assert report['features_sha256']==sha(ROOT/'features.parquet')
    c=conn()
    c.read_parquet(str(SOURCE/'features.parquet')).create_view('daily')
    expected=c.sql('''SELECT date,code,100*(p1/p6-1) AS D01,100*(p1/p21-1) AS D02,
        100*(price_1449/ma5-1) AS D03,100*(price_1449/ma20-1) AS D04 FROM daily ORDER BY date,code''').df()
    f=pd.read_parquet(ROOT/'features.parquet')
    a=pd.read_parquet(AFTERNOON/'features.parquet')
    pd.testing.assert_frame_equal(f[a.columns],a,check_exact=True)
    pd.testing.assert_frame_equal(f[expected.columns],expected,check_dtype=False,rtol=0,atol=2e-12)
    assert f.formula_input_valid.sum()==report['valid'] and len(f)==report['rows']
    assert not f.code.str[3:].str.startswith(('92','688','300','301')).any()
    assert f.isST.eq(0).all() and f.tradestatus.eq(1).all()
    result=dict(passed=True,feature_report_sha256=sha(ROOT/'feature_report.json'),
        rows=len(f),all_prior_price_context_rebuilt=True,afternoon_inputs_unchanged=True,
        native_history_alignment_verified=False,outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'feature_verification.json',result)
    return result


def inputs():
    proof=json.loads((ROOT/'feature_verification.json').read_text())
    report=json.loads((ROOT/'feature_report.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256']==sha(ROOT/'feature_report.json')
    assert report['features_sha256']==sha(ROOT/'features.parquet')
    proof=json.loads((SOURCE/'training_label_verification.json').read_text())
    report=json.loads((SOURCE/'training_label_report.json').read_text())
    assert proof['passed'] and proof['label_report_sha256']==sha(SOURCE/'training_label_report.json')
    assert report['labels_sha256']==sha(SOURCE/'training_labels.parquet')
    f=pd.read_parquet(ROOT/'features.parquet')
    labels=pd.read_parquet(SOURCE/'training_labels.parquet',columns=['date','code','next_date','known15','opportunity15'])
    assert labels.next_date.lt('2024-07-01').all()
    known=labels.loc[labels.known15].copy()
    base=known.groupby('date').opportunity15.mean()
    train=f.loc[f.formula_input_valid].merge(known,on=['date','code'],validate='one_to_one')
    train=train.sort_values(['date','code']).reset_index(drop=True)
    return f,train,known,base


def model():
    return study.model(ROOT,PROTOCOL,ROOT,inputs,EXPRESSIONS)


def verify_model():
    return study.verify_model(ROOT,PROTOCOL,ROOT,inputs,EXPRESSIONS)


def calibrate():
    return study.calibrate(ROOT,PROTOCOL,ROOT,inputs,EXPRESSIONS,HEADER)


def analyze():
    return study.analyze(ROOT,PROTOCOL)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['features','verify_features','model','verify_model','calibrate','analyze'])
    print(json.dumps(globals()[parser.parse_args().stage](),ensure_ascii=False,indent=2))
