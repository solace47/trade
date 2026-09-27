"""Evaluate the previously discussed fixed candidate in 2025 without retuning it."""
import argparse
import json
from pathlib import Path

import pandas as pd

from .corporate_cash import save_json,sha
from .tail_formula_joint import ROOT as SOURCE,EXPRESSIONS,HEADER
from .tail_formula_1000_analysis import select
from .tail_formula_1000_daily import analyze as common_analysis
from .tail_formula_intraday import conn

ROOT=Path('data/research/tail_formula_joint2025')
PROTOCOL=Path('config/tail_formula_joint2025_protocol.json')


def inputs():
    protocol=json.loads(PROTOCOL.read_text())
    assert protocol['source_report_sha256']==sha(SOURCE/'selection_report.json')
    source=json.loads((SOURCE/'selection_report.json').read_text())
    feature=json.loads((SOURCE/'feature_report.json').read_text())
    assert feature['features_sha256']==sha(SOURCE/'features.parquet')
    proof=json.loads((SOURCE/'feature_verification.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256']==sha(SOURCE/'feature_report.json')
    candidate=next(s for s in source['scores'] if s['node']==protocol['source_node'])
    return candidate,pd.read_parquet(SOURCE/'features.parquet')


def freeze():
    if (ROOT/'selection_report.json').exists():
        raise ValueError('Do not replace the fixed 2025 diagnostic candidate')
    ROOT.mkdir(parents=True,exist_ok=True)
    candidate,f=inputs()
    conditions=candidate['conditions']
    selected=f[['date','code','half','board','decision_shares']].copy()
    selected['selected']=select(f,conditions)
    selected.to_parquet(ROOT/'selection.parquet',index=False,compression='zstd')
    used=list(dict.fromkeys(x['feature'] for x in conditions))
    core=HEADER+'\n'.join(f'{n}:={EXPRESSIONS[n]};' for n in used)+'\nCORE:'+' AND '.join(
        f"{x['feature']}{x['op']}{x['threshold']:.2f}" for x in conditions)+';\n'
    (ROOT/'frozen_numeric_core.tdx').write_text(core)
    report=dict(protocol_sha256=sha(PROTOCOL),source_report_sha256=sha(SOURCE/'selection_report.json'),
        feature_report_sha256=sha(SOURCE/'feature_report.json'),selection_sha256=sha(ROOT/'selection.parquet'),
        core_sha256=sha(ROOT/'frozen_numeric_core.tdx'),conditions=conditions,source_node=candidate['node'],
        selected=int(selected.selected.sum()),by_half=selected.groupby('half').selected.agg(['size','sum']).reset_index().to_dict('records'),
        admission_is_diagnostic_only=True,original_quality_passed=False,new_2025_groups_read=False,
        new_2026_prices_read=False,no_exit_rules=True,multi_day_holding_study=False,software_compilation_verified=False)
    save_json(ROOT/'selection_report.json',report)
    return report


def verify():
    r=json.loads((ROOT/'selection_report.json').read_text())
    assert r['protocol_sha256']==sha(PROTOCOL)
    assert r['selection_sha256']==sha(ROOT/'selection.parquet')
    assert r['core_sha256']==sha(ROOT/'frozen_numeric_core.tdx')
    candidate,f=inputs()
    assert r['conditions']==candidate['conditions'] and r['source_node']==candidate['node']
    c=conn()
    c.register('features',f)
    condition=' AND '.join(f"{x['feature']}{x['op']}{x['threshold']}" for x in r['conditions'])
    expected=c.sql('SELECT date,code,half,board,decision_shares,formula_input_valid AND '+condition+
        ' AS selected FROM features ORDER BY date,code').df()
    actual=pd.read_parquet(ROOT/'selection.parquet')
    pd.testing.assert_frame_equal(actual,expected,check_dtype=False,check_exact=True)
    assert int(actual.selected.sum())==r['selected']
    proof=dict(passed=True,selection_report_sha256=sha(ROOT/'selection_report.json'),all_rows_rebuilt=len(actual),
        selected=int(actual.selected.sum()),unchanged_source_conditions=True,new_2025_groups_read=False,
        new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'selection_verification.json',proof)
    return proof


def analyze():
    return common_analysis(ROOT,PROTOCOL)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['freeze','verify','analyze'])
    print(json.dumps(globals()[p.parse_args().stage](),ensure_ascii=False,indent=2))
