"""Evaluate already fixed scores without making a count diagnostic a stop gate."""
import argparse
import json
from pathlib import Path

import pandas as pd

from .corporate_cash import save_json,sha
from .tail_formula_additive import native_core
from .tail_formula_complete import EXPRESSIONS,HEADER
from .tail_formula_intraday import conn
from .tail_formula_1000_daily import analyze as common_analysis

PROTOCOL=Path('config/tail_formula_complete_diagnostic_protocol.json')


def locations(variant):
    p=json.loads(PROTOCOL.read_text())['fixed_candidates'][variant]
    return Path('data/research/tail_formula_complete_'+variant+'_diagnostic'),Path(p['source']),p


def freeze(variant):
    root,source,p=locations(variant)
    if (root/'selection_report.json').exists():
        raise ValueError('Do not replace the fixed diagnostic selection')
    root.mkdir(parents=True,exist_ok=True)
    proof=json.loads((source/'selection_verification.json').read_text())
    assert proof['passed'] and proof['selection_report_sha256']==sha(source/'selection_report.json')
    s=json.loads((source/'selection_report.json').read_text())
    m=json.loads((source/'model_report.json').read_text())
    sr=json.loads((source/'score_report.json').read_text())
    assert sr['scores_sha256']==sha(source/'scores.parquet')
    cut=s['thresholds'][p['cut_id']]
    assert cut['threshold']==p['threshold']
    eligible=[x for x in s['thresholds'] if x['days']>=30 and x['known']>=100 and x['mean_selected_per_day']<=20]
    chosen=sorted(eligible,key=lambda x:(-x['lower'],-x['delta'],-x['known'],x['threshold']))[0]
    assert chosen['id']==cut['id']
    f=pd.read_parquet(source/'scores.parquet')
    out=f[['date','code','half','board','decision_shares']].copy()
    out['selected']=f.formula_input_valid&f.score.gt(cut['threshold'])
    out.to_parquet(root/'selection.parquet',index=False,compression='zstd')
    (root/'frozen_numeric_core.tdx').write_text(native_core(m,cut['threshold'],EXPRESSIONS,HEADER))
    r=dict(protocol_sha256=sha(PROTOCOL),source_selection_report_sha256=sha(source/'selection_report.json'),
        model_report_sha256=sha(source/'model_report.json'),score_report_sha256=sha(source/'score_report.json'),
        selection_sha256=sha(root/'selection.parquet'),core_sha256=sha(root/'frozen_numeric_core.tdx'),
        chosen_threshold=cut,selected=int(out.selected.sum()),
        by_half=out.groupby('half').selected.agg(['size','sum']).reset_index().to_dict('records'),
        diagnostic_only=True,relaxed_internal_p95_count_gate=True,new_2025H2_score_groups_read=False,
        new_2026_prices_read=False,no_exit_rules=True,software_compilation_verified=False)
    save_json(root/'selection_report.json',r)
    return r


def verify(variant):
    root,source,p=locations(variant);r=json.loads((root/'selection_report.json').read_text())
    for key,path in [('protocol_sha256',PROTOCOL),('source_selection_report_sha256',source/'selection_report.json'),
                     ('model_report_sha256',source/'model_report.json'),('score_report_sha256',source/'score_report.json'),
                     ('selection_sha256',root/'selection.parquet'),('core_sha256',root/'frozen_numeric_core.tdx')]:
        assert r[key]==sha(path)
    c=conn();c.read_parquet(str(source/'scores.parquet')).create_view('scores')
    expected=c.sql('SELECT date,code,half,board,decision_shares,formula_input_valid AND score>'+format(p['threshold'],'.17e')+
        ' AS selected FROM scores ORDER BY date,code').df()
    actual=pd.read_parquet(root/'selection.parquet').sort_values(['date','code']).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual,expected,check_exact=True,check_dtype=False)
    assert int(actual.selected.sum())==r['selected']
    proof=dict(passed=True,selection_report_sha256=sha(root/'selection_report.json'),rows=len(actual),
        all_selection_flags_rebuilt=True,existing_model_and_threshold_unchanged=True,
        new_2025H2_score_groups_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(root/'selection_verification.json',proof)
    return proof


def analyze(variant):
    return common_analysis(locations(variant)[0],PROTOCOL)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('variant',choices=['relative','risk'])
    p.add_argument('stage',choices=['freeze','verify','analyze'])
    a=p.parse_args()
    print(json.dumps(globals()[a.stage](a.variant),ensure_ascii=False,indent=2))
