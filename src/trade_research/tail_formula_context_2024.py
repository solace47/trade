"""Earlier time fold of the fixed intraday-market relative formula method."""
import argparse
import json
from pathlib import Path

import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_context as context
from . import tail_formula_recent as study
from . import tail_formula_relative as relative
from .corporate_cash import save_json,sha
from .tail_formula_1000_daily import analyze as common_analysis

ROOT=Path('data/research/tail_formula_context_2024')
PROTOCOL=Path('config/tail_formula_context_2024_protocol.json')
COMBINED=Path('data/research/tail_formula_context_2025')
H2=Path('data/research/tail_formula_context_relative')
H2_SELECTION_SHA='ea8befa79a78b46f1fb5a542f647a855d2149852fcf726e192f5bd26c46c71ca'
H2_MODEL_SHA='9818e066a849bf5d715aa395745cb4d210e817bfaa11f882bf490ef4ead886bf'
H2_OUTCOMES_PREVIOUSLY_SEEN=True


def setup():
    context.setup('relative')
    base.ROOT=ROOT;base.PROTOCOL=PROTOCOL;relative.PROTOCOL=PROTOCOL
    study.ROOT=ROOT;study.PROTOCOL=PROTOCOL


def combine():
    if (COMBINED/'selection_report.json').exists():
        raise ValueError('Do not replace the predeclared whole-year selection')
    assert not (ROOT/'analysis_report.json').exists(),'Freeze the whole-year linkage before H1 group outcomes'
    COMBINED.mkdir(parents=True,exist_ok=True)
    if H2_SELECTION_SHA is not None:
        assert sha(H2/'selection_report.json')==H2_SELECTION_SHA
    if H2_MODEL_SHA is not None:
        assert sha(H2/'model_report.json')==H2_MODEL_SHA
    if not H2_OUTCOMES_PREVIOUSLY_SEEN:
        assert not (H2/'analysis_report.json').exists()
    folds=[];frames=[]
    for path,half,start,end in [(ROOT,'2025H1','2025-01-01','2025-07-01'),(H2,'2025H2','2025-07-01','2026-01-01')]:
        s=json.loads((path/'selection_report.json').read_text());p=json.loads((path/'selection_verification.json').read_text())
        m=json.loads((path/'model_report.json').read_text())
        assert p['passed'] and p['selection_report_sha256']==sha(path/'selection_report.json')
        assert s['model_report_sha256']==sha(path/'model_report.json') and s['core_sha256']==sha(path/'frozen_numeric_core.tdx')
        assert s['selection_sha256']==sha(path/'selection.parquet') and m['last_observation']<start
        assert s['chosen_threshold']['training_quantile']==.995
        f=pd.read_parquet(path/'selection.parquet')
        assert f.loc[f.selected,'date'].ge(start).all() and f.loc[f.selected,'date'].lt(end).all()
        frames.append(f)
        core=COMBINED/f'frozen_numeric_core_{half}.tdx';core.write_bytes((path/'frozen_numeric_core.tdx').read_bytes())
        folds.append(dict(half=half,root=str(path),start=start,end=end,selection_report_sha256=sha(path/'selection_report.json'),
            model_report_sha256=sha(path/'model_report.json'),core_sha256=sha(core),selected=int(f.selected.sum())))
    pd.testing.assert_frame_equal(frames[0].drop(columns='selected'),frames[1].drop(columns='selected'),check_exact=True)
    out=frames[0].copy();out['selected']=frames[0].selected|frames[1].selected
    out.to_parquet(COMBINED/'selection.parquet',index=False,compression='zstd')
    r=dict(protocol_sha256=sha(PROTOCOL),folds=folds,selection_sha256=sha(COMBINED/'selection.parquet'),
        selected=int(out.selected.sum()),by_half=out.groupby('half').selected.agg(['size','sum']).reset_index().to_dict('records'),
        formula_method_updates_twice_per_year=True,identical_coefficients_all_year=False,H1_new_group_outcomes_read=False,
        H2_new_group_outcomes_previously_seen=H2_OUTCOMES_PREVIOUSLY_SEEN,year_2025_is_exploratory=True,new_2026_prices_read=False,
        no_exit_rules=True,software_compilation_verified=False)
    save_json(COMBINED/'selection_report.json',r)
    return r


def verify_combined():
    r=json.loads((COMBINED/'selection_report.json').read_text())
    assert r['protocol_sha256']==sha(PROTOCOL) and r['selection_sha256']==sha(COMBINED/'selection.parquet')
    for fold in r['folds']:
        path=Path(fold['root'])
        assert fold['selection_report_sha256']==sha(path/'selection_report.json')
        assert fold['model_report_sha256']==sha(path/'model_report.json')
        assert fold['core_sha256']==sha(COMBINED/f"frozen_numeric_core_{fold['half']}.tdx")
        assert json.loads((path/'model_report.json').read_text())['training_end']<=fold['start']
    c=base.conn();c.read_parquet(str(ROOT/'selection.parquet')).create_view('a');c.read_parquet(str(H2/'selection.parquet')).create_view('b')
    expected=c.sql('''SELECT a.date,a.code,a.half,a.board,a.decision_shares,
        (a.date>='2025-01-01' AND a.date<'2025-07-01' AND a.selected) OR
        (a.date>='2025-07-01' AND a.date<'2026-01-01' AND b.selected) AS selected
        FROM a JOIN b USING(date,code) ORDER BY date,code''').df()
    pd.testing.assert_frame_equal(pd.read_parquet(COMBINED/'selection.parquet'),expected,check_exact=True)
    assert int(expected.selected.sum())==r['selected']==sum(f['selected'] for f in r['folds'])
    proof=dict(passed=True,selection_report_sha256=sha(COMBINED/'selection_report.json'),rows=len(expected),
        all_time_windows_and_selection_flags_rebuilt=True,no_new_group_outcomes_read=True,new_2026_prices_read=False)
    save_json(COMBINED/'selection_verification.json',proof)
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['model','verify_model','scores','freeze','verify','analyze','diagnose','combine','verify_combined','analyze_combined'])
    a=p.parse_args();setup()
    if a.stage in ['combine','verify_combined']:
        r=globals()[a.stage]()
    elif a.stage=='analyze_combined':
        r=common_analysis(COMBINED,PROTOCOL)
    elif a.stage in ['model','verify_model']:
        r=getattr(relative,a.stage)('relative')
    elif a.stage in ['freeze','verify']:
        r=getattr(study,a.stage)()
    else:
        r=getattr(base,a.stage)()
    print(json.dumps(r,ensure_ascii=False,indent=2))
