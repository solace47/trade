"""Analyze unique frozen lists; reuse verified same-window control statistics."""
import argparse
import json
from pathlib import Path
import re
import subprocess

import pandas as pd

from trade_research import tail_formula_boundary_evaluation as evaluation
from trade_research.corporate_cash import save_json,sha
from reuse_tail_formula_selected_analysis import reuse


def run(stem):
    assert re.fullmatch(r'tail_formula_[a-z0-9_]+',stem)
    root=Path('data/research')/stem
    joint=json.loads((root/'joint_selection_freeze.json').read_text())
    assert joint['passed'] and joint['protocol_sha256']==sha(Path('config')/(stem+'_protocol.json'))
    master=json.loads((Path('config')/(stem+'_protocol.json')).read_text())
    assert len(joint['selections'])==3
    path=root/'analysis_dispatch_verification.json'
    if path.exists():
        recorded=json.loads(path.read_text())
        assert recorded['passed'] and recorded['joint_selection_freeze_sha256']==sha(root/'joint_selection_freeze.json')
        for record in recorded['records']:
            target=Path(record['root']);proof=json.loads((target/'analysis_verification.json').read_text())
            assert proof['passed'] and proof['analysis_report_sha256']==record['analysis_report_sha256']==sha(target/'analysis_report.json')
        print(json.dumps(dict(passed=True,existing_dispatch_reused=True,receipt_sha256=sha(path))))
        return
    records=[]
    for fold,argument in [('2024','2024'),('recent','recent'),('2025','combined')]:
        target=Path('data/research')/(stem+'_'+fold)
        frozen=next(x for x in joint['selections'] if x['root']==str(target))
        assert frozen['selection_report_sha256']==sha(target/'selection_report.json')
        assert frozen['selection_verification_sha256']==sha(target/'selection_verification.json')
        selected=pd.read_parquet(target/'selection.parquet')
        controls=[Path('data/research')/('tail_formula_before1000_model_'+fold),
            Path('data/research/tail_formula_before1000/evaluation')/('tail_formula_float_'+fold)]
        controls += [Path(path) for path in master.get('analysis_reuse_controls',{}).get(fold,[])]
        same=None
        for control in controls:
            if (control/'selection.parquet').exists() and selected.equals(pd.read_parquet(control/'selection.parquet')):
                same=control;break
        already=(target/'analysis_report.json').exists()
        if not already:
            if same is not None:
                reuse(target,same);evaluation.attach_labels(target)
                assert sha(target/'full_labels.parquet')==sha(same/'full_labels.parquet')
                report=json.loads((same/'analysis_report.json').read_text())
                assert report['reference_label']=='09:59'
                protocol=Path('config')/(stem+'_'+('combined' if fold=='2025' else fold)+'_protocol.json')
                report.update(protocol_sha256=sha(protocol),selection_report_sha256=sha(target/'selection_report.json'),
                    label_report_sha256=sha(target/'full_label_report.json'),reused_analysis_report_sha256=sha(same/'analysis_report.json'))
                (target/'daily_summary.parquet').symlink_to((same/'daily_summary.parquet').resolve())
                save_json(target/'analysis_report.json',report)
            else:
                with (target/'analysis_command.log').open('w') as log:
                    subprocess.run(['.venv/bin/python','-m','trade_research.'+stem+'_model','analyze','--fold',argument],stdout=log,check=True)
            with (target/'analysis_check_command.log').open('w') as log:
                subprocess.run(['.venv/bin/python','scripts/verify_tail_formula_before1000.py','analysis','--root',str(target)],stdout=log,check=True)
        proof=json.loads((target/'analysis_verification.json').read_text())
        assert proof['passed'] and proof['analysis_report_sha256']==sha(target/'analysis_report.json')
        report=json.loads((target/'analysis_report.json').read_text())
        assert report['selection_report_sha256']==sha(target/'selection_report.json')
        if same is not None:
            original=json.loads((same/'analysis_report.json').read_text())
            assert report['summaries']==original['summaries']
        records.append(dict(fold=fold,root=str(target),equal_selection_source=str(same) if same else None,
            already_verified_analysis=already,new_economic_aggregation_run=not already and same is None,
            analysis_report_sha256=sha(target/'analysis_report.json')))
        primary=[x for x in report['summaries'] if x['arm']=='formula' and x['bps']==15 and not x['sensitive'] and x['period']=='2025'][0]
        print(json.dumps(dict(**records[-1],primary=primary),ensure_ascii=False),flush=True)
    result=dict(passed=True,joint_selection_freeze_sha256=sha(root/'joint_selection_freeze.json'),records=records,
        full_selection_equality_checked_before_evaluation=True,identical_new_lists_reuse_canonical_statistics=True,
        new_2026_prices_read=False,no_exit_rules=True)
    save_json(path,result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stem',required=True)
    run(p.parse_args().stem)
