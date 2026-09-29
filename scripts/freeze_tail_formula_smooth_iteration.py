"""Freeze both completed larger-budget fits and all three lists together."""
import json
from pathlib import Path
import subprocess

import pandas as pd

from trade_research import tail_formula_smooth_iteration as study
from trade_research.corporate_cash import save_json, sha


def freeze():
    master = study.checked_sources()
    root = study.ROOT
    output = root / 'joint_selection_freeze.json'
    assert not output.exists()
    for fold in ['2024', 'recent']:
        target = root.parent / (study.STEM + '_' + fold)
        assert not (target / 'analysis_report.json').exists()
        for kind in ['model', 'score', 'selection']:
            proof = json.loads((target / (kind + '_verification.json')).read_text())
            assert proof['passed'] and proof[kind + '_report_sha256'] == sha(target / (kind + '_report.json'))
    root.mkdir(exist_ok=True)
    for stage in ['freeze', 'verify']:
        with (root / ('combined_' + stage + '.log')).open('w') as log:
            subprocess.run(['.venv/bin/python', '-m', 'trade_research.' + study.STEM + '_model',
                            stage, '--fold', 'combined'], stdout=log, check=True)
    records, checks = [], []
    for fold in ['2024', 'recent', '2025']:
        target = root.parent / (study.STEM + '_' + fold)
        report = json.loads((target / 'selection_report.json').read_text())
        proof = json.loads((target / 'selection_verification.json').read_text())
        assert proof['passed'] and proof['selection_report_sha256'] == sha(target / 'selection_report.json')
        assert report['selection_sha256'] == sha(target / 'selection.parquet')
        frame = pd.read_parquet(target / 'selection.parquet')
        selected = frame.loc[frame.selected]
        row = dict(root=str(target), fold=fold, selected=len(selected), days=selected.date.nunique(),
            selection_report_sha256=sha(target / 'selection_report.json'),
            selection_verification_sha256=sha(target / 'selection_verification.json'))
        if fold != '2025':
            model = json.loads((target / 'model_verification.json').read_text())
            row.update(model_report_sha256=sha(target / 'model_report.json'),
                model_verification_sha256=sha(target / 'model_verification.json'),
                parameter_checks=model['parameter_checks'], iterations=model['current_iterations'],
                max_gradient=model['max_gradient'], iteration_budget_reached=model['iteration_budget_reached'])
        records.append(row)
        controls = [root.parent / ('tail_formula_before1000_model_' + fold)]
        controls += [Path(p) for p in master['analysis_reuse_controls'][fold]]
        if fold == '2025':
            controls += [root.parent / 'tail_formula_before1000/evaluation/tail_formula_float_2025']
        for control in controls:
            checks.append(dict(root=str(target), control=str(control),
                selection_sha256=sha(target / 'selection.parquet'), control_selection_sha256=sha(control / 'selection.parquet'),
                full_frame_equal=frame.equals(pd.read_parquet(control / 'selection.parquet'))))
    save_json(output, dict(passed=True, protocol_sha256=sha(study.PROTOCOL), selections=records,
        full_frame_checks=checks, all_three_selections_frozen_together=True,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True))
    return dict(joint_sha256=sha(output), selections=records, full_frame_checks=checks)


if __name__ == '__main__':
    print(json.dumps(freeze(), ensure_ascii=False, indent=2))
