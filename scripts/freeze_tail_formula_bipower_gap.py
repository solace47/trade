"""Freeze both input-extension models and verify the reused controls together."""
import json
from pathlib import Path
import subprocess

import pandas as pd

from trade_research import tail_formula_bipower_gap as study
from trade_research.corporate_cash import save_json, sha


def checked_selection(root):
    report = json.loads((root / 'selection_report.json').read_text())
    proof = json.loads((root / 'selection_verification.json').read_text())
    assert proof['passed'] and proof['selection_report_sha256'] == sha(root / 'selection_report.json')
    assert report['selection_sha256'] == sha(root / 'selection.parquet')
    return pd.read_parquet(root / 'selection.parquet')


def freeze():
    study.checked_sources(); root = study.ROOT
    output = root / 'joint_selection_freeze.json'; assert not output.exists()
    for fold in ['2024', 'recent']:
        target = root.parent / (study.STEM + '_' + fold)
        assert not (target / 'analysis_report.json').exists()
        for kind in ['model', 'score', 'selection']:
            v = json.loads((target / (kind + '_verification.json')).read_text())
            assert v['passed'] and v[kind + '_report_sha256'] == sha(target / (kind + '_report.json'))
    for stage in ['freeze', 'verify']:
        with (root / ('combined_' + stage + '.log')).open('w') as log:
            subprocess.run(['.venv/bin/python', '-m', 'trade_research.' + study.STEM + '_model', stage,
                            '--fold', 'combined'], stdout=log, check=True)
    records, checks, control_checks = [], [], []
    for fold in ['2024', 'recent', '2025']:
        target = root.parent / (study.STEM + '_' + fold); selected = checked_selection(target)
        chosen = selected.loc[selected.selected]
        row = dict(root=str(target), fold=fold, selected=len(chosen), days=chosen.date.nunique(),
            selection_report_sha256=sha(target / 'selection_report.json'),
            selection_verification_sha256=sha(target / 'selection_verification.json'))
        if fold != '2025':
            m = json.loads((target / 'model_report.json').read_text())
            assert m['feature_names'] == list(study.EXPRESSIONS) and len(m['trees']) == 64
            row.update(model_report_sha256=sha(target / 'model_report.json'),
                model_verification_sha256=sha(target / 'model_verification.json'),
                score_report_sha256=sha(target / 'score_report.json'),
                score_verification_sha256=sha(target / 'score_verification.json'),
                numeric_core_sha256=sha(target / 'frozen_numeric_core.tdx'),
                new_input_node_uses=sum(t['feature'].count(48) for t in m['trees']))
        records.append(row)
        canonical = root.parent / ('tail_formula_before1000_model_' + fold)
        constant = root.parent / ('tail_formula_exchange_context_constant_' + fold)
        concentration = root.parent / ('tail_formula_jump_concentration_' + fold)
        baseline = checked_selection(canonical)
        for control in [constant, concentration]:
            pd.testing.assert_frame_equal(checked_selection(control), baseline, check_exact=True)
            control_checks.append(dict(control=str(control), canonical=str(canonical), full_frame_equal=True,
                control_selection_report_sha256=sha(control / 'selection_report.json'),
                canonical_selection_report_sha256=sha(canonical / 'selection_report.json')))
        fixed = root.parent / 'tail_formula_before1000/evaluation' / ('tail_formula_float_' + fold)
        for control in [canonical, constant, concentration, fixed]:
            if not (control / 'selection.parquet').exists():
                continue
            checks.append(dict(root=str(target), control=str(control),
                selection_sha256=sha(target / 'selection.parquet'), control_selection_sha256=sha(control / 'selection.parquet'),
                full_frame_equal=selected.equals(checked_selection(control))))
    save_json(root / 'constant_and_concentration_control_verification.json', dict(passed=True,
        comparisons=control_checks, same_dimension_controls_reuse_canonical_statistics=True))
    save_json(output, dict(passed=True, protocol_sha256=sha(study.PROTOCOL), selections=records, full_frame_checks=checks,
        constant_and_concentration_control_verification_sha256=sha(root / 'constant_and_concentration_control_verification.json'),
        all_two_models_and_three_lists_frozen_together=True, new_group_outcomes_read=False,
        new_2026_prices_read=False, no_exit_rules=True))
    return dict(joint_sha256=sha(output), selections=records, full_frame_checks=checks)


if __name__ == '__main__':
    print(json.dumps(freeze(), ensure_ascii=False, indent=2))
