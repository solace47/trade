"""Freeze the fixed removed-capital-input models against their existing controls."""
import json
from pathlib import Path
import subprocess

from trade_research import tail_formula_no_float as study
from trade_research.corporate_cash import save_json, sha
from freeze_tail_formula_bipower_gap import checked_selection


def freeze():
    master = study.checked_sources(); root = study.ROOT
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
    records, checks = [], []
    for fold in ['2024', 'recent', '2025']:
        target = root.parent / (study.STEM + '_' + fold); selected = checked_selection(target)
        chosen = selected.loc[selected.selected]
        row = dict(root=str(target), fold=fold, selected=len(chosen), days=chosen.date.nunique(),
            selection_report_sha256=sha(target / 'selection_report.json'),
            selection_verification_sha256=sha(target / 'selection_verification.json'))
        if fold != '2025':
            m = json.loads((target / 'model_report.json').read_text())
            assert m['feature_names'] == list(study.EXPRESSIONS) and len(m['trees']) == 64
            assert not any(i in [45, 46, 47] for t in m['trees'] for i in t['feature'])
            assert json.loads((target / 'model_verification.json').read_text())['three_financial_placeholders_unused_by_all_tree_nodes']
            row.update(model_report_sha256=sha(target / 'model_report.json'),
                model_verification_sha256=sha(target / 'model_verification.json'),
                score_report_sha256=sha(target / 'score_report.json'),
                score_verification_sha256=sha(target / 'score_verification.json'),
                numeric_core_sha256=sha(target / 'frozen_numeric_core.tdx'),
                financial_constant_node_uses=0)
        records.append(row)
        controls = [Path(p) for p in master['analysis_reuse_controls'][fold]]
        controls.append(root.parent / ('tail_formula_before1000_model_' + fold))
        if fold == '2025':
            controls.append(root.parent / 'tail_formula_before1000/evaluation/tail_formula_float_2025')
        for control in controls:
            checks.append(dict(root=str(target), control=str(control),
                selection_sha256=sha(target / 'selection.parquet'), control_selection_sha256=sha(control / 'selection.parquet'),
                full_frame_equal=selected.equals(checked_selection(control))))
    save_json(output, dict(passed=True, protocol_sha256=sha(study.PROTOCOL), selections=records, full_frame_checks=checks,
        all_two_models_and_three_lists_frozen_together=True, new_group_outcomes_read=False,
        new_2026_prices_read=False, no_exit_rules=True))
    return dict(joint_sha256=sha(output), selections=records, full_frame_checks=checks)


if __name__ == '__main__':
    print(json.dumps(freeze(), ensure_ascii=False, indent=2))
