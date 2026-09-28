"""Evaluate only the three predeclared annual positive-day population arms."""
import argparse
import json
from pathlib import Path
import subprocess

import pandas as pd

from trade_research import tail_formula_boundary_evaluation as evaluation
from trade_research import tail_formula_rising_population as study
from trade_research.corporate_cash import save_json, sha
from reuse_tail_formula_selected_analysis import reuse
from compare_tail_formula_same_dates import compare
from compare_tail_formula_shared_unknowns import compare as shared
from audit_tail_formula_reference_coverage import audit

STEM, ROOT, PROTOCOL = study.STEM, study.ROOT, study.PROTOCOL
ANNUAL = ['2025', 'control', 'population']
FOLDS = ['2024', 'recent']
FIXED = Path('data/research/tail_formula_before1000/evaluation/tail_formula_float_2025')


def root_for(fold):
    return Path('data/research') / (STEM + '_' + fold)


def checked_stage(root, stage):
    r = json.loads((root / (stage + '_report.json')).read_text())
    v = json.loads((root / (stage + '_verification.json')).read_text())
    assert v['passed'] and v[stage + '_report_sha256'] == sha(root / (stage + '_report.json'))
    if stage == 'selection':
        assert r['selection_sha256'] == sha(root / 'selection.parquet')
    return r


def run(stage, fold):
    root = root_for('2025' if fold == 'combined' else fold)
    root.mkdir(parents=True, exist_ok=True)
    with (root / (stage + '_command.log')).open('w') as log:
        subprocess.run(['.venv/bin/python', '-m', 'trade_research.' + STEM, stage, '--fold', fold], stdout=log, check=True)


def checked_joint():
    study.policy()
    joint = json.loads((ROOT / 'joint_selection_freeze.json').read_text())
    assert joint['protocol_sha256'] == sha(PROTOCOL)
    for file, digest in joint['files'].items():
        assert sha(Path(file)) == digest
    for fold in FOLDS + ANNUAL:
        checked_stage(root_for(fold), 'selection')
    return joint


def freeze():
    study.policy()
    assert not (ROOT / 'joint_selection_freeze.json').exists()
    assert not any((root_for(f) / 'analysis_report.json').exists() for f in FOLDS + ANNUAL)
    files, models, selections = {}, [], []
    for fold in FOLDS:
        root = root_for(fold)
        r = checked_stage(root, 'model')
        checked_stage(root, 'score')
        checked_stage(root, 'selection')
        assert len(r['feature_names']) == 48 and len(r['trees']) == 64
        for name in ['model_report.json', 'model_verification.json', 'score_report.json', 'score_verification.json', 'frozen_numeric_core.tdx']:
            files[str(root / name)] = sha(root / name)
        models.append(dict(fold=fold, rows=r['rows'], days=r['days'], model_report_sha256=sha(root / 'model_report.json')))
    for fold in ['combined', 'control', 'population']:
        run('freeze', fold)
        run('verify', fold)
    for fold in FOLDS + ANNUAL:
        root = root_for(fold)
        r = checked_stage(root, 'selection')
        for name in ['selection_report.json', 'selection_verification.json', 'selection.parquet']:
            files[str(root / name)] = sha(root / name)
        selections.append(dict(fold=fold, root=str(root), selected=r['selected'], days=pd.read_parquet(root / 'selection.parquet').query('selected').date.nunique()))
    for file in [PROTOCOL, study.COMBINED, ROOT / 'feature_report.json', ROOT / 'feature_verification.json'] + [Path('config') / (STEM + '_' + f + '_protocol.json') for f in FOLDS]:
        files[str(file)] = sha(file)
    report = dict(passed=True, protocol_sha256=sha(PROTOCOL), files=files, models=models, selections=selections,
        all_two_models_and_five_selections_frozen_together=True, annual_aggregations_only=ANNUAL,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'joint_selection_freeze.json', report)
    return dict(joint_sha256=sha(ROOT / 'joint_selection_freeze.json'), models=models, selections=selections)


def analyze():
    checked_joint()
    committed = subprocess.run(['git', 'show', 'HEAD:docs/selection-formula.md'], capture_output=True, text=True, check=True).stdout
    assert sha(ROOT / 'joint_selection_freeze.json') in committed, 'Commit the joint receipt before reading group outcomes'
    controls = [study.ORIGINAL, FIXED]
    records = []
    for group in ANNUAL:
        root = root_for(group)
        selected = pd.read_parquet(root / 'selection.parquet')
        comparisons = []
        same = None
        for candidate in controls:
            equal = selected.equals(pd.read_parquet(candidate / 'selection.parquet'))
            comparisons.append(dict(root=str(candidate), all_columns_and_keys_identical=equal))
            if equal and same is None:
                same = candidate
        # Save the full-frame decision before any economic aggregation.
        gate_file = root / 'analysis_reuse_precheck.json'
        if not gate_file.exists():
            save_json(gate_file, dict(passed=True, selection_report_sha256=sha(root / 'selection_report.json'),
                comparisons=comparisons, reuse_source=str(same) if same else None))
        else:
            gate = json.loads(gate_file.read_text())
            assert gate['selection_report_sha256'] == sha(root / 'selection_report.json') and gate['comparisons'] == comparisons
        already = (root / 'analysis_report.json').exists()
        if not already:
            if same is not None:
                reuse(root, same)
                evaluation.attach_labels(root)
                r = json.loads((same / 'analysis_report.json').read_text())
                r.update(protocol_sha256=sha(study.COMBINED), selection_report_sha256=sha(root / 'selection_report.json'),
                    label_report_sha256=sha(root / 'full_label_report.json'), reused_analysis_report_sha256=sha(same / 'analysis_report.json'))
                (root / 'daily_summary.parquet').symlink_to((same / 'daily_summary.parquet').resolve())
                save_json(root / 'analysis_report.json', r)
            else:
                run('analyze', 'combined' if group == '2025' else group)
        if not (root / 'analysis_verification.json').exists():
            with (root / 'analysis_check_command.log').open('w') as log:
                subprocess.run(['.venv/bin/python', 'scripts/verify_tail_formula_before1000.py', 'analysis', '--root', str(root)], stdout=log, check=True)
        r = checked_stage(root, 'analysis')
        assert r['selection_report_sha256'] == sha(root / 'selection_report.json')
        records.append(dict(group=group, root=str(root), reused_source=str(same) if same else None,
            new_economic_aggregation_run=not already and same is None, analysis_report_sha256=sha(root / 'analysis_report.json')))
        controls.append(root)
    save_json(ROOT / 'analysis_dispatch_verification.json', dict(passed=True, joint_sha256=sha(ROOT / 'joint_selection_freeze.json'),
        full_selection_equality_checked_before_evaluation=True, records=records, no_duplicate_half_year_aggregation=True,
        new_2026_prices_read=False, no_exit_rules=True))
    return records


def finish():
    checked_joint()
    reports = {}
    for group in ANNUAL:
        root = root_for(group)
        checked_stage(root, 'analysis')
        audit(root)
        for file in ['analysis_report.json', 'analysis_verification.json', 'reference_coverage_verification.json']:
            reports[str(root / file)] = sha(root / file)
    p = json.loads(Path('config/tail_formula_weekday_shared_unknowns_protocol.json').read_text())
    p['objective'] = '上涨群原48重训与只过滤、全部上涨群、同参数48及旧固定48完整比较；保留所有分母、共同和独有日期。'
    specs = []
    out = root_for('2025')
    for name, right in [('control', root_for('control')), ('population', root_for('population')), ('corrected48', study.ORIGINAL), ('fixed48', FIXED)]:
        same_file, shared_file = out / ('same_dates_' + name + '.json'), out / ('shared_unknowns_' + name + '.json')
        compare(out, right, same_file, intersection_only=True)
        specs.append(dict(left=str(out), right=str(right), left_analysis_sha256=sha(out / 'analysis_report.json'),
            right_analysis_sha256=sha(right / 'analysis_report.json'), output=str(shared_file)))
        reports[str(same_file)] = sha(same_file)
    p['comparisons'] = specs
    file = Path('config') / (STEM + '_shared_unknowns_protocol.json')
    assert not file.exists()
    save_json(file, p)
    p['protocol_sha256'] = sha(file)
    for spec in specs:
        shared(spec, p)
        reports[spec['output']] = sha(Path(spec['output']))
    for file in reports:
        assert json.loads(Path(file).read_text()).get('passed', True)
    complete = ROOT / 'complete_results_manifest.json'
    assert not complete.exists()
    save_json(complete, dict(passed=True, protocol_sha256=sha(PROTOCOL), reports=reports,
        three_annual_groups_and_four_comparisons_complete=True, new_2026_prices_read=False, no_exit_rules=True))
    return dict(completion_sha256=sha(complete))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['freeze', 'analyze', 'finish'])
    print(json.dumps(globals()[p.parse_args().stage](), ensure_ascii=False, indent=2))
