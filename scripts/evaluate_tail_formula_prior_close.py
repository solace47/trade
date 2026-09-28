"""Jointly freeze the prior-closing inputs study; aggregate each annual arm once."""
import argparse
import json
from pathlib import Path
import subprocess

import pandas as pd

from trade_research import tail_formula_boundary_evaluation as evaluation
from trade_research import tail_formula_prior_close as inputs
from trade_research.corporate_cash import save_json, sha
from evaluate_tail_formula_rising_population import checked_stage
from reuse_tail_formula_selected_analysis import reuse
from compare_tail_formula_same_dates import compare
from compare_tail_formula_shared_unknowns import compare as shared
from audit_tail_formula_reference_coverage import audit

STEM, ROOT, PROTOCOL = inputs.STEM, inputs.ROOT, inputs.PROTOCOL
ORIGINAL = Path('data/research/tail_formula_before1000_model_2025')
FIXED = Path('data/research/tail_formula_before1000/evaluation/tail_formula_float_2025')


def arms():
    r = json.loads((ROOT / 'feature_report.json').read_text())
    return ['joint'] + (['control'] if r['newly_invalid'] else [])


def root_for(arm, fold='2025'):
    return Path('data/research') / (STEM + '_' + arm + '_' + fold)


def run(stage, arm, fold='combined'):
    root = root_for(arm, '2025' if fold == 'combined' else fold)
    root.mkdir(parents=True, exist_ok=True)
    with (root / (stage + '_command.log')).open('w') as log:
        subprocess.run(['.venv/bin/python', '-m', 'trade_research.' + STEM + '_model', stage, '--arm', arm, '--fold', fold], stdout=log, check=True)


def checked_joint():
    inputs.raw.checked()
    joint = json.loads((ROOT / 'joint_selection_freeze.json').read_text())
    assert joint['protocol_sha256'] == sha(PROTOCOL) and joint['arms'] == arms()
    for file, digest in joint['files'].items():
        assert sha(Path(file)) == digest
    return joint


def freeze():
    inputs.raw.checked()
    assert not (ROOT / 'joint_selection_freeze.json').exists()
    assert not any((root_for(a, f) / 'analysis_report.json').exists() for a in arms() for f in ['2024', 'recent', '2025'])
    files, models, selections = {}, [], []
    for arm in arms():
        for fold in ['2024', 'recent']:
            root = root_for(arm, fold)
            r = checked_stage(root, 'model')
            checked_stage(root, 'score')
            checked_stage(root, 'selection')
            assert len(r['feature_names']) == (50 if arm == 'joint' else 48) and len(r['trees']) == 64
            for name in ['model_report.json', 'model_verification.json', 'score_report.json', 'score_verification.json', 'frozen_numeric_core.tdx']:
                files[str(root / name)] = sha(root / name)
            counts = {name: sum(tree['feature'].count(r['feature_names'].index(name)) for tree in r['trees']) for name in inputs.NEW_EXPRESSIONS} if arm == 'joint' else {}
            models.append(dict(arm=arm, fold=fold, rows=r['rows'], days=r['days'], splits_on_new_fields=counts,
                model_report_sha256=sha(root / 'model_report.json')))
        run('freeze', arm)
        run('verify', arm)
        for fold in ['2024', 'recent', '2025']:
            root = root_for(arm, fold)
            r = checked_stage(root, 'selection')
            for name in ['selection_report.json', 'selection_verification.json', 'selection.parquet']:
                files[str(root / name)] = sha(root / name)
            selections.append(dict(arm=arm, fold=fold, root=str(root), selected=r['selected'],
                days=pd.read_parquet(root / 'selection.parquet').query('selected').date.nunique()))
        for fold in ['2024', 'recent', 'combined']:
            file = Path('config') / (STEM + '_' + arm + '_' + fold + '_protocol.json')
            files[str(file)] = sha(file)
    if len(arms()) == 2:
        for fold in ['2024', 'recent']:
            a, b = [json.loads((root_for(arm, fold) / 'model_report.json').read_text()) for arm in arms()]
            for field in ['rows', 'days', 'training_start', 'training_end', 'last_observation', 'parameters']:
                assert a[field] == b[field]
    for file in [PROTOCOL, ROOT / 'feature_report.json', ROOT / 'feature_verification.json', ROOT / 'native_input_verification.json']:
        files[str(file)] = sha(file)
    if len(arms()) == 2:
        for name in ['feature_report.json', 'feature_verification.json']:
            file = inputs.CONTROL_INPUTS / name
            files[str(file)] = sha(file)
    save_json(ROOT / 'joint_selection_freeze.json', dict(passed=True, protocol_sha256=sha(PROTOCOL), arms=arms(), files=files,
        models=models, selections=selections, all_models_and_selections_frozen_together=True,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True))
    return dict(joint_sha256=sha(ROOT / 'joint_selection_freeze.json'), models=models, selections=selections)


def analyze():
    checked_joint()
    committed = subprocess.run(['git', 'show', 'HEAD:docs/selection-formula.md'], capture_output=True, text=True, check=True).stdout
    assert sha(ROOT / 'joint_selection_freeze.json') in committed
    controls, records = [ORIGINAL, FIXED], []
    for arm in arms():
        root = root_for(arm)
        checked_stage(root, 'selection')
        selection = pd.read_parquet(root / 'selection.parquet')
        comparisons = [dict(root=str(c), equal=selection.equals(pd.read_parquet(c / 'selection.parquet'))) for c in controls]
        same = next((Path(item['root']) for item in comparisons if item['equal']), None)
        file = root / 'analysis_reuse_precheck.json'
        gate = dict(passed=True, selection_report_sha256=sha(root / 'selection_report.json'), all_columns_and_keys_checked=True,
            comparisons=comparisons, reuse_source=str(same) if same else None)
        if file.exists():
            assert json.loads(file.read_text()) == gate
        else:
            save_json(file, gate)
        already = (root / 'analysis_report.json').exists()
        if not already:
            if same is not None:
                reuse(root, same)
                evaluation.attach_labels(root)
                r = json.loads((same / 'analysis_report.json').read_text())
                protocol = Path('config') / (STEM + '_' + arm + '_combined_protocol.json')
                r.update(protocol_sha256=sha(protocol), selection_report_sha256=sha(root / 'selection_report.json'),
                    label_report_sha256=sha(root / 'full_label_report.json'), reused_analysis_report_sha256=sha(same / 'analysis_report.json'))
                (root / 'daily_summary.parquet').symlink_to((same / 'daily_summary.parquet').resolve())
                save_json(root / 'analysis_report.json', r)
            else:
                run('analyze', arm)
        if not (root / 'analysis_verification.json').exists():
            with (root / 'analysis_check_command.log').open('w') as log:
                subprocess.run(['.venv/bin/python', 'scripts/verify_tail_formula_before1000.py', 'analysis', '--root', str(root)], stdout=log, check=True)
        r = checked_stage(root, 'analysis')
        assert r['selection_report_sha256'] == sha(root / 'selection_report.json')
        records.append(dict(arm=arm, root=str(root), reused_source=str(same) if same else None,
            new_economic_aggregation_run=not already and same is None, analysis_report_sha256=sha(root / 'analysis_report.json')))
        controls.append(root)
    save_json(ROOT / 'analysis_dispatch_verification.json', dict(passed=True, records=records, full_frame_reuse_checked_first=True,
        only_annual_aggregation_with_both_halves=True, new_2026_prices_read=False, no_exit_rules=True))
    return records


def finish():
    checked_joint()
    reports = {}
    for arm in arms():
        root = root_for(arm)
        checked_stage(root, 'analysis')
        audit(root)
        for name in ['analysis_report.json', 'analysis_verification.json', 'reference_coverage_verification.json']:
            reports[str(root / name)] = sha(root / name)
    p = json.loads(Path('config/tail_formula_weekday_shared_unknowns_protocol.json').read_text())
    p['objective'] = '前日收盘末段量价与同质量48、完整同参数48和旧固定48完整比较，保留全部未知及独有日期。'
    out, specs = root_for('joint'), []
    rights = ([('quality48', root_for('control'))] if len(arms()) == 2 else []) + [('corrected48', ORIGINAL), ('fixed48', FIXED)]
    for name, right in rights:
        path = out / ('same_dates_' + name + '.json')
        compare(out, right, path, intersection_only=True)
        reports[str(path)] = sha(path)
        specs.append(dict(left=str(out), right=str(right), left_analysis_sha256=sha(out / 'analysis_report.json'),
            right_analysis_sha256=sha(right / 'analysis_report.json'), output=str(out / ('shared_unknowns_' + name + '.json'))))
    p['comparisons'] = specs
    path = Path('config') / (STEM + '_shared_unknowns_protocol.json')
    assert not path.exists()
    save_json(path, p)
    p['protocol_sha256'] = sha(path)
    for spec in specs:
        shared(spec, p)
        reports[spec['output']] = sha(Path(spec['output']))
    for file in reports:
        assert json.loads(Path(file).read_text()).get('passed', True)
    path = ROOT / 'complete_results_manifest.json'
    assert not path.exists()
    save_json(path, dict(passed=True, protocol_sha256=sha(PROTOCOL), reports=reports, annual_arms=arms(), comparisons=len(specs),
        new_2026_prices_read=False, no_exit_rules=True))
    return dict(completion_sha256=sha(path))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['freeze', 'analyze', 'finish'])
    print(json.dumps(globals()[p.parse_args().stage](), ensure_ascii=False, indent=2))
