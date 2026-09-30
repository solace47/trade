"""Relative-target adapter; the four-model freezer and six comparisons are reused."""
import argparse
import importlib.util
import json
from pathlib import Path

from trade_research import tail_formula_stock_relative as study
from trade_research.corporate_cash import save_json, sha
from find_existing_tail_formula_models import find
import run_tail_formula_paired_study as shared

spec = importlib.util.spec_from_file_location('stock_relative_shared_runner', Path('scripts/run_tail_formula_stock_holdout.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.study = study
original_setup = runner.setup


def setup(fold, group):
    q = original_setup(fold, group)
    shared.relative.PROTOCOL = shared.base.PROTOCOL
    return q


runner.setup = setup


def protocols():
    p = shared.model.checked()
    metadata = json.loads((study.ROOT / 'metadata_lookup_verification.json').read_text())
    receipts = {str(study.INPUTS / file): sha(study.INPUTS / file) for file in
        ['feature_report.json', 'feature_verification.json', 'native_input_verification.json',
         'full_label_report.json', 'full_label_verification.json']}
    records = []
    for fold, dates in p['folds'].items():
        for group in [0, 1]:
            counts = next(r for r in metadata['records'] if r['fold'] == fold and r['group'] == group)
            arm = 'group' + str(group)
            q = dict(master_protocol_sha256=sha(shared.model.PROTOCOL), arm=arm, fold=fold, **dates,
                training_group=group, calibration_group=1-group,
                expected_training_rows=counts['training_rows'], expected_training_days=counts['training_days'],
                expected_last_observation=counts['last_observation'], expected_features=50,
                feature_names=list(study.ARMS['cross']), parameters=p['parameters'], model_max_depth=3,
                threshold=.995, variant='relative', target='own-group all-known day-centered opportunity15, then valid inputs and equal dates',
                projected_label_directory=str(study.model_root(fold, group) / 'labels'),
                input_receipts=receipts, no_training_period_selection=True,
                new_2026_prices_allowed=False, no_exit_rules=True)
            path = shared.model.fold_protocol(arm, fold)
            assert not path.exists(); path.parent.mkdir(parents=True, exist_ok=True); save_json(path, q)
            lookup = find(path, variant='relative')
            assert not lookup['matches'], 'Audit and reuse matching group-relative models before fitting'
            records.append(dict(arm=arm, fold=fold, protocol_sha256=sha(path), lookup=lookup))
    report = dict(passed=True, master_protocol_sha256=sha(shared.model.PROTOCOL), records=records,
        exactly_four_shared_models_allowed=True, same_model_control_never_refit=True,
        metadata_lookup_sha256=sha(study.ROOT / 'metadata_lookup_verification.json'),
        no_price_score_or_outcome_values_read=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(study.ROOT / 'prefit_lookup_verification.json', report)
    return dict(prefit_sha256=sha(study.ROOT / 'prefit_lookup_verification.json'), candidate_metadata_matches=0)


def fit():
    q = json.loads(shared.base.PROTOCOL.read_text()); root = shared.base.ROOT
    v = json.loads((shared.base.SOURCE / 'full_label_verification.json').read_text())
    r = json.loads((shared.base.SOURCE / 'full_label_report.json').read_text())
    assert v['passed'] and v['label_report_sha256'] == sha(shared.base.SOURCE / 'full_label_report.json')
    assert r['labels_sha256'] == sha(shared.base.SOURCE / 'full_labels.parquet')
    assert r['original_labels_sha256'] == sha(study.INPUTS / 'full_labels.parquet')
    assert r['training_group'] == q['training_group'] and r['fold'] == q['fold'] and r['fields'] == study.LABEL_FIELDS
    shared.relative.model('relative')
    m = json.loads((root / 'model_report.json').read_text())
    m.update(training_group=q['training_group'], original_label_report_sha256=sha(study.INPUTS / 'full_label_report.json'),
        stock_group_label_projection_verified=True, only_own_group_known_labels_in_day_baseline=True,
        year_2025_is_exploratory=True, uses_exposed_2025_training_labels=q['training_end'] > '2025-01-01')
    save_json(root / 'model_report.json', m)
    return {k: value for k, value in m.items() if k != 'trees'}


def checked_scores(fold, group):
    q = setup(fold, group); root = shared.base.ROOT
    m = json.loads((root / 'model_report.json').read_text()); s = json.loads((root / 'score_report.json').read_text())
    assert m['protocol_sha256'] == s['protocol_sha256'] == sha(shared.base.PROTOCOL)
    assert m['variant'] == q['variant'] == 'relative' and m['training_group'] == group
    assert m['only_own_group_known_labels_in_day_baseline'] and m['feature_names'] == q['feature_names']
    assert len(m['feature_names']) == 50 and s['rows'] == 1258085 and s['valid'] == 1117397
    assert m['feature_report_sha256'] == s['feature_report_sha256'] == sha(study.INPUTS / 'feature_report.json')
    assert m['label_report_sha256'] == sha(shared.base.SOURCE / 'full_label_report.json')
    assert m['rows'] == q['expected_training_rows'] and m['days'] == q['expected_training_days']
    assert m['last_observation'] == q['expected_last_observation'] < q['evaluation_start']
    assert all(m['parameters'][k] == value for k, value in q['parameters'].items())
    assert s['model_report_sha256'] == sha(root / 'model_report.json') and s['scores_sha256'] == sha(root / 'scores.parquet')
    for kind in ['model', 'score', 'calibration']:
        v = json.loads((root / (kind + '_verification.json')).read_text())
        assert v['passed'] and v[kind + '_report_sha256'] == sha(root / (kind + '_report.json'))
    r = json.loads((root / 'calibration_report.json').read_text())
    assert r['model_report_sha256'] == sha(root / 'model_report.json') and r['scores_sha256'] == sha(root / 'scores.parquet')
    assert r['quantile'] == .995 and r['training_group'] == group and r['calibration_group'] == 1-group
    return q, m, r


runner.checked_scores = checked_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['protocols', 'project', 'model', 'verify_model', 'scores', 'verify_scores',
        'calibrate', 'verify_calibration', 'freeze', 'analyze', 'finish'])
    parser.add_argument('--fold', choices=['2025h1', '2025h2']); parser.add_argument('--group', type=int, choices=[0, 1])
    args = parser.parse_args(); shared.configure(study.STEM)
    if args.stage == 'protocols': result = protocols()
    elif args.stage == 'freeze': result = runner.freeze()
    elif args.stage in ['analyze', 'finish']:
        for fold in ['2025h1', '2025h2']:
            for group in [0, 1]: checked_scores(fold, group)
        result = getattr(shared, args.stage)()
    else:
        assert args.fold and args.group is not None; setup(args.fold, args.group)
        if args.stage == 'project': result = study.project_labels(args.fold, args.group)
        elif args.stage == 'model': result = fit()
        elif args.stage == 'verify_model': result = shared.relative.verify_model('relative')
        elif args.stage == 'scores': result = shared.base.scores()
        elif args.stage == 'verify_scores': result = shared.verify_scores(expected_expressions=shared.base.EXPRESSIONS)
        else: result = getattr(study, args.stage)(args.fold, args.group)
    print(json.dumps(result, ensure_ascii=False, indent=2))
