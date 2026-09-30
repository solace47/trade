"""Absolute-loss adapter; reuse the exact fixed-cutoff freezer and economics."""
import argparse
import importlib.util
import json
from pathlib import Path

from trade_research import tail_formula_absolute_input_calibration as study
from trade_research import tail_formula_chrono_logit48 as binary
from trade_research.corporate_cash import save_json, sha
from find_existing_tail_formula_models import find
from run_tail_formula_absolute_morning import verify_model
import run_tail_formula_paired_study as shared

spec = importlib.util.spec_from_file_location('absolute_input_calibration_shared_runner',
    Path('scripts/run_tail_formula_input_calibration.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.study = study


def protocols():
    p = shared.model.checked()
    metadata = json.loads((study.ROOT / 'metadata_lookup_verification.json').read_text())
    receipts = {str(study.INPUTS / file): sha(study.INPUTS / file) for file in
        ['feature_report.json', 'feature_verification.json', 'native_input_verification.json',
         'full_label_report.json', 'full_label_verification.json']}
    records = []
    for fold, dates in p['folds'].items():
        counts = next(x for x in metadata['records'] if x['fold'] == fold)
        q = dict(master_protocol_sha256=sha(shared.model.PROTOCOL), arm='calibrated', fold=fold, **dates,
            expected_training_rows=counts['training_rows'], expected_training_days=counts['training_days'],
            expected_last_observation=counts['last_observation'], expected_features=50,
            feature_names=list(study.ARMS['calibrated']), parameters=p['parameters'], model_max_depth=3,
            threshold=.995, variant='absolute_logistic', target='known15 opportunity15; binary log-loss; equal date weights',
            input_receipts=receipts, no_training_period_selection=True, new_2026_prices_allowed=False, no_exit_rules=True)
        file = shared.model.fold_protocol('calibrated', fold)
        assert not file.exists()
        file.parent.mkdir(parents=True, exist_ok=True)
        save_json(file, q)
        lookup = find(file, variant='absolute_logistic')
        assert not lookup['matches'], 'Audit and reuse matching absolute models before fitting'
        records.append(dict(arm='calibrated', fold=fold, protocol_sha256=sha(file), lookup=lookup))
    r = dict(passed=True, master_protocol_sha256=sha(shared.model.PROTOCOL), records=records,
        exactly_two_new_models_allowed=True, same_model_control_never_refit=True,
        metadata_lookup_sha256=sha(study.ROOT / 'metadata_lookup_verification.json'),
        no_new_group_outcomes_read=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(study.ROOT / 'prefit_lookup_verification.json', r)
    return dict(prefit_sha256=sha(study.ROOT / 'prefit_lookup_verification.json'), candidate_metadata_matches=0)


def checked_scores(fold):
    q = runner.setup(fold)
    root = shared.base.ROOT
    m = json.loads((root / 'model_report.json').read_text())
    r = json.loads((root / 'score_report.json').read_text())
    assert m['protocol_sha256'] == r['protocol_sha256'] == sha(shared.base.PROTOCOL)
    assert m['feature_names'] == q['feature_names'] == list(study.ARMS['calibrated'])
    assert m['variant'] == q['variant'] == 'absolute_logistic' and len(m['feature_names']) == 50
    assert m['feature_report_sha256'] == r['feature_report_sha256'] == sha(study.INPUTS / 'feature_report.json')
    assert m['label_report_sha256'] == sha(study.INPUTS / 'full_label_report.json')
    assert m['rows'] == q['expected_training_rows'] and m['days'] == q['expected_training_days']
    assert m['last_observation'] == q['expected_last_observation'] < q['calibration_start']
    assert all(m['parameters'][k] == v for k, v in q['parameters'].items())
    assert m['thresholds'][3]['training_quantile'] == .995 and not m.get('no_model_fit_performed', False)
    assert r['model_report_sha256'] == sha(root / 'model_report.json') and r['scores_sha256'] == sha(root / 'scores.parquet')
    for kind in ['model', 'score']:
        v = json.loads((root / (kind + '_verification.json')).read_text())
        assert v['passed'] and v[kind + '_report_sha256'] == sha(root / (kind + '_report.json'))
    _, report, verification = study.calibration_paths(fold)
    c = json.loads(report.read_text())
    v = json.loads(verification.read_text())
    assert v['passed'] and v['calibration_report_sha256'] == sha(report)
    assert c['model_report_sha256'] == sha(root / 'model_report.json')
    assert c['scores_sha256'] == sha(root / 'scores.parquet') and c['quantile'] == .995
    assert c['calibration_end'] == q['evaluation_start']
    return q, m, c


runner.checked_scores = checked_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['protocols', 'model', 'verify_model', 'scores', 'verify_scores',
        'calibrate', 'verify_calibration', 'freeze', 'analyze', 'finish'])
    parser.add_argument('--fold', choices=['2025h1', '2025h2'])
    args = parser.parse_args()
    shared.configure(study.STEM)
    if args.stage == 'protocols':
        result = protocols()
    elif args.stage == 'freeze':
        result = runner.freeze()
    elif args.stage in ['analyze', 'finish']:
        for fold in ['2025h1', '2025h2']:
            checked_scores(fold)
        result = getattr(shared, args.stage)()
    else:
        assert args.fold
        runner.setup(args.fold)
        if args.stage == 'model':
            result = binary.model()
        elif args.stage == 'verify_model':
            result = verify_model()
        elif args.stage == 'scores':
            result = shared.base.scores()
        elif args.stage == 'verify_scores':
            result = shared.verify_scores(expected_expressions=shared.base.EXPRESSIONS)
        else:
            result = getattr(study, args.stage)(args.fold)
    print(json.dumps(result, ensure_ascii=False, indent=2))
