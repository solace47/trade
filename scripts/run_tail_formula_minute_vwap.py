"""Allow a proven evaluation-only validity delta without refitting identical controls."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trade_research.corporate_cash import save_json, sha
import run_tail_formula_paired_study as shared

STEM = 'tail_formula_minute_vwap'


def checked():
    study = shared.study; study.checked()
    p = json.loads(shared.model.PROTOCOL.read_text())
    assert p['input_protocol_sha256'] == sha(study.PROTOCOL) and p['arms'] == study.ARMS
    assert p['old_controls'] == {f: str(r) for f, r in shared.model.OLD.items()} and p['threshold'] == .995
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest, file
    fr = json.loads((study.INPUTS / 'feature_report.json').read_text())
    fv = json.loads((study.INPUTS / 'feature_verification.json').read_text())
    nv = json.loads((study.INPUTS / 'native_input_verification.json').read_text())
    assert fv['passed'] and nv['passed'] and fv['feature_report_sha256'] == nv['feature_report_sha256'] == sha(study.INPUTS / 'feature_report.json')
    assert fr['features_sha256'] == sha(study.INPUTS / 'features.parquet')
    assert fv['valid'] == p['expected_valid_score_rows'] == 1117396
    assert not fv['effective_input_intersection_unchanged']
    delta = json.loads(Path(p['domain_delta_verification']).read_text())
    assert delta['passed'] and delta['feature_verification_sha256'] == sha(study.INPUTS / 'feature_verification.json')
    assert delta['all_delta_keys_strictly_after_both_training_ends'] and delta['excluded_keys_not_in_original_q995_annual_selection']
    assert len(delta['excluded_keys']) == 1
    return p


def reuse(fold):
    q = shared.model.setup('control', fold)
    study = shared.study; base = shared.base; root = base.ROOT; parent = shared.model.OLD[fold]
    assert not (root / 'model_report.json').exists()
    cols = [*study.META, *study.ARMS['control']]
    new = pd.read_parquet(study.INPUTS / 'features.parquet', columns=cols)
    old = pd.read_parquet(study.prior.INPUTS / 'features.parquet', columns=cols)
    unchanged = [c for c in cols if c != 'formula_input_valid']
    pd.testing.assert_frame_equal(new[unchanged], old[unchanged], check_exact=True)
    changed = new.formula_input_valid.ne(old.formula_input_valid)
    assert changed.sum() == 1 and not new.loc[changed, 'formula_input_valid'].any()
    assert new.loc[changed, 'date'].ge('2025-07-01').all()
    pm = json.loads((parent / 'model_report.json').read_text()); sr = json.loads((parent / 'score_report.json').read_text())
    mv = json.loads((parent / 'model_verification.json').read_text()); sv = json.loads((parent / 'score_verification.json').read_text())
    assert mv['passed'] and mv['model_report_sha256'] == sha(parent / 'model_report.json')
    assert sv['passed'] and sv['score_report_sha256'] == sha(parent / 'score_report.json')
    assert pm['feature_report_sha256'] == sr['feature_report_sha256'] == sha(study.prior.INPUTS / 'feature_report.json')
    assert pm['label_report_sha256'] == sha(study.INPUTS / 'full_label_report.json')
    assert pm['variant'] == 'relative' and pm['feature_names'] == q['feature_names']
    assert all(pm['parameters'][k] == v for k, v in q['parameters'].items())
    fresh = shared.relative.training('relative')
    c = base.conn(); c.register('parent', old)
    fields = ','.join('f.' + n for n in study.ARMS['control'])
    prior = c.sql(f'''WITH labels AS(SELECT date,code,next_date,opportunity15,
        opportunity15-avg(opportunity15) OVER(PARTITION BY date) AS target
        FROM read_parquet('{study.prior.INPUTS}/full_labels.parquet')
        WHERE known15 AND date>='{q['training_start']}' AND next_date<'{q['training_end']}')
        SELECT date,code,next_date,opportunity15,target,1./count(*) OVER(PARTITION BY date) AS w,{fields}
        FROM parent f JOIN labels USING(date,code) WHERE formula_input_valid ORDER BY date,code''').df(); c.close()
    fields = ['date', 'code', 'next_date', 'opportunity15', *study.ARMS['control']]
    pd.testing.assert_frame_equal(fresh[fields], prior[fields], check_exact=True, check_dtype=False)
    np.testing.assert_allclose(fresh.target, prior.target, rtol=0, atol=2e-12)
    np.testing.assert_array_equal(1 / fresh.groupby('date').code.transform('size'), prior.w)
    assert len(prior) == q['expected_training_rows'] == pm['rows']
    assert prior.date.nunique() == q['expected_training_days'] == pm['days']
    assert prior.next_date.max() == q['expected_last_observation'] == pm['last_observation'] < q['evaluation_start']
    assert all(pm[k] == q[k] for k in ['training_start', 'training_end'])
    scores = pd.read_parquet(parent / 'scores.parquet')
    assert sr['scores_sha256'] == sha(parent / 'scores.parquet')
    pd.testing.assert_frame_equal(scores[study.META], old[study.META], check_exact=True)
    original = scores.copy()
    scores['formula_input_valid'] = new.formula_input_valid
    scores.loc[~scores.formula_input_valid, 'score'] = np.nan
    pd.testing.assert_frame_equal(scores[study.META], new[study.META], check_exact=True)
    pd.testing.assert_frame_equal(scores.loc[~changed], original.loc[~changed], check_exact=True)
    assert scores.loc[changed, 'score'].isna().all()
    root.mkdir(parents=True, exist_ok=True)
    m = dict(pm); m.update(protocol_sha256=sha(base.PROTOCOL), feature_report_sha256=sha(study.INPUTS / 'feature_report.json'),
        no_model_fit_performed=True, reused_model_report_sha256=sha(parent / 'model_report.json'), reused_source_root=str(parent))
    save_json(root / 'model_report.json', m)
    for field in ['parameters', 'trees', 'bias', 'feature_names', 'thresholds', 'rows', 'days', 'last_observation']:
        assert m[field] == pm[field]
    scores.to_parquet(root / 'scores.parquet', index=False, compression='zstd')
    pd.testing.assert_frame_equal(pd.read_parquet(root / 'scores.parquet'), scores, check_exact=True)
    r = dict(sr); r.update(protocol_sha256=sha(base.PROTOCOL), model_report_sha256=sha(root / 'model_report.json'),
        feature_report_sha256=sha(study.INPUTS / 'feature_report.json'), scores_sha256=sha(root / 'scores.parquet'),
        rows=len(scores), valid=int(scores.formula_input_valid.sum()), no_prediction_performed=True,
        only_one_evaluation_validity_flag_and_score_nan_projected=True)
    save_json(root / 'score_report.json', r)
    save_json(root / 'model_verification.json', dict(passed=True, model_report_sha256=sha(root / 'model_report.json'),
        reused_model_verification_sha256=sha(parent / 'model_verification.json'), rows=len(prior), days=pm['days'], node_checks=mv['node_checks'],
        exact_all_training_values_targets_weights_parameters_and_equations_before_reuse=True,
        new_2026_prices_read=False, no_exit_rules=True))
    save_json(root / 'score_verification.json', dict(passed=True, score_report_sha256=sha(root / 'score_report.json'),
        rows=len(scores), valid=int(scores.formula_input_valid.sum()), reused_score_verification_sha256=sha(parent / 'score_verification.json'),
        all_complete_original_scores_preserved_except_one_verified_invalid_evaluation_key=True,
        original_threshold_flag_checks=sv['threshold_flag_checks'], threshold_flag_checks=0,
        validity_mask_projection_and_unchanged_all_other_rows_checked=True, no_prediction_performed=True,
        new_2026_prices_read=False, no_exit_rules=True))
    proof = dict(passed=True, master_protocol_sha256=sha(shared.model.PROTOCOL), fold=fold, parent=str(parent),
        source_hashes={str(parent / f): sha(parent / f) for f in ['model_report.json', 'model_verification.json', 'scores.parquet', 'score_report.json', 'score_verification.json']},
        rows=len(prior), days=pm['days'], all_training_values_targets_weights_and_equations_exactly_matched=True,
        full_control_selection_equivalence_required_in_joint_freeze=True,
        no_refit_or_prediction=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(root / 'control_reuse_verification.json', proof)
    return dict(fold=fold, rows=len(prior), days=pm['days'], reused=True, invalid_evaluation_only_keys=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['protocols', 'reuse', 'model', 'verify_model', 'scores', 'verify_scores', 'freeze', 'analyze', 'finish'])
    parser.add_argument('--fold', choices=['2025h1', '2025h2']); args = parser.parse_args()
    shared.configure(STEM); shared.model.checked = checked
    if args.stage in ['protocols', 'freeze', 'analyze', 'finish']:
        result = getattr(shared, args.stage)()
    elif args.stage == 'reuse':
        assert args.fold; result = reuse(args.fold)
    else:
        assert args.fold; shared.model.setup('vwap', args.fold)
        if args.stage == 'model': result = shared.relative.model('relative')
        elif args.stage == 'verify_model': result = shared.relative.verify_model('relative')
        elif args.stage == 'scores': result = shared.base.scores()
        else: result = shared.verify_scores(expected_expressions=shared.base.EXPRESSIONS)
    print(json.dumps(result, ensure_ascii=False, indent=2))
