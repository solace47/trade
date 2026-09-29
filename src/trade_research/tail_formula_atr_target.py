"""Scale the existing relative 09:59 reference target by visible prior ATR."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_before1000_model as baseline
from . import tail_formula_boundary_evaluation as evaluation
from . import tail_formula_context_2024 as linkage
from . import tail_formula_endpoint_robust as engine
from . import tail_formula_recent as selection
from .corporate_cash import save_json, sha

STEM = 'tail_formula_atr_target'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
ORIGINAL_TRAINING = engine.training
ORIGINAL_INDEPENDENT_TRAINING = engine.independent_training
FOLD = None


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    assert p['scaling'] == 'after_full_finite_reference_date_centering'
    assert p['denominator'] == 'original_unencoded_V01' and p['loss'] == 'squared_error'
    assert not p['new_2026_prices_allowed']
    for file, digest in p['references'].items():
        assert sha(Path(file)) == digest
    for root, stage, key in [(baseline.inputs.ROOT, 'feature', 'feature_report_sha256'),
                             (baseline.labels.ROOT, 'full_label', 'label_report_sha256')]:
        r = json.loads((root / (stage + '_report.json')).read_text())
        v = json.loads((root / (stage + '_verification.json')).read_text())
        assert v['passed'] and v[key] == sha(root / (stage + '_report.json'))
        data = 'features' if stage == 'feature' else 'labels'
        file = root / ('features.parquet' if stage == 'feature' else 'full_labels.parquet')
        assert r[data + '_sha256'] == sha(file)
    return p


def setup(fold):
    global FOLD
    checked_sources(); FOLD = fold
    baseline.STEM = STEM; baseline.setup(fold)
    engine.ROOT = ROOT; engine.PROTOCOL = PROTOCOL; engine.VARIANT = 'atr_scaled_reference'
    engine.training = training; engine.independent_training = independent_training
    for name in ['2024', 'recent', 'combined']:
        p = json.loads((Path('config') / (STEM + '_' + name + '_protocol.json')).read_text())
        assert p['master_protocol_sha256'] == sha(PROTOCOL)
    if fold != 'combined':
        p = json.loads(base.PROTOCOL.read_text())
        old = json.loads((Path('config') / ('tail_formula_endpoint_squared_' + fold + '_protocol.json')).read_text())
        for key in ['training_start', 'training_end', 'evaluation_start', 'evaluation_end',
                    'expected_training_rows', 'original_training_rows', 'expected_reference_missing_known_rows',
                    'expected_training_days', 'expected_features', 'parameters', 'training_quantile']:
            assert p[key] == old[key]
        assert p['loss_arm'] == 'squared' and p['center_target'] and p.get('target_event') is None


def scale_target(relative_percentage_points, prior_atr_percentage_points):
    values, atr = np.broadcast_arrays(np.asarray(relative_percentage_points, float),
                                     np.asarray(prior_atr_percentage_points, float))
    assert np.isfinite(values).all() and np.isfinite(atr).all() and (atr > 0).all()
    result = values / atr
    assert np.isfinite(result).all()
    return result


def training():
    original = ORIGINAL_TRAINING()
    out = original.copy()
    out['target'] = scale_target(original.target, original.V01)
    pd.testing.assert_frame_equal(out.drop(columns='target'), original.drop(columns='target'), check_exact=True)
    return out


def independent_training():
    # Reuse the already audited raw-reference fees, full date baselines, finite
    # reference population, weights and original integer inputs. Restore our
    # target after this independent unscaled reconstruction even on failure.
    engine.training = ORIGINAL_TRAINING
    try:
        old_d, old_t, checks = ORIGINAL_INDEPENDENT_TRAINING()
    finally:
        engine.training = training
    f = base.feature_inputs()[['date', 'code', 'V01']]
    c = base.conn(); c.register('unscaled', old_d[['date', 'code', 'target']]); c.register('visible', f)
    expected = c.sql('SELECT date,code,target/V01 AS target,V01 FROM unscaled JOIN visible USING(date,code) '
                     'ORDER BY date,code').df(); c.close()
    d = old_d.copy(); d['target'] = expected.target.to_numpy(); t = training()
    pd.testing.assert_frame_equal(expected[['date', 'code']], old_d[['date', 'code']], check_exact=True)
    pd.testing.assert_frame_equal(d[['date', 'code']], t[['date', 'code']], check_exact=True)
    pd.testing.assert_frame_equal(t.drop(columns='target'), old_t.drop(columns='target'), check_exact=True)
    np.testing.assert_array_equal(expected.V01, t.V01)
    np.testing.assert_allclose(d.target, t.target, rtol=0, atol=2e-12)
    np.testing.assert_array_equal(d[list(base.EXPRESSIONS)].to_numpy(dtype='int32'), base.encode(t))
    np.testing.assert_array_equal(d.weight, 1/t.groupby('date').code.transform('size'))
    assert (expected.V01 > 0).all() and np.isfinite(expected.V01).all() and np.isfinite(d.target).all()
    control = Path('data/research') / ('tail_formula_endpoint_squared_' + FOLD)
    report = json.loads((control / 'model_report.json').read_text())
    proof = json.loads((control / 'model_verification.json').read_text())
    assert proof['passed'] and proof['model_report_sha256'] == sha(control / 'model_report.json')
    assert report['rows'] == len(t) and report['days'] == t.date.nunique()
    assert report['feature_names'] == list(base.EXPRESSIONS)
    checks.update(unscaled_control_model_sha256=sha(control / 'model_report.json'),
                  same_full_finite_reference_date_mean_before_division=True,
                  same_unscaled_training_keys_weights_and_integer_inputs=True,
                  final_scaled_date_means_not_forced_to_zero=True)
    return d, t, checks


def verify_inputs():
    path = ROOT / ('training_' + FOLD + '_verification.json'); assert not path.exists()
    d, t, checks = independent_training(); ROOT.mkdir(parents=True, exist_ok=True)
    p = json.loads(base.PROTOCOL.read_text())
    assert len(t) == p['expected_training_rows'] and t.date.nunique() == 241
    assert t.next_date.max() < p['training_end'] and t.date.min() >= p['training_start']
    proof = dict(passed=True, master_protocol_sha256=sha(PROTOCOL), fold_protocol_sha256=sha(base.PROTOCOL),
        implementation_sha256=sha(Path(__file__)), rows=len(t), days=t.date.nunique(),
        feature_report_sha256=sha(baseline.inputs.ROOT / 'feature_report.json'),
        label_report_sha256=sha(baseline.labels.ROOT / 'full_label_report.json'),
        checks=checks, last_observation=t.next_date.max(),
        no_future_reference_or_atr_used_in_inference_filter=True,
        no_target_clipping_floor_new_training_exclusions_or_exit_rules=True,
        new_group_outcomes_read=False, new_2026_prices_read=False)
    save_json(path, proof); return proof


def model():
    proof = json.loads((ROOT / ('training_' + FOLD + '_verification.json')).read_text())
    assert proof['passed'] and proof['implementation_sha256'] == sha(Path(__file__))
    result = engine.model('squared', FOLD)
    path = base.ROOT / 'model_report.json'; report = json.loads(path.read_text())
    extra = dict(unscaled_target_date_centered=True, final_scaled_target_date_centered=False,
                 target_divided_by_visible_prior_atr_after_centering=True)
    report.update(extra); result.update(extra); save_json(path, report)
    return result


def verify_model():
    report = json.loads((base.ROOT / 'model_report.json').read_text())
    assert report['unscaled_target_date_centered'] and not report['final_scaled_target_date_centered']
    assert report['target_divided_by_visible_prior_atr_after_centering']
    return engine.verify_model('squared', FOLD)


def main():
    from .tail_formula_offset_logit48 import verify_scores
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['verify_inputs', 'model', 'verify_model', 'scores', 'verify_scores', 'freeze', 'verify', 'analyze'])
    p.add_argument('--fold', choices=['2024', 'recent', 'combined'], default='2024')
    a = p.parse_args(); setup(a.fold)
    if a.stage == 'analyze':
        joint = json.loads((ROOT / 'joint_selection_freeze.json').read_text())
        assert joint['passed'] and joint['protocol_sha256'] == sha(PROTOCOL) and len(joint['selections']) == 3
        result = evaluation.analyze(linkage.COMBINED if a.fold == 'combined' else base.ROOT,
                                    linkage.PROTOCOL if a.fold == 'combined' else base.PROTOCOL)
    elif a.fold == 'combined':
        assert a.stage in ['freeze', 'verify']; result = linkage.combine() if a.stage == 'freeze' else linkage.verify_combined()
    elif a.stage == 'verify_inputs': result = verify_inputs()
    elif a.stage == 'model': result = model()
    elif a.stage == 'verify_model': result = verify_model()
    elif a.stage == 'verify_scores': result = verify_scores()
    elif a.stage == 'verify':
        report = json.loads((base.ROOT / 'model_report.json').read_text())
        assert (base.ROOT / 'frozen_numeric_core.tdx').read_text() == base.native_core(report, report['thresholds'][3]['threshold'], base.EXPRESSIONS, base.HEADER)
        result = selection.verify()
    elif a.stage == 'freeze': result = selection.freeze()
    else: result = base.scores()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
