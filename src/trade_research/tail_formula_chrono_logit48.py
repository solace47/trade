"""Absolute log-loss target under the unchanged chronological calibration."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from . import tail_formula_additive as base
from . import tail_formula_chrono48 as chronology
from . import tail_formula_long48 as history
from .corporate_cash import save_json, sha

STEM = 'tail_formula_chrono_logit48'


def configure():
    chronology.STEM = STEM


def setup(fold):
    configure(); p = chronology.setup(fold)
    old = Path('config') / ('tail_formula_chrono48_' + fold + '_protocol.json')
    assert p['relative_calibration_protocol_sha256'] == sha(old)
    reference = json.loads(old.read_text())
    for key in ['training_start', 'training_end', 'calibration_start', 'calibration_end',
                'evaluation_start', 'evaluation_end', 'quality_gates', 'bootstrap', 'training_quantiles']:
        assert p[key] == reference[key]
    assert p['variant'] == 'absolute_logistic'
    return p


def model():
    root = base.ROOT; p = json.loads(base.PROTOCOL.read_text())
    assert not (root / 'model_report.json').exists()
    t = base.training(start=p['training_start'], end=p['training_end'])
    assert t.date.ge(p['training_start']).all() and t.next_date.lt(p['training_end']).all()
    x = base.encode(t); y = t.opportunity15.to_numpy()
    assert np.isfinite(y).all() and set(np.unique(y)) == {0., 1.}
    w = 1 / t.groupby('date').code.transform('size').to_numpy()
    estimator = GradientBoostingClassifier(**p['parameters']).fit(x, y, sample_weight=w)
    trees = []
    for component in estimator.estimators_.ravel():
        a = component.tree_
        trees.append(dict(feature=a.feature.tolist(), threshold=a.threshold.tolist(),
            children_left=a.children_left.tolist(), children_right=a.children_right.tolist(),
            n_node_samples=a.n_node_samples.tolist(), weighted_n_node_samples=a.weighted_n_node_samples.tolist(),
            value=a.value.reshape(-1).tolist(), impurity=a.impurity.tolist()))
    prior = estimator.init_.class_prior_
    r = dict(protocol_sha256=sha(base.PROTOCOL), feature_report_sha256=sha(base.FEATURES / 'feature_report.json'),
        label_report_sha256=sha(base.SOURCE / 'full_label_report.json'), rows=len(t), days=t.date.nunique(),
        last_observation=t.next_date.max(), parameters=estimator.get_params(), feature_names=list(base.EXPRESSIONS),
        learning_rate=.05, bias=float(np.log(prior[1] / prior[0])), class_prior=prior.tolist(), trees=trees,
        training_start=p['training_start'], training_end=p['training_end'], variant='absolute_logistic',
        new_2025_score_groups_read=False, new_2025H2_score_groups_read=False,
        new_2026_prices_read=False, no_exit_rules=True)
    score = base.predict(x, r)
    np.testing.assert_allclose(score, estimator.decision_function(x), rtol=0, atol=2e-12)
    r['thresholds'] = [dict(id=i, training_quantile=q, threshold=float(np.quantile(score, q)))
                       for i, q in enumerate(base.QUANTILES)]
    root.mkdir(parents=True, exist_ok=True); save_json(root / 'model_report.json', r)
    return {k: v for k, v in r.items() if k != 'trees'}


def verify_model():
    root = base.ROOT; p = json.loads(base.PROTOCOL.read_text())
    r = json.loads((root / 'model_report.json').read_text())
    assert r['feature_names'] == list(base.EXPRESSIONS) and len(r['feature_names']) == 48
    assert r['variant'] == 'absolute_logistic'
    assert all(r['parameters'][k] == value for k, value in p['parameters'].items())
    for k in ['training_start', 'training_end']:
        assert r[k] == p[k]
    assert r['last_observation'] < p['calibration_start']
    c = base.conn()
    keys = c.execute('''SELECT l.date,l.code,l.next_date FROM read_parquet(?) l
        JOIN read_parquet(?) f USING(date,code) WHERE l.date>=? AND l.next_date<?
        AND l.known15 AND f.formula_input_valid ORDER BY date,code''',
        [str(base.SOURCE / 'full_labels.parquet'), str(base.FEATURES / 'features.parquet'),
         p['training_start'], p['training_end']]).df()
    assert len(keys) == r['rows'] and keys.date.nunique() == r['days'] and keys.next_date.max() == r['last_observation']
    c.close()
    for tree in r['trees']:
        depths = {0: 0}
        for node, left in enumerate(tree['children_left']):
            assert depths[node] <= p['parameters']['max_depth']
            if left >= 0:
                depths[left] = depths[tree['children_right'][node]] = depths[node] + 1
    # The original binary-loss verifier independently reconstructs each Newton
    # leaf and residual variance; only its input date scope is adapted here.
    original_loader = base.training
    def scoped_loader():
        t = original_loader(start=p['training_start'], end=p['training_end'])
        assert t[['date', 'code', 'next_date']].equals(keys)
        return t
    base.training = scoped_loader
    try:
        proof = base.verify_model()
    finally:
        base.training = original_loader
    proof.update(training_start=p['training_start'], training_end=p['training_end'],
        all_training_keys_and_depth_bounds_rebuilt=True, unchanged_original_48_inputs=True)
    save_json(root / 'model_verification.json', proof); return proof


def verifier():
    spec = importlib.util.spec_from_file_location('independent_chronological_check', Path('scripts/verify_tail_formula_chrono48.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('fold', choices=['2024', 'recent', 'combined'])
    p.add_argument('stage', choices=['model', 'verify_model', 'scores', 'verify_scores', 'calibrate',
        'verify_calibration', 'freeze', 'verify_selection', 'combine', 'analyze'])
    p.add_argument('--control', action='store_true'); a = p.parse_args(); configure()
    protocol = Path('config') / (STEM + '_' + a.fold + '_protocol.json')
    if a.fold != 'combined':
        setup(a.fold)
    else:
        assert a.stage in ['combine', 'verify_selection', 'analyze']
    part = '2025' if a.fold == 'combined' else a.fold
    root = Path('data/research') / (STEM + '_' + part + ('_control' if a.control else ''))
    if a.stage in ['model', 'verify_model']:
        result = globals()[a.stage]()
    elif a.stage == 'verify_scores':
        result = history.verify_scores()
    elif a.stage == 'scores':
        result = base.scores()
    elif a.stage == 'verify_calibration':
        result = verifier().calibration(root, protocol)
    elif a.stage == 'verify_selection':
        result = getattr(verifier(), 'combined' if a.fold == 'combined' else 'selection')(root, protocol)
    elif a.stage == 'analyze':
        for suffix in ['', '_control']:
            combined = Path('data/research') / (STEM + '_2025' + suffix)
            proof = json.loads((combined / 'selection_verification.json').read_text())
            assert proof['passed'] and proof['selection_report_sha256'] == sha(combined / 'selection_report.json')
        assert json.loads((root / 'selection_report.json').read_text())['selected'] > 0
        result = chronology.common_analysis(root, protocol)
    else:
        result = getattr(chronology, a.stage)()
    print(json.dumps(result, ensure_ascii=False, indent=2))
