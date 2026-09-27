"""Grow the original relative-opportunity rules with a fixed leaf day-weight floor."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from . import tail_formula_additive as base
from . import tail_formula_context_2024 as linkage
from . import tail_formula_float as inputs
from . import tail_formula_recent as study
from . import tail_formula_relative as relative
from .corporate_cash import save_json, sha

STEM = 'tail_formula_day_weight48'


def setup(fold):
    if fold == 'combined':
        protocol = Path('config') / (STEM + '_combined_protocol.json')
        p = json.loads(protocol.read_text())
        paths = [Path('config') / (STEM + '_' + f + '_protocol.json') for f in ['2024', 'recent']]
        assert p['fold_protocols'] == [str(x) for x in paths]
        for path, name in zip(paths, ['2024', 'recent']):
            for file in ['model_report.json', 'selection_report.json']:
                assert json.loads((Path('data/research') / (STEM + '_' + name) / file).read_text())['protocol_sha256'] == sha(path)
        linkage.ROOT = Path('data/research') / (STEM + '_2024')
        linkage.H2 = Path('data/research') / (STEM + '_recent')
        linkage.COMBINED = Path('data/research') / (STEM + '_2025')
        linkage.PROTOCOL = protocol
        linkage.H2_SELECTION_SHA = None; linkage.H2_MODEL_SHA = None
        linkage.H2_OUTCOMES_PREVIOUSLY_SEEN = False
    else:
        inputs.setup(fold)
        root = Path('data/research') / (STEM + '_' + fold)
        protocol = Path('config') / (STEM + '_' + fold + '_protocol.json')
        base.ROOT = root; base.PROTOCOL = protocol; relative.PROTOCOL = protocol
        study.ROOT = root; study.PROTOCOL = protocol
        p = json.loads(protocol.read_text())
        assert p['feature_report_sha256'] == sha(inputs.ROOT / 'feature_report.json')
        assert p['label_report_sha256'] == sha(base.SOURCE / 'full_label_report.json')


def model():
    root = base.ROOT
    if (root / 'model_report.json').exists():
        raise ValueError('Do not replace the frozen growth-constrained model')
    p = json.loads(base.PROTOCOL.read_text())
    train = relative.training('relative'); x = base.encode(train); y = train.target.to_numpy()
    w = 1 / train.groupby('date').code.transform('size').to_numpy()
    dates, date_ids = np.unique(train.date.to_numpy(), return_inverse=True)
    assert len(dates) == p['expected_training_days'] == 241
    assert p['minimum_leaf_day_weight'] == p['minimum_leaf_training_days'] == 20
    assert p['parameters']['min_weight_fraction_leaf'] == 20 / len(dates)
    np.testing.assert_allclose(np.bincount(date_ids, weights=w), 1., rtol=0, atol=2e-12)
    estimator = GradientBoostingRegressor(**p['parameters']).fit(x, y, sample_weight=w)
    trees = []
    for component in estimator.estimators_.ravel():
        t = component.tree_
        tree = dict(feature=t.feature.tolist(), threshold=t.threshold.tolist(),
            children_left=t.children_left.tolist(), children_right=t.children_right.tolist(),
            n_node_samples=t.n_node_samples.tolist(), weighted_n_node_samples=t.weighted_n_node_samples.tolist(),
            value=t.value.reshape(-1).tolist(), impurity=t.impurity.tolist())
        # The sparse training decision path is independent of the verifier's
        # dense recursive masks, including every internal node and final leaf.
        path = component.decision_path(x).tocsc()
        support = []
        for node in range(t.node_count):
            rows = path.indices[path.indptr[node]:path.indptr[node+1]]
            support.append(int(np.unique(date_ids[rows]).size))
            if t.children_left[node] < 0:
                assert support[-1] >= 20 and w[rows].sum() >= 20 - 1e-8
        tree['training_days'] = support; trees.append(tree)
    start, end, _ = relative.training_scope()
    r = dict(protocol_sha256=sha(base.PROTOCOL), feature_report_sha256=sha(base.FEATURES / 'feature_report.json'),
        label_report_sha256=sha(base.SOURCE / 'full_label_report.json'), rows=len(train), days=len(dates),
        last_observation=train.next_date.max(), parameters=estimator.get_params(),
        feature_names=list(base.EXPRESSIONS), variant='relative', learning_rate=.05,
        bias=float(np.ravel(estimator.init_.constant_)[0]), trees=trees,
        training_start=start, training_end=end, minimum_leaf_day_weight=20,
        constraint_applied_during_split_search=True, post_fit_pruning=False,
        new_2025_score_groups_read=bool(train.date.ge('2025-01-01').any()),
        new_2025H2_score_groups_read=bool(train.date.ge('2025-07-01').any()),
        new_2026_prices_read=False, no_exit_rules=True)
    manual = base.predict(x, r)
    np.testing.assert_allclose(manual, estimator.predict(x), rtol=0, atol=2e-12)
    r['thresholds'] = [dict(id=i, training_quantile=q, threshold=float(np.quantile(manual, q)))
                       for i, q in enumerate(base.QUANTILES)]
    root.mkdir(parents=True, exist_ok=True); save_json(root / 'model_report.json', r)
    return {key: value for key, value in r.items() if key != 'trees'}


def verify_model():
    r = json.loads((base.ROOT / 'model_report.json').read_text())
    p = json.loads(base.PROTOCOL.read_text())
    assert r['feature_names'] == list(base.EXPRESSIONS) and len(r['feature_names']) == 48
    assert r['days'] == p['expected_training_days'] == 241
    assert r['minimum_leaf_day_weight'] == p['minimum_leaf_day_weight'] == 20
    assert r['constraint_applied_during_split_search'] and not r['post_fit_pruning']
    assert all(r['parameters'][key] == value for key, value in p['parameters'].items())
    assert r['parameters']['min_weight_fraction_leaf'] == 20 / r['days']
    # Existing verifier reconstructs all labels/encodings/weights in SQL and
    # every residual, node count, date support, mass and prediction via masks.
    proof = relative.verify_model('relative')
    masses = []; dates = []
    for tree in r['trees']:
        for node, left in enumerate(tree['children_left']):
            if left < 0:
                masses.append(tree['weighted_n_node_samples'][node])
                dates.append(tree['training_days'][node])
    assert min(masses) >= 20 - 1e-8 and min(dates) >= 20
    proof.update(all_final_leaf_day_weights_checked=True,
        leaf_checks=len(masses), minimum_leaf_day_weight_observed=min(masses),
        minimum_leaf_dates_observed=min(dates), unchanged_original_relative_objective=True)
    save_json(base.ROOT / 'model_verification.json', proof)
    return proof


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('fold', choices=['2024', 'recent', 'combined'])
    p.add_argument('stage', choices=['model', 'verify_model', 'scores', 'freeze', 'verify', 'analyze'])
    a = p.parse_args(); setup(a.fold)
    if a.stage == 'analyze':
        assert (Path('data/research') / (STEM + '_2025') / 'selection_verification.json').exists()
    if a.fold == 'combined':
        assert a.stage in ['freeze', 'verify', 'analyze']
        result = (linkage.common_analysis(linkage.COMBINED, linkage.PROTOCOL) if a.stage == 'analyze'
                  else getattr(linkage, 'combine' if a.stage == 'freeze' else 'verify_combined')())
    elif a.stage in ['model', 'verify_model']:
        result = globals()[a.stage]()
    elif a.stage in ['freeze', 'verify']:
        result = getattr(study, a.stage)()
    else:
        result = getattr(base, a.stage)()
    print(json.dumps(result, ensure_ascii=False, indent=2))
