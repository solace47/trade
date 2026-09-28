"""Reweight fixed binary pairs by expected top-ten rank-swap discounts."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from . import tail_formula_additive as base
from . import tail_formula_before1000_model as adapter
from . import tail_formula_float as inputs
from . import tail_formula_pairwise as uniform
from . import tail_formula_relative as relative
from .corporate_cash import save_json, sha

STEM = 'tail_formula_top_pairs_raw'
MASTER = Path('config/tail_formula_top_pairs_protocol.json')
ROOT = Path('data/research/tail_formula_top_pairs')
PARAMETERS = dict(loss='within_date_binary_logistic_with_expected_top10_swap_discount',
    n_estimators=64, learning_rate=.05, max_depth=3, min_samples_leaf=300,
    subsample=1., random_state=20260927, criterion='friedman_mse', pair_passes=8,
    sigmoid_scale=1., newton_leaf_clip=2., discount_top_k=10)


def checked_sources():
    p = json.loads(MASTER.read_text())
    assert p['parameters'] == PARAMETERS and not p['new_2026_prices_allowed']
    for path, digest in p['references'].items():
        assert sha(Path(path)) == digest
    r = json.loads((inputs.ROOT / 'feature_report.json').read_text())
    v = json.loads((inputs.ROOT / 'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(inputs.ROOT / 'feature_report.json')
    assert r['features_sha256'] == sha(inputs.ROOT / 'features.parquet')
    assert (r['rows'], r['valid']) == (1258085, 1117397)
    records = []
    for fold in ['2024', 'recent']:
        path = Path('data/research') / ('tail_formula_pairwise_' + fold)
        m = json.loads((path / 'model_report.json').read_text())
        proof = json.loads((path / 'model_verification.json').read_text())
        cfg = json.loads((Path('config') / (STEM+'_'+fold+'_protocol.json')).read_text())
        assert proof['passed'] and proof['model_report_sha256'] == sha(path / 'model_report.json')
        assert m['training_pairs_sha256'] == sha(path / 'training_pairs.parquet')
        assert m['training_days_sha256'] == sha(path / 'training_days.parquet')
        assert cfg['inputs_protocol_sha256'] == sha(MASTER) and cfg['parameters'] == PARAMETERS
        assert cfg['feature_report_sha256'] == m['feature_report_sha256'] == sha(inputs.ROOT / 'feature_report.json')
        assert cfg['label_report_sha256'] == m['label_report_sha256']
        assert m['feature_names'] == list(inputs.EXPRESSIONS)
        assert (m['rows'], m['days']) == (cfg['expected_rows'], 241)
        assert m['training_start'] == cfg['training_start'] and m['training_end'] == cfg['training_end']
        assert m['parameters'] == uniform.PARAMETERS
        for key, value in PARAMETERS.items():
            if key not in ['loss', 'discount_top_k']:
                assert m['parameters'][key] == value
        records.append(dict(fold=fold, model_report_sha256=sha(path/'model_report.json'),
            training_pairs_sha256=sha(path/'training_pairs.parquet'),
            training_days_sha256=sha(path/'training_days.parquet'),
            model_verification_sha256=sha(path/'model_verification.json'), rows=m['rows'], days=m['days']))
    return records


def verify_inputs():
    records = checked_sources()
    ROOT.mkdir(parents=True, exist_ok=True)
    out = ROOT / 'input_reuse_verification.json'
    assert not out.exists()
    save_json(out, dict(passed=True, protocol_sha256=sha(MASTER), original_pairs=records,
        original_inputs_labels_and_uniform_pair_verifications_reused=True,
        only_training_pair_weights_changed=True, new_group_outcomes_read=False,
        new_2026_prices_read=False, no_exit_rules=True))
    return dict(passed=True, input_reuse_sha256=sha(out), original_pairs=records)


def setup(fold):
    checked_sources()
    v = json.loads((ROOT / 'input_reuse_verification.json').read_text())
    assert v['passed'] and v['protocol_sha256'] == sha(MASTER)
    adapter.STEM = STEM
    adapter.setup(fold)


def rank_geometry(score, day_ids, top_k=10):
    """Mean discount and expected |discount_i-discount_j| inside each tie."""
    score, day_ids = np.asarray(score), np.asarray(day_ids)
    assert np.isfinite(score).all() and len(score) == len(day_ids)
    order = np.lexsort((-score, day_ids))
    s, d = score[order], day_ids[order]
    starts = np.flatnonzero(np.r_[True, (s[1:] != s[:-1]) | (d[1:] != d[:-1])])
    sizes = np.diff(np.r_[starts, len(s)])
    group = np.repeat(np.arange(len(starts)), sizes)
    date_starts = np.flatnonzero(np.r_[True, d[1:] != d[:-1]])
    ranks = np.arange(len(s)) - np.repeat(date_starts, np.diff(np.r_[date_starts, len(s)])) + 1
    discount = np.where(ranks <= top_k, 1 / np.log2(ranks + 1), 0.)
    means = np.add.reduceat(discount, starts) / sizes
    within = np.zeros(len(starts))
    numerator = 2 * np.add.reduceat((sizes[group] - 1 - 2 * (np.arange(len(s)) - starts[group])) * discount, starts)
    multiple = sizes > 1
    within[multiple] = numerator[multiple] / (sizes[multiple] * (sizes[multiple] - 1))
    assert (within >= 0).all()
    mean_out, within_out = np.empty(len(s)), np.empty(len(s))
    mean_out[order], within_out[order] = means[group], within[group]
    return mean_out, within_out


def rank_weights(score, day_ids, positive, negative, top_k=10):
    mean, within = rank_geometry(score, day_ids, top_k)
    delta = np.where(score[positive] == score[negative], within[positive],
                     np.abs(mean[positive] - mean[negative]))
    pair_day = day_ids[positive]
    np.testing.assert_array_equal(pair_day, day_ids[negative])
    totals = np.bincount(pair_day, weights=delta, minlength=int(day_ids.max()) + 1)
    assert (totals[pair_day] > 0).all(), 'An informative date must retain positive pair mass'
    return delta / totals[pair_day]


def model(fold):
    root = base.ROOT
    assert not (root / 'model_report.json').exists()
    p = json.loads(base.PROTOCOL.read_text())
    start, end, _ = relative.training_scope()
    t = base.training(start=start, end=end)
    assert (len(t), t.date.nunique()) == (p['expected_rows'], 241) and t.next_date.max() < end
    source = Path('data/research') / ('tail_formula_pairwise_' + fold)
    pairs = pd.read_parquet(source / 'training_pairs.parquet')
    days = pd.read_parquet(source / 'training_days.parquet')
    positive, negative = pairs.positive.to_numpy('int64'), pairs.negative.to_numpy('int64')
    np.testing.assert_array_equal(t.date.to_numpy()[positive], pairs.date)
    np.testing.assert_array_equal(t.date.to_numpy()[negative], pairs.date)
    assert t.opportunity15.to_numpy()[positive].all() and not t.opportunity15.to_numpy()[negative].any()
    x = base.encode(t)
    w = 1 / t.groupby('date').code.transform('size').to_numpy()
    day_ids = pd.factorize(t.date, sort=True)[0]
    score = np.zeros(len(t)); trees = []; trace = []
    rng = np.random.RandomState(20260927)
    for iteration in range(64):
        weight = rank_weights(score, day_ids, positive, negative)
        if iteration == 0:
            np.testing.assert_allclose(weight, pairs.weight, rtol=0, atol=2e-15)
        gradient, curvature, loss = uniform.derivatives(score, positive, negative, weight)
        residual = gradient / w
        np.testing.assert_allclose(np.bincount(day_ids, weights=gradient), 0, rtol=0, atol=2e-12)
        estimator = DecisionTreeRegressor(criterion='friedman_mse', max_depth=3,
            min_samples_leaf=300, random_state=rng).fit(x, residual, sample_weight=w)
        q = estimator.tree_; leaves = estimator.apply(x)
        tree = dict(feature=q.feature.tolist(), threshold=q.threshold.tolist(), children_left=q.children_left.tolist(),
            children_right=q.children_right.tolist(), n_node_samples=q.n_node_samples.tolist(),
            weighted_n_node_samples=q.weighted_n_node_samples.tolist(), gradient_value=q.value.reshape(-1).tolist(),
            value=q.value.reshape(-1).tolist(), impurity=q.impurity.tolist(), leaf_numerator={}, leaf_curvature={})
        for node in np.flatnonzero(q.children_left < 0):
            value, numerator, denominator = uniform.leaf_step(node, leaves, gradient, curvature, positive, negative)
            tree['value'][node] = value
            tree['leaf_numerator'][str(node)] = numerator
            tree['leaf_curvature'][str(node)] = denominator
        trees.append(tree)
        score += .05 * np.asarray(tree['value'])[leaves]
        loss_after = float(np.sum(weight * np.logaddexp(0., score[negative] - score[positive])))
        trace.append(dict(iteration=iteration, loss_before=loss, loss_after=loss_after,
            gradient_sum=float(gradient.sum()), nonzero_weight_pairs=int(np.count_nonzero(weight)),
            max_pair_weight=float(weight.max()),
            clipped_leaves=sum(abs(tree['value'][n]) == 2 for n in np.flatnonzero(q.children_left < 0))))
        assert np.isfinite(score).all() and np.isfinite(loss_after)
    root.mkdir(parents=True, exist_ok=True)
    for name in ['training_pairs.parquet', 'training_days.parquet']:
        (root / name).symlink_to((source / name).resolve())
    r = dict(protocol_sha256=sha(base.PROTOCOL), inputs_protocol_sha256=sha(MASTER),
        feature_report_sha256=sha(base.FEATURES/'feature_report.json'), label_report_sha256=sha(base.SOURCE/'full_label_report.json'),
        input_reuse_verification_sha256=sha(ROOT/'input_reuse_verification.json'), rows=len(t), days=t.date.nunique(),
        last_observation=t.next_date.max(), parameters=PARAMETERS, feature_names=list(base.EXPRESSIONS),
        variant='same_date_binary_top10_pairs', learning_rate=.05, bias=0., trees=trees, trace=trace,
        training_start=start, training_end=end, training_pairs_sha256=sha(root/'training_pairs.parquet'),
        training_days_sha256=sha(root/'training_days.parquet'), pairs=len(pairs), informative_days=int(days.informative.sum()),
        constant_label_days=days.loc[~days.informative,'date'].tolist(),
        within_iteration_weights_fixed_when_comparing_loss=True, scores_are_not_win_probabilities=True,
        new_2025_score_groups_read=bool(t.date.ge('2025-01-01').any()), new_2025H2_score_groups_read=False,
        new_2026_prices_read=False, no_exit_rules=True)
    exported = base.predict(x, r)
    np.testing.assert_allclose(score, exported, rtol=0, atol=2e-12)
    r['thresholds'] = [dict(id=i, training_quantile=q, threshold=float(np.quantile(exported,q)))
                       for i,q in enumerate(base.QUANTILES)]
    save_json(root/'model_report.json', r)
    return {k:v for k,v in r.items() if k not in ['trees','trace']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['inputs','model','verify_model','scores','verify_scores'])
    parser.add_argument('--fold', choices=['2024','recent'], default='2024')
    args = parser.parse_args()
    if args.stage == 'inputs':
        result = verify_inputs()
    else:
        setup(args.fold)
        if args.stage == 'model':
            result = model(args.fold)
        elif args.stage == 'verify_model':
            spec = importlib.util.spec_from_file_location('verify_top_pairs', 'scripts/verify_tail_formula_top_pairs.py')
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            result = module.verify(args.fold)
        elif args.stage == 'verify_scores':
            from .tail_formula_offset_logit48 import verify_scores
            result = verify_scores()
        else:
            result = base.scores()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
