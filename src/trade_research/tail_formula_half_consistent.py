"""Boost only a residual direction shared by two disjoint training halves."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

from . import tail_formula_additive as base
from . import tail_formula_before1000 as labels
from . import tail_formula_context_2024 as linkage
from . import tail_formula_float as inputs
from . import tail_formula_half_consensus as previous
from . import tail_formula_recent as study
from . import tail_formula_relative as relative
from .corporate_cash import save_json, sha

STEM = 'tail_formula_half_consistent'
PROTOCOL = Path('config') / (STEM + '_protocol.json')
PARAMETERS = dict(n_estimators=64, max_depth=3, min_samples_leaf=300, learning_rate=.05,
                  random_state=20260927, criterion='friedman_mse', subsample=1.)


def policy():
    p = json.loads(PROTOCOL.read_text())
    assert p['feature_report_sha256'] == sha(inputs.ROOT / 'feature_report.json')
    assert p['label_report_sha256'] == sha(labels.ROOT / 'full_label_report.json')
    assert p['label_verification_sha256'] == sha(labels.ROOT / 'full_label_verification.json')
    assert p['previous_consensus_protocol_sha256'] == sha(previous.PROTOCOL)
    assert p['parameters'] == PARAMETERS and p['training_quantile'] == .995
    assert p['variants'] == ['ordinary', 'consistent'] and not p['new_2026_prices_allowed']
    return p


def root(variant, fold):
    return Path('data/research') / f'{STEM}_{variant}_{fold}'


def setup(variant, fold):
    p = policy(); assert variant in p['variants']
    config = Path('config') / f'{STEM}_{variant}_{fold}_protocol.json'
    q = json.loads(config.read_text()); assert q['parent_protocol_sha256'] == sha(PROTOCOL)
    if fold == 'combined':
        linkage.ROOT = root(variant, '2024'); linkage.H2 = root(variant, 'recent')
        linkage.COMBINED = root(variant, '2025'); linkage.PROTOCOL = config
        linkage.H2_SELECTION_SHA = linkage.H2_MODEL_SHA = None; linkage.H2_OUTCOMES_PREVIOUSLY_SEEN = False
    else:
        assert all(q[k] == v for k, v in p['folds'][fold].items())
        assert q['variant'] == variant and q['parameters'] == PARAMETERS
        base.ROOT = study.ROOT = root(variant, fold)
        base.PROTOCOL = relative.PROTOCOL = study.PROTOCOL = config
        base.FEATURES = inputs.ROOT; base.EXPRESSIONS = inputs.EXPRESSIONS; base.HEADER = inputs.HEADER
        base.SOURCE = labels.ROOT
    return q


def training():
    q = json.loads(base.PROTOCOL.read_text()); split = q['training_split']
    t = relative.training('relative')
    t = t.loc[t.date.ge(split) | t.next_date.lt(split)].reset_index(drop=True)
    t['training_group'] = t.date.ge(split).astype(int)
    assert len(t) == q['expected_rows'] and t.date.nunique() == q['expected_days']
    assert t.groupby('training_group').size().tolist() == q['group_rows']
    assert t.groupby('training_group').date.nunique().tolist() == q['group_days']
    return t


def shared_direction(means):
    if any(x is None or not np.isfinite(x) for x in means):
        return 0.
    if min(means) > 0:
        return float(min(means))
    if max(means) < 0:
        return float(max(means))
    return 0.


def group_stats(residual, weights, group, mask):
    counts, totals, means = [], [], []
    for g in [0, 1]:
        take = mask & (group == g); weight = float(weights[take].sum())
        counts.append(int(take.sum())); totals.append(weight)
        means.append(float(np.average(residual[take], weights=weights[take])) if weight else None)
    return dict(rows=counts, weights=totals, means=means)


def group_loss(y, score, weights, group):
    return [float(np.average((y[group == g] - score[group == g]) ** 2, weights=weights[group == g]))
            for g in [0, 1]]


def model(variant):
    assert not (base.ROOT / 'model_report.json').exists(), 'Do not refit a frozen model'
    q = json.loads(base.PROTOCOL.read_text()); t = training(); x = base.encode(t)
    y = t.target.to_numpy(); w = 1 / t.groupby('date').code.transform('size').to_numpy()
    group = t.training_group.to_numpy(); all_rows = np.ones(len(t), dtype=bool)
    bias_stats = group_stats(y, w, group, all_rows)
    reference = None
    if variant == 'ordinary':
        reference = GradientBoostingRegressor(loss='squared_error', **PARAMETERS).fit(x, y, sample_weight=w)
        bias = float(np.ravel(reference.init_.constant_)[0])
    else:
        bias = shared_direction(bias_stats['means'])
    score = np.full(len(t), bias); trees = []; losses = []; rng = np.random.RandomState(20260927)
    zero_leaves = 0; leaves = 0
    for index in range(64):
        residual = y - score
        before = group_loss(y, score, w, group)
        estimator = (reference.estimators_[index, 0] if reference is not None else
            DecisionTreeRegressor(criterion='friedman_mse', max_depth=3, min_samples_leaf=300,
                                  random_state=rng).fit(x, residual, sample_weight=w))
        tr = estimator.tree_
        tree = dict(feature=tr.feature.tolist(), threshold=tr.threshold.tolist(), children_left=tr.children_left.tolist(),
            children_right=tr.children_right.tolist(), n_node_samples=tr.n_node_samples.tolist(),
            weighted_n_node_samples=tr.weighted_n_node_samples.tolist(), raw_value=tr.value.reshape(-1).tolist(),
            value=tr.value.reshape(-1).tolist(), impurity=tr.impurity.tolist(), half_leaf_stats={})
        terminal = estimator.apply(x)
        for node in np.flatnonzero(tr.children_left < 0):
            mask = terminal == node; stats = group_stats(residual, w, group, mask)
            update = shared_direction(stats['means']) if variant == 'consistent' else float(tree['raw_value'][node])
            tree['value'][node] = update; tree['half_leaf_stats'][str(node)] = stats
            leaves += 1; zero_leaves += int(update == 0)
        score += .05 * np.asarray(tree['value'])[terminal]
        after = group_loss(y, score, w, group)
        if variant == 'consistent':
            assert np.all(np.asarray(after) <= np.asarray(before) + 2e-12)
        losses.append(dict(tree=index, before=before, after=after)); trees.append(tree)
    r = dict(protocol_sha256=sha(base.PROTOCOL), feature_report_sha256=sha(inputs.ROOT / 'feature_report.json'),
        label_report_sha256=sha(labels.ROOT / 'full_label_report.json'), variant=variant,
        parameters=PARAMETERS, feature_names=list(inputs.EXPRESSIONS), rows=len(t), days=t.date.nunique(),
        training_start=q['training_start'], training_split=q['training_split'], training_end=q['training_end'],
        group_rows=q['group_rows'], group_days=q['group_days'], last_observation=t.next_date.max(),
        learning_rate=.05, bias=bias, bias_half_stats=bias_stats, trees=trees, half_losses=losses,
        leaves=leaves, zero_update_leaves=zero_leaves, score_is_not_probability=True,
        sklearn_ordinary_predictions_reproduced=reference is not None,
        new_2025_score_groups_read=bool(t.date.ge('2025-01-01').any()), new_2025H2_score_groups_read=False,
        new_2026_prices_read=False, no_exit_rules=True)
    np.testing.assert_allclose(score, base.predict(x, r), rtol=0, atol=2e-12)
    if reference is not None:
        np.testing.assert_allclose(score, reference.predict(x), rtol=0, atol=2e-12)
    r['thresholds'] = [dict(id=i, training_quantile=v, threshold=float(np.quantile(score, v)))
                       for i, v in enumerate(base.QUANTILES)]
    base.ROOT.mkdir(parents=True, exist_ok=True); save_json(base.ROOT / 'model_report.json', r)
    return {k: v for k, v in r.items() if k not in ['trees', 'half_losses']}


def verify_model(variant):
    r = json.loads((base.ROOT / 'model_report.json').read_text()); q = json.loads(base.PROTOCOL.read_text())
    for key, path in [('protocol_sha256', base.PROTOCOL), ('feature_report_sha256', inputs.ROOT / 'feature_report.json'),
                      ('label_report_sha256', labels.ROOT / 'full_label_report.json')]:
        assert r[key] == sha(path)
    assert r['parameters'] == PARAMETERS and r['variant'] == variant and r['learning_rate'] == .05
    assert all(r[k] == q[k] for k in ['training_start', 'training_split', 'training_end', 'group_rows', 'group_days'])
    names = list(inputs.EXPRESSIONS); assert names == r['feature_names'] and len(names) == 48
    c = base.conn(); encoded = ','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::INT AS {n}' for n in names)
    d = c.sql(f'''WITH known AS (
        SELECT date,code,next_date,opportunity15 FROM read_parquet('{labels.ROOT}/full_labels.parquet')
        WHERE known15 AND date>='{q['training_start']}' AND next_date<'{q['training_end']}'
        AND (date>='{q['training_split']}' OR next_date<'{q['training_split']}')),
        targets AS(SELECT *,opportunity15-avg(opportunity15) OVER(PARTITION BY date) AS target FROM known),
        joined AS(SELECT date,code,next_date,target,{encoded},
            (date>='{q['training_split']}')::INT AS training_group
            FROM targets JOIN read_parquet('{inputs.ROOT}/features.parquet') f USING(date,code)
            WHERE formula_input_valid)
        SELECT *,1./count(*) OVER(PARTITION BY date) AS w FROM joined ORDER BY date,code''').df(); c.close()
    original = training()
    pd.testing.assert_frame_equal(d[['date', 'code', 'next_date', 'training_group']],
                                  original[['date', 'code', 'next_date', 'training_group']], check_exact=True, check_dtype=False)
    assert len(d) == r['rows'] == q['expected_rows'] and d.date.nunique() == r['days'] == q['expected_days']
    assert d.next_date.max() == r['last_observation'] < q['training_end']
    x = d[names].to_numpy(dtype='int32'); y = d.target.to_numpy(); w = d.w.to_numpy(); group = d.training_group.to_numpy()
    np.testing.assert_array_equal(x, base.encode(original)); np.testing.assert_allclose(y, original.target, rtol=0, atol=2e-12)

    def independent_stats(mask, residual):
        rows, weights, means = [], [], []
        for g in [0, 1]:
            ix = np.flatnonzero(mask & (group == g)); total = np.sum(w[ix])
            rows.append(len(ix)); weights.append(float(total))
            means.append(float(np.dot(w[ix], residual[ix]) / total) if len(ix) else None)
        return dict(rows=rows, weights=weights, means=means)

    def independently_bound(means):
        a, b = means
        if a is None or b is None or a * b <= 0:
            return 0.
        return np.sign(a) * min(abs(a), abs(b))

    def check_stats(expected, actual):
        assert expected['rows'] == actual['rows']
        np.testing.assert_allclose(expected['weights'], actual['weights'], rtol=0, atol=1e-8)
        for a, b in zip(expected['means'], actual['means']):
            if a is None:
                assert b is None
            else:
                np.testing.assert_allclose(a, b, rtol=0, atol=2e-10)

    bias_stats = independent_stats(np.ones(len(d), dtype=bool), y); check_stats(bias_stats, r['bias_half_stats'])
    bias = independently_bound(bias_stats['means']) if variant == 'consistent' else np.dot(w, y) / w.sum()
    np.testing.assert_allclose(bias, r['bias'], rtol=0, atol=2e-12)
    score = np.full(len(d), r['bias']); checks = 0; leaves = 0; zeros = 0
    assert len(r['trees']) == len(r['half_losses']) == 64
    for index, tree in enumerate(r['trees']):
        residual = y - score; masks = {0: np.ones(len(d), dtype=bool)}; depth = {0: 0}; step = np.empty(len(d))
        assert len(tree['feature']) <= 15
        for node, left in enumerate(tree['children_left']):
            mask = masks[node]; weights = w[mask]; mean = np.average(residual[mask], weights=weights)
            assert int(mask.sum()) == tree['n_node_samples'][node] and depth[node] <= 3
            np.testing.assert_allclose(weights.sum(), tree['weighted_n_node_samples'][node], rtol=0, atol=1e-8)
            np.testing.assert_allclose(mean, tree['raw_value'][node], rtol=0, atol=2e-10)
            np.testing.assert_allclose(np.average((residual[mask] - mean) ** 2, weights=weights),
                                       tree['impurity'][node], rtol=0, atol=2e-10)
            if left < 0:
                assert mask.sum() >= 300
                stats = independent_stats(mask, residual); check_stats(stats, tree['half_leaf_stats'][str(node)])
                value = independently_bound(stats['means']) if variant == 'consistent' else mean
                np.testing.assert_allclose(value, tree['value'][node], rtol=0, atol=2e-10)
                step[mask] = value; leaves += 1; zeros += int(value == 0)
            else:
                np.testing.assert_allclose(tree['value'][node], mean, rtol=0, atol=2e-10)
                right = tree['children_right'][node]; lower = x[:, tree['feature'][node]] <= tree['threshold'][node]
                masks[left], masks[right] = mask & lower, mask & ~lower
                depth[left] = depth[right] = depth[node] + 1
            checks += 1
        before = [np.dot(w[group == g], (y[group == g] - score[group == g]) ** 2) / w[group == g].sum() for g in [0, 1]]
        score += .05 * step
        after = [np.dot(w[group == g], (y[group == g] - score[group == g]) ** 2) / w[group == g].sum() for g in [0, 1]]
        receipt = r['half_losses'][index]; assert receipt['tree'] == index
        np.testing.assert_allclose(before, receipt['before'], rtol=0, atol=2e-10)
        np.testing.assert_allclose(after, receipt['after'], rtol=0, atol=2e-10)
        if variant == 'consistent':
            assert np.all(np.asarray(after) <= np.asarray(before) + 2e-12)
    assert leaves == r['leaves'] and zeros == r['zero_update_leaves']
    np.testing.assert_allclose(score, base.predict(x, r), rtol=0, atol=2e-10)
    for quantile, cut in zip(base.QUANTILES, r['thresholds']):
        assert quantile == cut['training_quantile']
        np.testing.assert_allclose(np.quantile(score, quantile), cut['threshold'], rtol=0, atol=2e-10)
    proof = dict(passed=True, model_report_sha256=sha(base.ROOT / 'model_report.json'), rows=len(d), node_checks=checks,
        target_before_feature_intersection_rebuilt=True, purged_boundary_and_date_weights_rebuilt=True,
        all_half_leaf_means_and_updates_rebuilt=True, all_128_half_losses_rebuilt=True,
        both_half_losses_never_increase=variant == 'consistent', zero_update_leaves=zeros,
        new_2025_score_groups_read=bool(d.date.ge('2025-01-01').any()), new_2025H2_score_groups_read=False,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(base.ROOT / 'model_verification.json', proof); return proof


if __name__ == '__main__':
    from .tail_formula_offset_logit48 import verify_scores
    from .tail_formula_boundary_evaluation import analyze
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['model', 'verify_model', 'scores', 'verify_scores', 'freeze', 'verify', 'analyze'])
    p.add_argument('--variant', choices=['ordinary', 'consistent'], required=True)
    p.add_argument('--fold', choices=['2024', 'recent', 'combined'], required=True)
    a = p.parse_args(); setup(a.variant, a.fold)
    if a.stage == 'analyze':
        assert all((root(v, f) / 'selection_verification.json').exists()
                   for v in ['ordinary', 'consistent'] for f in ['2024', 'recent', '2025'])
        result = analyze(linkage.COMBINED if a.fold == 'combined' else base.ROOT,
                         linkage.PROTOCOL if a.fold == 'combined' else base.PROTOCOL)
    elif a.fold == 'combined':
        assert a.stage in ['freeze', 'verify']
        result = linkage.combine() if a.stage == 'freeze' else linkage.verify_combined()
    elif a.stage in ['model', 'verify_model']:
        result = globals()[a.stage](a.variant)
    elif a.stage == 'verify_scores':
        result = verify_scores()
    elif a.stage in ['freeze', 'verify']:
        result = getattr(study, a.stage)()
    else:
        result = base.scores()
    print(json.dumps(result, ensure_ascii=False, indent=2))
