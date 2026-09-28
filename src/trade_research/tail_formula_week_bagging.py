"""Average independent shallow trees fitted to whole-week bootstrap samples."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from . import tail_formula_additive as base
from . import tail_formula_before1000_model as original
from . import tail_formula_boundary_evaluation as evaluation
from . import tail_formula_context_2024 as linkage
from . import tail_formula_recent as study
from . import tail_formula_relative as relative
from .corporate_cash import save_json, sha

STEM = 'tail_formula_week_bagging'
PROTOCOL = Path('config') / (STEM + '_protocol.json')
PARAMETERS = dict(n_estimators=64, max_depth=3, min_samples_leaf=300,
    random_state=20260927, criterion='friedman_mse', bootstrap_unit='calendar_week',
    bootstrap_with_replacement=True, aggregation='equal_tree_mean')
EXPORT_MULTIPLIER = .05  # Existing native IF exporter; not a boosting step.


def setup(fold):
    original.STEM = STEM
    original.setup(fold)
    for f in ['2024', 'recent']:
        p = json.loads((Path('config') / (STEM + '_' + f + '_protocol.json')).read_text())
        assert p['week_bagging_protocol_sha256'] == sha(PROTOCOL)
        assert p['parameters'] == PARAMETERS


def multiplicities(week_values, drawn):
    counts = pd.Series(drawn).value_counts().to_dict()
    return np.asarray([counts.get(week, 0) for week in week_values], dtype='int32')


def model():
    root = base.ROOT
    assert not (root / 'model_report.json').exists(), 'Do not refit a frozen model'
    t = relative.training('relative'); x = base.encode(t); y = t.target.to_numpy()
    dates = pd.to_datetime(t.date)
    weeks = (dates - pd.to_timedelta(dates.dt.weekday, unit='D')).dt.strftime('%Y-%m-%d').to_numpy()
    unique = np.unique(weeks); w = 1 / t.groupby('date').code.transform('size').to_numpy()
    rng = np.random.default_rng(PARAMETERS['random_state']); trees = []; schedule = []
    score = np.zeros(len(t)); n = PARAMETERS['n_estimators']
    for i in range(n):
        drawn = rng.choice(unique, size=len(unique), replace=True).tolist()
        mult = multiplicities(weeks, drawn); active = mult > 0
        estimator = DecisionTreeRegressor(criterion='friedman_mse', max_depth=3,
            min_samples_leaf=300, random_state=PARAMETERS['random_state'] + i).fit(
                x[active], y[active], sample_weight=(w * mult)[active])
        q = estimator.tree_
        raw = q.value.reshape(-1)
        tree = dict(feature=q.feature.tolist(), threshold=q.threshold.tolist(),
            children_left=q.children_left.tolist(), children_right=q.children_right.tolist(),
            n_node_samples=q.n_node_samples.tolist(), weighted_n_node_samples=q.weighted_n_node_samples.tolist(),
            raw_value=raw.tolist(), value=(raw / (n * EXPORT_MULTIPLIER)).tolist(), impurity=q.impurity.tolist())
        trees.append(tree)
        prediction = estimator.predict(x)
        np.testing.assert_allclose(prediction, raw[base.leaf_indices(x, tree)], rtol=0, atol=2e-12)
        score += prediction / n
        schedule.append(dict(tree=i, drawn_week_starts=drawn, unique_weeks=len(set(drawn)),
            rows=int(active.sum()), days=int(t.loc[active, 'date'].nunique()), weighted_days=float((w * mult).sum())))
    start, end, _ = relative.training_scope()
    p = json.loads(base.PROTOCOL.read_text())
    assert len(t) == p['expected_rows'] and t.date.nunique() == p['expected_training_days'] == 241
    assert len(unique) == p['expected_training_weeks'] == 52 and t.next_date.max() < end
    r = dict(protocol_sha256=sha(base.PROTOCOL), feature_report_sha256=sha(base.FEATURES / 'feature_report.json'),
        label_report_sha256=sha(base.SOURCE / 'full_label_report.json'), rows=len(t), days=t.date.nunique(),
        last_observation=t.next_date.max(), parameters=PARAMETERS, feature_names=list(base.EXPRESSIONS),
        variant='independent_week_bootstrap_average', learning_rate=EXPORT_MULTIPLIER, bias=0., trees=trees,
        training_start=start, training_end=end, training_week_starts=unique.tolist(), draws_per_tree=len(unique),
        sampling_schedule=schedule, tree_values_are_export_scaled=True, aggregation_weight=1/n,
        export_multiplier_is_not_a_boosting_step=True, all_trees_fit_original_target=True,
        new_2025_score_groups_read=bool(t.date.ge('2025-01-01').any()), new_2025H2_score_groups_read=False,
        new_2026_prices_read=False, no_exit_rules=True)
    exported_score = base.predict(x, r)
    np.testing.assert_allclose(score, exported_score, rtol=0, atol=2e-12)
    r['thresholds'] = [dict(id=i, training_quantile=q, threshold=float(np.quantile(exported_score, q)))
                       for i, q in enumerate(base.QUANTILES)]
    root.mkdir(parents=True, exist_ok=True); save_json(root / 'model_report.json', r)
    return {k: v for k, v in r.items() if k not in ['trees', 'sampling_schedule', 'training_week_starts']}


def verify_model():
    root = base.ROOT; r = json.loads((root / 'model_report.json').read_text())
    p = json.loads(base.PROTOCOL.read_text()); start, end, where = relative.training_scope()
    for key, path in [('protocol_sha256', base.PROTOCOL), ('feature_report_sha256', base.FEATURES / 'feature_report.json'),
                      ('label_report_sha256', base.SOURCE / 'full_label_report.json')]:
        assert r[key] == sha(path)
    assert r['parameters'] == PARAMETERS == p['parameters']
    assert r['training_start'] == start and r['training_end'] == end and r['last_observation'] < end
    assert r['variant'] == 'independent_week_bootstrap_average' and r['bias'] == 0.
    assert r['learning_rate'] == .05 and r['aggregation_weight'] == 1/64
    assert r['tree_values_are_export_scaled'] and r['export_multiplier_is_not_a_boosting_step']
    assert r['all_trees_fit_original_target']
    names = list(base.EXPRESSIONS); assert len(names) == 48 and names == r['feature_names']
    f = base.feature_inputs()[['date', 'code', 'formula_input_valid', *names]]
    c = base.conn(); c.register('features', f)
    c.execute(f'''CREATE VIEW targets AS WITH l AS(SELECT date,code,opportunity15 AS utility
        FROM read_parquet('{base.SOURCE}/full_labels.parquet') WHERE {where} AND known15)
        SELECT date,code,utility-avg(utility) OVER(PARTITION BY date) AS target FROM l''')
    encoded = ','.join(f'floor(least(greatest(100*{name}+10000+.000001,0),999999))::INT AS {name}' for name in names)
    d = c.sql('''SELECT date,code,target,1./count(*) OVER(PARTITION BY date) AS w,
        strftime(date_trunc('week',date::DATE),'%Y-%m-%d') AS week_start,''' + encoded + '''
        FROM features JOIN targets USING(date,code) WHERE formula_input_valid ORDER BY date,code''').df(); c.close()
    train = relative.training('relative')
    pd.testing.assert_frame_equal(d[['date', 'code']], train[['date', 'code']], check_exact=True)
    assert len(d) == r['rows'] == p['expected_rows'] and d.date.nunique() == r['days'] == 241
    assert train.next_date.max() == r['last_observation']
    np.testing.assert_allclose(d.target, train.target, rtol=0, atol=2e-12)
    x = d[names].to_numpy(dtype='int32'); y = d.target.to_numpy(); w = d.w.to_numpy()
    np.testing.assert_array_equal(x, base.encode(train))
    weeks = sorted(d.week_start.unique()); assert len(weeks) == r['draws_per_tree'] == 52
    assert weeks == r['training_week_starts'] and len(r['trees']) == len(r['sampling_schedule']) == 64
    rng = np.random.default_rng(20260927); score = np.zeros(len(d)); checks = 0
    for i, (tree, item) in enumerate(zip(r['trees'], r['sampling_schedule'])):
        drawn = rng.choice(weeks, size=len(weeks), replace=True).tolist()
        assert item['tree'] == i and item['drawn_week_starts'] == drawn
        # Independently count the scheduled week draws; no call to the producer helper.
        counts = {week: drawn.count(week) for week in weeks}
        mult = d.week_start.map(counts).to_numpy(); active = mult > 0; weights = w * mult
        assert item['unique_weeks'] == sum(value > 0 for value in counts.values())
        assert item['rows'] == int(active.sum()) and item['days'] == d.loc[active, 'date'].nunique()
        np.testing.assert_allclose(item['weighted_days'], weights.sum(), rtol=0, atol=1e-9)
        assert pd.Series(mult).groupby(d.week_start).nunique().eq(1).all()
        masks = {0: np.ones(len(d), dtype=bool)}; levels = {0: 0}; terminal = np.empty(len(d))
        assert len(tree['feature']) <= 15
        for node, left in enumerate(tree['children_left']):
            mask = masks[node]; fit = mask & active; ww = weights[fit]
            assert int(fit.sum()) == tree['n_node_samples'][node] and levels[node] <= 3
            mean = np.average(y[fit], weights=ww)
            variance = np.average((y[fit] - mean)**2, weights=ww)
            np.testing.assert_allclose(ww.sum(), tree['weighted_n_node_samples'][node], rtol=0, atol=1e-8)
            np.testing.assert_allclose(mean, tree['raw_value'][node], rtol=0, atol=2e-10)
            np.testing.assert_allclose(mean/64/.05, tree['value'][node], rtol=0, atol=2e-10)
            np.testing.assert_allclose(variance, tree['impurity'][node], rtol=0, atol=2e-10)
            right = tree['children_right'][node]
            if left < 0:
                assert right < 0 and fit.sum() >= 300; terminal[mask] = mean
            else:
                split = x[:, tree['feature'][node]] <= tree['threshold'][node]
                masks[left], masks[right] = mask & split, mask & ~split
                levels[left] = levels[right] = levels[node] + 1
            checks += 1
        score += terminal / 64
    np.testing.assert_allclose(score, base.predict(x, r), rtol=0, atol=2e-10)
    for q, item in zip(base.QUANTILES, r['thresholds']):
        assert q == item['training_quantile']
        np.testing.assert_allclose(np.quantile(score, q), item['threshold'], rtol=0, atol=2e-10)
    proof = dict(passed=True, model_report_sha256=sha(root / 'model_report.json'), rows=len(d), node_checks=checks,
        all_original_targets_integer_inputs_and_date_weights_rebuilt=True,
        all_week_draws_multiplicities_and_whole_week_memberships_rebuilt=True,
        all_independent_target_means_variances_and_native_export_scaling_rebuilt=True,
        training_weeks=len(weeks), draws_per_tree=52,
        minimum_unique_weeks=min(s['unique_weeks'] for s in r['sampling_schedule']),
        maximum_unique_weeks=max(s['unique_weeks'] for s in r['sampling_schedule']),
        minimum_sampled_days=min(s['days'] for s in r['sampling_schedule']),
        maximum_sampled_days=max(s['days'] for s in r['sampling_schedule']),
        training_start=start, training_end=end, new_2025H2_score_groups_read=False,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(root / 'model_verification.json', proof); return proof


if __name__ == '__main__':
    from .tail_formula_offset_logit48 import verify_scores
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['model', 'verify_model', 'scores', 'verify_scores', 'freeze', 'verify', 'analyze'])
    p.add_argument('--fold', choices=['2024', 'recent', 'combined'], default='2024')
    a = p.parse_args(); setup(a.fold)
    if a.stage == 'analyze':
        assert all((Path('data/research') / (STEM + '_' + f) / 'selection_verification.json').exists()
                   for f in ['2024', 'recent', '2025'])
        root = linkage.COMBINED if a.fold == 'combined' else base.ROOT
        result = evaluation.analyze(root, linkage.PROTOCOL if a.fold == 'combined' else base.PROTOCOL)
    elif a.fold == 'combined':
        assert a.stage in ['freeze', 'verify']
        result = linkage.combine() if a.stage == 'freeze' else linkage.verify_combined()
    elif a.stage in ['model', 'verify_model']:
        result = globals()[a.stage]()
    elif a.stage == 'verify_scores':
        result = verify_scores()
    elif a.stage in ['freeze', 'verify']:
        result = getattr(study, a.stage)()
    else:
        result = base.scores()
    print(json.dumps(result, ensure_ascii=False, indent=2))
