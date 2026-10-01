"""Fixed same-day relative opportunity target and its independent tree audit."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from . import tail_formula_additive as base
from .research_io import save_json, sha

PROTOCOL = Path('config/tail_formula_emotion_transition_model_protocol.json')

def training_scope():
    from datetime import date
    r = json.loads(PROTOCOL.read_text())
    start, end = (r.get('training_start'), r.get('training_end', '2025-01-01'))
    assert date.fromisoformat(end).isoformat() == end
    assert start is None or date.fromisoformat(start).isoformat() == start
    where = f"next_date<'{end}'" + (f" AND date>='{start}'" if start else '')
    return (start, end, where)

def training(variant):
    assert variant == 'relative'
    start, end, where = training_scope()
    t = base.training(start=start, end=end)
    c = base.conn()
    labels = c.execute(f'SELECT date,code,opportunity15 FROM read_parquet(?)\n        WHERE {where} AND known15', [str(base.SOURCE / 'full_labels.parquet')]).df()
    c.close()
    assert np.isfinite(labels.opportunity15).all() and labels.opportunity15.isin([0, 1]).all()
    labels['target'] = labels.opportunity15 - labels.groupby('date').opportunity15.transform('mean')
    t = t.merge(labels[['date', 'code', 'target']], on=['date', 'code'], validate='one_to_one')
    return t.sort_values(['date', 'code']).reset_index(drop=True)

def model(variant):
    assert variant == 'relative'
    root = base.ROOT
    if (root / 'model_report.json').exists():
        raise ValueError('Do not refit the frozen relative model')
    root.mkdir(parents=True, exist_ok=True)
    train = training(variant)
    start, end, _ = training_scope()
    x = base.encode(train)
    y = train.target.to_numpy()
    w = 1 / train.groupby('date').code.transform('size').to_numpy()
    depth = json.loads(PROTOCOL.read_text()).get('model_max_depth', 2)
    assert depth == 3
    model = GradientBoostingRegressor(loss='squared_error', n_estimators=64, learning_rate=0.05, max_depth=depth, min_samples_leaf=300, subsample=1.0, random_state=20260927)
    model.fit(x, y, sample_weight=w)
    trees = []
    for estimator in model.estimators_.ravel():
        t = estimator.tree_
        trees.append(dict(feature=t.feature.tolist(), threshold=t.threshold.tolist(), children_left=t.children_left.tolist(), children_right=t.children_right.tolist(), n_node_samples=t.n_node_samples.tolist(), weighted_n_node_samples=t.weighted_n_node_samples.tolist(), value=t.value.reshape(-1).tolist(), impurity=t.impurity.tolist()))
    r = dict(protocol_sha256=sha(PROTOCOL), feature_report_sha256=sha(base.FEATURES / 'feature_report.json'), label_report_sha256=sha(base.SOURCE / 'full_label_report.json'), rows=len(train), days=train.date.nunique(), last_observation=train.next_date.max(), parameters=model.get_params(), feature_names=list(base.EXPRESSIONS), variant=variant, learning_rate=0.05, bias=float(np.ravel(model.init_.constant_)[0]), trees=trees, training_start=start, training_end=end, new_2025_score_groups_read=bool(train.date.ge('2025-01-01').any()), new_2025H2_score_groups_read=bool(train.date.ge('2025-07-01').any()), new_2026_prices_read=False, no_exit_rules=True)
    manual = base.predict(x, r)
    np.testing.assert_allclose(manual, model.predict(x), rtol=0, atol=2e-12)
    r['thresholds'] = [dict(id=i, training_quantile=q, threshold=float(np.quantile(manual, q))) for i, q in enumerate(base.QUANTILES)]
    save_json(root / 'model_report.json', r)
    return {k: v for k, v in r.items() if k != 'trees'}

def verify_model(variant, encoding_multiplier=1, training_cost_bps=15):
    assert variant == 'relative'
    root = base.ROOT
    r = json.loads((root / 'model_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['variant'] == variant
    assert r['feature_report_sha256'] == sha(base.FEATURES / 'feature_report.json')
    assert r['label_report_sha256'] == sha(base.SOURCE / 'full_label_report.json')
    start, end, where = training_scope()
    assert r.get('training_start') == start and r.get('training_end', '2025-01-01') == end
    config = json.loads(PROTOCOL.read_text())
    assert encoding_multiplier == 1 and config.get('encoding_multiplier', 1) == 1
    depth = config.get('model_max_depth', 2)
    minimum_days = config.get('minimum_leaf_training_days', 0)
    assert depth == 3 and minimum_days == 0
    assert r['parameters']['max_depth'] == depth
    columns = ['date', 'code', 'formula_input_valid', *base.EXPRESSIONS]
    assert len({name.casefold() for name in columns}) == len(columns)
    f = base.feature_inputs()[columns]
    c = base.conn()
    c.register('features', f)
    assert training_cost_bps == 15
    utility = 'opportunity15'
    target = 'utility-avg(utility) OVER(PARTITION BY date)'
    c.execute(f"CREATE VIEW targets AS WITH l AS(SELECT date,code,{utility} AS utility\n            FROM read_parquet('{base.SOURCE}/full_labels.parquet') WHERE {where} AND known15)\n            SELECT date,code,{target} AS target FROM l")
    names = list(base.EXPRESSIONS)
    multiplier = '' if encoding_multiplier == 1 else '10*'
    encoded = ','.join((f'floor({multiplier}least(greatest(100*{n}+10000+.000001,0),999999))::INT AS {n}' for n in names))
    d = c.sql('SELECT date,code,target,1./count(*) OVER(PARTITION BY date) AS w,' + encoded + '\n        FROM features JOIN targets USING(date,code) WHERE formula_input_valid ORDER BY date,code').df()
    original = training(variant)
    assert original.next_date.lt(end).all() and (start is None or original.date.ge(start).all())
    assert r['new_2025H2_score_groups_read'] == bool(original.date.ge('2025-07-01').any())
    np.testing.assert_allclose(d.target, original.target, rtol=0, atol=2e-12)
    assert d[['date', 'code']].equals(original[['date', 'code']])
    x = d[names].to_numpy(dtype='int32')
    y = d.target.to_numpy()
    w = d.w.to_numpy()
    np.testing.assert_array_equal(x, base.encode(original))
    np.testing.assert_allclose(r['bias'], np.average(y, weights=w), rtol=0, atol=2e-12)
    score = np.full(len(d), r['bias'])
    checks = 0
    minimum_observed_days = len(d)
    dates = np.unique(d.date.to_numpy(), return_inverse=True)[1] if minimum_days else None
    assert len(r['trees']) == 64 and r['learning_rate'] == 0.05
    for tree in r['trees']:
        assert len(tree['feature']) <= 2 ** (depth + 1) - 1
        residual = y - score
        masks = {0: np.ones(len(d), dtype=bool)}
        levels = {0: 0}
        terminal = np.empty(len(d))
        for i in range(len(tree['feature'])):
            assert levels[i] <= depth
            mask = masks[i]
            weights = w[mask]
            if minimum_days:
                observed_days = len(np.unique(dates[mask]))
                assert tree['training_days'][i] == observed_days
            assert int(mask.sum()) == tree['n_node_samples'][i]
            mean = np.average(residual[mask], weights=weights)
            variance = np.average((residual[mask] - mean) ** 2, weights=weights)
            np.testing.assert_allclose(weights.sum(), tree['weighted_n_node_samples'][i], rtol=0, atol=1e-08)
            np.testing.assert_allclose(mean, tree['value'][i], rtol=0, atol=2e-10)
            np.testing.assert_allclose(variance, tree['impurity'][i], rtol=0, atol=2e-10)
            left, right = (tree['children_left'][i], tree['children_right'][i])
            if left < 0:
                assert mask.sum() >= 300
                if minimum_days:
                    assert observed_days >= minimum_days
                    minimum_observed_days = min(minimum_observed_days, observed_days)
                terminal[mask] = mean
            else:
                lower = x[:, tree['feature'][i]] <= tree['threshold'][i]
                masks[left], masks[right] = (mask & lower, mask & ~lower)
                levels[left] = levels[right] = levels[i] + 1
            checks += 1
        score += 0.05 * terminal
    for q, t in zip(base.QUANTILES, r['thresholds']):
        assert q == t['training_quantile']
        np.testing.assert_allclose(np.quantile(score, q), t['threshold'], rtol=0, atol=2e-10)
    proof = dict(passed=True, model_report_sha256=sha(root / 'model_report.json'), variant=variant, rows=len(d), node_checks=checks, all_targets_integer_inputs_day_weights_residual_means_and_variances_rebuilt=True, training_start=start, training_end=end, new_2025_score_groups_read=bool(original.date.ge('2025-01-01').any()), new_2025H2_score_groups_read=bool(original.date.ge('2025-07-01').any()), new_2026_prices_read=False, no_exit_rules=True)
    if encoding_multiplier != 1:
        proof['encoding_multiplier'] = encoding_multiplier
    if training_cost_bps != 15:
        proof['training_opportunity_cost_bps'] = training_cost_bps
        proof['original_known15_training_population_preserved'] = True
    if minimum_days:
        proof.update(minimum_leaf_training_days_required=minimum_days, minimum_leaf_training_days_observed=minimum_observed_days, all_node_date_support_rebuilt=True)
    save_json(root / 'model_verification.json', proof)
    return proof
