"""Learn relative stock log odds against a fixed, training-only daily baseline."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.tree import DecisionTreeRegressor

from . import tail_formula_additive as base
from . import tail_formula_context_2024 as linkage
from . import tail_formula_float as inputs
from . import tail_formula_recent as study
from . import tail_formula_relative as relative
from .corporate_cash import save_json, sha

STEM = 'tail_formula_offset_logit48'
PARAMETERS = dict(loss='binomial_log_loss_with_fixed_training_day_offset',
    n_estimators=64, learning_rate=.05, max_depth=3, min_samples_leaf=300,
    subsample=1., random_state=20260927, criterion='friedman_mse', newton_leaf_clip=2.)


def setup(fold):
    protocol = Path('config') / f'{STEM}_{fold}_protocol.json'
    p = json.loads(protocol.read_text())
    if fold == 'combined':
        linkage.ROOT = Path('data/research') / f'{STEM}_2024'
        linkage.H2 = Path('data/research') / f'{STEM}_recent'
        linkage.COMBINED = Path('data/research') / f'{STEM}_2025'
        linkage.PROTOCOL = protocol
        linkage.H2_SELECTION_SHA = linkage.H2_MODEL_SHA = None
        linkage.H2_OUTCOMES_PREVIOUSLY_SEEN = False
        assert [sha(Path(x)) for x in p['fold_protocols']] == p['fold_protocol_sha256']
    else:
        inputs.setup(fold)
        root = Path('data/research') / f'{STEM}_{fold}'
        base.ROOT = study.ROOT = root
        base.PROTOCOL = relative.PROTOCOL = study.PROTOCOL = protocol
        assert p['parameters'] == PARAMETERS
        assert p['feature_report_sha256'] == sha(base.FEATURES / 'feature_report.json')
        assert p['label_report_sha256'] == sha(base.SOURCE / 'full_label_report.json')
        assert p['training_end'] == p['evaluation_start'] <= '2025-07-01'


def date_baselines(labels):
    """Use all known base-pool labels, before applying formula input validity."""
    known = labels.loc[labels.known15]
    assert np.isfinite(known.opportunity15).all() and known.opportunity15.isin([0, 1]).all()
    return known.groupby('date', as_index=False).agg(
        rows=('opportunity15', 'size'), success=('opportunity15', 'sum'),
        rate=('opportunity15', 'mean')).sort_values('date').reset_index(drop=True)


def training():
    start, end, where = relative.training_scope()
    daily = date_baselines(base.labels(where))
    t = base.training(start=start, end=end).merge(
        daily[['date', 'rate']], on='date', how='left', validate='many_to_one')
    assert np.isfinite(t.rate).all() and t.rate.between(0, 1).all()
    return t.sort_values(['date', 'code']).reset_index(drop=True), daily


def probabilities(rate, score):
    # logit(0/1) is -/+ infinity, so these constant-response days contribute
    # exactly zero gradient and curvature without dropping their observations.
    return expit(logit(np.asarray(rate, dtype=float)) + np.asarray(score, dtype=float))


def newton_update(residual, probability, weights):
    numerator = float(np.sum(weights * residual))
    curvature = float(np.sum(weights * probability * (1 - probability)))
    if curvature == 0:
        assert numerator == 0
        raw = 0.
    else:
        raw = numerator / curvature
    return float(np.clip(raw, -PARAMETERS['newton_leaf_clip'], PARAMETERS['newton_leaf_clip']))


def model():
    root = base.ROOT
    assert not (root / 'model_report.json').exists(), 'Do not refit a frozen model'
    t, daily = training()
    x = base.encode(t); y = t.opportunity15.to_numpy(); rate = t.rate.to_numpy()
    w = 1 / t.groupby('date').code.transform('size').to_numpy()
    score = np.zeros(len(t)); trees = []; clipped = 0; zero_curvature = 0
    rng = np.random.RandomState(PARAMETERS['random_state'])
    for _ in range(PARAMETERS['n_estimators']):
        probability = probabilities(rate, score)
        residual = y - probability
        estimator = DecisionTreeRegressor(criterion='friedman_mse', max_depth=3,
            min_samples_leaf=300, random_state=rng).fit(x, residual, sample_weight=w)
        q = estimator.tree_
        tree = dict(feature=q.feature.tolist(), threshold=q.threshold.tolist(),
            children_left=q.children_left.tolist(), children_right=q.children_right.tolist(),
            n_node_samples=q.n_node_samples.tolist(), weighted_n_node_samples=q.weighted_n_node_samples.tolist(),
            value=q.value.reshape(-1).tolist(), gradient_value=q.value.reshape(-1).tolist(),
            impurity=q.impurity.tolist())
        leaves = estimator.apply(x)
        np.testing.assert_array_equal(leaves, base.leaf_indices(x, tree))
        for node in np.flatnonzero(q.children_left < 0):
            mask = leaves == node
            value = newton_update(residual[mask], probability[mask], w[mask])
            curvature = np.sum(w[mask] * probability[mask] * (1 - probability[mask]))
            zero_curvature += int(curvature == 0)
            clipped += int(curvature > 0 and abs(np.sum(w[mask] * residual[mask])) > 2 * curvature)
            tree['value'][node] = value
        trees.append(tree)
        score += .05 * np.asarray(tree['value'])[leaves]
    start, end, _ = relative.training_scope()
    r = dict(protocol_sha256=sha(base.PROTOCOL), feature_report_sha256=sha(base.FEATURES / 'feature_report.json'),
        label_report_sha256=sha(base.SOURCE / 'full_label_report.json'), rows=len(t), days=t.date.nunique(),
        last_observation=t.next_date.max(), parameters=PARAMETERS, feature_names=list(base.EXPRESSIONS),
        variant='fixed_training_day_offset_logistic', learning_rate=.05, bias=0., trees=trees,
        training_start=start, training_end=end, training_baselines=daily.to_dict('records'),
        constant_baseline_days=int(daily.rate.isin([0, 1]).sum()),
        constant_baseline_training_rows=int(t.rate.isin([0, 1]).sum()),
        clipped_leaf_updates=clipped, zero_curvature_leaf_updates=zero_curvature,
        score_is_not_probability=True, training_offsets_never_used_for_inference=True,
        new_2025_score_groups_read=bool(t.date.ge('2025-01-01').any()), new_2025H2_score_groups_read=False,
        new_2026_prices_read=False, no_exit_rules=True)
    assert t.date.ge(start).all() and t.next_date.lt(end).all()
    np.testing.assert_allclose(score, base.predict(x, r), rtol=0, atol=2e-12)
    r['thresholds'] = [dict(id=i, training_quantile=q, threshold=float(np.quantile(score, q)))
                       for i, q in enumerate(base.QUANTILES)]
    root.mkdir(parents=True, exist_ok=True); save_json(root / 'model_report.json', r)
    return {k: v for k, v in r.items() if k not in ['trees', 'training_baselines']}


def verify_model():
    root = base.ROOT; r = json.loads((root / 'model_report.json').read_text())
    for key, path in [('protocol_sha256', base.PROTOCOL),
                      ('feature_report_sha256', base.FEATURES / 'feature_report.json'),
                      ('label_report_sha256', base.SOURCE / 'full_label_report.json')]:
        assert r[key] == sha(path)
    start, end, where = relative.training_scope()
    assert r['training_start'] == start and r['training_end'] == end and r['last_observation'] < end
    assert r['parameters'] == PARAMETERS and r['bias'] == 0 and r['learning_rate'] == .05
    assert r['variant'] == 'fixed_training_day_offset_logistic'
    assert r['score_is_not_probability'] and r['training_offsets_never_used_for_inference']
    names = list(base.EXPRESSIONS); assert r['feature_names'] == names and len(names) == 48
    c = base.conn()
    c.read_parquet(str(base.FEATURES / 'features.parquet')).create_view('features')
    c.sql(f"SELECT * FROM read_parquet('{base.SOURCE}/full_labels.parquet') WHERE {where} AND known15").create_view('labels')
    c.sql('SELECT date,count(*) AS rows,sum(opportunity15) AS success,avg(opportunity15) AS rate '
          'FROM labels GROUP BY date').create_view('baselines')
    daily = c.sql('SELECT * FROM baselines ORDER BY date').df()
    pd.testing.assert_frame_equal(daily, pd.DataFrame(r['training_baselines']), check_dtype=False, rtol=0, atol=2e-12)
    encoded = ','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::INT AS {n}' for n in names)
    d = c.sql(f'''SELECT l.date,l.code,l.next_date,l.opportunity15,b.rate,{encoded},
        1./count(*) OVER(PARTITION BY l.date) AS w FROM labels l
        JOIN baselines b USING(date) JOIN features f ON f.date=l.date AND f.code=l.code
        WHERE f.formula_input_valid ORDER BY l.date,l.code''').df()
    original, original_daily = training()
    pd.testing.assert_frame_equal(daily, original_daily, check_dtype=False, rtol=0, atol=2e-12)
    pd.testing.assert_frame_equal(d[['date', 'code', 'next_date']], original[['date', 'code', 'next_date']], check_exact=True)
    assert len(d) == r['rows'] and d.date.nunique() == r['days'] and d.next_date.max() == r['last_observation']
    y = d.opportunity15.to_numpy(); rate = d.rate.to_numpy(); w = d.w.to_numpy()
    x = d[names].to_numpy(dtype='int32'); np.testing.assert_array_equal(x, base.encode(original))
    np.testing.assert_allclose(rate, original.rate, rtol=0, atol=2e-12)
    score = np.zeros(len(d)); checks = clipped = zero_curvature = 0
    assert len(r['trees']) == 64
    for tree in r['trees']:
        # An independent odds-ratio form, with no call to the producer's
        # logit/expit or Newton helper. The fixed cap bounds |F| by 6.4.
        odds_factor = np.exp(score)
        probability = rate * odds_factor / (1 - rate + rate * odds_factor)
        assert np.isfinite(probability).all()
        residual = y - probability
        masks = {0: np.ones(len(d), dtype=bool)}; levels = {0: 0}; terminal = np.empty(len(d))
        assert len(tree['feature']) <= 15
        for node, left in enumerate(tree['children_left']):
            mask = masks[node]; weights = w[mask]
            assert int(mask.sum()) == tree['n_node_samples'][node] and levels[node] <= 3
            mean = np.average(residual[mask], weights=weights)
            variance = np.average((residual[mask] - mean) ** 2, weights=weights)
            np.testing.assert_allclose(weights.sum(), tree['weighted_n_node_samples'][node], rtol=0, atol=1e-8)
            np.testing.assert_allclose(mean, tree['gradient_value'][node], rtol=0, atol=2e-10)
            np.testing.assert_allclose(variance, tree['impurity'][node], rtol=0, atol=2e-10)
            if left < 0:
                assert mask.sum() >= 300
                numerator = np.dot(weights, residual[mask])
                curvature = np.dot(weights, probability[mask] * (1 - probability[mask]))
                if curvature == 0:
                    assert numerator == 0
                    value = 0.; zero_curvature += 1
                else:
                    raw = numerator / curvature
                    value = min(2., max(-2., raw)); clipped += int(abs(raw) > 2.)
                terminal[mask] = value
            else:
                value = mean
                right = tree['children_right'][node]
                lower = x[:, tree['feature'][node]] <= tree['threshold'][node]
                masks[left], masks[right] = mask & lower, mask & ~lower
                levels[left] = levels[right] = levels[node] + 1
            np.testing.assert_allclose(value, tree['value'][node], rtol=0, atol=3e-10)
            checks += 1
        score += .05 * terminal
    c.close()
    assert r['constant_baseline_days'] == int(daily.rate.isin([0, 1]).sum())
    assert r['constant_baseline_training_rows'] == int(d.rate.isin([0, 1]).sum())
    assert r['clipped_leaf_updates'] == clipped and r['zero_curvature_leaf_updates'] == zero_curvature
    assert len(r['thresholds']) == len(base.QUANTILES)
    for index, (q, cut) in enumerate(zip(base.QUANTILES, r['thresholds'])):
        assert cut['id'] == index and cut['training_quantile'] == q
        np.testing.assert_allclose(np.quantile(score, q), cut['threshold'], rtol=0, atol=3e-10)
    np.testing.assert_allclose(score, base.predict(x, r), rtol=0, atol=3e-10)
    assert r['new_2025_score_groups_read'] == bool(d.date.ge('2025-01-01').any())
    assert not d.date.ge('2025-07-01').any()
    proof = dict(passed=True, model_report_sha256=sha(root / 'model_report.json'),
        rows=len(d), node_checks=checks, training_start=start, training_end=end,
        all_full_pool_daily_baselines_encodings_and_date_weights_rebuilt=True,
        all_probability_gradients_variances_and_capped_newton_updates_independently_rebuilt=True,
        all_scores_and_training_quantiles_rebuilt=True, clipped_leaf_updates=clipped,
        zero_curvature_leaf_updates=zero_curvature,
        new_2025_score_groups_read=bool(d.date.ge('2025-01-01').any()),
        new_2025H2_score_groups_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(root / 'model_verification.json', proof); return proof


def verify_scores(expected_expressions=None, encoding_multiplier=1):
    spec = importlib.util.spec_from_file_location('independent_additive_scores', Path('scripts/verify_tail_formula_additive.py'))
    module = importlib.util.module_from_spec(spec)
    # The shared score verifier imports another verifier in the scripts folder.
    import sys
    sys.path.insert(0, str(Path('scripts').resolve()))
    try:
        spec.loader.exec_module(module)
        module.ROOT = base.ROOT; module.FEATURES = base.FEATURES
        module.SOURCE = base.SOURCE; module.PROTOCOL = base.PROTOCOL
        return module.scores(expected_expressions=expected_expressions, encoding_multiplier=encoding_multiplier)
    finally:
        sys.path.pop(0)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('fold', choices=['2024', 'recent', 'combined'])
    p.add_argument('stage', choices=['model', 'verify_model', 'scores', 'verify_scores', 'freeze', 'verify', 'analyze'])
    a = p.parse_args(); setup(a.fold)
    if a.stage == 'analyze':
        combined = Path('data/research') / f'{STEM}_2025'
        proof = json.loads((combined / 'selection_verification.json').read_text())
        assert proof['passed'] and proof['selection_report_sha256'] == sha(combined / 'selection_report.json')
        root = combined if a.fold == 'combined' else base.ROOT
        result = linkage.common_analysis(root, Path('config') / f'{STEM}_{a.fold}_protocol.json')
    elif a.fold == 'combined':
        assert a.stage in ['freeze', 'verify']
        result = getattr(linkage, 'combine' if a.stage == 'freeze' else 'verify_combined')()
    elif a.stage in ['model', 'verify_model', 'verify_scores']:
        result = globals()[a.stage]()
    elif a.stage in ['freeze', 'verify']:
        result = getattr(study, a.stage)()
    else:
        result = base.scores()
    print(json.dumps(result, ensure_ascii=False, indent=2))
