"""Reweight four training quarters by their loss improvement, not future returns."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from . import tail_formula_additive as base
from . import tail_formula_before1000 as labels
from . import tail_formula_boundary_evaluation as evaluation
from . import tail_formula_context_2024 as linkage
from . import tail_formula_float as inputs
from . import tail_formula_prior_day as adapter
from . import tail_formula_recent as study
from . import tail_formula_relative as relative
from .corporate_cash import save_json, sha

STEM = 'tail_formula_quarter_robust'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
PARAMETERS = dict(n_estimators=64, max_depth=3, min_samples_leaf=300,
    learning_rate=.05, criterion='friedman_mse', random_state=20260927,
    subsample=1., temperature_fraction=.1)


def policy():
    p = json.loads(PROTOCOL.read_text())
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    assert p['parameters'] == PARAMETERS and p['training_quantile'] == .995
    assert p['expected_features'] == 48 and not p['new_2026_prices_allowed']
    return p


def setup(fold):
    p = policy()
    adapter.STEM = STEM
    adapter.ROOT = inputs.ROOT
    adapter.EXPRESSIONS = inputs.EXPRESSIONS
    adapter.HEADER = inputs.HEADER
    adapter.COMBINED_PROTOCOL = Path('config') / (STEM + '_combined_protocol.json')
    adapter.setup(fold)
    base.SOURCE = labels.ROOT
    for name in ['2024', 'recent']:
        q = json.loads((Path('config') / (STEM + '_' + name + '_protocol.json')).read_text())
        assert q['master_protocol_sha256'] == sha(PROTOCOL)
        assert q['parameters'] == PARAMETERS
        assert q['feature_report_sha256'] == sha(inputs.ROOT / 'feature_report.json')
        assert q['label_report_sha256'] == sha(labels.ROOT / 'full_label_report.json')
        assert all(q[k] == v for k, v in p['folds'][name].items())


def training():
    p = json.loads(base.PROTOCOL.read_text())
    t = relative.training('relative')[['date', 'code', 'next_date', 'target', *inputs.EXPRESSIONS]].copy()
    t['quarter'] = pd.to_datetime(t.date).dt.to_period('Q').astype(str)
    assert len(t) == p['expected_rows'] and t.date.nunique() == p['expected_training_days'] == 241
    assert t.quarter.nunique() == 4
    return t


def independent_training():
    p = json.loads(base.PROTOCOL.read_text())
    names = list(inputs.EXPRESSIONS)
    f = base.feature_inputs()[['date', 'code', 'formula_input_valid', *names]]
    c = base.conn()
    c.register('features', f)
    expressions = ','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::INT AS {n}' for n in names)
    d = c.sql(f'''WITH known AS(SELECT date,code,next_date,opportunity15
        FROM read_parquet('{labels.ROOT}/full_labels.parquet')
        WHERE known15 AND date>='{p['training_start']}' AND next_date<'{p['training_end']}'),
        targets AS(SELECT *,opportunity15-avg(opportunity15) OVER(PARTITION BY date) AS target FROM known),
        joined AS(SELECT date,code,next_date,target,{expressions},
            concat(year(date::DATE),'Q',quarter(date::DATE)) AS quarter
            FROM targets JOIN features USING(date,code) WHERE formula_input_valid)
        SELECT *,1./count(*) OVER(PARTITION BY date) AS w FROM joined ORDER BY date,code''').df()
    c.close()
    t = training()
    pd.testing.assert_frame_equal(d[['date', 'code', 'next_date', 'quarter']], t[['date', 'code', 'next_date', 'quarter']], check_exact=True)
    np.testing.assert_array_equal(d[names].to_numpy(dtype='int32'), base.encode(t))
    np.testing.assert_allclose(d.target, t.target, rtol=0, atol=2e-12)
    np.testing.assert_allclose(d.w, 1/t.groupby('date').code.transform('size'), rtol=0, atol=0)
    return d


def verify_inputs():
    assert not (base.ROOT / 'training_input_verification.json').exists()
    d = independent_training()
    proof = dict(passed=True, protocol_sha256=sha(base.PROTOCOL),
        feature_report_sha256=sha(inputs.ROOT / 'feature_report.json'), label_report_sha256=sha(labels.ROOT / 'full_label_report.json'),
        rows=len(d), days=d.date.nunique(), first_date=d.date.min(), last_observation=d.next_date.max(),
        quarter_days=d.groupby('quarter').date.nunique().to_dict(), quarter_rows=d.groupby('quarter').size().to_dict(),
        all_training_keys_targets_encodings_quarters_and_date_weights_sql_rebuilt=True,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    base.ROOT.mkdir(parents=True, exist_ok=True)
    save_json(base.ROOT / 'training_input_verification.json', proof)
    return proof


def group_risks(y, score, weights, groups, count):
    return np.bincount(groups, weights=weights*(y-score)**2, minlength=count) / np.bincount(groups, weights=weights, minlength=count)


def tilted_distribution(risks, initial, priors, temperature):
    # Preserve the ordinary date weights bit for bit at initialization.
    if temperature is None or np.array_equal(risks, initial):
        return priors.copy(), np.ones(len(priors)), 0. if temperature is not None else float(np.dot(priors, risks-initial))
    logits = np.log(priors) + (risks-initial)/temperature
    peak = logits.max()
    raw = np.exp(logits-peak)
    probabilities = raw/raw.sum()
    objective = temperature*(peak+np.log(raw.sum()))
    return probabilities, probabilities/priors, float(objective)


def tree_dict(estimator):
    t = estimator.tree_
    return dict(feature=t.feature.tolist(), threshold=t.threshold.tolist(), children_left=t.children_left.tolist(),
        children_right=t.children_right.tolist(), n_node_samples=t.n_node_samples.tolist(),
        weighted_n_node_samples=t.weighted_n_node_samples.tolist(), value=t.value.reshape(-1).tolist(), impurity=t.impurity.tolist())


def fit(x, y, weights, groups, priors, parameters, tilted=True):
    bias = float(np.average(y, weights=weights))
    score = np.full(len(y), bias)
    count = len(priors)
    initial = group_risks(y, score, weights, groups, count)
    temperature = float(parameters['temperature_fraction']*np.dot(priors, initial)) if tilted else None
    assert temperature is None or temperature > 0
    rng = np.random.RandomState(parameters['random_state'])
    trees, trace = [], []
    for index in range(parameters['n_estimators']):
        before = group_risks(y, score, weights, groups, count)
        probabilities, multipliers, objective_before = tilted_distribution(before, initial, priors, temperature)
        stage_weights = weights*multipliers[groups]
        estimator = DecisionTreeRegressor(criterion=parameters['criterion'], max_depth=parameters['max_depth'],
            min_samples_leaf=parameters['min_samples_leaf'], random_state=rng).fit(x, y-score, sample_weight=stage_weights)
        tree = tree_dict(estimator)
        step = np.asarray(tree['value'])[base.leaf_indices(x, tree)]
        np.testing.assert_allclose(step, estimator.predict(x), rtol=0, atol=2e-12)
        score += parameters['learning_rate']*step
        after = group_risks(y, score, weights, groups, count)
        _, _, objective_after = tilted_distribution(after, initial, priors, temperature)
        trace.append(dict(tree=index, before=before.tolist(), after=after.tolist(), probabilities=probabilities.tolist(),
            multipliers=multipliers.tolist(), objective_before=objective_before, objective_after=objective_after))
        trees.append(tree)
    return dict(bias=bias, learning_rate=parameters['learning_rate'], trees=trees, initial_group_risks=initial.tolist(),
        temperature=temperature, group_priors=priors.tolist(), training_trace=trace)


def model():
    assert not (base.ROOT / 'model_report.json').exists(), 'Do not refit a frozen model'
    proof = json.loads((base.ROOT / 'training_input_verification.json').read_text())
    assert proof['passed'] and proof['protocol_sha256'] == sha(base.PROTOCOL)
    assert proof['feature_report_sha256'] == sha(inputs.ROOT / 'feature_report.json')
    assert proof['label_report_sha256'] == sha(labels.ROOT / 'full_label_report.json')
    p = json.loads(base.PROTOCOL.read_text())
    t = training()
    x, y = base.encode(t), t.target.to_numpy()
    w = 1/t.groupby('date').code.transform('size').to_numpy()
    quarters, group = np.unique(t.quarter.to_numpy(), return_inverse=True)
    days = t.groupby('quarter').date.nunique().reindex(quarters).to_numpy()
    r = fit(x, y, w, group, days/days.sum(), PARAMETERS)
    fold = '2024' if p['training_start'] == '2024-01-01' else 'recent'
    control = Path('data/research') / ('tail_formula_before1000_model_' + fold)
    old = json.loads((control / 'model_report.json').read_text())
    np.testing.assert_allclose(r['bias'], old['bias'], rtol=0, atol=2e-12)
    for key in ['feature', 'threshold', 'children_left', 'children_right', 'n_node_samples']:
        np.testing.assert_array_equal(r['trees'][0][key], old['trees'][0][key])
    for key in ['weighted_n_node_samples', 'value', 'impurity']:
        np.testing.assert_allclose(r['trees'][0][key], old['trees'][0][key], rtol=0, atol=2e-10)
    r.update(protocol_sha256=sha(base.PROTOCOL), training_input_verification_sha256=sha(base.ROOT / 'training_input_verification.json'),
        feature_report_sha256=sha(inputs.ROOT / 'feature_report.json'),
        label_report_sha256=sha(labels.ROOT / 'full_label_report.json'), parameters=PARAMETERS,
        feature_names=list(inputs.EXPRESSIONS), variant='quarter_robust', rows=len(t), days=t.date.nunique(),
        training_start=p['training_start'], training_end=p['training_end'], last_observation=t.next_date.max(),
        training_quarters=quarters.tolist(), quarter_days=days.tolist(),
        quarter_rows=t.groupby('quarter').size().reindex(quarters).tolist(),
        first_tree_control_model_sha256=sha(control / 'model_report.json'), first_tree_matches_frozen_ordinary_control=True,
        objective_increase_steps=sum(z['objective_after'] > z['objective_before']+2e-12 for z in r['training_trace']),
        score_is_not_probability=True, new_2025_score_groups_read=bool(t.date.ge('2025-01-01').any()),
        new_2025H2_score_groups_read=False, new_2026_prices_read=False, no_exit_rules=True)
    scores = base.predict(x, r)
    r['thresholds'] = [dict(id=i, training_quantile=q, threshold=float(np.quantile(scores, q))) for i, q in enumerate(base.QUANTILES)]
    base.ROOT.mkdir(parents=True, exist_ok=True)
    save_json(base.ROOT / 'model_report.json', r)
    return {k: v for k, v in r.items() if k not in ['trees', 'training_trace']}


def verify_model():
    r = json.loads((base.ROOT / 'model_report.json').read_text())
    p = json.loads(base.PROTOCOL.read_text())
    for key, file in [('protocol_sha256', base.PROTOCOL), ('feature_report_sha256', inputs.ROOT / 'feature_report.json'),
                      ('label_report_sha256', labels.ROOT / 'full_label_report.json'),
                      ('training_input_verification_sha256', base.ROOT / 'training_input_verification.json')]:
        assert r[key] == sha(file)
    assert r['parameters'] == PARAMETERS and r['feature_names'] == list(inputs.EXPRESSIONS)
    assert r['variant'] == 'quarter_robust' and r['learning_rate'] == .05
    d = independent_training()
    x = d[list(inputs.EXPRESSIONS)].to_numpy(dtype='int32')
    y, w = d.target.to_numpy(), d.w.to_numpy()
    quarters = sorted(d.quarter.unique())
    masks = [d.quarter.eq(q).to_numpy() for q in quarters]
    days = [d.loc[m, 'date'].nunique() for m in masks]
    priors = np.asarray(days)/sum(days)
    assert quarters == r['training_quarters'] and days == r['quarter_days']
    assert [int(m.sum()) for m in masks] == r['quarter_rows']
    assert len(d) == r['rows'] == p['expected_rows'] and sum(days) == r['days'] == 241
    assert r['training_start'] == p['training_start'] and r['training_end'] == p['training_end']
    assert d.next_date.max() == r['last_observation'] < p['training_end'] and d.date.min() >= p['training_start']
    bias = float(np.average(y, weights=w))
    np.testing.assert_allclose(bias, r['bias'], rtol=0, atol=2e-12)
    score = np.full(len(d), bias)
    losses = lambda z: np.array([np.average((y[m]-z[m])**2, weights=w[m]) for m in masks])
    initial = losses(score)
    temperature = sum(float(pi*loss) for pi, loss in zip(priors, initial))*.1
    np.testing.assert_allclose(initial, r['initial_group_risks'], rtol=0, atol=2e-12)
    np.testing.assert_allclose(priors, r['group_priors'], rtol=0, atol=0)
    np.testing.assert_allclose(temperature, r['temperature'], rtol=0, atol=2e-12)
    def probabilities_and_objective(risks):
        v = [math.log(pi)+(loss-origin)/temperature for pi, loss, origin in zip(priors, risks, initial)]
        peak = max(v)
        raw = [math.exp(z-peak) for z in v]
        total = math.fsum(raw)
        return np.array([z/total for z in raw]), temperature*(peak+math.log(total))
    checks, increases = 0, 0
    assert len(r['trees']) == len(r['training_trace']) == 64
    for index, (tree, trace) in enumerate(zip(r['trees'], r['training_trace'])):
        before = losses(score)
        probs, objective_before = probabilities_and_objective(before)
        multipliers = probs/priors
        np.testing.assert_allclose(before, trace['before'], rtol=0, atol=2e-10)
        np.testing.assert_allclose(probs, trace['probabilities'], rtol=0, atol=2e-10)
        np.testing.assert_allclose(multipliers, trace['multipliers'], rtol=0, atol=2e-10)
        stage_weights = np.empty(len(d))
        for mask, factor in zip(masks, multipliers):
            stage_weights[mask] = w[mask]*factor
        residual = y-score
        branches = {0: np.ones(len(d), dtype=bool)}
        levels = {0: 0}
        step = np.empty(len(d))
        assert len(tree['feature']) <= 15 and trace['tree'] == index
        for node, left in enumerate(tree['children_left']):
            take = branches[node]
            ww = stage_weights[take]
            mean = np.average(residual[take], weights=ww)
            variance = np.average((residual[take]-mean)**2, weights=ww)
            assert int(take.sum()) == tree['n_node_samples'][node] and levels[node] <= 3
            np.testing.assert_allclose(ww.sum(), tree['weighted_n_node_samples'][node], rtol=0, atol=1e-8)
            np.testing.assert_allclose(mean, tree['value'][node], rtol=0, atol=2e-10)
            np.testing.assert_allclose(variance, tree['impurity'][node], rtol=0, atol=2e-10)
            if left < 0:
                assert take.sum() >= 300
                step[take] = mean
            else:
                right = tree['children_right'][node]
                cut = x[:, tree['feature'][node]] <= tree['threshold'][node]
                branches[left], branches[right] = take & cut, take & ~cut
                levels[left] = levels[right] = levels[node]+1
            checks += 1
        score += .05*step
        after = losses(score)
        _, objective_after = probabilities_and_objective(after)
        np.testing.assert_allclose(after, trace['after'], rtol=0, atol=2e-10)
        np.testing.assert_allclose([objective_before, objective_after], [trace['objective_before'], trace['objective_after']], rtol=0, atol=2e-10)
        increases += int(objective_after > objective_before+2e-12)
    assert increases == r['objective_increase_steps']
    np.testing.assert_allclose(score, base.predict(x, r), rtol=0, atol=2e-10)
    for q, cut in zip(base.QUANTILES, r['thresholds']):
        assert cut['training_quantile'] == q
        np.testing.assert_allclose(cut['threshold'], np.quantile(score, q), rtol=0, atol=2e-10)
    fold = '2024' if p['training_start'] == '2024-01-01' else 'recent'
    path = Path('data/research') / ('tail_formula_before1000_model_' + fold) / 'model_report.json'
    assert sha(path) == r['first_tree_control_model_sha256'] and r['first_tree_matches_frozen_ordinary_control']
    old = json.loads(path.read_text())
    for key in ['feature', 'threshold', 'children_left', 'children_right', 'n_node_samples']:
        np.testing.assert_array_equal(r['trees'][0][key], old['trees'][0][key])
    for key in ['weighted_n_node_samples', 'value', 'impurity']:
        np.testing.assert_allclose(r['trees'][0][key], old['trees'][0][key], rtol=0, atol=2e-10)
    assert r['new_2025_score_groups_read'] == bool(d.date.ge('2025-01-01').any())
    assert not d.date.ge('2025-07-01').any() and not r['new_2025H2_score_groups_read']
    proof = dict(passed=True, model_report_sha256=sha(base.ROOT / 'model_report.json'), rows=len(d), node_checks=checks,
        all_training_targets_encodings_quarters_and_original_date_weights_sql_rebuilt=True,
        all_64_group_losses_dynamic_weights_and_node_means_variances_rebuilt=True,
        all_quantiles_and_native_tree_predictions_rebuilt=True, actual_first_tree_matches_ordinary_control=True,
        training_start=p['training_start'], training_end=p['training_end'],
        objective_increase_steps=increases, new_2025_score_groups_read=r['new_2025_score_groups_read'],
        new_2025H2_score_groups_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(base.ROOT / 'model_verification.json', proof)
    return proof


def main():
    from .tail_formula_offset_logit48 import verify_scores
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['verify_inputs', 'model', 'verify_model', 'scores', 'verify_scores', 'freeze', 'verify', 'analyze'])
    p.add_argument('--fold', choices=['2024', 'recent', 'combined'], default='2024')
    a = p.parse_args()
    setup(a.fold)
    if a.stage == 'analyze':
        for fold in ['2024', 'recent', '2025']:
            root = Path('data/research') / (STEM + '_' + fold)
            v = json.loads((root / 'selection_verification.json').read_text())
            assert v['passed'] and v['selection_report_sha256'] == sha(root / 'selection_report.json')
        result = evaluation.analyze(linkage.COMBINED if a.fold == 'combined' else base.ROOT,
            linkage.PROTOCOL if a.fold == 'combined' else base.PROTOCOL)
    elif a.fold == 'combined':
        assert a.stage in ['freeze', 'verify']
        result = linkage.combine() if a.stage == 'freeze' else linkage.verify_combined()
    elif a.stage in ['verify_inputs', 'model', 'verify_model']:
        result = globals()[a.stage]()
    elif a.stage == 'verify_scores':
        result = verify_scores()
    elif a.stage == 'freeze':
        result = study.freeze()
    elif a.stage == 'verify':
        m = json.loads((base.ROOT / 'model_report.json').read_text())
        assert (base.ROOT / 'frozen_numeric_core.tdx').read_text() == base.native_core(m, m['thresholds'][3]['threshold'], base.EXPRESSIONS, base.HEADER)
        result = study.verify()
    else:
        result = base.scores()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
