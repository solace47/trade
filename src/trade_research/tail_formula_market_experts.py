"""Two visible-index-state experts and a pooled 128-tree capacity control."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from . import tail_formula_additive as base
from . import tail_formula_before1000 as labels
from . import tail_formula_boundary_evaluation as evaluation
from . import tail_formula_context_2024 as linkage
from . import tail_formula_float as inputs
from . import tail_formula_prior_day as adapter
from . import tail_formula_recent as study
from . import tail_formula_relative as relative
from .corporate_cash import save_json, sha

STEM = 'tail_formula_market_experts'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
GATE_INDEX = list(inputs.EXPRESSIONS).index('J01')
GATE = 10000
LR = .05


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    for key, path in [
        ('feature_report_sha256', inputs.ROOT / 'feature_report.json'),
        ('feature_verification_sha256', inputs.ROOT / 'feature_verification.json'),
        ('features_sha256', inputs.ROOT / 'features.parquet'),
        ('boundary_protocol_sha256', labels.PROTOCOL),
        ('label_report_sha256', labels.ROOT / 'full_label_report.json'),
        ('label_verification_sha256', labels.ROOT / 'full_label_verification.json'),
        ('labels_sha256', labels.ROOT / 'full_labels.parquet'),
    ]:
        assert p[key] == sha(path), (key, path)
    assert p['gate']['feature'] == 'J01' and p['gate']['threshold'] == GATE
    assert p['gate']['expression'] == inputs.EXPRESSIONS['J01']
    return p


def setup(arm, fold):
    master = checked_sources()
    adapter.STEM = STEM + '_' + arm
    adapter.ROOT = inputs.ROOT
    adapter.EXPRESSIONS = inputs.EXPRESSIONS
    adapter.HEADER = inputs.HEADER
    adapter.COMBINED_PROTOCOL = Path('config') / (adapter.STEM + '_combined_protocol.json')
    adapter.setup(fold)
    base.SOURCE = labels.ROOT
    for f in ['2024', 'recent']:
        p = json.loads((Path('config') / (adapter.STEM + '_' + f + '_protocol.json')).read_text())
        assert p['master_protocol_sha256'] == sha(PROTOCOL)
        assert p['arm'] == arm and p['expected_features'] == len(inputs.EXPRESSIONS) == 48
        assert p['gate'] == master['gate']
        assert p['parameters'] == dict(master['parameters_common'],
            n_estimators=master['arms'][arm]['n_estimators_per_component'])


def state_masks(x, arm):
    if arm == 'pooled':
        return [('all', np.ones(len(x), dtype=bool))]
    assert arm == 'split'
    upper = x[:, GATE_INDEX] > GATE
    return [('lower', ~upper), ('upper', upper)]


def gated_tree(tree, upper):
    """Make the inactive expert contribute exactly zero using ordinary IF nodes."""
    return dict(
        feature=[GATE_INDEX, -2, *tree['feature']],
        threshold=[float(GATE), -2., *tree['threshold']],
        children_left=[1 if upper else 2, -1,
            *[i + 2 if i >= 0 else -1 for i in tree['children_left']]],
        children_right=[2 if upper else 1, -1,
            *[i + 2 if i >= 0 else -1 for i in tree['children_right']]],
        value=[0., 0., *tree['value']],
    )


def flatten(components, arm):
    if arm == 'pooled':
        return components[0]['bias'], components[0]['trees']
    assert [c['state'] for c in components] == ['lower', 'upper']
    bias_tree = dict(feature=[GATE_INDEX, -2, -2], threshold=[float(GATE), -2., -2.],
        children_left=[1, -1, -1], children_right=[2, -1, -1],
        value=[0., components[0]['bias'] / LR, components[1]['bias'] / LR])
    trees = [bias_tree]
    for component in components:
        trees.extend(gated_tree(t, component['state'] == 'upper') for t in component['trees'])
    return 0., trees


def model(arm):
    assert not (base.ROOT / 'model_report.json').exists(), 'Do not refit frozen models'
    p = json.loads(base.PROTOCOL.read_text())
    train = relative.training('relative')
    x = base.encode(train)
    y = train.target.to_numpy()
    # Keep the ORIGINAL full-intersection date weight after partitioning.
    w = 1 / train.groupby('date').code.transform('size').to_numpy()
    scores = np.empty(len(train))
    components = []
    for state, mask in state_masks(x, arm):
        assert mask.sum() >= 600
        fitted = GradientBoostingRegressor(**p['parameters']).fit(x[mask], y[mask], sample_weight=w[mask])
        trees = []
        for estimator in fitted.estimators_.ravel():
            t = estimator.tree_
            trees.append({name: getattr(t, name).reshape(-1).tolist() for name in [
                'feature', 'threshold', 'children_left', 'children_right',
                'n_node_samples', 'weighted_n_node_samples', 'value', 'impurity']})
        component = dict(state=state, rows=int(mask.sum()), days=train.loc[mask, 'date'].nunique(),
            weight_sum=float(w[mask].sum()), bias=float(fitted.init_.constant_.ravel()[0]),
            learning_rate=LR, trees=trees)
        scores[mask] = fitted.predict(x[mask])
        np.testing.assert_allclose(base.predict(x[mask], component), scores[mask], rtol=0, atol=2e-12)
        components.append(component)
    bias, trees = flatten(components, arm)
    r = dict(protocol_sha256=sha(base.PROTOCOL), master_protocol_sha256=sha(PROTOCOL),
        feature_report_sha256=sha(inputs.ROOT / 'feature_report.json'),
        label_report_sha256=sha(labels.ROOT / 'full_label_report.json'),
        implementation_sha256=sha(Path(__file__)), arm=arm, variant='relative',
        rows=len(train), days=train.date.nunique(), last_observation=train.next_date.max(),
        training_start=p['training_start'], training_end=p['training_end'],
        feature_names=list(inputs.EXPRESSIONS), parameters=p['parameters'], gate=p['gate'],
        learning_rate=LR, bias=bias, trees=trees, components=components,
        new_2025_score_groups_read=bool(train.date.ge('2025-01-01').any()),
        new_2025H2_score_groups_read=False, new_2026_prices_read=False, no_exit_rules=True)
    manual = base.predict(x, r)
    np.testing.assert_allclose(manual, scores, rtol=0, atol=2e-12)
    r['thresholds'] = [dict(id=i, training_quantile=q, threshold=float(np.quantile(manual, q)))
        for i, q in enumerate(base.QUANTILES)]
    base.ROOT.mkdir(parents=True, exist_ok=True)
    save_json(base.ROOT / 'model_report.json', r)
    return {k: v for k, v in r.items() if k not in ['trees', 'components']}


def independent_training():
    """SQL builds means BEFORE the feature intersection, and weights BEFORE the gate."""
    start, end, where = relative.training_scope()
    names = list(inputs.EXPRESSIONS)
    f = base.feature_inputs()[['date', 'code', 'formula_input_valid', *names]]
    c = base.conn()
    c.register('features', f)
    enc = ','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::INT AS {n}' for n in names)
    d = c.sql(f'''WITH l AS (
            SELECT date,code,opportunity15-avg(opportunity15) OVER(PARTITION BY date) AS target
            FROM read_parquet('{labels.ROOT}/full_labels.parquet') WHERE {where} AND known15),
        full_training AS (SELECT date,code,target,1./count(*) OVER(PARTITION BY date) AS w,{enc}
            FROM features JOIN l USING(date,code) WHERE formula_input_valid)
        SELECT *,CASE WHEN J01>{GATE} THEN 'upper' ELSE 'lower' END AS state
        FROM full_training ORDER BY date,code''').df()
    c.close()
    original = relative.training('relative')
    pd.testing.assert_frame_equal(d[['date', 'code']], original[['date', 'code']], check_exact=True)
    np.testing.assert_allclose(d.target, original.target, rtol=0, atol=2e-12)
    np.testing.assert_array_equal(d[names].to_numpy(dtype='int32'), base.encode(original))
    np.testing.assert_allclose(d.w, 1 / original.groupby('date').code.transform('size'), rtol=0, atol=0)
    assert original.next_date.lt(end).all() and original.date.ge(start).all()
    return d, original


def verify_component(component, x, y, w, iterations):
    """Rebuild every fitted node's residual statistics, independent of sklearn prediction."""
    assert len(component['trees']) == iterations
    np.testing.assert_allclose(component['bias'], np.average(y, weights=w), rtol=0, atol=2e-12)
    score = np.full(len(y), component['bias'])
    checks = 0
    for t in component['trees']:
        residual = y - score
        masks = {0: np.ones(len(y), dtype=bool)}
        levels = {0: 0}
        terminal = np.full(len(y), np.nan)
        assert len(t['feature']) <= 15
        for i in range(len(t['feature'])):
            assert levels[i] <= 3
            mask = masks[i]
            weights = w[mask]
            assert int(mask.sum()) == t['n_node_samples'][i]
            mean = np.average(residual[mask], weights=weights)
            variance = np.average((residual[mask] - mean) ** 2, weights=weights)
            np.testing.assert_allclose(weights.sum(), t['weighted_n_node_samples'][i], rtol=0, atol=1e-8)
            np.testing.assert_allclose(mean, t['value'][i], rtol=0, atol=2e-10)
            np.testing.assert_allclose(variance, t['impurity'][i], rtol=0, atol=2e-10)
            left, right = t['children_left'][i], t['children_right'][i]
            if left < 0:
                assert right < 0 and mask.sum() >= 300
                terminal[mask] = mean
            else:
                lower = x[:, t['feature'][i]] <= t['threshold'][i]
                masks[left], masks[right] = mask & lower, mask & ~lower
                levels[left] = levels[right] = levels[i] + 1
            checks += 1
        assert np.isfinite(terminal).all()
        score += LR * terminal
    return score, checks


def verify_model(arm):
    p = json.loads(base.PROTOCOL.read_text())
    r = json.loads((base.ROOT / 'model_report.json').read_text())
    for key, path in [('protocol_sha256', base.PROTOCOL), ('master_protocol_sha256', PROTOCOL),
        ('implementation_sha256', Path(__file__)), ('feature_report_sha256', inputs.ROOT / 'feature_report.json'),
        ('label_report_sha256', labels.ROOT / 'full_label_report.json')]:
        assert r[key] == sha(path)
    assert r['arm'] == arm and r['parameters'] == p['parameters'] and r['gate'] == p['gate']
    assert r['feature_names'] == list(inputs.EXPRESSIONS) and r['learning_rate'] == LR
    assert (r['training_start'], r['training_end']) == (p['training_start'], p['training_end'])
    d, original = independent_training()
    assert r['rows'] == len(d) and r['days'] == d.date.nunique() == p['expected_training_days'] == 241
    assert r['last_observation'] == original.next_date.max() < p['evaluation_start']
    x = d[list(inputs.EXPRESSIONS)].to_numpy(dtype='int32')
    y, w = d.target.to_numpy(), d.w.to_numpy()
    states = ['all'] if arm == 'pooled' else ['lower', 'upper']
    assert [q['state'] for q in r['components']] == states
    score = np.full(len(d), np.nan)
    checks = 0
    coverage = []
    for component in r['components']:
        mask = np.ones(len(d), dtype=bool) if arm == 'pooled' else d.state.eq(component['state']).to_numpy()
        assert component['rows'] == int(mask.sum()) and component['days'] == d.loc[mask, 'date'].nunique()
        np.testing.assert_allclose(component['weight_sum'], w[mask].sum(), rtol=0, atol=1e-10)
        score[mask], count = verify_component(component, x[mask], y[mask], w[mask], p['parameters']['n_estimators'])
        checks += count
        coverage.append({k: v for k, v in component.items() if k not in ['trees', 'bias', 'learning_rate']})
    bias, trees = flatten(r['components'], arm)
    assert bias == r['bias'] and trees == r['trees']
    assert len(trees) == (129 if arm == 'split' else 128)
    np.testing.assert_allclose(score, base.predict(x, r), rtol=0, atol=2e-10)
    for q, threshold in zip(base.QUANTILES, r['thresholds']):
        assert q == threshold['training_quantile']
        np.testing.assert_allclose(np.quantile(score, q), threshold['threshold'], rtol=0, atol=2e-10)
    proof = dict(passed=True, model_report_sha256=sha(base.ROOT / 'model_report.json'), arm=arm,
        rows=len(d), days=d.date.nunique(), node_checks=checks, components=coverage,
        original_targets_integer_inputs_global_weights_and_states_sql_rebuilt=True,
        every_component_node_residual_mean_variance_and_support_rebuilt=True,
        component_scores_equal_flattened_IF_scores=True, thresholds_use_entire_training_population=True,
        new_2025_score_groups_read=r['new_2025_score_groups_read'], new_2025H2_score_groups_read=False,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(base.ROOT / 'model_verification.json', proof)
    return proof


def selection_gate(write=False):
    records = []
    for arm in ['split', 'pooled']:
        for fold in ['2024', 'recent', '2025']:
            root = Path('data/research') / (STEM + '_' + arm + '_' + fold)
            v = json.loads((root / 'selection_verification.json').read_text())
            assert v['passed'] and v['selection_report_sha256'] == sha(root / 'selection_report.json')
            s = json.loads((root / 'selection_report.json').read_text())
            assert s['selection_sha256'] == sha(root / 'selection.parquet')
            if write:
                assert not (root / 'analysis_report.json').exists()
            records.append(dict(arm=arm, fold=fold, root=str(root), selected=s['selected'],
                selection_report_sha256=sha(root / 'selection_report.json'),
                selection_verification_sha256=sha(root / 'selection_verification.json')))
    for fold in ['2024', 'recent']:
        reports = [json.loads((Path('data/research') / (STEM + '_' + arm + '_' + fold) / 'model_report.json').read_text())
            for arm in ['split', 'pooled']]
        original = json.loads((Path('data/research') / ('tail_formula_before1000_model_' + fold) / 'model_report.json').read_text())
        for key in ['rows', 'days', 'last_observation', 'training_start', 'training_end', 'feature_report_sha256', 'label_report_sha256']:
            assert reports[0][key] == reports[1][key] == original[key]
    result = dict(passed=True, master_protocol_sha256=sha(PROTOCOL), selections=records,
        all_four_models_and_six_selections_frozen_together=True, training_populations_equal_original64=True,
        new_group_outcomes_read=False, year_2025_is_exploratory=True, new_2026_prices_read=False, no_exit_rules=True)
    if write:
        ROOT.mkdir(parents=True, exist_ok=True)
        assert not (ROOT / 'joint_selection_freeze.json').exists()
        save_json(ROOT / 'joint_selection_freeze.json', result)
    else:
        assert json.loads((ROOT / 'joint_selection_freeze.json').read_text()) == result
    return result


if __name__ == '__main__':
    from .tail_formula_offset_logit48 import verify_scores
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['model', 'verify_model', 'scores', 'verify_scores', 'freeze', 'verify', 'gate', 'analyze'])
    p.add_argument('--arm', choices=['split', 'pooled'], required=True)
    p.add_argument('--fold', choices=['2024', 'recent', 'combined'], default='2024')
    a = p.parse_args()
    setup(a.arm, a.fold)
    if a.stage == 'analyze':
        selection_gate()
        result = evaluation.analyze(linkage.COMBINED if a.fold == 'combined' else base.ROOT,
            linkage.PROTOCOL if a.fold == 'combined' else base.PROTOCOL)
    elif a.stage == 'gate':
        result = selection_gate(write=True)
    elif a.fold == 'combined':
        assert a.stage in ['freeze', 'verify']
        result = linkage.combine() if a.stage == 'freeze' else linkage.verify_combined()
    elif a.stage in ['model', 'verify_model']:
        result = globals()[a.stage](a.arm)
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
