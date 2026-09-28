"""Purely additive nonlinear stock scores: fixed sums of one-split trees."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from . import tail_formula_additive as base
from . import tail_formula_before1000_model as adapter
from . import tail_formula_boundary_evaluation as evaluation
from . import tail_formula_context_2024 as linkage
from . import tail_formula_recent as study
from . import tail_formula_relative as relative
from .corporate_cash import save_json, sha

STEM = 'tail_formula_stumps'
PROTOCOL = Path('config') / (STEM + '_protocol.json')


def setup(fold):
    adapter.STEM = STEM; adapter.setup(fold)
    combined = json.loads((Path('config') / (STEM + '_combined_protocol.json')).read_text())
    control = Path(combined['control'])
    for stage in ['selection', 'analysis']:
        assert combined['control_' + stage + '_report_sha256'] == sha(control / (stage + '_report.json'))
        proof = json.loads((control / (stage + '_verification.json')).read_text())
        assert proof['passed'] and proof[stage + '_report_sha256'] == sha(control / (stage + '_report.json'))
    for name in ['2024', 'recent']:
        p = json.loads((Path('config') / (STEM + '_' + name + '_protocol.json')).read_text())
        assert p['structure_protocol_sha256'] == sha(PROTOCOL)
        assert p['parameters']['n_estimators'] == 256 and p['parameters']['max_depth'] == 1
        assert p['parameters']['learning_rate'] == .05 and p['parameters']['min_samples_leaf'] == 300


def model():
    assert not (base.ROOT / 'model_report.json').exists(), 'Do not refit fixed stumps'
    p = json.loads(base.PROTOCOL.read_text())
    train = relative.training('relative'); x = base.encode(train); y = train.target.to_numpy()
    w = 1/train.groupby('date').code.transform('size').to_numpy()
    fitted = GradientBoostingRegressor(**p['parameters']).fit(x, y, sample_weight=w)
    trees = []
    for estimator in fitted.estimators_.ravel():
        tree = estimator.tree_
        assert tree.node_count in [1, 3] and tree.max_depth <= 1
        trees.append(dict(feature=tree.feature.tolist(), threshold=tree.threshold.tolist(),
            children_left=tree.children_left.tolist(), children_right=tree.children_right.tolist(),
            n_node_samples=tree.n_node_samples.tolist(), weighted_n_node_samples=tree.weighted_n_node_samples.tolist(),
            value=tree.value.reshape(-1).tolist(), impurity=tree.impurity.tolist()))
    r = dict(protocol_sha256=sha(base.PROTOCOL), structure_protocol_sha256=sha(PROTOCOL),
        feature_report_sha256=sha(base.FEATURES / 'feature_report.json'),
        label_report_sha256=sha(base.SOURCE / 'full_label_report.json'), rows=len(train), days=train.date.nunique(),
        last_observation=train.next_date.max(), parameters=fitted.get_params(), feature_names=list(base.EXPRESSIONS),
        variant='relative', learning_rate=.05, bias=float(np.ravel(fitted.init_.constant_)[0]), trees=trees,
        training_start=p['training_start'], training_end=p['training_end'], purely_additive_no_cross_feature_interactions=True,
        new_2025_score_groups_read=bool(train.date.ge('2025-01-01').any()), new_2025H2_score_groups_read=False,
        new_2026_prices_read=False, no_exit_rules=True)
    scores = base.predict(x, r)
    np.testing.assert_allclose(scores, fitted.predict(x), rtol=0, atol=2e-12)
    r['thresholds'] = [dict(id=i, training_quantile=q, threshold=float(np.quantile(scores,q))) for i,q in enumerate(base.QUANTILES)]
    base.ROOT.mkdir(parents=True, exist_ok=True); save_json(base.ROOT / 'model_report.json', r)
    return {k:v for k,v in r.items() if k != 'trees'}


def verify_model():
    p = json.loads(base.PROTOCOL.read_text()); r = json.loads((base.ROOT / 'model_report.json').read_text())
    for key, path in [('protocol_sha256', base.PROTOCOL), ('structure_protocol_sha256', PROTOCOL),
                      ('feature_report_sha256', base.FEATURES / 'feature_report.json'),
                      ('label_report_sha256', base.SOURCE / 'full_label_report.json')]:
        assert r[key] == sha(path)
    assert all(r['parameters'][k] == v for k,v in p['parameters'].items())
    assert len(r['trees']) == 256 and r['learning_rate'] == .05
    assert r['feature_names'] == list(base.EXPRESSIONS) and len(r['feature_names']) == 48
    assert r['purely_additive_no_cross_feature_interactions'] is True
    assert r['new_2025H2_score_groups_read'] is False and r['new_2026_prices_read'] is False
    assert r['no_exit_rules'] is True
    start, end = p['training_start'], p['training_end']
    f = base.feature_inputs()[['date','code','formula_input_valid',*base.EXPRESSIONS]]
    c = base.conn(); c.register('visible', f)
    enc = ','.join(f'floor(least(greatest(100*{name}+10000+.000001,0),999999))::INT AS {name}' for name in base.EXPRESSIONS)
    c.execute(f'''CREATE VIEW targets AS WITH a AS(SELECT date,code,next_date,opportunity15
        FROM read_parquet('{base.SOURCE}/full_labels.parquet')
        WHERE date>='{start}' AND next_date<'{end}' AND known15)
        SELECT date,code,next_date,opportunity15-avg(opportunity15) OVER(PARTITION BY date) AS target FROM a''')
    d = c.sql('SELECT date,code,next_date,target,1./count(*) OVER(PARTITION BY date) AS w,'+enc+
              ' FROM visible JOIN targets USING(date,code) WHERE formula_input_valid ORDER BY date,code').df(); c.close()
    expected = relative.training('relative')
    pd.testing.assert_frame_equal(d[['date','code','next_date']], expected[['date','code','next_date']], check_exact=True)
    np.testing.assert_allclose(d.target, expected.target, rtol=0, atol=2e-12)
    x = d[list(base.EXPRESSIONS)].to_numpy('int32'); np.testing.assert_array_equal(x, base.encode(expected))
    y = d.target.to_numpy(); w = d.w.to_numpy()
    assert len(d) == r['rows'] and d.date.nunique() == r['days'] == p['expected_training_days'] == 241
    assert d.next_date.max() == r['last_observation'] < end
    assert r['new_2025_score_groups_read'] == bool(d.date.ge('2025-01-01').any())
    assert (r['training_start'], r['training_end']) == (start, end)
    np.testing.assert_allclose(r['bias'], np.sum(w*y)/w.sum(), rtol=0, atol=2e-12)
    score = np.full(len(d), r['bias']); checks = 0
    for tree in r['trees']:
        residual = y-score; size = len(tree['feature'])
        assert size in [1,3]
        assert all(len(tree[key]) == size for key in ['threshold', 'children_left', 'children_right',
                   'n_node_samples', 'weighted_n_node_samples', 'value', 'impurity'])
        if size == 1:
            assert tree['children_left'] == tree['children_right'] == [-1]
            masks = [np.ones(len(d), dtype=bool)]
        else:
            assert tree['children_left'] == [1,-1,-1] and tree['children_right'] == [2,-1,-1]
            feature = tree['feature'][0]; assert 0 <= feature < x.shape[1]
            left = x[:,feature] <= tree['threshold'][0]
            masks = [np.ones(len(d), dtype=bool), left, ~left]
        update = np.empty(len(d))
        for node, mask in enumerate(masks):
            weights = w[mask]; values = residual[mask]; mass = weights.sum()
            mean = np.sum(weights*values)/mass
            variance = np.sum(weights*(values-mean)**2)/mass
            assert int(mask.sum()) == tree['n_node_samples'][node]
            np.testing.assert_allclose(mass, tree['weighted_n_node_samples'][node], rtol=0, atol=1e-8)
            np.testing.assert_allclose([mean,variance], [tree['value'][node],tree['impurity'][node]], rtol=0, atol=2e-10)
            if size == 1 or node > 0:
                assert mask.sum() >= 300 and tree['feature'][node] < 0
                update[mask] = mean
            checks += 1
        score += .05*update
    np.testing.assert_allclose(score, base.predict(x,r), rtol=0, atol=2e-10)
    assert len(r['thresholds']) == len(base.QUANTILES)
    for i, (q, threshold) in enumerate(zip(base.QUANTILES, r['thresholds'])):
        assert threshold['id'] == i and threshold['training_quantile'] == q
        np.testing.assert_allclose(np.quantile(score,q), threshold['threshold'], rtol=0, atol=2e-10)
    proof = dict(passed=True, model_report_sha256=sha(base.ROOT / 'model_report.json'), rows=len(d), node_checks=checks,
        all_sql_keys_targets_integer_inputs_and_day_weights_verified=True,
        every_stump_shape_residual_mean_variance_and_leaf_support_verified=True,
        all_training_scores_and_quantiles_verified=True, purely_additive_no_cross_feature_interactions=True,
        new_2025_score_groups_read=bool(d.date.ge('2025-01-01').any()), new_2025H2_score_groups_read=False,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(base.ROOT / 'model_verification.json', proof); return proof


if __name__ == '__main__':
    from .tail_formula_offset_logit48 import verify_scores
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['model','verify_model','scores','verify_scores','freeze','verify','analyze'])
    parser.add_argument('--fold', choices=['2024','recent','combined'], default='2024')
    args = parser.parse_args(); setup(args.fold)
    if args.stage == 'analyze':
        for fold in ['2024','recent','2025']:
            root = Path('data/research') / (STEM+'_'+fold)
            v = json.loads((root / 'selection_verification.json').read_text())
            assert v['passed'] and v['selection_report_sha256'] == sha(root / 'selection_report.json')
        result = evaluation.analyze(linkage.COMBINED if args.fold == 'combined' else base.ROOT,
                                    linkage.PROTOCOL if args.fold == 'combined' else base.PROTOCOL)
    elif args.fold == 'combined':
        assert args.stage in ['freeze','verify']
        result = linkage.combine() if args.stage == 'freeze' else linkage.verify_combined()
    elif args.stage in ['model','verify_model']:
        result = globals()[args.stage]()
    elif args.stage == 'verify_scores':
        result = verify_scores()
    elif args.stage == 'freeze':
        result = study.freeze()
    elif args.stage == 'verify':
        r = json.loads((base.ROOT / 'model_report.json').read_text())
        assert (base.ROOT / 'frozen_numeric_core.tdx').read_text() == base.native_core(r,r['thresholds'][3]['threshold'],base.EXPRESSIONS,base.HEADER)
        result = study.verify()
    else:
        result = base.scores()
    print(json.dumps(result,ensure_ascii=False,indent=2))
