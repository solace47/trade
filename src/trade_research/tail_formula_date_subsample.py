"""Squared-error boosting with fixed whole-date half-samples at every stage."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
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

STEM = 'tail_formula_date_subsample'
PROTOCOL = Path('config') / (STEM + '_protocol.json')


def setup(fold):
    p = json.loads(PROTOCOL.read_text())
    assert p['sample_fraction'] == .5 and p['sampling_unit'] == 'whole_signal_date'
    assert not p['new_2026_prices_allowed'] and sklearn.__version__ == '1.7.2'
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    combined = Path('config') / (STEM + '_combined_protocol.json')
    q = json.loads(combined.read_text())
    assert q['master_protocol_sha256'] == sha(PROTOCOL)
    control = Path(q['control'])
    for stage in ['selection', 'analysis']:
        proof = json.loads((control / (stage + '_verification.json')).read_text())
        assert proof['passed'] and proof[stage + '_report_sha256'] == q['control_' + stage + '_report_sha256'] == sha(control / (stage + '_report.json'))
    adapter.STEM = STEM
    adapter.ROOT, adapter.EXPRESSIONS, adapter.HEADER = inputs.ROOT, inputs.EXPRESSIONS, inputs.HEADER
    adapter.COMBINED_PROTOCOL = combined
    adapter.setup(fold)
    base.SOURCE = labels.ROOT
    for name in ['2024', 'recent']:
        q = json.loads((Path('config') / (STEM + '_' + name + '_protocol.json')).read_text())
        assert q['master_protocol_sha256'] == sha(PROTOCOL) and q['parameters'] == p['parameters']
        assert q['sample_fraction'] == .5 and q['expected_sampled_days_per_round'] == 120
        assert q['feature_report_sha256'] == sha(inputs.ROOT / 'feature_report.json')
        assert q['label_report_sha256'] == sha(labels.ROOT / 'full_label_report.json')


def schedule(dates):
    unique = sorted(set(dates))
    assert len(unique) == 241
    return [sorted(sorted(unique, key=lambda date: (
        hashlib.sha256(f'20260927|{i}|{date}'.encode()).hexdigest(), date))[:len(unique) // 2])
        for i in range(64)]


def independent_training():
    p = json.loads(base.PROTOCOL.read_text())
    start, end, where = relative.training_scope()
    names = list(inputs.EXPRESSIONS)
    c = base.conn()
    c.register('features', base.feature_inputs()[['date', 'code', 'formula_input_valid', *names]])
    c.execute(f'''CREATE VIEW targets AS SELECT date,code,
        opportunity15-avg(opportunity15) OVER(PARTITION BY date) AS target
        FROM read_parquet('{labels.ROOT}/full_labels.parquet') WHERE {where} AND known15''')
    fields = ','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::INT AS {n}' for n in names)
    d = c.sql('SELECT date,code,target,1./count(*) OVER(PARTITION BY date) AS w,' + fields +
              ' FROM features JOIN targets USING(date,code) WHERE formula_input_valid ORDER BY date,code').df()
    c.register('training', d[['date', 'code']])
    sql = c.sql('''WITH dates AS(SELECT DISTINCT date FROM training), ranked AS(
        SELECT i,date,row_number() OVER(PARTITION BY i ORDER BY
        sha256('20260927|'||i::VARCHAR||'|'||date),date) AS position
        FROM dates CROSS JOIN range(64) stages(i))
        SELECT i,date FROM ranked WHERE position<=120 ORDER BY i,date''').df()
    c.close()
    t = relative.training('relative')
    assert len(d) == p['expected_training_rows'] and d.date.nunique() == p['expected_training_days'] == 241
    assert t.next_date.max() < end and t.date.min() >= start
    pd.testing.assert_frame_equal(d[['date', 'code']], t[['date', 'code']], check_exact=True)
    np.testing.assert_allclose(d.target, t.target, rtol=0, atol=2e-12)
    np.testing.assert_allclose(d.w, 1 / t.groupby('date').code.transform('size'), rtol=0, atol=0)
    np.testing.assert_array_equal(d[names].to_numpy(dtype='int32'), base.encode(t))
    planned = schedule(t.date)
    for i, dates in enumerate(planned):
        assert dates == sql.loc[sql.i.eq(i), 'date'].tolist() and len(dates) == 120
    return d, t, planned


def verify_inputs():
    d, t, planned = independent_training()
    participation = pd.Series([date for dates in planned for date in dates]).value_counts()
    assert set(participation.index) == set(t.date.unique())
    result = dict(passed=True, protocol_sha256=sha(base.PROTOCOL), master_protocol_sha256=sha(PROTOCOL),
        feature_report_sha256=sha(inputs.ROOT / 'feature_report.json'),
        label_report_sha256=sha(labels.ROOT / 'full_label_report.json'), rows=len(t), days=t.date.nunique(),
        first_signal=t.date.min(), last_observation=t.next_date.max(),
        sampled_days_per_round=120, sampling_schedule=planned,
        min_rounds_per_date=int(participation.min()), max_rounds_per_date=int(participation.max()),
        all_training_keys_targets_date_weights_integer_inputs_and_sha_schedules_rebuilt=True,
        original_training_intersection_unchanged=True, selection_is_not_out_of_fold_validation=True,
        new_2026_prices_read=False, no_exit_rules=True)
    base.ROOT.mkdir(parents=True, exist_ok=True)
    save_json(base.ROOT / 'training_verification.json', result)
    return {k: v for k, v in result.items() if k != 'sampling_schedule'}


def model():
    root = base.ROOT
    assert not (root / 'model_report.json').exists()
    proof = json.loads((root / 'training_verification.json').read_text())
    assert proof['passed'] and proof['protocol_sha256'] == sha(base.PROTOCOL)
    assert proof['master_protocol_sha256'] == sha(PROTOCOL)
    p = json.loads(base.PROTOCOL.read_text())
    t = relative.training('relative')
    x, y = base.encode(t), t.target.to_numpy()
    w = 1 / t.groupby('date').code.transform('size').to_numpy()
    planned = schedule(t.date)
    assert planned == proof['sampling_schedule']
    bias = float(np.average(y, weights=w))
    score = np.full(len(t), bias)
    trees, counts = [], []
    rng = np.random.RandomState(20260927)
    for dates in planned:
        active = t.date.isin(dates).to_numpy()
        estimator = DecisionTreeRegressor(criterion='friedman_mse', max_depth=3,
            min_samples_leaf=300, random_state=rng)
        estimator.fit(x[active], (y - score)[active], sample_weight=w[active])
        q = estimator.tree_
        tree = {key: getattr(q, key).tolist() for key in ['feature', 'threshold', 'children_left',
            'children_right', 'n_node_samples', 'weighted_n_node_samples', 'impurity']}
        tree['value'] = q.value.reshape(-1).tolist()
        trees.append(tree)
        pred = estimator.predict(x)
        np.testing.assert_allclose(pred, q.value.reshape(-1)[base.leaf_indices(x, tree)], rtol=0, atol=2e-12)
        score += .05 * pred
        counts.append(int(active.sum()))
    assert len(t) == p['expected_training_rows'] and t.date.nunique() == 241
    r = dict(protocol_sha256=sha(base.PROTOCOL), master_protocol_sha256=sha(PROTOCOL),
        feature_report_sha256=sha(inputs.ROOT / 'feature_report.json'),
        label_report_sha256=sha(labels.ROOT / 'full_label_report.json'),
        training_verification_sha256=sha(root / 'training_verification.json'),
        rows=len(t), days=t.date.nunique(), last_observation=t.next_date.max(),
        parameters=p['parameters'], feature_names=list(inputs.EXPRESSIONS),
        variant='relative_whole_date_subsample_half', learning_rate=.05, bias=bias, trees=trees,
        sampling_schedule=planned, sampled_rows_per_round=counts,
        training_start=p['training_start'], training_end=p['training_end'],
        new_2025_score_groups_read=bool(t.date.ge('2025-01-01').any()), new_2025H2_score_groups_read=False,
        new_2026_prices_read=False, no_exit_rules=True)
    np.testing.assert_allclose(base.predict(x, r), score, rtol=0, atol=2e-12)
    r['thresholds'] = [dict(id=i, training_quantile=q, threshold=float(np.quantile(score, q)))
                       for i, q in enumerate(base.QUANTILES)]
    save_json(root / 'model_report.json', r)
    return {k: v for k, v in r.items() if k not in ['trees', 'sampling_schedule']}


def verify_model():
    root = base.ROOT
    p = json.loads(base.PROTOCOL.read_text())
    r = json.loads((root / 'model_report.json').read_text())
    for key, path in [('protocol_sha256', base.PROTOCOL), ('master_protocol_sha256', PROTOCOL),
                      ('feature_report_sha256', inputs.ROOT / 'feature_report.json'),
                      ('label_report_sha256', labels.ROOT / 'full_label_report.json'),
                      ('training_verification_sha256', root / 'training_verification.json')]:
        assert r[key] == sha(path)
    assert r['variant'] == 'relative_whole_date_subsample_half' and r['parameters'] == p['parameters']
    assert r['feature_names'] == list(inputs.EXPRESSIONS) and len(r['trees']) == 64 and r['learning_rate'] == .05
    d, t, planned = independent_training()
    assert planned == r['sampling_schedule']
    assert len(d) == r['rows'] and d.date.nunique() == r['days'] == 241
    assert r['last_observation'] == t.next_date.max() < r['training_end'] == p['training_end']
    assert r['training_start'] == p['training_start']
    x = d[list(inputs.EXPRESSIONS)].to_numpy(dtype='int32')
    y, w = d.target.to_numpy(), d.w.to_numpy()
    np.testing.assert_allclose(r['bias'], np.average(y, weights=w), rtol=0, atol=2e-12)
    score = np.full(len(d), r['bias'])
    nodes, leaves = 0, 0
    for stage, (tree, dates) in enumerate(zip(r['trees'], planned)):
        active = d.date.isin(dates).to_numpy()
        assert int(active.sum()) == r['sampled_rows_per_round'][stage]
        assert d.loc[active, 'date'].nunique() == 120
        residual = y - score
        masks, levels = {0: np.ones(len(d), dtype=bool)}, {0: 0}
        terminal = np.empty(len(d))
        assert len(tree['feature']) <= 15
        for node, left in enumerate(tree['children_left']):
            mask = masks[node]
            fit = mask & active
            ww = w[fit]
            assert fit.sum() == tree['n_node_samples'][node] and levels[node] <= 3
            mean = np.average(residual[fit], weights=ww)
            variance = np.average((residual[fit] - mean) ** 2, weights=ww)
            np.testing.assert_allclose(ww.sum(), tree['weighted_n_node_samples'][node], rtol=0, atol=1e-8)
            np.testing.assert_allclose(mean, tree['value'][node], rtol=0, atol=2e-10)
            np.testing.assert_allclose(variance, tree['impurity'][node], rtol=0, atol=2e-8)
            right = tree['children_right'][node]
            if left < 0:
                assert right < 0 and fit.sum() >= 300
                terminal[mask] = mean
                leaves += 1
            else:
                split = x[:, tree['feature'][node]] <= tree['threshold'][node]
                masks[left], masks[right] = mask & split, mask & ~split
                levels[left] = levels[right] = levels[node] + 1
            nodes += 1
        score += .05 * terminal
    np.testing.assert_allclose(score, base.predict(x, r), rtol=0, atol=2e-10)
    for q, item in zip(base.QUANTILES, r['thresholds']):
        assert q == item['training_quantile']
        np.testing.assert_allclose(np.quantile(score, q), item['threshold'], rtol=0, atol=2e-10)
    result = dict(passed=True, model_report_sha256=sha(root / 'model_report.json'),
        rows=len(d), days=241, node_checks=nodes, leaf_checks=leaves,
        all_targets_inputs_date_weights_and_whole_date_schedule_rebuilt=True,
        all_sampled_node_residuals_weights_variances_and_sequential_updates_rebuilt=True,
        all_full_training_scores_and_quantiles_rebuilt=True,
        sampled_days_per_round=120, minimum_sampled_rows=min(r['sampled_rows_per_round']),
        maximum_sampled_rows=max(r['sampled_rows_per_round']), new_2026_prices_read=False, no_exit_rules=True)
    save_json(root / 'model_verification.json', result)
    return result


if __name__ == '__main__':
    from .tail_formula_offset_logit48 import verify_scores
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['verify_inputs', 'model', 'verify_model', 'scores', 'verify_scores', 'freeze', 'verify', 'analyze'])
    parser.add_argument('--fold', choices=['2024', 'recent', 'combined'], default='2024')
    args = parser.parse_args()
    setup(args.fold)
    if args.stage == 'analyze':
        for fold in ['2024', 'recent', '2025']:
            root = Path('data/research') / (STEM + '_' + fold)
            v = json.loads((root / 'selection_verification.json').read_text())
            assert v['passed'] and v['selection_report_sha256'] == sha(root / 'selection_report.json')
        result = evaluation.analyze(linkage.COMBINED if args.fold == 'combined' else base.ROOT,
                                    linkage.PROTOCOL if args.fold == 'combined' else base.PROTOCOL)
    elif args.fold == 'combined':
        assert args.stage in ['freeze', 'verify']
        result = linkage.combine() if args.stage == 'freeze' else linkage.verify_combined()
    elif args.stage == 'verify_scores':
        result = verify_scores()
    elif args.stage == 'freeze':
        result = study.freeze()
    elif args.stage == 'verify':
        r = json.loads((base.ROOT / 'model_report.json').read_text())
        assert (base.ROOT / 'frozen_numeric_core.tdx').read_text() == base.native_core(r, r['thresholds'][3]['threshold'], inputs.EXPRESSIONS, inputs.HEADER)
        result = study.verify()
    elif args.stage == 'scores':
        result = base.scores()
    else:
        result = globals()[args.stage]()
    print(json.dumps(result, ensure_ascii=False, indent=2))
