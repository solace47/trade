"""Revalue fixed independent trees using only their unsampled training weeks."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_before1000_model as original
from . import tail_formula_boundary_evaluation as evaluation
from . import tail_formula_context_2024 as linkage
from . import tail_formula_recent as study
from . import tail_formula_relative as relative
from . import tail_formula_week_bagging as parent
from .corporate_cash import save_json, sha

STEM = 'tail_formula_oob_leaf'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
PARAMETERS = {**parent.PARAMETERS, 'leaf_estimation': 'unused_weeks_with_ancestor_fallback',
    'minimum_estimation_rows': 300, 'minimum_estimation_dates': 20}
GEOMETRY = ['feature', 'threshold', 'children_left', 'children_right']


def setup(fold):
    p = json.loads(PROTOCOL.read_text())
    for file, digest in p['references'].items():
        assert sha(Path(file)) == digest
    assert p['parameters'] == PARAMETERS and not p['new_2026_prices_allowed']
    original.STEM = STEM
    original.setup(fold)
    for f in ['2024', 'recent']:
        q = json.loads((Path('config') / (STEM + '_' + f + '_protocol.json')).read_text())
        assert q['oob_leaf_protocol_sha256'] == sha(PROTOCOL) and q['parameters'] == PARAMETERS


def source_model():
    q = json.loads(base.PROTOCOL.read_text())
    fold = '2024' if q['evaluation_start'] == '2025-01-01' else 'recent'
    source = Path('data/research') / (parent.STEM + '_' + fold)
    r = json.loads((source / 'model_report.json').read_text())
    v = json.loads((source / 'model_verification.json').read_text())
    assert v['passed'] and v['model_report_sha256'] == q['source_model_report_sha256'] == sha(source / 'model_report.json')
    assert r['feature_names'] == list(base.EXPRESSIONS) and r['parameters'] == parent.PARAMETERS
    assert r['variant'] == 'independent_week_bootstrap_average'
    for key in ['rows', 'days']:
        assert r[key] == q['expected_' + ('training_days' if key == 'days' else key)]
    return source, r


def verify_inputs():
    source, r = source_model()
    assert not (base.ROOT / 'training_input_verification.json').exists()
    start, end, where = relative.training_scope()
    f = base.feature_inputs()[['date', 'code', 'formula_input_valid', *base.EXPRESSIONS]]
    c = base.conn()
    c.register('features', f)
    encoded = ','.join(f'floor(least(greatest(100*{name}+10000+.000001,0),999999))::INT AS {name}' for name in base.EXPRESSIONS)
    c.execute(f'''CREATE VIEW targets AS WITH l AS(SELECT date,code,opportunity15 AS utility
        FROM read_parquet('{base.SOURCE}/full_labels.parquet') WHERE {where} AND known15)
        SELECT date,code,utility-avg(utility) OVER(PARTITION BY date) AS target FROM l''')
    d = c.sql('''SELECT date,code,target,1./count(*) OVER(PARTITION BY date) AS w,
        strftime(date_trunc('week',date::DATE),'%Y-%m-%d') AS week_start,''' + encoded + '''
        FROM features JOIN targets USING(date,code) WHERE formula_input_valid ORDER BY date,code''').df()
    c.close()
    train = relative.training('relative')
    pd.testing.assert_frame_equal(d[['date', 'code']], train[['date', 'code']], check_exact=True)
    np.testing.assert_allclose(d.target, train.target, rtol=0, atol=2e-12)
    np.testing.assert_allclose(d.w, 1/train.groupby('date').code.transform('size'), rtol=0, atol=0)
    np.testing.assert_array_equal(d[list(base.EXPRESSIONS)].to_numpy(dtype='int32'), base.encode(train))
    weeks = sorted(d.week_start.unique())
    assert len(d) == r['rows'] and d.date.nunique() == r['days'] == 241 and len(weeks) == 52
    assert weeks == r['training_week_starts'] and train.next_date.max() == r['last_observation'] < end
    rng = np.random.default_rng(20260927)
    trace = []
    for i, item in enumerate(r['sampling_schedule']):
        drawn = rng.choice(weeks, size=len(weeks), replace=True).tolist()
        assert drawn == item['drawn_week_starts'] and item['tree'] == i
        remaining = sorted(set(weeks)-set(drawn))
        oob = d.week_start.isin(remaining)
        assert not set(remaining).intersection(drawn)
        assert (~oob).sum() == item['rows'] and d.loc[~oob, 'date'].nunique() == item['days']
        assert d.loc[oob, 'date'].nunique() >= 20 and oob.sum() >= 300
        trace.append(dict(tree=i, unused_week_starts=remaining, estimation_rows=int(oob.sum()),
            estimation_dates=d.loc[oob, 'date'].nunique(), splitting_rows=item['rows'], splitting_dates=item['days']))
    base.ROOT.mkdir(parents=True, exist_ok=True)
    d.to_parquet(base.ROOT / 'verified_training.parquet', index=False, compression='zstd')
    proof = dict(passed=True, protocol_sha256=sha(base.PROTOCOL), source_model_report_sha256=sha(source / 'model_report.json'),
        feature_report_sha256=sha(base.FEATURES / 'feature_report.json'), label_report_sha256=sha(base.SOURCE / 'full_label_report.json'),
        training_sha256=sha(base.ROOT / 'verified_training.parquet'), rows=len(d), days=d.date.nunique(), trace=trace,
        training_start=start, training_end=end, last_observation=train.next_date.max(),
        all_inputs_targets_date_weights_and_integer_values_independently_rebuilt=True,
        every_estimation_week_is_excluded_from_its_tree_structure=True,
        new_2025H2_score_groups_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(base.ROOT / 'training_input_verification.json', proof)
    return {k: v for k, v in proof.items() if k != 'trace'}


def checked_inputs():
    source, r = source_model()
    v = json.loads((base.ROOT / 'training_input_verification.json').read_text())
    assert v['passed'] and v['protocol_sha256'] == sha(base.PROTOCOL)
    assert v['source_model_report_sha256'] == sha(source / 'model_report.json')
    assert v['feature_report_sha256'] == sha(base.FEATURES / 'feature_report.json')
    assert v['label_report_sha256'] == sha(base.SOURCE / 'full_label_report.json')
    assert v['training_sha256'] == sha(base.ROOT / 'verified_training.parquet')
    return r, v, pd.read_parquet(base.ROOT / 'verified_training.parquet')


def revalue(tree, x, y, weights, dates, oob, minimum_rows=300, minimum_dates=20):
    leaf = base.leaf_indices(x, tree)
    parents = {0: -1}
    descendants = {}
    def walk(node):
        left, right = tree['children_left'][node], tree['children_right'][node]
        if left < 0:
            descendants[node] = [node]
        else:
            parents[left] = parents[right] = node
            descendants[node] = walk(left) + walk(right)
        return descendants[node]
    walk(0)
    rows, days, mass, means = [], [], [], []
    for node in range(len(tree['feature'])):
        mask = oob & np.isin(leaf, descendants[node])
        w = weights[mask]
        rows.append(int(mask.sum()))
        days.append(len(np.unique(dates[mask])))
        mass.append(float(w.sum()))
        means.append(float(np.dot(w, y[mask])/w.sum()) if mask.any() else None)
    supported = [a >= minimum_rows and b >= minimum_dates for a, b in zip(rows, days)]
    if not supported[0]:
        raise ValueError('No supported out-of-bag root; no old value or zero fallback is permitted')
    chosen = []
    for node in range(len(tree['feature'])):
        use = node
        while not supported[use]:
            use = parents[use]
        chosen.append(use)
    raw = np.asarray([means[node] for node in chosen])
    return {**{key: copy.deepcopy(tree[key]) for key in GEOMETRY}, 'raw_value': raw.tolist(),
        'value': (raw/64/.05).tolist(), 'estimation_rows': rows, 'estimation_dates': days,
        'estimation_weight': mass, 'unbacked_means': means, 'value_source_node': chosen}


def model():
    source, proof, d = checked_inputs()
    assert not (base.ROOT / 'model_report.json').exists()
    x = d[list(base.EXPRESSIONS)].to_numpy(dtype='int32')
    y, w, dates = d.target.to_numpy(), d.w.to_numpy(), d.date.to_numpy()
    trees, counts = [], []
    for tree, schedule in zip(source['trees'], source['sampling_schedule']):
        oob = parent.multiplicities(d.week_start.to_numpy(), schedule['drawn_week_starts']) == 0
        result = revalue(tree, x, y, w, dates, oob)
        trees.append(result)
        counts.append(sum(left < 0 and node != result['value_source_node'][node] for node, left in enumerate(tree['children_left'])))
    r = dict(protocol_sha256=sha(base.PROTOCOL), inputs_protocol_sha256=sha(PROTOCOL),
        source_model_report_sha256=proof['source_model_report_sha256'], training_input_verification_sha256=sha(base.ROOT / 'training_input_verification.json'),
        feature_report_sha256=proof['feature_report_sha256'], label_report_sha256=proof['label_report_sha256'],
        rows=proof['rows'], days=proof['days'], last_observation=proof['last_observation'], training_start=proof['training_start'], training_end=proof['training_end'],
        feature_names=list(base.EXPRESSIONS), parameters=PARAMETERS, variant='unused_week_leaf_estimation',
        bias=0., learning_rate=.05, aggregation_weight=1/64, trees=trees,
        fallback_leaves_by_tree=counts, fallback_leaves=sum(counts), original_structure_and_sampling_reused=True,
        all_estimation_targets_are_original_not_boosting_residuals=True, new_tree_structures_fitted=False,
        new_2025H2_score_groups_read=False, new_2026_prices_read=False, no_exit_rules=True)
    score = base.predict(x, r)
    r['thresholds'] = [dict(id=i, training_quantile=q, threshold=float(np.quantile(score, q))) for i, q in enumerate(base.QUANTILES)]
    save_json(base.ROOT / 'model_report.json', r)
    return {k: v for k, v in r.items() if k != 'trees'}


def verify_model():
    source, input_proof, d = checked_inputs()
    r = json.loads((base.ROOT / 'model_report.json').read_text())
    assert r['protocol_sha256'] == sha(base.PROTOCOL) and r['inputs_protocol_sha256'] == sha(PROTOCOL)
    assert r['training_input_verification_sha256'] == sha(base.ROOT / 'training_input_verification.json')
    assert r['source_model_report_sha256'] == input_proof['source_model_report_sha256']
    for name in ['feature_report_sha256', 'label_report_sha256', 'rows', 'days', 'last_observation', 'training_start', 'training_end']:
        assert r[name] == input_proof[name]
    assert r['parameters'] == PARAMETERS and r['feature_names'] == list(base.EXPRESSIONS)
    assert r['bias'] == 0 and r['learning_rate'] == .05 and r['aggregation_weight'] == 1/64 and len(r['trees']) == 64
    x = d[r['feature_names']].to_numpy(dtype='int32')
    y, w = d.target.to_numpy(), d.w.to_numpy()
    dates, weeks = d.date.to_numpy(), d.week_start.to_numpy()
    score = np.zeros(len(d))
    checks, fallbacks = 0, []
    for tree, old, item in zip(r['trees'], source['trees'], source['sampling_schedule']):
        assert all(tree[k] == old[k] for k in GEOMETRY)
        drawn = set(item['drawn_week_starts'])
        oob = np.fromiter((week not in drawn for week in weeks), dtype=bool, count=len(d))
        masks, parents, means, eligible = {0: np.ones(len(d), bool)}, {0: -1}, {}, {}
        for node, left in enumerate(tree['children_left']):
            mask = masks[node] & oob
            count, day_count = int(mask.sum()), len(set(dates[mask]))
            ww = w[mask]
            mean = float(np.average(y[mask], weights=ww)) if count else None
            assert count == tree['estimation_rows'][node] and day_count == tree['estimation_dates'][node]
            np.testing.assert_allclose(ww.sum(), tree['estimation_weight'][node], rtol=0, atol=1e-10)
            if mean is None:
                assert tree['unbacked_means'][node] is None
            else:
                np.testing.assert_allclose(mean, tree['unbacked_means'][node], rtol=0, atol=2e-12)
            means[node] = mean
            eligible[node] = count >= 300 and day_count >= 20
            if left >= 0:
                right = tree['children_right'][node]
                split = x[:, tree['feature'][node]] <= tree['threshold'][node]
                masks[left], masks[right] = masks[node] & split, masks[node] & ~split
                parents[left] = parents[right] = node
            checks += 1
        assert eligible[0]
        count = 0
        for node, left in enumerate(tree['children_left']):
            ancestors = []
            cur = node
            while cur >= 0:
                ancestors.append(cur)
                cur = parents[cur]
            chosen = next(a for a in ancestors if eligible[a])
            assert chosen == tree['value_source_node'][node]
            np.testing.assert_allclose(means[chosen], tree['raw_value'][node], rtol=0, atol=2e-12)
            np.testing.assert_allclose(means[chosen]/64/.05, tree['value'][node], rtol=0, atol=2e-12)
            if left < 0:
                count += chosen != node
                score[masks[node]] += means[chosen]/64
        fallbacks.append(count)
    assert fallbacks == r['fallback_leaves_by_tree'] and sum(fallbacks) == r['fallback_leaves']
    np.testing.assert_allclose(score, base.predict(x, r), rtol=0, atol=2e-12)
    for q, item in zip(base.QUANTILES, r['thresholds']):
        assert q == item['training_quantile']
        np.testing.assert_allclose(np.quantile(score, q), item['threshold'], rtol=0, atol=2e-12)
    proof = dict(passed=True, model_report_sha256=sha(base.ROOT / 'model_report.json'), rows=len(d), node_checks=checks,
        all_source_geometry_identical=True, all_unused_week_node_support_and_means_independently_rebuilt=True,
        every_leaf_uses_nearest_eligible_ancestor_without_old_values=True, fallback_leaves=sum(fallbacks),
        original_target_mean_and_export_scaling_verified=True, new_2025H2_score_groups_read=False, new_2026_prices_read=False, no_exit_rules=True)
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
        assert a.fold == 'combined'
        for f in ['2024', 'recent', '2025']:
            root = Path('data/research') / (STEM + '_' + f)
            v = json.loads((root / 'selection_verification.json').read_text())
            assert v['passed'] and v['selection_report_sha256'] == sha(root / 'selection_report.json')
        result = evaluation.analyze(linkage.COMBINED, linkage.PROTOCOL)
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
        r = json.loads((base.ROOT / 'model_report.json').read_text())
        assert (base.ROOT / 'frozen_numeric_core.tdx').read_text() == base.native_core(r, r['thresholds'][3]['threshold'], base.EXPRESSIONS, base.HEADER)
        result = study.verify()
    else:
        result = base.scores()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
