"""Pointwise opportunity learning with equal within-date class mass."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from . import tail_formula_additive as base
from . import tail_formula_before1000_model as original
from . import tail_formula_boundary_evaluation as evaluation
from . import tail_formula_context_2024 as linkage
from . import tail_formula_recent as study
from . import tail_formula_relative as relative
from .corporate_cash import save_json, sha

STEM = 'tail_formula_class_balance'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
PARAMETERS = dict(n_estimators=64, max_depth=3, min_samples_leaf=300, learning_rate=.05,
    subsample=1., random_state=20260927, loss='squared_error', criterion='friedman_mse')


def setup(fold):
    p = json.loads(PROTOCOL.read_text())
    for path, digest in p['references'].items():
        assert sha(Path(path)) == digest
    assert p['parameters'] == PARAMETERS and p['training_quantile'] == .995
    assert not p['new_2026_prices_allowed']
    original.STEM = STEM
    original.setup(fold)
    for f in ['2024', 'recent']:
        q = json.loads((Path('config') / (STEM + '_' + f + '_protocol.json')).read_text())
        assert q['class_balance_protocol_sha256'] == sha(PROTOCOL) and q['parameters'] == PARAMETERS


def balance(frame):
    assert frame.opportunity15.isin([0, 1]).all()
    out = frame.copy()
    groups = out.groupby('date').opportunity15
    out['classes'] = groups.transform('nunique')
    out['class_rows'] = out.groupby(['date', 'opportunity15']).opportunity15.transform('size')
    out['date_rows'] = groups.transform('size')
    out['centre'] = np.where(out.classes.eq(2), .5, out.opportunity15)
    out['target'] = out.opportunity15 - out.centre
    out['w'] = 1./(out.classes*out.class_rows)
    return out


def training():
    start, end, _ = relative.training_scope()
    t = base.training(start=start, end=end)
    return balance(t)


def independent_training():
    start, end, where = relative.training_scope()
    names = list(base.EXPRESSIONS)
    f = base.feature_inputs()[['date', 'code', 'formula_input_valid', *names]]
    c = base.conn()
    c.register('features', f)
    encoded = ','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::INT AS {n}' for n in names)
    d = c.sql(f'''WITH joined AS(SELECT date,code,next_date,opportunity15,{encoded}
        FROM read_parquet('{base.SOURCE}/full_labels.parquet') JOIN features USING(date,code)
        WHERE {where} AND known15 AND formula_input_valid),
        counts AS(SELECT *,count(*) OVER(PARTITION BY date,opportunity15) AS class_rows,
        count(*) OVER(PARTITION BY date) AS date_rows,
        min(opportunity15) OVER(PARTITION BY date) AS low,max(opportunity15) OVER(PARTITION BY date) AS high FROM joined)
        SELECT date,code,next_date,opportunity15,{','.join(names)},class_rows,date_rows,
        1+high-low AS classes,(high+low)/2. AS centre,opportunity15-(high+low)/2. AS target,
        1./((1+high-low)*class_rows) AS w FROM counts ORDER BY date,code''').df()
    c.close()
    t = training()
    keys = ['date', 'code', 'next_date']
    pd.testing.assert_frame_equal(d[keys], t[keys], check_exact=True)
    for name in ['opportunity15', 'class_rows', 'date_rows', 'classes', 'centre', 'target', 'w']:
        np.testing.assert_array_equal(d[name], t[name])
    np.testing.assert_array_equal(d[names].to_numpy(dtype='int32'), base.encode(t))
    p = json.loads(base.PROTOCOL.read_text())
    assert len(d) == p['expected_rows'] and d.date.nunique() == p['expected_training_days'] == 241
    assert d.date.min() >= start and d.next_date.max() < end
    np.testing.assert_allclose(d.groupby('date').w.sum(), 1., rtol=0, atol=2e-12)
    np.testing.assert_allclose((d.w*d.target).groupby(d.date).sum(), 0., rtol=0, atol=2e-12)
    return d


def verify_inputs():
    assert not (base.ROOT / 'training_input_verification.json').exists()
    d = independent_training()
    counts = pd.crosstab(d.date, d.opportunity15).reindex(columns=[0., 1.], fill_value=0)
    counts.columns = ['negative', 'positive']
    counts = counts.reset_index()
    counts['classes'] = counts[['negative', 'positive']].gt(0).sum(axis=1)
    base.ROOT.mkdir(parents=True, exist_ok=True)
    d.to_parquet(base.ROOT / 'verified_training.parquet', index=False, compression='zstd')
    counts.to_parquet(base.ROOT / 'training_days.parquet', index=False, compression='zstd')
    result = dict(passed=True, protocol_sha256=sha(base.PROTOCOL),
        feature_report_sha256=sha(base.FEATURES / 'feature_report.json'),
        label_report_sha256=sha(base.SOURCE / 'full_label_report.json'),
        training_sha256=sha(base.ROOT / 'verified_training.parquet'), training_days_sha256=sha(base.ROOT / 'training_days.parquet'),
        rows=len(d), days=d.date.nunique(), last_observation=d.next_date.max(),
        two_class_days=int(counts.classes.eq(2).sum()), single_class_days=counts.loc[counts.classes.eq(1), 'date'].tolist(),
        smallest_present_class=int(d.class_rows.min()), maximum_row_day_weight=float(d.w.max()),
        all_original_training_keys_labels_and_48_encodings_preserved=True,
        all_date_class_counts_centred_targets_and_weights_independently_sql_rebuilt=True,
        every_date_has_unit_weight_and_zero_weighted_target_mean=True,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(base.ROOT / 'training_input_verification.json', result)
    return result


def checked_inputs():
    p = json.loads((base.ROOT / 'training_input_verification.json').read_text())
    assert p['passed'] and p['protocol_sha256'] == sha(base.PROTOCOL)
    assert p['feature_report_sha256'] == sha(base.FEATURES / 'feature_report.json')
    assert p['label_report_sha256'] == sha(base.SOURCE / 'full_label_report.json')
    assert p['training_sha256'] == sha(base.ROOT / 'verified_training.parquet')
    assert p['training_days_sha256'] == sha(base.ROOT / 'training_days.parquet')
    return p, pd.read_parquet(base.ROOT / 'verified_training.parquet')


def model():
    proof, d = checked_inputs()
    assert not (base.ROOT / 'model_report.json').exists()
    # The producer recomputes the Pandas transformation; the saved SQL table is
    # the independent input witness, not an alternative feature population.
    t = training()
    x, y, weights = base.encode(t), t.target.to_numpy(), t.w.to_numpy()
    np.testing.assert_array_equal(x, d[list(base.EXPRESSIONS)].to_numpy(dtype='int32'))
    np.testing.assert_array_equal(y, d.target)
    np.testing.assert_array_equal(weights, d.w)
    fitted = GradientBoostingRegressor(**PARAMETERS).fit(x, y, sample_weight=weights)
    trees = []
    for estimator in fitted.estimators_.ravel():
        q = estimator.tree_
        trees.append(dict(feature=q.feature.tolist(), threshold=q.threshold.tolist(), children_left=q.children_left.tolist(),
            children_right=q.children_right.tolist(), n_node_samples=q.n_node_samples.tolist(),
            weighted_n_node_samples=q.weighted_n_node_samples.tolist(), value=q.value.reshape(-1).tolist(), impurity=q.impurity.tolist()))
    p = json.loads(base.PROTOCOL.read_text())
    result = dict(protocol_sha256=sha(base.PROTOCOL), master_protocol_sha256=sha(PROTOCOL),
        training_input_verification_sha256=sha(base.ROOT / 'training_input_verification.json'),
        feature_report_sha256=proof['feature_report_sha256'], label_report_sha256=proof['label_report_sha256'],
        rows=proof['rows'], days=proof['days'], last_observation=proof['last_observation'],
        training_start=p['training_start'], training_end=p['training_end'], parameters=fitted.get_params(),
        feature_names=list(base.EXPRESSIONS), variant='within_day_class_balanced_pointwise', learning_rate=.05,
        bias=float(fitted.init_.constant_.ravel()[0]), trees=trees, score_is_not_win_probability=True,
        new_2025_score_groups_read=bool(t.date.ge('2025-01-01').any()), new_2025H2_score_groups_read=False,
        new_2026_prices_read=False, no_exit_rules=True)
    score = base.predict(x, result)
    np.testing.assert_allclose(score, fitted.predict(x), rtol=0, atol=2e-12)
    result['thresholds'] = [dict(id=i, training_quantile=q, threshold=float(np.quantile(score, q))) for i, q in enumerate(base.QUANTILES)]
    save_json(base.ROOT / 'model_report.json', result)
    return {k: v for k, v in result.items() if k != 'trees'}


def verify_model():
    proof, d = checked_inputs()
    r = json.loads((base.ROOT / 'model_report.json').read_text())
    p = json.loads(base.PROTOCOL.read_text())
    for key, path in [('protocol_sha256', base.PROTOCOL), ('master_protocol_sha256', PROTOCOL),
                      ('training_input_verification_sha256', base.ROOT / 'training_input_verification.json')]:
        assert r[key] == sha(path)
    for key in ['feature_report_sha256', 'label_report_sha256', 'rows', 'days', 'last_observation']:
        assert r[key] == proof[key]
    assert r['rows'] == p['expected_rows'] and r['days'] == 241
    assert r['training_start'] == p['training_start'] and r['training_end'] == p['training_end']
    assert r['variant'] == 'within_day_class_balanced_pointwise' and r['feature_names'] == list(base.EXPRESSIONS)
    assert all(r['parameters'][key] == value for key, value in PARAMETERS.items())
    assert len(r['trees']) == 64 and r['learning_rate'] == .05
    x = d[list(base.EXPRESSIONS)].to_numpy(dtype='int32')
    y, w = d.target.to_numpy(), d.w.to_numpy()
    bias = float(np.average(y, weights=w))
    np.testing.assert_allclose(bias, r['bias'], rtol=0, atol=2e-12)
    score = np.full(len(d), bias)
    checks = 0
    for tree in r['trees']:
        residual = y-score
        masks, levels = {0: np.ones(len(d), bool)}, {0: 0}
        step = np.empty(len(d))
        assert len(tree['feature']) <= 15
        for node, left in enumerate(tree['children_left']):
            take = masks[node]
            weights = w[take]
            mean = np.average(residual[take], weights=weights)
            variance = np.average((residual[take]-mean)**2, weights=weights)
            assert int(take.sum()) == tree['n_node_samples'][node] and levels[node] <= 3
            np.testing.assert_allclose(weights.sum(), tree['weighted_n_node_samples'][node], rtol=0, atol=1e-8)
            np.testing.assert_allclose(mean, tree['value'][node], rtol=0, atol=2e-10)
            np.testing.assert_allclose(variance, tree['impurity'][node], rtol=0, atol=2e-10)
            if left < 0:
                assert take.sum() >= 300
                step[take] = mean
            else:
                right = tree['children_right'][node]
                cut = x[:, tree['feature'][node]] <= tree['threshold'][node]
                masks[left], masks[right] = take & cut, take & ~cut
                levels[left] = levels[right] = levels[node]+1
            checks += 1
        score += .05*step
    np.testing.assert_allclose(score, base.predict(x, r), rtol=0, atol=2e-10)
    assert len(r['thresholds']) == len(base.QUANTILES)
    for i, (q, item) in enumerate(zip(base.QUANTILES, r['thresholds'])):
        assert item['id'] == i and item['training_quantile'] == q
        np.testing.assert_allclose(np.quantile(score, q), item['threshold'], rtol=0, atol=2e-10)
    result = dict(passed=True, model_report_sha256=sha(base.ROOT / 'model_report.json'), rows=len(d), node_checks=checks,
        all_node_counts_class_weights_residual_means_and_variances_rebuilt=True,
        all_exported_training_scores_and_quantiles_rebuilt=True,
        score_is_not_win_probability=True, new_2025H2_score_groups_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(base.ROOT / 'model_verification.json', result)
    return result


def main():
    from .tail_formula_offset_logit48 import verify_scores
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['verify_inputs', 'model', 'verify_model', 'scores', 'verify_scores', 'freeze', 'verify', 'analyze'])
    parser.add_argument('--fold', choices=['2024', 'recent', 'combined'], default='2024')
    a = parser.parse_args()
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
