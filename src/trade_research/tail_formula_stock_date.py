"""Opportunity targets residualized against historical stock and date effects."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsmr

from . import tail_formula_additive as base
from . import tail_formula_before1000_model as original
from . import tail_formula_boundary_evaluation as evaluation
from . import tail_formula_context_2024 as linkage
from . import tail_formula_recent as study
from . import tail_formula_relative as relative
from .corporate_cash import save_json, sha

STEM = 'tail_formula_stock_date'
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
        assert q['stock_date_protocol_sha256'] == sha(PROTOCOL) and q['parameters'] == PARAMETERS


def projection(frame):
    """Weighted alternating projections; weights give every date unit mass."""
    out = frame.copy()
    assert out.opportunity15.isin([0, 1]).all() and not out[['date','code']].duplicated().any()
    di, dates = pd.factorize(out.date, sort=True)
    si, stocks = pd.factorize(out.code, sort=True)
    dcount = np.bincount(di)
    weights = 1./dcount[di]
    dmass = np.bincount(di, weights=weights)
    smass = np.bincount(si, weights=weights)
    residual = out.opportunity15.to_numpy(float).copy()
    for iterations in range(1, 10001):
        residual -= (np.bincount(di, weights=weights*residual)/dmass)[di]
        residual -= (np.bincount(si, weights=weights*residual)/smass)[si]
        date_mean = np.bincount(di, weights=weights*residual)/dmass
        stock_mean = np.bincount(si, weights=weights*residual)/smass
        maximum = max(np.abs(date_mean).max(), np.abs(stock_mean).max())
        if maximum <= 1e-13:
            break
    else:
        raise AssertionError('Two-way projection did not reach the fixed tolerance')
    out['date_rows'] = dcount[di]
    out['stock_rows'] = np.bincount(si)[si]
    out['w'] = weights
    out['date_only_target'] = out.opportunity15-out.groupby('date').opportunity15.transform('mean')
    out['target'] = residual
    out.attrs['projection_iterations'] = iterations
    out.attrs['max_weighted_group_residual'] = float(maximum)
    return out


def least_squares_target(frame):
    """Solve the same projection via a sparse weighted two-factor design matrix."""
    di, dates = pd.factorize(frame.date, sort=True)
    si, stocks = pd.factorize(frame.code, sort=True)
    weights = frame.w.to_numpy(float)
    n = len(frame)
    design = coo_matrix((np.tile(np.sqrt(weights), 2),
        (np.tile(np.arange(n), 2), np.r_[di, si+len(dates)])), shape=(n,len(dates)+len(stocks))).tocsr()
    fit = lsmr(design, np.sqrt(weights)*frame.opportunity15.to_numpy(float),
               atol=1e-14, btol=1e-14, conlim=1e12, maxiter=10000)
    assert fit[1] in [0,1,2], 'Independent sparse solver failed to converge'
    residual = frame.opportunity15.to_numpy(float)-fit[0][di]-fit[0][si+len(dates)]
    maximum = max(np.max(np.abs(np.bincount(di,weights=weights*residual)/np.bincount(di,weights=weights))),
        np.max(np.abs(np.bincount(si,weights=weights*residual)/np.bincount(si,weights=weights))))
    assert maximum <= 2e-10
    return residual, dict(stop_code=int(fit[1]),iterations=int(fit[2]),max_weighted_group_residual=float(maximum))


def training():
    start, end, _ = relative.training_scope()
    return projection(base.training(start=start, end=end))


def independent_training():
    start, end, where = relative.training_scope()
    names = list(base.EXPRESSIONS)
    f = base.feature_inputs()[['date', 'code', 'formula_input_valid', *names]]
    c = base.conn();c.register('features', f)
    encoded = ','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::INT AS {n}' for n in names)
    d = c.sql(f"""WITH joined AS(SELECT date,code,next_date,opportunity15,{encoded}
        FROM read_parquet('{base.SOURCE}/full_labels.parquet') JOIN features USING(date,code)
        WHERE {where} AND known15 AND formula_input_valid)
        SELECT *,count(*) OVER(PARTITION BY date) AS date_rows,
        count(*) OVER(PARTITION BY code) AS stock_rows,
        1./count(*) OVER(PARTITION BY date) AS w,
        opportunity15-avg(opportunity15) OVER(PARTITION BY date) AS date_only_target
        FROM joined ORDER BY date,code""").df()
    c.close()
    t = training()
    pd.testing.assert_frame_equal(d[['date','code','next_date']], t[['date','code','next_date']], check_exact=True)
    for name in ['opportunity15', 'date_rows', 'stock_rows', 'date_only_target', 'w']:
        np.testing.assert_array_equal(d[name], t[name])
    np.testing.assert_array_equal(d[names].to_numpy(dtype='int32'), base.encode(t))
    residual, diagnostics = least_squares_target(d)
    np.testing.assert_allclose(residual, t.target, rtol=0, atol=2e-10)
    d['independent_target'] = residual
    # Keep the exact producer target fixed; its mathematical equivalence has
    # just been verified by the independent sparse least-squares implementation.
    d['target'] = t.target.to_numpy()
    diagnostics.update(projection_iterations=t.attrs['projection_iterations'],
        projection_max_weighted_group_residual=t.attrs['max_weighted_group_residual'],
        maximum_target_difference=float(np.max(np.abs(residual-t.target.to_numpy()))))
    p = json.loads(base.PROTOCOL.read_text())
    assert len(d) == p['expected_rows'] and d.date.nunique() == p['expected_training_days'] == 241
    assert d.date.min() >= start and d.next_date.max() < end
    np.testing.assert_allclose(d.groupby('date').w.sum(), 1., rtol=0, atol=2e-12)
    return d, diagnostics


def verify_inputs():
    assert not (base.ROOT / 'training_input_verification.json').exists()
    d, diagnostics = independent_training()
    def group_records(key):
        f=d[[key,'target','w']].copy();f['weighted_target']=f.target*f.w
        a=f.groupby(key).agg(rows=('target','size'),mass=('w','sum'),weighted_total=('weighted_target','sum')).reset_index()
        a['weighted_mean']=a.weighted_total/a.mass
        assert a.weighted_mean.abs().max() <= 1e-13
        return a
    days,stocks=group_records('date'),group_records('code')
    base.ROOT.mkdir(parents=True, exist_ok=True)
    d.to_parquet(base.ROOT / 'verified_training.parquet', index=False, compression='zstd')
    days.to_parquet(base.ROOT / 'training_days.parquet', index=False, compression='zstd')
    stocks.to_parquet(base.ROOT / 'training_stocks.parquet', index=False, compression='zstd')
    result = dict(passed=True, protocol_sha256=sha(base.PROTOCOL),
        feature_report_sha256=sha(base.FEATURES / 'feature_report.json'),
        label_report_sha256=sha(base.SOURCE / 'full_label_report.json'),
        training_sha256=sha(base.ROOT / 'verified_training.parquet'), training_days_sha256=sha(base.ROOT / 'training_days.parquet'),
        training_stocks_sha256=sha(base.ROOT / 'training_stocks.parquet'),
        rows=len(d),days=len(days),stocks=len(stocks),last_observation=d.next_date.max(),
        minimum_stock_rows=int(stocks.rows.min()),maximum_stock_rows=int(stocks.rows.max()),
        single_row_stocks=int(stocks.rows.eq(1).sum()),target_min=float(d.target.min()),target_max=float(d.target.max()),
        solvers=diagnostics,all_original_training_keys_labels_and_48_encodings_preserved=True,
        date_weights_and_single_date_targets_independently_sql_rebuilt=True,
        both_date_and_stock_weighted_target_means_zero=True,independent_sparse_least_squares_residuals_match=True,
        existing_same_intersection_date_only_control_reused=True,
        new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(base.ROOT / 'training_input_verification.json', result)
    return result


def checked_inputs():
    p = json.loads((base.ROOT / 'training_input_verification.json').read_text())
    assert p['passed'] and p['protocol_sha256'] == sha(base.PROTOCOL)
    assert p['feature_report_sha256'] == sha(base.FEATURES / 'feature_report.json')
    assert p['label_report_sha256'] == sha(base.SOURCE / 'full_label_report.json')
    assert p['training_sha256'] == sha(base.ROOT / 'verified_training.parquet')
    assert p['training_days_sha256'] == sha(base.ROOT / 'training_days.parquet')
    assert p['training_stocks_sha256'] == sha(base.ROOT / 'training_stocks.parquet')
    return p, pd.read_parquet(base.ROOT / 'verified_training.parquet')


def model():
    proof, d = checked_inputs()
    assert not (base.ROOT / 'model_report.json').exists()
    # Recompute the exact producer projection and bind it to the independently
    # verified sparse least-squares input witness before fitting.
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
        feature_names=list(base.EXPRESSIONS), variant='stock_and_date_fixed_effect_residual', learning_rate=.05,
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
    assert r['variant'] == 'stock_and_date_fixed_effect_residual' and r['feature_names'] == list(base.EXPRESSIONS)
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
        all_node_counts_date_weights_residual_means_and_variances_rebuilt=True,
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
