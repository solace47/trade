"""Fixed lower-quartile next-morning price-space model; no exit optimization."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from . import tail_formula_additive as base
from . import tail_formula_float as original
from .corporate_cash import save_json, sha
from .tail_formula_1000_daily import analyze as common_analysis

STEM = 'tail_formula_quantile48'


def root(fold):
    return Path('data/research') / (STEM + '_' + ('2025' if fold == 'combined' else fold))


def protocol(fold):
    return Path('config') / (STEM + '_' + fold + '_protocol.json')


def setup(fold):
    original.setup(fold)
    base.ROOT = root(fold); base.PROTOCOL = protocol(fold)


def config(fold):
    p = json.loads(protocol(fold).read_text())
    assert p['selection_score_cut'] == 0 and p['parameters']['alpha'] == .25
    assert p['feature_report_sha256'] == sha(original.ROOT / 'feature_report.json')
    assert p['label_report_sha256'] == sha(base.SOURCE / 'full_label_report.json')
    return p


def training(fold):
    p = config(fold)
    t = base.training(start=p['training_start'], end=p['training_end'])
    l = pd.read_parquet(base.SOURCE / 'full_labels.parquet', columns=['date', 'code', 'sustained_return15'],
                       filters=[('date', '>=', p['training_start']), ('next_date', '<', p['training_end'])])
    out = t.merge(l, on=['date', 'code'], validate='one_to_one').sort_values(['date', 'code']).reset_index(drop=True)
    assert len(out) == len(t) and np.isfinite(out.sustained_return15).all()
    return out


def model(fold):
    path = root(fold); p = config(fold)
    if (path / 'model_report.json').exists():
        raise ValueError('Do not refit the frozen quartile model')
    path.mkdir(parents=True, exist_ok=True)
    t = training(fold); x = base.encode(t); y = t.sustained_return15.to_numpy()
    w = 1 / t.groupby('date').code.transform('size').to_numpy()
    m = GradientBoostingRegressor(**p['parameters']).fit(x, y, sample_weight=w)
    trees = []
    for estimator in m.estimators_.ravel():
        tree = estimator.tree_
        r = {key: getattr(tree, key).tolist() for key in ['feature', 'threshold', 'children_left', 'children_right',
                                                       'n_node_samples', 'weighted_n_node_samples', 'impurity']}
        r['value'] = tree.value.reshape(-1).tolist(); trees.append(r)
    r = dict(protocol_sha256=sha(protocol(fold)), feature_report_sha256=p['feature_report_sha256'],
             label_report_sha256=p['label_report_sha256'], variant='absolute_lower_quartile_space',
             rows=len(t), days=t.date.nunique(), last_observation=t.next_date.max(),
             training_start=p['training_start'], training_end=p['training_end'], parameters=m.get_params(),
             feature_names=list(original.EXPRESSIONS), learning_rate=.05, bias=float(m.init_.constant_.ravel()[0]),
             trees=trees, selection_score_cut=0., alpha=.25,
             new_2025_score_groups_read=bool(t.date.ge('2025-01-01').any()), new_2025H2_score_groups_read=False,
             new_2026_prices_read=False, no_exit_rules=True)
    score = base.predict(x, r)
    np.testing.assert_allclose(score, m.predict(x), rtol=0, atol=2e-12)
    # Retain the common score-verifier interface. None of these quantiles selects stocks.
    r['thresholds'] = [dict(id=i, training_quantile=q, threshold=float(np.quantile(score, q)))
                       for i, q in enumerate(base.QUANTILES)]
    r['thresholds_are_diagnostics_not_selection'] = True
    save_json(path / 'model_report.json', r)
    return {k: v for k, v in r.items() if k != 'trees'}


def weighted_quartile(values, weights):
    # Independently implement the inverse weighted CDF, rather than calling sklearn.
    order = np.argsort(values, kind='stable')
    cumulative = np.cumsum(weights[order])
    index = np.searchsorted(cumulative, .25 * cumulative[-1], side='left')
    return float(values[order[min(index, len(order) - 1)]])


def verify_model(fold):
    p = config(fold); path = root(fold); r = json.loads((path / 'model_report.json').read_text())
    assert r['protocol_sha256'] == sha(protocol(fold)) and r['variant'] == 'absolute_lower_quartile_space'
    for key in ['feature_report_sha256', 'label_report_sha256', 'training_start', 'training_end']:
        assert r[key] == p[key]
    assert r['feature_names'] == list(original.EXPRESSIONS) and r['selection_score_cut'] == 0 and r['alpha'] == .25
    for key, value in p['parameters'].items():
        assert r['parameters'][key] == value
    # Source checks and a SQL join independently rebuild the exact training intersection.
    base.feature_inputs(); base.labels(f"date>='{p['training_start']}' AND next_date<'{p['training_end']}'")
    c = base.conn(); names = list(original.EXPRESSIONS)
    expressions = ','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::INT AS {n}' for n in names)
    d = c.sql(f"SELECT date,code,next_date,sustained_return15 AS target,1./count(*) OVER(PARTITION BY date) AS w,{expressions} "
              f"FROM read_parquet('{original.ROOT}/features.parquet') f JOIN read_parquet('{base.SOURCE}/full_labels.parquet') l "
              f"USING(date,code) WHERE formula_input_valid AND known15 AND date>='{p['training_start']}' "
              f"AND next_date<'{p['training_end']}' ORDER BY date,code").df()
    c.close(); actual = training(fold)
    pd.testing.assert_frame_equal(d[['date', 'code', 'next_date']], actual[['date', 'code', 'next_date']], check_exact=True)
    np.testing.assert_array_equal(d.target, actual.sustained_return15)
    np.testing.assert_array_equal(d[names].to_numpy(), base.encode(actual))
    assert len(d) == r['rows'] and d.date.nunique() == r['days'] and d.next_date.max() == r['last_observation']
    assert r['last_observation'] < p['evaluation_start']
    x = d[names].to_numpy(); y = d.target.to_numpy(); w = d.w.to_numpy(); dates = d.date.to_numpy()
    np.testing.assert_allclose(r['bias'], weighted_quartile(y, w), rtol=0, atol=2e-12)
    score = np.full(len(d), r['bias']); checks = 0; minimum_days = len(d); leaf_checks = 0
    assert len(r['trees']) == 64 and r['learning_rate'] == .05
    for tree in r['trees']:
        residual = y - score
        # Pinball subgradient at equality follows the installed learner's declared convention.
        gradient = np.where(y >= score, .25, -.75)
        masks = {0: np.ones(len(d), dtype=bool)}; levels = {0: 0}; terminal = np.empty(len(d))
        assert len(tree['feature']) <= 15
        for node, left in enumerate(tree['children_left']):
            mask = masks[node]; weights = w[mask]; assert levels[node] <= 3
            assert mask.sum() == tree['n_node_samples'][node]
            mean = np.average(gradient[mask], weights=weights)
            variance = np.average((gradient[mask] - mean) ** 2, weights=weights)
            np.testing.assert_allclose(weights.sum(), tree['weighted_n_node_samples'][node], rtol=0, atol=1e-8)
            np.testing.assert_allclose(variance, tree['impurity'][node], rtol=0, atol=2e-10)
            if left < 0:
                assert mask.sum() >= 300
                value = weighted_quartile(residual[mask], weights)
                minimum_days = min(minimum_days, len(np.unique(dates[mask]))); leaf_checks += 1
                # Use the exported, already checked value to preserve exact subgradient ties next round.
                terminal[mask] = tree['value'][node]
            else:
                value = mean
                right = tree['children_right'][node]
                lower = x[:, tree['feature'][node]] <= tree['threshold'][node]
                masks[left], masks[right] = mask & lower, mask & ~lower
                levels[left] = levels[right] = levels[node] + 1
            np.testing.assert_allclose(value, tree['value'][node], rtol=0, atol=2e-12)
            checks += 1
        score += .05 * terminal
    np.testing.assert_allclose(score, base.predict(x, r), rtol=0, atol=2e-12)
    for q, item in zip(base.QUANTILES, r['thresholds']):
        assert q == item['training_quantile']
        np.testing.assert_allclose(np.quantile(score, q), item['threshold'], rtol=0, atol=2e-12)
    v = dict(passed=True, model_report_sha256=sha(path / 'model_report.json'), rows=len(d), days=d.date.nunique(),
             node_checks=checks, weighted_quantile_leaf_checks=leaf_checks, minimum_leaf_days_observed=minimum_days,
             all_training_targets_encodings_weights_pinball_gradients_and_weighted_quantiles_rebuilt=True,
             score_quantiles_are_not_selection_thresholds=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(path / 'model_verification.json', v); return v


def freeze(fold):
    path = root(fold); p = config(fold)
    if (path / 'selection_report.json').exists():
        raise ValueError('Do not replace the frozen lower-quartile selection')
    assert all(not (root(f) / 'analysis_report.json').exists() for f in ['2024', 'recent', 'combined'])
    for kind in ['model', 'score']:
        v = json.loads((path / (kind + '_verification.json')).read_text())
        assert v['passed'] and v[kind + '_report_sha256'] == sha(path / (kind + '_report.json'))
    s = json.loads((path / 'score_report.json').read_text())
    assert s['scores_sha256'] == sha(path / 'scores.parquet')
    m = json.loads((path / 'model_report.json').read_text()); assert m['last_observation'] < p['evaluation_start']
    f = pd.read_parquet(path / 'scores.parquet'); out = f[['date', 'code', 'half', 'board', 'decision_shares']].copy()
    out['selected'] = f.formula_input_valid & f.score.gt(0) & f.date.ge(p['evaluation_start']) & f.date.lt(p['evaluation_end'])
    out.to_parquet(path / 'selection.parquet', index=False, compression='zstd')
    (path / 'frozen_numeric_core.tdx').write_text(base.native_core(m, 0., original.EXPRESSIONS, original.HEADER))
    counts = out.loc[out.selected].groupby('date').size()
    r = dict(protocol_sha256=sha(protocol(fold)), model_report_sha256=sha(path / 'model_report.json'),
             score_report_sha256=sha(path / 'score_report.json'), selection_sha256=sha(path / 'selection.parquet'),
             core_sha256=sha(path / 'frozen_numeric_core.tdx'), selected=int(out.selected.sum()), days=len(counts),
             max_daily=int(counts.max()) if len(counts) else 0,
             chosen_threshold=dict(kind='fixed_positive_lower_quartile', threshold=0., training_quantile=None),
             evaluation_start=p['evaluation_start'], evaluation_end=p['evaluation_end'],
             new_group_outcomes_read=False, year_2025_is_exploratory=True, new_2026_prices_read=False,
             no_exit_rules=True, software_compilation_verified=False)
    save_json(path / 'selection_report.json', r); return r


def verify(fold):
    p = config(fold); path = root(fold); r = json.loads((path / 'selection_report.json').read_text())
    for key, file in [('protocol_sha256', protocol(fold)), ('model_report_sha256', path / 'model_report.json'),
                      ('score_report_sha256', path / 'score_report.json'), ('selection_sha256', path / 'selection.parquet'),
                      ('core_sha256', path / 'frozen_numeric_core.tdx')]:
        assert r[key] == sha(file)
    assert r['chosen_threshold'] == dict(kind='fixed_positive_lower_quartile', threshold=0., training_quantile=None)
    for kind in ['model', 'score']:
        v = json.loads((path / (kind + '_verification.json')).read_text())
        assert v['passed'] and v[kind + '_report_sha256'] == r[kind + '_report_sha256']
    m = json.loads((path / 'model_report.json').read_text())
    assert r['evaluation_start'] == p['evaluation_start'] and r['evaluation_end'] == p['evaluation_end']
    assert m['last_observation'] < p['evaluation_start']
    c = base.conn()
    expected = c.sql(f"SELECT date,code,half,board,decision_shares,formula_input_valid AND score>0 "
                     f"AND date>='{p['evaluation_start']}' AND date<'{p['evaluation_end']}' AS selected "
                     f"FROM read_parquet('{path}/scores.parquet') ORDER BY date,code").df(); c.close()
    pd.testing.assert_frame_equal(expected, pd.read_parquet(path / 'selection.parquet'), check_exact=True)
    assert int(expected.selected.sum()) == r['selected'] and expected.loc[expected.selected, 'date'].nunique() == r['days']
    assert (path / 'frozen_numeric_core.tdx').read_text() == base.native_core(m, 0., original.EXPRESSIONS, original.HEADER)
    v = dict(passed=True, selection_report_sha256=sha(path / 'selection_report.json'), rows=len(expected),
             all_fixed_zero_score_flags_rebuilt=True, no_training_period_selection=True,
             new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(path / 'selection_verification.json', v); return v


def combined(stage):
    path = root('combined'); p = json.loads(protocol('combined').read_text())
    assert p['fold_protocols'] == [str(protocol(f)) for f in ['2024', 'recent']]
    frames = []; folds = []
    for f in ['2024', 'recent']:
        setup(f); verify(f)
        s = json.loads((root(f) / 'selection_report.json').read_text())
        assert s['chosen_threshold']['threshold'] == p['selection_score_cut'] == 0 and p['alpha'] == .25
        frames.append(pd.read_parquet(root(f) / 'selection.parquet'))
        folds.append(dict(fold=f, protocol_sha256=sha(protocol(f)), selection_report_sha256=sha(root(f) / 'selection_report.json'),
                          model_report_sha256=sha(root(f) / 'model_report.json'), core_sha256=s['core_sha256'], selected=s['selected']))
    pd.testing.assert_frame_equal(frames[0].drop(columns='selected'), frames[1].drop(columns='selected'), check_exact=True)
    assert not (frames[0].selected & frames[1].selected).any()
    if stage == 'freeze':
        assert not (path / 'selection_report.json').exists()
        assert all(not (root(f) / 'analysis_report.json').exists() for f in ['2024', 'recent', 'combined'])
        path.mkdir(parents=True, exist_ok=True)
        out = frames[0].copy(); out['selected'] |= frames[1].selected
        out.to_parquet(path / 'selection.parquet', index=False, compression='zstd')
        for f in ['2024', 'recent']:
            (path / ('frozen_numeric_core_' + f + '.tdx')).write_bytes((root(f) / 'frozen_numeric_core.tdx').read_bytes())
        r = dict(protocol_sha256=sha(protocol('combined')), folds=folds, selection_sha256=sha(path / 'selection.parquet'),
                 selected=int(out.selected.sum()), days=out.loc[out.selected, 'date'].nunique(),
                 new_group_outcomes_read=False, year_2025_is_exploratory=True, new_2026_prices_read=False, no_exit_rules=True)
        save_json(path / 'selection_report.json', r); return r
    r = json.loads((path / 'selection_report.json').read_text())
    assert r['protocol_sha256'] == sha(protocol('combined')) and r['folds'] == folds
    assert r['selection_sha256'] == sha(path / 'selection.parquet')
    c = base.conn()
    queries = [f"SELECT * FROM read_parquet('{root(f)}/selection.parquet')" for f in ['2024', 'recent']]
    expected = c.sql('SELECT date,code,half,board,decision_shares,bool_or(selected) AS selected FROM (' +
                     ' UNION ALL '.join(queries) + ') GROUP BY date,code,half,board,decision_shares ORDER BY date,code').df(); c.close()
    pd.testing.assert_frame_equal(expected, pd.read_parquet(path / 'selection.parquet'), check_exact=True)
    assert int(expected.selected.sum()) == r['selected'] and expected.loc[expected.selected, 'date'].nunique() == r['days']
    for f in ['2024', 'recent']:
        assert sha(path / ('frozen_numeric_core_' + f + '.tdx')) == sha(root(f) / 'frozen_numeric_core.tdx')
    v = dict(passed=True, selection_report_sha256=sha(path / 'selection_report.json'), rows=len(expected),
             all_time_folds_and_selection_flags_rebuilt=True, new_group_outcomes_read=False,
             new_2026_prices_read=False, no_exit_rules=True)
    save_json(path / 'selection_verification.json', v); return v


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('fold', choices=['2024', 'recent', 'combined'])
    p.add_argument('stage', choices=['model', 'verify_model', 'scores', 'freeze', 'verify', 'analyze'])
    a = p.parse_args()
    if a.stage == 'analyze':
        assert (root('combined') / 'selection_verification.json').exists()
        if json.loads((root(a.fold) / 'selection_report.json').read_text())['selected']:
            result = common_analysis(root(a.fold), protocol(a.fold))
        else:
            result = dict(no_candidates=True, no_performance_estimated=True)
    elif a.fold == 'combined':
        assert a.stage in ['freeze', 'verify']; result = combined(a.stage)
    else:
        setup(a.fold)
        result = base.scores() if a.stage == 'scores' else globals()[a.stage](a.fold)
    print(json.dumps(result, ensure_ascii=False, indent=2))
