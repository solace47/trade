"""A date-level index forecast gates, but never reorders, the frozen 48-input pool."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from . import tail_formula_additive as base
from . import tail_formula_float as original
from .corporate_cash import save_json, sha
from .tail_formula_1000_daily import analyze as common_analysis

PROTOCOL = Path('config/tail_formula_market_gate_protocol.json')
FEATURES = Path('data/research/tail_formula_long48')
NAMES = ['J02', 'J03', 'J04']


def root(fold):
    return Path('data/research/tail_formula_market_gate_' + fold)


def config():
    p = json.loads(PROTOCOL.read_text())
    assert p['feature_root'] == str(FEATURES) and p['features'] == NAMES
    for kind, report, proof, key, data in [
        ('feature', 'feature_report.json', 'feature_verification.json', 'features_sha256', 'features.parquet'),
        ('label', 'full_label_report.json', 'full_label_verification.json', 'labels_sha256', 'full_labels.parquet'),
    ]:
        r = json.loads((FEATURES / report).read_text())
        v = json.loads((FEATURES / proof).read_text())
        assert p[kind + '_report_sha256'] == sha(FEATURES / report)
        assert v['passed'] and v[kind + '_report_sha256'] == sha(FEATURES / report)
        assert r[key] == sha(FEATURES / data)
    return p


def source(fold):
    p = config(); cfg = p['folds'][fold]; prior = Path(cfg['source_root'])
    for name, digest in cfg['source_sha256'].items():
        assert sha(prior / name) == digest
    for kind in ['model', 'score', 'selection']:
        v = json.loads((prior / (kind + '_verification.json')).read_text())
        assert v['passed'] and v[kind + '_report_sha256'] == sha(prior / (kind + '_report.json'))
    r = json.loads((prior / 'selection_report.json').read_text())
    assert r['selection_sha256'] == sha(prior / 'selection.parquet')
    assert r['chosen_threshold']['training_quantile'] == .995
    m = json.loads((prior / 'model_report.json').read_text())
    assert m['last_observation'] < cfg['evaluation_start'] and m['feature_names'] == list(original.EXPRESSIONS)
    return p, cfg, prior


def market_inputs():
    f = pd.read_parquet(FEATURES / 'features.parquet', columns=['date', 'code', 'formula_input_valid'] + NAMES)
    f = f.loc[f.formula_input_valid].copy()
    f['exchange'] = f.code.str[:2]
    assert np.isfinite(f[NAMES]).all().all() and f.date.lt('2026-01-01').all()
    g = f.groupby(['date', 'exchange'], sort=True)
    assert g[NAMES].nunique().eq(1).all().all()
    out = g[NAMES].first().reset_index()
    assert out.groupby('date').exchange.nunique().eq(2).all()
    assert set(out.exchange) == {'sh', 'sz'}
    return out


def training(fold):
    p, cfg, _ = source(fold)
    # Aggregate the entire known base before intersecting with index inputs.
    l = pd.read_parquet(FEATURES / 'full_labels.parquet',
                       columns=['date', 'next_date', 'known15', 'opportunity15'],
                       filters=[('date', '>=', cfg['training_start']), ('next_date', '<', cfg['training_end'])])
    l = l.loc[l.known15]
    assert l.opportunity15.isin([0, 1]).all()
    d = l.groupby('date').agg(target=('opportunity15', 'mean'), known=('known15', 'size'),
                              next_date=('next_date', 'max')).reset_index()
    out = market_inputs().merge(d, on='date', validate='many_to_one').sort_values(['date', 'exchange']).reset_index(drop=True)
    assert out.groupby('date').size().eq(2).all() and out.date.nunique() == len(d)
    assert out.next_date.lt(cfg['training_end']).all() and out.date.ge(cfg['training_start']).all()
    out['weight'] = .5
    return p, cfg, out


def encoded(frame):
    return np.floor(np.clip(frame[NAMES].to_numpy() * 100 + 10000 + .000001, 0, 999999)).astype('int32')


def model(fold):
    path = root(fold)
    if (path / 'model_report.json').exists():
        raise ValueError('Do not refit the frozen market gate')
    path.mkdir(parents=True, exist_ok=True)
    p, cfg, t = training(fold)
    x = encoded(t); y = t.target.to_numpy(); w = t.weight.to_numpy()
    m = GradientBoostingRegressor(**p['parameters']).fit(x, y, sample_weight=w)
    trees = []
    for estimator in m.estimators_.ravel():
        tree = estimator.tree_
        trees.append({k: getattr(tree, k).tolist() for k in
                      ['feature', 'threshold', 'children_left', 'children_right', 'n_node_samples',
                       'weighted_n_node_samples', 'impurity']})
        trees[-1]['value'] = tree.value.reshape(-1).tolist()
    t.to_parquet(path / 'market_training.parquet', index=False, compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL), feature_report_sha256=p['feature_report_sha256'],
             label_report_sha256=p['label_report_sha256'], training_sha256=sha(path / 'market_training.parquet'),
             training_start=cfg['training_start'], training_end=cfg['training_end'],
             last_observation=t.next_date.max(), rows=len(t), days=t.date.nunique(), feature_names=NAMES,
             parameters=m.get_params(), learning_rate=.05, bias=float(m.init_.constant_.ravel()[0]),
             trees=trees, gate_threshold=float(m.init_.constant_.ravel()[0]),
             gate_threshold_is_training_date_mean=True, stock_model_refitted=False,
             new_2025_score_groups_read=bool(t.date.ge('2025-01-01').any()),
             new_2025H2_score_groups_read=False, new_2026_prices_read=False, no_exit_rules=True)
    np.testing.assert_allclose(base.predict(x, r), m.predict(x), rtol=0, atol=2e-12)
    save_json(path / 'model_report.json', r)
    return {k: v for k, v in r.items() if k != 'trees'}


def sql_inputs(c):
    c.execute(f"CREATE VIEW raw_inputs AS SELECT date,substr(code,1,2) AS exchange,{','.join(NAMES)} "
              f"FROM read_parquet('{FEATURES}/features.parquet') WHERE formula_input_valid")
    for name in NAMES:
        assert c.sql(f'SELECT count(*) FROM (SELECT date,exchange FROM raw_inputs GROUP BY ALL '
                     f'HAVING count(DISTINCT {name})<>1)').fetchone()[0] == 0
    c.execute('CREATE VIEW market_inputs AS SELECT date,exchange,' +
              ','.join(f'max({n}) AS {n}' for n in NAMES) + ' FROM raw_inputs GROUP BY date,exchange')
    assert c.sql('SELECT count(*) FROM (SELECT date FROM market_inputs GROUP BY date HAVING count(*)<>2)').fetchone()[0] == 0


def verify_model(fold):
    p, cfg, _ = source(fold); path = root(fold)
    r = json.loads((path / 'model_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['feature_names'] == NAMES
    for name in ['feature_report_sha256', 'label_report_sha256']:
        assert r[name] == p[name]
    for name, value in p['parameters'].items():
        assert r['parameters'][name] == value
    assert r['training_sha256'] == sha(path / 'market_training.parquet')
    assert r['training_start'] == cfg['training_start'] and r['training_end'] == cfg['training_end']
    c = base.conn(); sql_inputs(c)
    d = c.sql(f"WITH targets AS (SELECT date,avg(opportunity15) AS target,count(*) AS known,max(next_date) AS next_date "
              f"FROM read_parquet('{FEATURES}/full_labels.parquet') WHERE known15 "
              f"AND date>='{cfg['training_start']}' AND next_date<'{cfg['training_end']}' GROUP BY date) "
              f"SELECT m.*,t.target,t.known,t.next_date,0.5::DOUBLE AS weight FROM market_inputs m "
              'JOIN targets t USING(date) ORDER BY date,exchange').df()
    pd.testing.assert_frame_equal(d, pd.read_parquet(path / 'market_training.parquet'), check_dtype=False, rtol=0, atol=2e-12)
    assert len(d) == r['rows'] and d.date.nunique() == r['days'] and d.next_date.max() == r['last_observation']
    c.register('training', d)
    expressions = ','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::INT AS {n}' for n in NAMES)
    x = c.sql('SELECT ' + expressions + ' FROM training ORDER BY date,exchange').df().to_numpy()
    c.close()
    np.testing.assert_array_equal(x, encoded(d))
    y = d.target.to_numpy(); w = d.weight.to_numpy(); dates = d.date.to_numpy()
    np.testing.assert_allclose(r['bias'], d.groupby('date').target.first().mean(), rtol=0, atol=2e-12)
    assert r['gate_threshold'] == r['bias'] and len(r['trees']) == 32 and r['learning_rate'] == .05
    score = np.full(len(d), r['bias']); checks = 0; minimum_days = len(d)
    for tree in r['trees']:
        assert len(tree['feature']) <= 7
        residual = y - score; masks = {0: np.ones(len(d), dtype=bool)}; levels = {0: 0}
        terminal = np.empty(len(d))
        for node, left in enumerate(tree['children_left']):
            mask = masks[node]; weights = w[mask]; assert levels[node] <= 2
            assert int(mask.sum()) == tree['n_node_samples'][node]
            mean = np.average(residual[mask], weights=weights)
            variance = np.average((residual[mask] - mean) ** 2, weights=weights)
            np.testing.assert_allclose(weights.sum(), tree['weighted_n_node_samples'][node], rtol=0, atol=1e-10)
            np.testing.assert_allclose(mean, tree['value'][node], rtol=0, atol=2e-12)
            np.testing.assert_allclose(variance, tree['impurity'][node], rtol=0, atol=2e-12)
            if left < 0:
                days = len(np.unique(dates[mask])); minimum_days = min(minimum_days, days)
                assert mask.sum() >= 40 and days >= p['minimum_leaf_distinct_days']
                terminal[mask] = mean
            else:
                right = tree['children_right'][node]
                lower = x[:, tree['feature'][node]] <= tree['threshold'][node]
                masks[left], masks[right] = mask & lower, mask & ~lower
                levels[left] = levels[right] = levels[node] + 1
            checks += 1
        score += .05 * terminal
    np.testing.assert_allclose(score, base.predict(x, r), rtol=0, atol=2e-12)
    proof = dict(passed=True, model_report_sha256=sha(path / 'model_report.json'), rows=len(d),
                 days=d.date.nunique(), node_checks=checks, minimum_leaf_days=minimum_days,
                 all_daily_base_targets_weights_integer_inputs_and_residuals_independently_rebuilt=True,
                 new_2026_prices_read=False, no_exit_rules=True)
    save_json(path / 'model_verification.json', proof); return proof


def scores(fold):
    config(); path = root(fold)
    if (path / 'score_report.json').exists():
        raise ValueError('Do not replace frozen market scores')
    v = json.loads((path / 'model_verification.json').read_text())
    assert v['passed'] and v['model_report_sha256'] == sha(path / 'model_report.json')
    m = json.loads((path / 'model_report.json').read_text())
    out = market_inputs(); out['market_score'] = base.predict(encoded(out), m)
    out['gate'] = out.market_score.gt(m['gate_threshold'])
    out.to_parquet(path / 'market_scores.parquet', index=False, compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL), model_report_sha256=sha(path / 'model_report.json'),
             feature_report_sha256=sha(FEATURES / 'feature_report.json'), scores_sha256=sha(path / 'market_scores.parquet'),
             rows=len(out), new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(path / 'score_report.json', r); return r


def score_sql(tree, node=0):
    if tree['children_left'][node] < 0:
        return repr(.05 * tree['value'][node])
    n = NAMES[tree['feature'][node]]
    value = f'floor(least(greatest(100*{n}+10000+.000001,0),999999))'
    return (f"CASE WHEN {value}<={int(np.floor(tree['threshold'][node]))} THEN " +
            score_sql(tree, tree['children_left'][node]) + ' ELSE ' +
            score_sql(tree, tree['children_right'][node]) + ' END')


def verify_scores(fold):
    config(); path = root(fold)
    r = json.loads((path / 'score_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['model_report_sha256'] == sha(path / 'model_report.json')
    assert r['feature_report_sha256'] == sha(FEATURES / 'feature_report.json')
    assert r['scores_sha256'] == sha(path / 'market_scores.parquet')
    m = json.loads((path / 'model_report.json').read_text())
    c = base.conn(); sql_inputs(c)
    expression = repr(m['bias']) + '+' + '+'.join('(' + score_sql(t) + ')' for t in m['trees'])
    expected = c.sql(f'SELECT *,market_score>{m["gate_threshold"]!r} AS gate FROM '
                     f'(SELECT *,{expression} AS market_score FROM market_inputs) ORDER BY date,exchange').df()
    c.close(); actual = pd.read_parquet(path / 'market_scores.parquet')
    pd.testing.assert_frame_equal(actual, expected, check_exact=False, atol=2e-12, rtol=0)
    np.testing.assert_array_equal(actual.gate, expected.gate)
    assert len(actual) == r['rows'] and actual.date.lt('2026-01-01').all()
    v = dict(passed=True, score_report_sha256=sha(path / 'score_report.json'), rows=len(actual),
             all_index_groups_tree_scores_and_gate_flags_independently_rebuilt=True,
             new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(path / 'score_verification.json', v); return v


def numeric_core(prior, model):
    lines = (prior / 'frozen_numeric_core.tdx').read_text().splitlines()
    assert lines[-1].startswith('CORE:SC>')
    stock_condition = lines.pop()[5:-1]
    indices = [list(original.EXPRESSIONS).index(n) for n in NAMES]
    for i, tree in enumerate(model['trees'], 1):
        tree = deepcopy(tree)
        tree['feature'] = [indices[n] if n >= 0 else n for n in tree['feature']]
        lines.append(f'MK{i:02d}:=' + base.native_tree(tree) + ';')
    lines.append('MG:=' + repr(model['bias']) + '+' + '+'.join(f'MK{i:02d}' for i in range(1, 33)) + ';')
    lines.append(f'CORE:({stock_condition}) AND MG>{model["gate_threshold"]!r};')
    return '\n'.join(lines) + '\n'


def freeze(fold):
    path = root(fold)
    if (path / 'selection_report.json').exists():
        raise ValueError('Do not replace the frozen market-gated selection')
    assert not any((root(f) / 'analysis_report.json').exists() for f in ['2024', 'recent', '2025'])
    path.mkdir(parents=True, exist_ok=True)
    if fold == '2025':
        a = pd.read_parquet(root('2024') / 'selection.parquet')
        b = pd.read_parquet(root('recent') / 'selection.parquet')
        pd.testing.assert_frame_equal(a.drop(columns='selected'), b.drop(columns='selected'), check_exact=True)
        assert not (a.selected & b.selected).any()
        out = a.copy(); out['selected'] |= b.selected
        details = dict(fold_selection_report_sha256={f: sha(root(f) / 'selection_report.json') for f in ['2024', 'recent']})
        old = pd.read_parquet(Path('data/research/tail_formula_float_2025/selection.parquet'))
    else:
        _, cfg, prior = source(fold)
        v = json.loads((path / 'score_verification.json').read_text())
        assert v['passed'] and v['score_report_sha256'] == sha(path / 'score_report.json')
        old = pd.read_parquet(prior / 'selection.parquet')
        out = old.assign(exchange=old.code.str[:2]).merge(
            pd.read_parquet(path / 'market_scores.parquet')[['date', 'exchange', 'gate']],
            on=['date', 'exchange'], how='left', validate='many_to_one')
        assert out.gate.notna().all()
        out['selected'] &= out.gate
        out = out[list(old.columns)]
        model = json.loads((path / 'model_report.json').read_text())
        (path / 'frozen_numeric_core.tdx').write_text(numeric_core(prior, model))
        details = dict(source_selection_report_sha256=sha(prior / 'selection_report.json'),
                       score_report_sha256=sha(path / 'score_report.json'), model_report_sha256=sha(path / 'model_report.json'),
                       core_sha256=sha(path / 'frozen_numeric_core.tdx'))
        assert out.loc[out.selected, 'date'].ge(cfg['evaluation_start']).all()
        assert out.loc[out.selected, 'date'].lt(cfg['evaluation_end']).all()
    out.to_parquet(path / 'selection.parquet', index=False, compression='zstd')
    counts = out.loc[out.selected].groupby('date').size()
    old_days = set(old.loc[old.selected, 'date'])
    r = dict(protocol_sha256=sha(PROTOCOL), selection_sha256=sha(path / 'selection.parquet'), **details,
             selected=int(counts.sum()), days=len(counts), max_daily=int(counts.max()) if len(counts) else 0,
             original_selected=int(old.selected.sum()), original_days=len(old_days),
             fully_filtered_dates=sorted(old_days - set(counts.index)), stock_model_refitted=False,
             original_results_previously_seen=True, new_group_outcomes_read=False, year_2025_is_exploratory=True,
             new_2026_prices_read=False, no_exit_rules=True, software_compilation_verified=False)
    save_json(path / 'selection_report.json', r); return r


def verify(fold):
    config(); path = root(fold); r = json.loads((path / 'selection_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['selection_sha256'] == sha(path / 'selection.parquet')
    c = base.conn()
    if fold == '2025':
        queries = []
        for f in ['2024', 'recent']:
            v = json.loads((root(f) / 'selection_verification.json').read_text())
            assert v['passed'] and v['selection_report_sha256'] == r['fold_selection_report_sha256'][f] == sha(root(f) / 'selection_report.json')
            queries.append(f"SELECT * FROM read_parquet('{root(f)}/selection.parquet')")
        expected = c.sql('SELECT date,code,half,board,decision_shares,bool_or(selected) AS selected FROM (' +
                         ' UNION ALL '.join(queries) + ') GROUP BY date,code,half,board,decision_shares ORDER BY date,code').df()
        prior = Path('data/research/tail_formula_float_2025')
        v = json.loads((prior / 'selection_verification.json').read_text())
        assert v['passed'] and v['selection_report_sha256'] == sha(prior / 'selection_report.json')
        assert json.loads((prior / 'selection_report.json').read_text())['selection_sha256'] == sha(prior / 'selection.parquet')
    else:
        _, _, prior = source(fold)
        assert r['source_selection_report_sha256'] == sha(prior / 'selection_report.json')
        for kind in ['model', 'score']:
            v = json.loads((path / (kind + '_verification.json')).read_text())
            assert v['passed'] and v[kind + '_report_sha256'] == r[kind + '_report_sha256'] == sha(path / (kind + '_report.json'))
        m = json.loads((path / 'model_report.json').read_text())
        expected = c.sql(f"SELECT s.* EXCLUDE(selected),s.selected AND p.market_score>{m['gate_threshold']!r} AS selected "
                         f"FROM read_parquet('{prior}/selection.parquet') s LEFT JOIN read_parquet('{path}/market_scores.parquet') p "
                         'ON s.date=p.date AND substr(s.code,1,2)=p.exchange ORDER BY s.date,s.code').df()
        assert expected.selected.notna().all()
        assert r['core_sha256'] == sha(path / 'frozen_numeric_core.tdx')
        assert (path / 'frozen_numeric_core.tdx').read_text() == numeric_core(prior, m)
    c.close()
    actual = pd.read_parquet(path / 'selection.parquet'); old = pd.read_parquet(prior / 'selection.parquet')
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=True)
    pd.testing.assert_frame_equal(actual.drop(columns='selected'), old.drop(columns='selected'), check_exact=True)
    assert not (actual.selected & ~old.selected).any()
    counts = actual.loc[actual.selected].groupby('date').size(); old_days = set(old.loc[old.selected, 'date'])
    assert r['selected'] == int(counts.sum()) and r['days'] == len(counts)
    assert r['max_daily'] == (int(counts.max()) if len(counts) else 0)
    assert r['original_days'] == len(old_days) and r['original_selected'] == int(old.selected.sum())
    assert r['fully_filtered_dates'] == sorted(old_days - set(counts.index))
    v = dict(passed=True, selection_report_sha256=sha(path / 'selection_report.json'), rows=len(actual),
             all_gate_flags_and_removed_dates_independently_rebuilt=True, all_original_keys_retained=True,
             stock_model_refitted=False, new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(path / 'selection_verification.json', v); return v


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('fold', choices=['2024', 'recent', '2025'])
    p.add_argument('stage', choices=['model', 'verify_model', 'scores', 'verify_scores', 'freeze', 'verify', 'analyze'])
    a = p.parse_args()
    if a.stage == 'analyze':
        assert (root('2025') / 'selection_verification.json').exists()
        if not json.loads((root(a.fold) / 'selection_report.json').read_text())['selected']:
            result = dict(no_candidates=True, no_performance_estimated=True)
        else:
            result = common_analysis(root(a.fold), PROTOCOL)
    else:
        assert a.fold != '2025' or a.stage in ['freeze', 'verify']
        result = globals()[a.stage](a.fold)
    print(json.dumps(result, ensure_ascii=False, indent=2))
