"""A fixed three-index, date-level absolute-reference gate on existing stock cores."""
import argparse
import ast
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from trade_research import tail_formula_additive as base
from trade_research import tail_formula_boundary_evaluation as evaluation
from trade_research.corporate_cash import save_json, sha
from trade_research.reference_gain_accounting import weekly_interval
from freeze_tail_formula_bipower_gap import checked_selection
from audit_tail_formula_reference_coverage import audit
from compare_tail_formula_same_dates import compare
from compare_tail_formula_shared_unknowns import compare as shared_compare
import evaluate_tail_formula_pool_scope as reuse_common

ROOT = Path('data/research/tail_formula_market_reference')
PROTOCOL = Path('config/tail_formula_market_reference_protocol.json')
NAMES = ['J02', 'J03', 'J04']
META = ['date', 'code', 'half', 'board', 'decision_shares', 'formula_input_valid']
KEYS = META[:-1]
LABEL_FIELDS = ['date', 'code', 'next_date', 'known15', 'known_no_trade', 'mark_0959_return15']


def checked():
    p = json.loads(PROTOCOL.read_text())
    assert p['feature_names'] == NAMES and p['ridge_alpha'] == 1 and p['gate_threshold'] == 0
    assert p['maximum_new_market_fits'] == 4 and not p['new_stock_fit_allowed']
    assert not p['new_2026_prices_allowed'] and p['no_exit_rules']
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest, file
    for root in p['feature_roots']:
        path = Path(root)
        r = json.loads((path / 'feature_report.json').read_text())
        v = json.loads((path / 'feature_verification.json').read_text())
        assert v['passed'] and v['feature_report_sha256'] == sha(path / 'feature_report.json')
        assert r['features_sha256'] == sha(path / 'features.parquet')
    for root in p['label_roots']:
        path = Path(root)
        r = json.loads((path / 'full_label_report.json').read_text())
        v = json.loads((path / 'full_label_verification.json').read_text())
        assert v['passed'] and v['label_report_sha256'] == sha(path / 'full_label_report.json')
        assert r['labels_sha256'] == sha(path / 'full_labels.parquet')
    return p


def visible_features(p):
    history = pd.read_parquet(Path(p['feature_roots'][0]) / 'features.parquet',
        columns=META + NAMES, filters=[('date', '<', '2024-01-01')])
    current = pd.read_parquet(Path(p['feature_roots'][1]) / 'features.parquet', columns=META + NAMES)
    assert len(history) == 557044 and len(current) == 1258085
    out = pd.concat([history, current], ignore_index=True)
    assert not out.duplicated(['date', 'code']).any() and out.date.lt('2026-01-01').all()
    assert out.code.str[:2].isin(['sh', 'sz']).all() and out.board.eq('main').all()
    out['exchange'] = out.code.str[:2]
    return out


def visible_index_groups(frame):
    """Only visible validity determines inputs; no label or fill filtering."""
    valid = frame.loc[frame.formula_input_valid].copy()
    assert np.isfinite(valid[NAMES].to_numpy(float)).all()
    grouped = valid.groupby(['date', 'exchange'], sort=True)
    assert grouped[NAMES].nunique(dropna=False).eq(1).all().all()
    out = grouped[NAMES].first().reset_index()
    assert out.groupby('date').exchange.nunique().eq(2).all()
    return out


def encode(frame):
    return np.floor(np.clip(100 * frame[NAMES].to_numpy(float) + 10000 + .000001, 0, 999999))


def target_groups(labels, start, end, observation_end):
    """Compute the full known reference pool before any input intersection."""
    d = labels.loc[labels.date.ge(start) & labels.date.lt(end) & labels.next_date.lt(observation_end)].copy()
    d['exchange'] = d.code.str[:2]
    d['reference'] = d.mark_0959_return15.where(d.known15 & np.isfinite(d.mark_0959_return15))
    d['reference_observation'] = d.next_date.where(d.reference.notna())
    out = d.groupby(['date', 'exchange'], sort=True).agg(pool_rows=('code', 'size'),
        known_rows=('known15', 'sum'), no_trade_rows=('known_no_trade', 'sum'),
        reference_rows=('reference', 'count'), target=('reference', 'mean'),
        last_observation=('reference_observation', 'max')).reset_index()
    out['target'] *= 100
    assert out.reference_rows.gt(0).all() and np.isfinite(out.target).all()
    assert out.groupby('date').exchange.nunique().eq(2).all()
    return out


def read_labels(p, start, end):
    frames = []
    for root in p['label_roots']:
        d = pd.read_parquet(Path(root) / 'full_labels.parquet', columns=LABEL_FIELDS,
            filters=[('date', '>=', start), ('date', '<', end)])
        frames.append(d)
    result = pd.concat(frames, ignore_index=True).sort_values(['date', 'code']).reset_index(drop=True)
    assert not result.duplicated(['date', 'code']).any()
    return result


def sql_inputs(c, p):
    a, b = [str(Path(r) / 'features.parquet') for r in p['feature_roots']]
    columns = ','.join(META + NAMES)
    c.sql(f"SELECT {columns} FROM read_parquet('{a}') WHERE date<'2024-01-01' UNION ALL SELECT {columns} FROM read_parquet('{b}')").create_view('visible')
    counts = c.sql('SELECT date,substr(code,1,2) AS exchange,' + ','.join(f'count(DISTINCT {n}) AS {n}' for n in NAMES) +
        ' FROM visible WHERE formula_input_valid GROUP BY date,exchange').df()
    assert counts[NAMES].eq(1).all().all()
    c.sql('SELECT date,substr(code,1,2) AS exchange,' + ','.join(f'min({n}) AS {n}' for n in NAMES) +
        ' FROM visible WHERE formula_input_valid GROUP BY date,exchange').create_view('market_inputs')


def sql_targets(c, p, spec, observation_end):
    sources = [str(Path(r) / 'full_labels.parquet') for r in p['label_roots']]
    fields = ','.join(LABEL_FIELDS)
    union = ' UNION ALL '.join(f"SELECT {fields} FROM read_parquet('{r}') WHERE date>='{spec['start']}' AND date<'{spec['end']}' AND next_date<'{observation_end}'" for r in sources)
    return c.sql(f'''SELECT date,substr(code,1,2) AS exchange,count(*) AS pool_rows,
        sum(known15::INT) AS known_rows,sum(known_no_trade::INT) AS no_trade_rows,
        count(*) FILTER(WHERE known15 AND isfinite(mark_0959_return15)) AS reference_rows,
        100*avg(mark_0959_return15) FILTER(WHERE known15 AND isfinite(mark_0959_return15)) AS target,
        max(next_date) FILTER(WHERE known15 AND isfinite(mark_0959_return15)) AS last_observation
        FROM ({union}) GROUP BY date,exchange ORDER BY date,exchange''').df()


def project():
    p = checked(); assert not (ROOT / 'input_verification.json').exists()
    committed = subprocess.run(['git', 'show', 'HEAD:docs/selection-formula.md'], text=True, capture_output=True, check=True).stdout
    assert sha(PROTOCOL) in committed, 'Commit the actual protocol before reading new training targets'
    ROOT.mkdir(parents=True, exist_ok=True)
    f = visible_features(p); actual = visible_index_groups(f)
    c = base.conn(); sql_inputs(c, p)
    expected = c.sql('SELECT * FROM market_inputs ORDER BY date,exchange').df()
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    c.register('projected_inputs', actual)
    integers = c.sql('SELECT ' + ','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))' for n in NAMES) +
        ' FROM projected_inputs ORDER BY date,exchange').fetchall()
    np.testing.assert_array_equal(encode(actual), np.asarray(integers))
    actual.to_parquet(ROOT / 'market_inputs.parquet', index=False, compression='zstd')
    training = []
    for fold, spec in p['folds'].items():
        start, end = spec['training_start'], spec['training_end']
        labels = read_labels(p, start, end)
        expected_keys = f.loc[f.date.ge(start) & f.date.lt(end), ['date', 'code']].reset_index(drop=True)
        pd.testing.assert_frame_equal(labels[['date', 'code']], expected_keys, check_exact=True)
        targets = target_groups(labels, start, end, spec['evaluation_start'])
        expected_targets = sql_targets(c, p, dict(start=start, end=end), spec['evaluation_start'])
        pd.testing.assert_frame_equal(targets, expected_targets, check_dtype=False, rtol=0, atol=2e-12)
        d = targets.merge(actual, on=['date', 'exchange'], validate='one_to_one').sort_values(['date', 'exchange']).reset_index(drop=True)
        assert len(d) == len(targets) and d.groupby('date').exchange.nunique().eq(2).all()
        assert d.last_observation.max() < spec['evaluation_start']
        d['weight'] = .5 / d.date.nunique()
        path = ROOT / 'models' / fold; path.mkdir(parents=True, exist_ok=True)
        d.to_parquet(path / 'training.parquet', index=False, compression='zstd')
        training.append(dict(fold=fold, rows=len(d), days=d.date.nunique(), last_observation=d.last_observation.max(),
            training_sha256=sha(path / 'training.parquet')))
    c.close()
    lookup = []
    for root in p['prior_model_roots']:
        path = Path(root) / 'model_report.json'
        m = json.loads(path.read_text())
        lookup.append(dict(root=root, sha256=sha(path), feature_names=m.get('feature_names'),
            model_family=m.get('model_family'), variant=m.get('variant'), parameters=m.get('parameters'),
            exact_three_index_reference_ridge_match=m.get('feature_names') == NAMES and
                m.get('ridge_alpha') == 1 and 'coefficients' in m))
    assert not any(r['exact_three_index_reference_ridge_match'] for r in lookup)
    save_json(ROOT / 'input_verification.json', dict(passed=True, protocol_sha256=sha(PROTOCOL),
        visible_keys=len(f), index_rows=len(actual), market_inputs_sha256=sha(ROOT / 'market_inputs.parquet'),
        training=training, prior_model_lookup=lookup, all_market_groups_values_encodings_targets_and_keys_independently_verified=True,
        targets_full_known_reference_pool_before_valid_input_intersection=True,
        no_new_holdout_group_economics_read=True, new_2026_prices_read=False, no_exit_rules=True))
    return dict(input_verification_sha256=sha(ROOT / 'input_verification.json'), index_rows=len(actual), training=training)


def predict(x, model):
    out = np.full(len(x), model['bias'], dtype=float)
    for i, (coefficient, mean, scale) in enumerate(zip(model['coefficients'], model['means'], model['scales'])):
        out += coefficient * (x[:, i] - mean) / scale
    return out


def fit(fold):
    p = checked(); path = ROOT / 'models' / fold
    assert not (path / 'model_report.json').exists()
    receipt = json.loads((ROOT / 'input_verification.json').read_text())
    assert receipt['passed'] and receipt['protocol_sha256'] == sha(PROTOCOL)
    r = next(r for r in receipt['training'] if r['fold'] == fold)
    assert r['training_sha256'] == sha(path / 'training.parquet')
    d = pd.read_parquet(path / 'training.parquet'); x = encode(d); y = d.target.to_numpy(float); w = d.weight.to_numpy(float)
    np.testing.assert_allclose(w, np.full(len(d), .5 / d.date.nunique()), rtol=0, atol=1e-15)
    means = np.average(x, axis=0, weights=w); centered = x - means
    scales = np.sqrt(np.average(centered * centered, axis=0, weights=w)); constant = scales <= 1e-12; scales[constant] = 1
    z = centered / scales; bias = float(np.average(y, weights=w))
    coefficients = np.linalg.solve(z.T @ (z * w[:, None]) + np.eye(3), z.T @ (w * (y - bias)))
    m = dict(protocol_sha256=sha(PROTOCOL), input_verification_sha256=sha(ROOT / 'input_verification.json'),
        training_sha256=sha(path / 'training.parquet'), fold=fold, **p['folds'][fold], model_family=p['model_family'],
        feature_names=NAMES, ridge_alpha=1, bias=bias, means=means.tolist(), scales=scales.tolist(),
        coefficients=coefficients.tolist(), constant_columns=[n for n, yes in zip(NAMES, constant) if yes],
        rows=len(d), days=d.date.nunique(), last_observation=d.last_observation.max(), gate_threshold=0,
        unit='percentage_points_of_net_0959_reference', no_stock_model_refit=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(path / 'model_report.json', m)
    c = base.conn(); c.register('training', d.assign(X0=x[:, 0], X1=x[:, 1], X2=x[:, 2]))
    sql_means = np.array(c.sql('SELECT ' + ','.join(f'sum(weight*X{i})/sum(weight)' for i in range(3)) + ' FROM training').fetchone())
    sql_scales = np.sqrt(np.array(c.sql('SELECT ' + ','.join(f'sum(weight*(X{i}-{mu:.17e})*(X{i}-{mu:.17e}))/sum(weight)' for i, mu in enumerate(sql_means)) + ' FROM training').fetchone()))
    sql_scales[sql_scales <= 1e-12] = 1; c.close()
    np.testing.assert_allclose(sql_means, means, rtol=0, atol=2e-7); np.testing.assert_allclose(sql_scales, scales, rtol=0, atol=2e-7)
    independent = Ridge(alpha=1, fit_intercept=True, solver='cholesky').fit((x-sql_means)/sql_scales, y, sample_weight=w)
    np.testing.assert_allclose(independent.coef_, coefficients, rtol=0, atol=2e-10)
    np.testing.assert_allclose(independent.intercept_, bias, rtol=0, atol=2e-10)
    np.testing.assert_allclose(independent.predict((x-sql_means)/sql_scales), predict(x, m), rtol=0, atol=2e-10)
    save_json(path / 'model_verification.json', dict(passed=True, model_report_sha256=sha(path / 'model_report.json'),
        all_three_coefficients_independent_cholesky_and_sql_moments_verified=True,
        mature_date_exchange_targets_and_equal_date_weights_verified=True, new_2026_prices_read=False))
    return {k: m[k] for k in ['fold', 'rows', 'days', 'last_observation', 'bias', 'coefficients']}


def score_expression(model, variables):
    terms = [f'({coefficient:.17e}*({name}-{mean:.17e})/{scale:.17e})'
        for name, coefficient, mean, scale in zip(variables, model['coefficients'], model['means'], model['scales'])]
    return f'{model["bias"]:.17e}' + '+' + '+'.join(terms)


def checked_model(fold):
    p = checked(); path = ROOT / 'models' / fold
    model = json.loads((path / 'model_report.json').read_text())
    v = json.loads((path / 'model_verification.json').read_text())
    assert v['passed'] and v['model_report_sha256'] == sha(path / 'model_report.json')
    assert model['protocol_sha256'] == sha(PROTOCOL) and model['training_sha256'] == sha(path / 'training.parquet')
    assert model['input_verification_sha256'] == sha(ROOT / 'input_verification.json')
    assert model['feature_names'] == NAMES and model['last_observation'] < p['folds'][fold]['evaluation_start']
    return p, model, path


def scores(fold):
    p, model, path = checked_model(fold); assert not (path / 'score_report.json').exists()
    r = json.loads((ROOT / 'input_verification.json').read_text())
    assert r['passed'] and r['market_inputs_sha256'] == sha(ROOT / 'market_inputs.parquet')
    out = pd.read_parquet(ROOT / 'market_inputs.parquet'); out['score'] = predict(encode(out), model)
    out['gate'] = out.score.gt(0)
    c = base.conn(); sql_inputs(c, p)
    variables = [f'floor(least(greatest(100*{n}+10000+.000001,0),999999))' for n in NAMES]
    expression = score_expression(model, variables)
    expected = c.sql(f'SELECT *,score>0 AS gate FROM (SELECT *,{expression} AS score FROM market_inputs) ORDER BY date,exchange').df(); c.close()
    pd.testing.assert_frame_equal(out, expected, check_dtype=False, rtol=0, atol=2e-10)
    np.testing.assert_array_equal(out.gate, expected.gate)
    out.to_parquet(path / 'market_scores.parquet', index=False, compression='zstd')
    save_json(path / 'score_report.json', dict(protocol_sha256=sha(PROTOCOL), model_report_sha256=sha(path / 'model_report.json'),
        market_inputs_sha256=sha(ROOT / 'market_inputs.parquet'), scores_sha256=sha(path / 'market_scores.parquet'), rows=len(out),
        no_holdout_labels_read=True, no_stock_prediction=True, new_2026_prices_read=False))
    save_json(path / 'score_verification.json', dict(passed=True, score_report_sha256=sha(path / 'score_report.json'),
        all_index_integer_scores_and_zero_gate_flags_independently_sql_verified=True, new_2026_prices_read=False))
    return dict(fold=fold, rows=len(out), score_report_sha256=sha(path / 'score_report.json'))


def native_values(expression, x):
    tree = ast.parse(expression, mode='eval')
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div,
        ast.UAdd, ast.USub, ast.Name, ast.Load, ast.Constant)
    assert all(isinstance(n, allowed) for n in ast.walk(tree))
    assert {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} <= {'X39', 'X40', 'X41'}
    return eval(compile(tree, '<fixed-native-reference>', 'eval'), {'__builtins__': {}},
        {'X39': x[:, 0], 'X40': x[:, 1], 'X41': x[:, 2]})


def freeze():
    p = checked(); joint = ROOT / 'joint_selection_freeze.json'; assert not joint.exists()
    f = pd.read_parquet(Path(p['feature_roots'][1]) / 'features.parquet', columns=META)
    parents = {year: checked_selection(Path(root)) for year, root in p['original_selection_roots'].items()}
    for d in parents.values(): pd.testing.assert_frame_equal(d[KEYS], f[KEYS], check_exact=True)
    flags = {year: np.zeros(len(f), dtype=bool) for year in parents}; receipts = {}
    for fold, spec in p['folds'].items():
        _, model, path = checked_model(fold); year = fold[:4]
        sr = json.loads((path / 'score_report.json').read_text()); sv = json.loads((path / 'score_verification.json').read_text())
        assert sv['passed'] and sv['score_report_sha256'] == sha(path / 'score_report.json')
        assert sr['scores_sha256'] == sha(path / 'market_scores.parquet') and sr['model_report_sha256'] == sha(path / 'model_report.json')
        market = pd.read_parquet(path / 'market_scores.parquet')
        mask = f.date.ge(spec['evaluation_start']) & f.date.lt(spec['evaluation_end'])
        stock_root = Path(spec['stock_root']); stock_model = json.loads((stock_root / 'model_report.json').read_text())
        assert stock_model['last_observation'] < spec['evaluation_start']
        for kind in ['model', 'score']:
            v = json.loads((stock_root / (kind + '_verification.json')).read_text())
            assert v['passed'] and v[kind + '_report_sha256'] == sha(stock_root / (kind + '_report.json'))
        d = pd.read_parquet(stock_root / 'scores.parquet', filters=[('date', '>=', spec['evaluation_start']), ('date', '<', spec['evaluation_end'])])
        pd.testing.assert_frame_equal(d[META], f.loc[mask].reset_index(drop=True), check_exact=True)
        cutoff = stock_model['thresholds'][3]; assert cutoff['training_quantile'] == .995
        np.testing.assert_array_equal(parents[year].loc[mask, 'selected'].to_numpy(),
            (d.formula_input_valid & d.score.gt(cutoff['threshold'])).to_numpy())
        current = f.loc[mask, ['date', 'code']].copy(); current['exchange'] = current.code.str[:2]
        current = current.merge(market[['date', 'exchange', 'gate']], on=['date', 'exchange'], how='left', validate='many_to_one')
        assert len(current) == mask.sum() and current.gate.notna().all()
        flags[year][mask] = parents[year].loc[mask, 'selected'].to_numpy() & current.gate.to_numpy()
        c = base.conn(); c.register('parent', parents[year]); c.register('market', market)
        expected = c.sql(f'''SELECT p.date,p.code,(p.selected AND m.gate) AS selected FROM parent p JOIN market m
            ON p.date=m.date AND substr(p.code,1,2)=m.exchange WHERE p.date>='{spec['evaluation_start']}'
            AND p.date<'{spec['evaluation_end']}' ORDER BY p.date,p.code''').df(); c.close()
        np.testing.assert_array_equal(flags[year][mask], expected.selected)
        original = Path(spec['stock_core']).read_text(); assert original.count('CORE:') == 1
        condition = original.rsplit('CORE:', 1)[1].strip(); assert condition.endswith(';')
        expr = score_expression(model, ['X39', 'X40', 'X41'])
        np.testing.assert_allclose(native_values(expr, encode(market)), market.score.to_numpy(), rtol=0, atol=2e-10)
        text = original.rsplit('CORE:', 1)[0] + 'MG:=' + expr + ';\nCORE:(' + condition[:-1] + ') AND MG>0;\n'
        assert text.startswith(original.rsplit('CORE:', 1)[0])
        core = ROOT / (fold + '_frozen_numeric_core.tdx'); assert not core.exists(); core.write_text(text)
        for file in ['training.parquet', 'model_report.json', 'model_verification.json', 'market_scores.parquet', 'score_report.json', 'score_verification.json']:
            receipts[str(path / file)] = sha(path / file)
        receipts[str(core)] = sha(core)
    selections = []
    for year, old in parents.items():
        for name, value in [('kept', flags[year]), ('removed', old.selected.to_numpy() & ~flags[year])]:
            group = name + year; root = ROOT / group; root.mkdir(parents=True, exist_ok=True)
            out = f[KEYS].copy(); out['selected'] = value
            assert out.loc[out.selected, 'date'].str.startswith(year).all()
            out.to_parquet(root / 'selection.parquet', index=False, compression='zstd')
            save_json(root / 'selection_report.json', dict(protocol_sha256=sha(PROTOCOL),
                selection_sha256=sha(root / 'selection.parquet'), rows=len(out), selected=int(out.selected.sum()),
                group=group, original_selection_root=p['original_selection_roots'][year], new_group_outcomes_read=False,
                no_fill_or_label_filtering=True, new_2026_prices_read=False, no_exit_rules=True))
            save_json(root / 'selection_verification.json', dict(passed=True,
                selection_report_sha256=sha(root / 'selection_report.json'),
                all_full_keys_metadata_original_flags_and_index_gate_memberships_independently_verified=True))
            selections.append(dict(group=group, root=str(root), rows=len(out), selected=int(out.selected.sum()),
                days=out.loc[out.selected, 'date'].nunique(), selection_report_sha256=sha(root / 'selection_report.json'),
                selection_verification_sha256=sha(root / 'selection_verification.json')))
        np.testing.assert_array_equal(flags[year] | (old.selected.to_numpy() & ~flags[year]), old.selected)
    save_json(joint, dict(passed=True, protocol_sha256=sha(PROTOCOL), selections=selections,
        source_hashes=receipts, all_four_models_cores_and_full_annual_complements_frozen_together=True,
        original_stock_models_predictions_and_thresholds_unchanged=True, native_three_term_equations_verified=True,
        native_client_data_parity_verified=False, software_compilation_verified=False,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True))
    return dict(joint_sha256=sha(joint), selections=selections)


def checked_joint():
    p = checked(); path = ROOT / 'joint_selection_freeze.json'; joint = json.loads(path.read_text())
    assert joint['passed'] and joint['protocol_sha256'] == sha(PROTOCOL)
    committed = subprocess.run(['git', 'show', 'HEAD:docs/selection-formula.md'], text=True, capture_output=True, check=True).stdout
    assert sha(path) in committed
    for file, digest in joint['source_hashes'].items(): assert sha(Path(file)) == digest, file
    for item in joint['selections']:
        root = Path(item['root']); checked_selection(root)
        assert item['selection_report_sha256'] == sha(root / 'selection_report.json')
        assert item['selection_verification_sha256'] == sha(root / 'selection_verification.json')
    evaluation.source.ROOT = Path(p['label_roots'][1])
    master = dict(original_control=p['original_selection_roots']['2024'],
        controls={year: root for year, root in p['original_selection_roots'].items() if year != '2024'})
    return master, joint


def checked_analysis(root):
    frame = checked_selection(root)
    assert frame.date.lt('2026-01-01').all() and frame.loc[frame.selected, 'date'].ge('2024-01-01').all()
    r = json.loads((root / 'analysis_report.json').read_text()); v = json.loads((root / 'analysis_verification.json').read_text())
    assert v['passed'] and v['analysis_report_sha256'] == sha(root / 'analysis_report.json')
    assert r['selection_report_sha256'] == sha(root / 'selection_report.json')
    assert r['daily_summary_sha256'] == sha(root / 'daily_summary.parquet') and r['reference_label'] == '09:59'
    lr = json.loads((root / 'full_label_report.json').read_text()); lv = json.loads((root / 'full_label_verification.json').read_text())
    assert lv['passed'] and lv['label_report_sha256'] == r['label_report_sha256'] == sha(root / 'full_label_report.json')
    assert lr['labels_sha256'] == sha(root / 'full_labels.parquet')
    return frame, r


def analyze():
    p = checked(); master, joint = checked_joint()
    # Metadata first: only complete-frame matches can enter the reuse registry.
    matches = {}; receipts = {}; scans = 0
    for item in joint['selections']:
        frame = checked_selection(Path(item['root'])); same = []
        for root in p['prior_analysis_roots']:
            root = Path(root); sr = json.loads((root / 'selection_report.json').read_text()); scans += 1
            if sr.get('selected') != item['selected']: continue
            candidate, ar = checked_analysis(root)
            if frame.equals(candidate):
                same.append(str(root))
                for file in ['selection_report.json', 'selection_verification.json', 'analysis_report.json', 'analysis_verification.json']:
                    receipts[str(root / file)] = sha(root / file)
        matches[item['group']] = same
    save_json(ROOT / 'analysis_reuse_lookup.json', dict(passed=True, joint_sha256=sha(ROOT / 'joint_selection_freeze.json'),
        complete_selection_frame_matches=matches, metadata_scans=scans, source_hashes=receipts,
        no_count_only_statistical_reuse=True, new_2026_prices_read=False))
    master['controls'].update({f'cached{i}': root for i, root in enumerate(sorted({r for names in matches.values() for r in names}))})
    reuse_common.PROTOCOL = PROTOCOL; reuse_common.inputs.ROOT = ROOT
    reuse_common.checked_joint = lambda: (master, joint)
    reuse_common.checked_analysis = checked_analysis
    return reuse_common.analyze()


def calendar_difference(calendar, left, right, left_column, right_column):
    """Shared calendar/week draws, with separate active-day denominators."""
    dates = pd.Index(sorted(set(calendar)), name='date')
    a = left.set_index('date').reindex(dates); b = right.set_index('date').reindex(dates)
    aa, bb = a.rows.notna(), b.rows.notna()
    undefined = int((aa & a[left_column].isna()).sum() + (bb & b[right_column].isna()).sum())
    base = dict(calendar_days=len(dates), left_days=int(aa.sum()), right_days=int(bb.sum()), undefined_active_days=undefined)
    if not aa.any() or not bb.any() or undefined:
        return dict(**base, difference=None, ci=None, zero_denominator_bootstrap_draws=None)
    av, bv = a[left_column].fillna(0), b[right_column].fillna(0)
    point = float(av.sum()/aa.sum()-bv.sum()/bb.sum())
    blocks = pd.DataFrame(dict(a=av, b=bv, na=aa.astype(int), nb=bb.astype(int)), index=dates)
    blocks['week'] = pd.to_datetime(dates).to_period('W-SUN').astype(str)
    groups = blocks.groupby('week')[['a', 'b', 'na', 'nb']].sum()
    if len(groups) < 2: return dict(**base, difference=point, ci=None, zero_denominator_bootstrap_draws=None)
    indices = np.random.default_rng(20260926).integers(len(groups), size=(10000, len(groups)))
    boot = groups.to_numpy()[indices].sum(axis=1); bad = (boot[:, 2] == 0) | (boot[:, 3] == 0)
    interval = None if bad.any() else np.quantile(boot[:, 0]/boot[:, 2]-boot[:, 1]/boot[:, 3], [.025, .975]).tolist()
    return dict(**base, difference=point, ci=interval, zero_denominator_bootstrap_draws=int(bad.sum()))


def forecast_diagnostic(p):
    target = ROOT / 'forecast_diagnostic.json'; assert not target.exists()
    records = []; receipts = {}; c = base.conn()
    for fold, spec in p['folds'].items():
        _, model, path = checked_model(fold)
        labels = read_labels(p, spec['evaluation_start'], spec['evaluation_end'])
        actual = target_groups(labels, spec['evaluation_start'], spec['evaluation_end'], '2026-01-01')
        expected = sql_targets(c, p, dict(start=spec['evaluation_start'], end=spec['evaluation_end']), '2026-01-01')
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, rtol=0, atol=2e-12)
        scores = pd.read_parquet(path / 'market_scores.parquet')
        d = actual.merge(scores[['date', 'exchange', 'score']], on=['date', 'exchange'], validate='one_to_one')
        assert len(d) == len(actual) and d.groupby('date').exchange.nunique().eq(2).all()
        d['mse'] = (d.score-d.target)**2; d['constant_mse'] = (model['bias']-d.target)**2
        d['mse_delta'] = d.mse-d.constant_mse
        c.register('records', d)
        expected_daily = c.sql(f'''SELECT date,avg((score-target)*(score-target)) AS mse,
            avg(({model['bias']!r}-target)*({model['bias']!r}-target)) AS constant_mse,
            avg((score-target)*(score-target)-({model['bias']!r}-target)*({model['bias']!r}-target)) AS mse_delta
            FROM records GROUP BY date ORDER BY date''').df()
        daily = d.groupby('date', sort=True)[['mse', 'constant_mse', 'mse_delta']].mean().reset_index()
        pd.testing.assert_frame_equal(daily, expected_daily, check_dtype=False, rtol=0, atol=2e-12)
        daily.to_parquet(path / 'forecast_daily.parquet', index=False, compression='zstd')
        receipts[str(path / 'forecast_daily.parquet')] = sha(path / 'forecast_daily.parquet')
        records.append(dict(fold=fold, rows=len(d), days=d.date.nunique(),
            mse=float(daily.mse.mean()), constant_mse=float(daily.constant_mse.mean()),
            mse_delta=float(daily.mse_delta.mean()), mse_delta_ci=weekly_interval(daily.set_index('date').mse_delta),
            forecast_is_net_reference_not_realized_sell_profit=True))
    c.close()
    out = dict(passed=True, protocol_sha256=sha(PROTOCOL), joint_sha256=sha(ROOT / 'joint_selection_freeze.json'),
        records=records, source_hashes=receipts, full_known_exchange_targets_and_mse_independently_sql_verified=True,
        no_historical_2023_economics_reported=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(target, out); return out


def finish():
    p = checked(); _, joint = checked_joint(); assert not (ROOT / 'complete_results_manifest.json').exists()
    roots = {r['group']: Path(r['root']) for r in joint['selections']}
    roots.update({f'original{y}': Path(r) for y, r in p['original_selection_roots'].items()})
    receipts = {}; cache = {}
    for name, root in roots.items():
        cache[name] = checked_analysis(root)
        if not (root / 'reference_coverage_verification.json').exists(): audit(root, [2024, 2025])
        ref = json.loads((root / 'reference_coverage_verification.json').read_text())
        assert ref['passed'] and ref['analysis_report_sha256'] == sha(root / 'analysis_report.json')
        for file in ['selection.parquet', 'selection_report.json', 'selection_verification.json', 'analysis_report.json',
                     'analysis_verification.json', 'daily_summary.parquet', 'full_label_report.json', 'full_label_verification.json',
                     'reference_coverage_verification.json']:
            receipts[str(root / file)] = sha(root / file)
    def equivalent(left, right):
        a, ar = cache[left]; b, br = cache[right]
        return a.equals(b) and all(ar[k] == br[k] for k in ['summaries', 'daily_summary_sha256', 'label_report_sha256'])
    done = []
    for left, right in p['planned_comparisons']:
        pair = left + '_minus_' + right
        prior = next((r for r in done if equivalent(left, r['left']) and equivalent(right, r['right'])), None)
        paths = [ROOT / ('same_dates_' + pair + '.json'), ROOT / ('shared_unknowns_' + pair + '.json')]
        if prior: paths = [Path(x) for x in prior['outputs']]
        else:
            if not paths[0].exists(): compare(roots[left], roots[right], paths[0], periods=p['periods'], intersection_only=True)
            if not paths[1].exists():
                shared_compare(dict(left=str(roots[left]), right=str(roots[right]), output=str(paths[1]),
                    left_analysis_sha256=sha(roots[left] / 'analysis_report.json'), right_analysis_sha256=sha(roots[right] / 'analysis_report.json')),
                    dict(labels_sha256=sha(evaluation.source.ROOT / 'full_labels.parquet'), periods=p['periods'], signal_range=['2024-01-01', '2025-12-31']))
        for file in paths: assert json.loads(file.read_text())['passed']; receipts[str(file)] = sha(file)
        done.append(dict(left=left, right=right, outputs=[str(x) for x in paths], reused=prior is not None))
    calendar_records = []
    for year in ['2024', '2025']:
        daily = {}
        for name in ['kept' + year, 'removed' + year, 'original' + year]:
            d = pd.read_parquet(roots[name] / 'daily_summary.parquet')
            daily[name] = d.loc[d.arm.eq('formula') & d.bps.eq(15) & ~d.sensitive & d.date.str.startswith(year)]
        calendar = daily['original' + year].date
        for name in ['kept' + year, 'removed' + year]:
            for a, b in [('rate', 'rate'), ('mean_reference', 'mean_reference'), ('bad3', 'bad3'), ('lower', 'upper')]:
                r = calendar_difference(calendar, daily[name], daily['original' + year], a, b)
                calendar_records.append(dict(left=name, right='original' + year, metric=a + '_minus_' + b, **r))
    save_json(ROOT / 'calendar_comparisons.json', dict(passed=True, records=calendar_records,
        original_full_calendar_week_draws_and_active_day_denominators_preserved=True,
        known_metrics_are_conditional_on_known_labels=True,
        lower_minus_upper_is_conservative_not_shared_unknown_cancellation=True,
        zero_denominator_draws_retained_not_conditioned_away=True, no_same_date_only_benefit_claim=True))
    forecast = forecast_diagnostic(p)
    def main(name, period):
        return next(s for s in cache[name][1]['summaries'] if s['arm'] == 'formula' and s['bps'] == 15 and not s['sensitive'] and s['period'] == period)
    gates = dict(each_half_at_least_20_signal_days=all(main('kept' + h[:4], h)['days'] >= 20 for h in p['halves']),
        all_four_half_reference_means_positive=all((main('kept' + h[:4], h)['mean_reference'] or -1) > 0 for h in p['halves']),
        both_year_opportunity_above_original_and_bad3_not_above=all(
            main('kept' + y, y)['rate'] is not None and main('kept' + y, y)['bad3'] is not None and
            main('kept' + y, y)['rate'] > main('original' + y, y)['rate'] and
            main('kept' + y, y)['bad3'] <= main('original' + y, y)['bad3'] for y in ['2024', '2025']),
        both_year_calendar_conservative_lower_ci_positive=all(
            r['ci'] is not None and r['ci'][0] > 0 for r in calendar_records if r['left'].startswith('kept') and r['metric'] == 'lower_minus_upper'),
        all_four_forecast_mse_below_own_training_constant=all(r['mse_delta'] < 0 for r in forecast['records']))
    save_json(ROOT / 'market_reference_gate.json', dict(passed=True, criteria=gates,
        supports_further_validation=all(gates.values()), no_automatic_2026_evaluation=True, no_publish_claim=True,
        original_prior_failure_gates_unchanged=True, new_2026_prices_read=False, no_exit_rules=True))
    for file in ['analysis_dispatch_verification.json', 'analysis_reuse_lookup.json', 'calendar_comparisons.json',
                 'forecast_diagnostic.json', 'market_reference_gate.json']:
        receipts[str(ROOT / file)] = sha(ROOT / file)
    save_json(ROOT / 'complete_results_manifest.json', dict(passed=True, protocol_sha256=sha(PROTOCOL),
        joint_sha256=sha(ROOT / 'joint_selection_freeze.json'), source_hashes=receipts, comparisons=done,
        original_stock_statistics_reused_after_full_frame_and_label_checks=True,
        no_duplicate_half_year_aggregation=True, year_2024_is_exploratory=True, year_2025_is_exploratory=True,
        no_publish_claim=True, new_2026_prices_read=False, no_exit_rules=True))
    return dict(completion_sha256=sha(ROOT / 'complete_results_manifest.json'), criteria=gates, forecast=forecast['records'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['project', 'model', 'scores', 'freeze', 'analyze', 'finish'])
    parser.add_argument('--fold', choices=['2024h1', '2024h2', '2025h1', '2025h2'])
    args = parser.parse_args()
    if args.stage in ['model', 'scores']:
        assert args.fold; result = fit(args.fold) if args.stage == 'model' else scores(args.fold)
    else: result = globals()[args.stage]()
    print(json.dumps(result, ensure_ascii=False, indent=2))
