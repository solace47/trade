"""Frozen linear/quadratic ridge comparison on the original 48 visible inputs."""
import argparse
import ast
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from . import tail_formula_additive as base
from . import tail_formula_boundary_evaluation as evaluation
from . import tail_formula_context_2024 as linkage
from . import tail_formula_float as original
from . import tail_formula_relative as relative
from .corporate_cash import save_json, sha

STEM = 'tail_formula_polynomial_ridge'
MASTER = Path('config') / (STEM + '_protocol.json')
ROOT = Path('data/research') / STEM
BLOCK = 8192
GROUP = 32
ARM = None


def setup(arm, fold):
    global ARM
    assert arm in ['linear', 'quadratic'] and fold in ['2024', 'recent', 'combined']
    ARM = arm
    base.FEATURES = original.ROOT
    base.EXPRESSIONS = original.EXPRESSIONS
    base.HEADER = original.HEADER
    base.SOURCE = Path('data/research/tail_formula_before1000')
    prefix = STEM + '_' + arm
    base.ROOT = Path('data/research') / (prefix + '_' + ('2025' if fold == 'combined' else fold))
    base.PROTOCOL = Path('config') / (prefix + '_' + fold + '_protocol.json')
    relative.PROTOCOL = base.PROTOCOL
    if fold == 'combined':
        linkage.ROOT = Path('data/research') / (prefix + '_2024')
        linkage.H2 = Path('data/research') / (prefix + '_recent')
        linkage.COMBINED = base.ROOT
        linkage.PROTOCOL = base.PROTOCOL
        linkage.H2_SELECTION_SHA = None
        linkage.H2_MODEL_SHA = None
        linkage.H2_OUTCOMES_PREVIOUSLY_SEEN = False


def config():
    p = json.loads(base.PROTOCOL.read_text())
    master = json.loads(MASTER.read_text())
    assert p['inputs_protocol_sha256'] == sha(MASTER)
    for path, digest in master['references'].items():
        assert sha(Path(path)) == digest
    assert p['feature_report_sha256'] == sha(base.FEATURES / 'feature_report.json')
    assert p['label_report_sha256'] == sha(base.SOURCE / 'full_label_report.json')
    assert p['expected_features'] == 48 and p['expected_terms'] == master['expected_terms'][ARM]
    assert p['ridge_alpha'] == master['ridge_alpha'] == 1
    assert p['standardized_clip'] == master['standardized_clip'] == 5
    assert p['scale_floor'] == master['scale_floor'] == 1e-12
    assert p['training_quantile'] == .995 and p['training_end'] == p['evaluation_start']
    assert p['window_end'] == '09:59'
    return p


def dictionary(width, arm):
    terms = [[i] for i in range(width)]
    if arm == 'quadratic':
        terms += [[i, j] for i in range(width) for j in range(i, width)]
    else:
        assert arm == 'linear'
    return terms


def basis(z, terms):
    out = np.empty((len(z), len(terms)), dtype=float)
    for k, term in enumerate(terms):
        out[:, k] = z[:, term[0]] if len(term) == 1 else z[:, term[0]] * z[:, term[1]]
    return out


def standardized(x, m):
    return np.clip((x - np.asarray(m['input_means'])) / np.asarray(m['input_scales']), -5, 5)


def blocks(n, size=BLOCK):
    for start in range(0, n, size):
        yield slice(start, min(start + size, n))


def fit_numeric(x, y, w, arm, block=BLOCK):
    """Bounded-memory fit; preprocessing and penalty do not use evaluation rows."""
    w = np.asarray(w, dtype=float)
    assert np.isfinite(x).all() and np.isfinite(y).all() and (w > 0).all()
    w = w / w.sum()
    means = np.average(x, axis=0, weights=w)
    scales = np.sqrt(np.average((x - means) ** 2, axis=0, weights=w))
    constant = scales <= 1e-12
    scales[constant] = 1
    terms = dictionary(x.shape[1], arm)
    m = dict(arm=arm, terms=terms, input_means=means.tolist(), input_scales=scales.tolist(),
             input_constant_indices=np.flatnonzero(constant).tolist(), ridge_alpha=1,
             standardized_clip=5, scale_floor=1e-12)
    z = standardized(x, m)
    term_means = np.zeros(len(terms))
    for part in blocks(len(x), block):
        term_means += w[part] @ basis(z[part], terms)
    term_variance = np.zeros(len(terms))
    for part in blocks(len(x), block):
        b = basis(z[part], terms) - term_means
        term_variance += w[part] @ (b * b)
    term_scales = np.sqrt(term_variance)
    term_constant = term_scales <= 1e-12
    term_scales[term_constant] = 1
    bias = float(w @ y)
    gram = np.zeros((len(terms), len(terms)))
    rhs = np.zeros(len(terms))
    for part in blocks(len(x), block):
        b = (basis(z[part], terms) - term_means) / term_scales
        gram += b.T @ (b * w[part, None])
        rhs += b.T @ (w[part] * (y[part] - bias))
    coefficients = np.linalg.solve(gram + np.eye(len(terms)), rhs)
    m.update(bias=bias, term_means=term_means.tolist(), term_scales=term_scales.tolist(),
             term_constant_indices=np.flatnonzero(term_constant).tolist(),
             coefficients=coefficients.tolist(), weights_sum=float(w.sum()))
    return m


def predict(x, m):
    out = np.empty(len(x))
    terms = m['terms']
    for part in blocks(len(x)):
        z = standardized(x[part], m)
        score = np.full(len(z), m['bias'])
        for start in range(0, len(terms), GROUP):
            subtotal = np.zeros(len(z))
            for k in range(start, min(start + GROUP, len(terms))):
                term = terms[k]
                value = z[:, term[0]] if len(term) == 1 else z[:, term[0]] * z[:, term[1]]
                subtotal += m['coefficients'][k] * (value - m['term_means'][k]) / m['term_scales'][k]
            score += subtotal
        out[part] = score
    return out


def model():
    p = config()
    root = base.ROOT
    assert not (root / 'model_report.json').exists(), 'Do not refit a frozen model'
    t = relative.training('relative')
    expected = next(v for v in json.loads(MASTER.read_text())['training'] if v['start'] == p['training_start'])
    assert len(t) == expected['rows'] and t.date.nunique() == expected['days']
    assert t.date.ge(p['training_start']).all() and t.next_date.lt(p['training_end']).all()
    w = 1 / t.groupby('date').code.transform('size').to_numpy() / t.date.nunique()
    x = base.encode(t)
    with threadpool_limits(limits=2):
        m = fit_numeric(x, t.target.to_numpy(), w, ARM)
    m.update(protocol_sha256=sha(base.PROTOCOL), inputs_protocol_sha256=sha(MASTER),
             feature_report_sha256=sha(base.FEATURES / 'feature_report.json'),
             label_report_sha256=sha(base.SOURCE / 'full_label_report.json'),
             model_family='date_equal_clipped_polynomial_ridge', feature_names=list(base.EXPRESSIONS),
             rows=len(t), days=t.date.nunique(), training_start=p['training_start'], training_end=p['training_end'],
             last_observation=t.next_date.max(), new_2025_score_groups_read=bool(t.date.ge('2025-01-01').any()),
             new_2025H2_score_groups_read=False, new_2026_prices_read=False, no_exit_rules=True,
             score_is_not_probability=True)
    score = predict(x, m)
    m['thresholds'] = [dict(id=0, training_quantile=.995, threshold=float(np.quantile(score, .995)))]
    save_json(root / 'model_report.json', m)
    return dict(rows=m['rows'], days=m['days'], arm=ARM, terms=len(m['terms']),
                model_report_sha256=sha(root / 'model_report.json'))


def checked_model():
    p = config()
    m = json.loads((base.ROOT / 'model_report.json').read_text())
    for key, path in [('protocol_sha256', base.PROTOCOL), ('inputs_protocol_sha256', MASTER),
                      ('feature_report_sha256', base.FEATURES / 'feature_report.json'),
                      ('label_report_sha256', base.SOURCE / 'full_label_report.json')]:
        assert m[key] == sha(path)
    assert m['arm'] == ARM and m['model_family'] == 'date_equal_clipped_polynomial_ridge'
    assert m['feature_names'] == list(base.EXPRESSIONS) and m['terms'] == dictionary(48, ARM)
    assert m['ridge_alpha'] == 1 and m['standardized_clip'] == 5 and m['scale_floor'] == 1e-12
    for names, count in [(['input_means', 'input_scales'], 48),
                         (['term_means', 'term_scales', 'coefficients'], p['expected_terms'])]:
        for name in names:
            assert len(m[name]) == count and np.isfinite(m[name]).all()
    assert min(m['input_scales']) > 0 and min(m['term_scales']) > 0
    assert m['last_observation'] < p['evaluation_start']
    assert m['training_start'] == p['training_start'] and m['training_end'] == p['training_end']
    return p, m


def register_features(c):
    # Arrow projects exact names before DuckDB's case-insensitive identifier handling.
    import pyarrow.parquet as pq
    names = ['date', 'code', 'half', 'board', 'decision_shares', 'formula_input_valid', *base.EXPRESSIONS]
    c.register('features', pq.read_table(base.FEATURES / 'features.parquet', columns=names))


def training_sql(c, p):
    encoded = ','.join(f'floor(least(greatest(100*{name}+10000+.000001,0),999999))::INT AS {name}'
                       for name in base.EXPRESSIONS)
    c.execute(f"""CREATE VIEW training AS WITH l AS (
        SELECT date,code,opportunity15 FROM read_parquet('{base.SOURCE}/full_labels.parquet')
        WHERE known15 AND date>='{p['training_start']}' AND next_date<'{p['training_end']}'),
        target AS (SELECT date,code,opportunity15-avg(opportunity15) OVER(PARTITION BY date) AS target FROM l),
        joined AS (SELECT f.date,f.code,target,1./count(*) OVER(PARTITION BY f.date) AS raw_w,{encoded}
        FROM features f JOIN target USING(date,code) WHERE formula_input_valid)
        SELECT * EXCLUDE(raw_w),raw_w/sum(raw_w) OVER() AS w FROM joined""")


def verify_model():
    p, m = checked_model()
    c = base.conn()
    register_features(c)
    training_sql(c, p)
    d = c.sql('SELECT * FROM training ORDER BY date,code').df()
    t = relative.training('relative')
    pd.testing.assert_frame_equal(d[['date', 'code']], t[['date', 'code']], check_exact=True)
    np.testing.assert_allclose(d.target, t.target, rtol=0, atol=2e-12)
    np.testing.assert_array_equal(d[list(base.EXPRESSIONS)].to_numpy(), base.encode(t))
    w = 1 / t.groupby('date').code.transform('size').to_numpy() / t.date.nunique()
    w /= w.sum()
    np.testing.assert_allclose(d.w, w, rtol=0, atol=2e-15)
    names = list(base.EXPRESSIONS)
    means = np.array(c.sql('SELECT ' + ','.join(f'sum(w*{n})/sum(w)' for n in names) + ' FROM training').fetchone())
    variances = c.sql('SELECT ' + ','.join(f'sum(w*({n}-({mu:.17e}))*({n}-({mu:.17e})))/sum(w)'
                     for n, mu in zip(names, means)) + ' FROM training').fetchone()
    scales = np.sqrt(variances)
    constants = scales <= 1e-12
    scales[constants] = 1
    np.testing.assert_allclose(means, m['input_means'], rtol=0, atol=2e-7)
    np.testing.assert_allclose(scales, m['input_scales'], rtol=0, atol=2e-7)
    assert np.flatnonzero(constants).tolist() == m['input_constant_indices']
    # Verify with an independently enumerated full dictionary and SQL input transforms.
    exprs = [f'least(greatest(({n}-({mu:.17e}))/({scale:.17e}),-5e0),5e0) AS Z{i}'
             for i, (n, mu, scale) in enumerate(zip(names, m['input_means'], m['input_scales']))]
    z = c.sql('SELECT ' + ','.join(exprs) + ' FROM training ORDER BY date,code').df().to_numpy()
    reference_terms = [(i,) for i in range(48)]
    if ARM == 'quadratic':
        reference_terms.extend((i, j) for i in range(48) for j in range(48) if j >= i)
    assert [list(v) for v in reference_terms] == m['terms']
    def reference_basis(part):
        return np.column_stack([np.prod(z[part, term], axis=1) if len(term) > 1
                                else z[part, term[0]] for term in reference_terms])
    rw = d.w.to_numpy()
    nterms = len(reference_terms)
    bm = np.zeros(nterms)
    with threadpool_limits(limits=2):
        for part in blocks(len(d), 4096):
            bm += np.einsum('i,ij->j', rw[part], reference_basis(part))
        bv = np.zeros(nterms)
        for part in blocks(len(d), 4096):
            centered = reference_basis(part) - bm
            bv += np.einsum('i,ij,ij->j', rw[part], centered, centered)
        bs = np.sqrt(bv)
        constants = bs <= 1e-12
        bs[constants] = 1
        np.testing.assert_allclose(bm, m['term_means'], rtol=0, atol=2e-10)
        np.testing.assert_allclose(bs, m['term_scales'], rtol=0, atol=2e-10)
        assert np.flatnonzero(constants).tolist() == m['term_constant_indices']
        gradient = np.zeros(nterms)
        intercept_gradient = 0.
        reference_scores = np.empty(len(d))
        coef = np.asarray(m['coefficients'])
        for part in blocks(len(d), 4096):
            normalized = (reference_basis(part) - bm) / bs
            pred = float(m['bias']) + normalized @ coef
            reference_scores[part] = pred
            residual = d.target.to_numpy()[part] - pred
            gradient += np.einsum('i,ij->j', rw[part] * residual, normalized)
            intercept_gradient += float(rw[part] @ residual)
        gradient -= coef  # alpha = 1, strictly convex in all penalized coefficients.
    assert np.max(np.abs(gradient)) < 2e-10 and abs(intercept_gradient) < 2e-10
    np.testing.assert_allclose(m['bias'], rw @ d.target.to_numpy(), rtol=0, atol=2e-12)
    actual = predict(d[names].to_numpy(), m)
    np.testing.assert_allclose(actual, reference_scores, rtol=0, atol=2e-10)
    np.testing.assert_allclose(np.quantile(reference_scores, .995), m['thresholds'][0]['threshold'], rtol=0, atol=2e-10)
    assert len(d) == m['rows'] and d.date.nunique() == m['days'] == 241
    assert t.next_date.max() == m['last_observation'] and abs(m['weights_sum'] - 1) < 2e-12
    c.close()
    proof = dict(passed=True, model_report_sha256=sha(base.ROOT / 'model_report.json'), rows=len(d),
                 days=d.date.nunique(), terms=nterms, all_targets_encodings_date_weights_sql_verified=True,
                 all_input_and_basis_statistics_verified=True, all_normal_equations_verified=True,
                 max_normal_equation_residual=float(np.max(np.abs(gradient))),
                 intercept_gradient=intercept_gradient,
                 max_training_score_difference=float(np.max(np.abs(actual - reference_scores))),
                 new_2026_prices_read=False, no_exit_rules=True)
    save_json(base.ROOT / 'model_verification.json', proof)
    return proof


def native_definitions(m):
    definitions = []
    for i, name in enumerate(m['feature_names'], 1):
        definitions.append((f'X{i:02d}', f'INTPART(MIN(MAX(100*{name}+10000+0.000001,0),999999))'))
    for i, (mean, scale) in enumerate(zip(m['input_means'], m['input_scales']), 1):
        definitions.append((f'PZ{i:02d}', f'MIN(MAX((X{i:02d}-({mean:.17e}))/({scale:.17e}),-5),5)'))
    groups = []
    for start in range(0, len(m['terms']), GROUP):
        expressions = []
        for k in range(start, min(start + GROUP, len(m['terms']))):
            term = m['terms'][k]
            value = '*'.join(f'PZ{i+1:02d}' for i in term)
            expressions.append(f'(({m["coefficients"][k]:.17e})*(({value})-({m["term_means"][k]:.17e}))/({m["term_scales"][k]:.17e}))')
        name = f'PG{len(groups)+1:02d}'
        groups.append(name)
        definitions.append((name, '+'.join(expressions)))
    definitions.append(('SC', f'({m["bias"]:.17e})+' + '+'.join(groups)))
    return definitions


def native_core(m):
    return (base.HEADER + '\n'.join(f'{k}:={v};' for k, v in base.EXPRESSIONS.items()) + '\n'
            + '\n'.join(f'{k}:={v};' for k, v in native_definitions(m))
            + f'\nCORE:SC>{m["thresholds"][0]["threshold"]:.17e};\n')


def flatten_addition(node):
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return flatten_addition(node.left) + flatten_addition(node.right)
    return [node]


def number_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    assert isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd))
    return number_node(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)


def verify_native(core, m):
    definitions = re.findall(r'^([A-Z][A-Z0-9]*):=([^;]+);$', core, re.M)
    assert len(definitions) == len({n for n, _ in definitions}), 'Native variable names must be unique'
    d = dict(definitions)
    assert [n for n in d if n.startswith('PZ')] == [f'PZ{i:02d}' for i in range(1, 49)]
    for i, (mean, scale) in enumerate(zip(m['input_means'], m['input_scales']), 1):
        expr = ast.parse(d[f'PZ{i:02d}'], mode='eval').body
        assert isinstance(expr, ast.Call) and expr.func.id == 'MIN' and number_node(expr.args[1]) == 5
        low = expr.args[0]
        assert isinstance(low, ast.Call) and low.func.id == 'MAX' and number_node(low.args[1]) == -5
        div = low.args[0]
        assert isinstance(div.op, ast.Div) and number_node(div.right) == scale
        sub = div.left
        assert isinstance(sub.op, ast.Sub) and sub.left.id == f'X{i:02d}' and number_node(sub.right) == mean
    groups = [(n, ast.parse(e, mode='eval').body) for n, e in definitions if re.fullmatch(r'PG\d{2}', n)]
    terms = [term for _, expr in groups for term in flatten_addition(expr)]
    assert len(terms) == len(m['terms'])
    for k, node in enumerate(terms):
        assert isinstance(node.op, ast.Div) and number_node(node.right) == m['term_scales'][k]
        mul = node.left
        assert isinstance(mul.op, ast.Mult) and number_node(mul.left) == m['coefficients'][k]
        sub = mul.right
        assert isinstance(sub.op, ast.Sub) and number_node(sub.right) == m['term_means'][k]
        term = m['terms'][k]
        if len(term) == 1:
            assert sub.left.id == f'PZ{term[0]+1:02d}'
        else:
            product = sub.left
            assert isinstance(product.op, ast.Mult)
            assert [product.left.id, product.right.id] == [f'PZ{i+1:02d}' for i in term]
    score_terms = flatten_addition(ast.parse(d['SC'], mode='eval').body)
    assert number_node(score_terms[0]) == m['bias']
    assert [v.id for v in score_terms[1:]] == [name for name, _ in groups]
    assert core.endswith(f'CORE:SC>{m["thresholds"][0]["threshold"]:.17e};\n')
    return d


def scores():
    p, m = checked_model()
    root = base.ROOT
    assert not (root / 'score_report.json').exists(), 'Do not replace frozen scores'
    proof = json.loads((root / 'model_verification.json').read_text())
    assert proof['passed'] and proof['model_report_sha256'] == sha(root / 'model_report.json')
    f = base.feature_inputs()
    out = f[['date', 'code', 'half', 'board', 'decision_shares', 'formula_input_valid']].copy()
    out['score'] = np.nan
    out.loc[f.formula_input_valid, 'score'] = predict(base.encode(f.loc[f.formula_input_valid]), m)
    out.to_parquet(root / 'scores.parquet', index=False, compression='zstd')
    # Freeze the exact native arithmetic before its independent full SQL replay.
    core = native_core(m)
    verify_native(core, m)
    (root / 'numeric_score.tdx').write_text(core)
    report = dict(protocol_sha256=sha(base.PROTOCOL), model_report_sha256=sha(root / 'model_report.json'),
                  feature_report_sha256=sha(base.FEATURES / 'feature_report.json'),
                  scores_sha256=sha(root / 'scores.parquet'), numeric_score_sha256=sha(root / 'numeric_score.tdx'),
                  rows=len(out), valid=int(f.formula_input_valid.sum()), no_outcome_based_scoring=True,
                  new_2026_prices_read=False, no_exit_rules=True)
    save_json(root / 'score_report.json', report)
    return report


def verify_scores():
    p, m = checked_model()
    root = base.ROOT
    report = json.loads((root / 'score_report.json').read_text())
    for key, path in [('protocol_sha256', base.PROTOCOL), ('model_report_sha256', root / 'model_report.json'),
                      ('feature_report_sha256', base.FEATURES / 'feature_report.json'),
                      ('scores_sha256', root / 'scores.parquet'), ('numeric_score_sha256', root / 'numeric_score.tdx')]:
        assert report[key] == sha(path)
    core = (root / 'numeric_score.tdx').read_text()
    assert core == native_core(m)
    native = verify_native(core, m)
    def sql(expr):
        return re.sub(r'\bINTPART\(', 'floor(', re.sub(r'\bMIN\(', 'least(', re.sub(r'\bMAX\(', 'greatest(', expr)))
    c = base.conn()
    register_features(c)
    encoded = ','.join(sql(native[f'X{i:02d}']) + f'::DOUBLE AS X{i:02d}' for i in range(1, 49))
    c.sql('SELECT date,code,' + encoded + ' FROM features WHERE formula_input_valid').create_view('encoded')
    transformed = ','.join(sql(native[f'PZ{i:02d}']) + f' AS PZ{i:02d}' for i in range(1, 49))
    c.sql('SELECT date,code,' + transformed + ' FROM encoded').create_view('transformed')
    groups = [n for n in native if re.fullmatch(r'PG\d{2}', n)]
    c.sql('SELECT date,code,' + ','.join(sql(native[n]) + ' AS ' + n for n in groups)
          + ' FROM transformed').create_view('grouped')
    c.sql('SELECT date,code,' + sql(native['SC']) + ' AS score FROM grouped').create_view('rebuilt')
    expected = c.sql('''SELECT f.date,f.code,f.half,f.board,f.decision_shares,f.formula_input_valid,r.score
        FROM features f LEFT JOIN rebuilt r USING(date,code) ORDER BY date,code''').df()
    actual = pd.read_parquet(root / 'scores.parquet')
    pd.testing.assert_frame_equal(actual.drop(columns='score'), expected.drop(columns='score'), check_exact=True)
    np.testing.assert_allclose(actual.score, expected.score, rtol=0, atol=2e-10, equal_nan=True)
    assert actual.score.notna().equals(actual.formula_input_valid)
    cut = m['thresholds'][0]['threshold']
    np.testing.assert_array_equal(actual.score.gt(cut), expected.score.gt(cut))
    training_sql(c, p)
    c.register('replayed_scores', expected[['date', 'code', 'score']])
    training_scores = c.sql('SELECT score FROM replayed_scores JOIN training USING(date,code) ORDER BY date,code').df()
    np.testing.assert_allclose(np.quantile(training_scores.score, .995), cut, rtol=0, atol=2e-10)
    c.close()
    proof = dict(passed=True, score_report_sha256=sha(root / 'score_report.json'), rows=len(actual),
                 valid=int(actual.formula_input_valid.sum()), all_native_arithmetic_sql_replayed=True,
                 all_threshold_flags_and_training_quantile_rebuilt=True,
                 max_score_difference=float((actual.score - expected.score).abs().max()),
                 new_2026_prices_read=False, no_exit_rules=True, software_compilation_verified=False)
    save_json(root / 'score_verification.json', proof)
    return proof


def freeze():
    p, m = checked_model()
    root = base.ROOT
    assert not (root / 'selection_report.json').exists(), 'Do not replace frozen selections'
    report = json.loads((root / 'score_report.json').read_text())
    proof = json.loads((root / 'score_verification.json').read_text())
    assert proof['passed'] and proof['score_report_sha256'] == sha(root / 'score_report.json')
    assert report['scores_sha256'] == sha(root / 'scores.parquet')
    f = pd.read_parquet(root / 'scores.parquet')
    out = f[['date', 'code', 'half', 'board', 'decision_shares']].copy()
    out['selected'] = (f.date.ge(p['evaluation_start']) & f.date.lt(p['evaluation_end'])
                       & f.formula_input_valid & f.score.gt(m['thresholds'][0]['threshold']))
    out.to_parquet(root / 'selection.parquet', index=False, compression='zstd')
    core = root / 'frozen_numeric_core.tdx'
    core.write_bytes((root / 'numeric_score.tdx').read_bytes())
    assert sha(core) == report['numeric_score_sha256']
    r = dict(protocol_sha256=sha(base.PROTOCOL), model_report_sha256=sha(root / 'model_report.json'),
             score_report_sha256=sha(root / 'score_report.json'), selection_sha256=sha(root / 'selection.parquet'),
             core_sha256=sha(core), chosen_threshold=m['thresholds'][0], selected=int(out.selected.sum()),
             by_half=out.groupby('half').selected.agg(['size', 'sum']).reset_index().to_dict('records'),
             evaluation_start=p['evaluation_start'], evaluation_end=p['evaluation_end'],
             new_group_outcomes_read=False, year_2025_is_exploratory=True, new_2026_prices_read=False,
             no_exit_rules=True, software_compilation_verified=False)
    save_json(root / 'selection_report.json', r)
    return r


def verify():
    p, m = checked_model()
    root = base.ROOT
    r = json.loads((root / 'selection_report.json').read_text())
    for key, path in [('protocol_sha256', base.PROTOCOL), ('model_report_sha256', root / 'model_report.json'),
                      ('score_report_sha256', root / 'score_report.json'), ('selection_sha256', root / 'selection.parquet'),
                      ('core_sha256', root / 'frozen_numeric_core.tdx')]:
        assert r[key] == sha(path)
    c = base.conn()
    c.read_parquet(str(root / 'scores.parquet')).create_view('scores')
    cut = m['thresholds'][0]
    assert r['chosen_threshold'] == cut and cut['training_quantile'] == .995
    expected = c.sql(f"""SELECT date,code,half,board,decision_shares,
        date>='{p['evaluation_start']}' AND date<'{p['evaluation_end']}' AND formula_input_valid
        AND score>{cut['threshold']:.17e} AS selected FROM scores ORDER BY date,code""").df()
    c.close()
    pd.testing.assert_frame_equal(pd.read_parquet(root / 'selection.parquet'), expected, check_exact=True)
    assert int(expected.selected.sum()) == r['selected']
    core = (root / 'frozen_numeric_core.tdx').read_text()
    assert core == native_core(m)
    verify_native(core, m)
    assert expected.loc[expected.selected, 'date'].ge(m['training_end']).all()
    proof = dict(passed=True, selection_report_sha256=sha(root / 'selection_report.json'), rows=len(expected),
                 all_selection_flags_rebuilt=True, all_native_coefficients_transforms_dictionary_verified=True,
                 no_training_period_selection=True, no_new_group_outcomes_read=True,
                 new_2026_prices_read=False, no_exit_rules=True)
    save_json(root / 'selection_verification.json', proof)
    return proof


def analyze():
    joint = json.loads((ROOT / 'joint_selection_freeze.json').read_text())
    assert joint['passed'] and joint['protocol_sha256'] == sha(MASTER) and len(joint['selections']) == 6
    frozen = next(v for v in joint['selections'] if v['root'] == str(base.ROOT))
    assert frozen['selection_report_sha256'] == sha(base.ROOT / 'selection_report.json')
    return evaluation.analyze(base.ROOT, base.PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['model', 'verify_model', 'scores', 'verify_scores', 'freeze', 'verify', 'analyze'])
    parser.add_argument('--arm', choices=['linear', 'quadratic'], required=True)
    parser.add_argument('--fold', choices=['2024', 'recent', 'combined'], default='2024')
    args = parser.parse_args()
    setup(args.arm, args.fold)
    if args.fold == 'combined' and args.stage != 'analyze':
        assert args.stage in ['freeze', 'verify']
        result = getattr(linkage, 'combine' if args.stage == 'freeze' else 'verify_combined')()
    else:
        result = globals()[args.stage]()
    print(json.dumps(result, ensure_ascii=False, indent=2))
