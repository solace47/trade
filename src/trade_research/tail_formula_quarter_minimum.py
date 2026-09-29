"""Use the minimum of four immutable, previously verified quarterly components."""
import argparse
import ast
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_context_2024 as linkage
from . import tail_formula_float as inputs
from . import tail_formula_polynomial_ridge as reuse
from . import tail_formula_quarter_ensemble as previous
from . import tail_formula_relative as relative
from .corporate_cash import save_json, sha

STEM = 'tail_formula_quarter_minimum'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
FOLD = None


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    for file, digest in p['references'].items():
        assert sha(Path(file)) == digest
    assert p['components'] == 4 and p['training_quantile'] == .995
    assert p['new_model_fitting_allowed'] is False
    for fold, folder in p['source_models'].items():
        root = Path(folder)
        m = json.loads((root / 'model_report.json').read_text())
        v = json.loads((root / 'model_verification.json').read_text())
        assert v['passed'] and v['model_report_sha256'] == sha(root / 'model_report.json')
        assert v['every_component_node_residual_statistics_rebuilt']
        assert m['feature_names'] == list(inputs.EXPRESSIONS)
        assert len(m['components']) == 4 and all(len(c['trees']) == 64 for c in m['components'])
        assert [c['omitted_quarter'] for c in m['components']] == p['quarter_sets'][fold]
        for c in m['components']:
            file = root / 'components' / (c['omitted_quarter'] + '.json')
            assert m['component_hashes'][str(file)] == sha(file)
            assert json.loads(file.read_text()) == c and c['learning_rate'] == .05
    return p


def setup(fold):
    global FOLD
    FOLD = fold
    base.FEATURES = inputs.ROOT; base.EXPRESSIONS = inputs.EXPRESSIONS; base.HEADER = inputs.HEADER
    base.SOURCE = Path('data/research/tail_formula_before1000')
    base.ROOT = ROOT.parent / (STEM + '_' + ('2025' if fold == 'combined' else fold))
    base.PROTOCOL = Path('config') / (STEM + '_' + fold + '_protocol.json')
    relative.PROTOCOL = base.PROTOCOL
    # Only model-independent artifact routines are reused, in fresh processes.
    reuse.checked_model = checked_model; reuse.predict = predict
    reuse.native_core = native_core; reuse.verify_native = verify_native
    if fold == 'combined':
        linkage.ROOT = ROOT.parent / (STEM + '_2024')
        linkage.H2 = ROOT.parent / (STEM + '_recent')
        linkage.COMBINED = base.ROOT; linkage.PROTOCOL = base.PROTOCOL
        linkage.H2_SELECTION_SHA = None; linkage.H2_MODEL_SHA = None
        linkage.H2_OUTCOMES_PREVIOUSLY_SEEN = False


def config():
    master = checked_sources(); p = json.loads(base.PROTOCOL.read_text())
    assert p['inputs_protocol_sha256'] == sha(PROTOCOL)
    assert p['parameters'] == master['parameters'] and p['expected_features'] == 48
    assert p['training_quantile'] == .995 and p['window_end'] == '09:59'
    assert p['training_end'] == p['evaluation_start']
    assert p['omitted_quarters'] == master['quarter_sets'][FOLD]
    assert p['source_root'] == master['source_models'][FOLD]
    return p


def predict(x, m):
    scores = [base.predict(x, c) for c in m['components']]
    assert len(scores) == 4 and all(np.isfinite(s).all() for s in scores)
    return np.minimum.reduce(scores)


def model():
    p = config(); root = base.ROOT
    assert not (root / 'model_report.json').exists()
    source = Path(p['source_root']); old = json.loads((source / 'model_report.json').read_text())
    t = relative.training('relative')
    for key, value in [('rows', len(t)), ('days', t.date.nunique()), ('last_observation', t.next_date.max()),
                       ('training_start', p['training_start']), ('training_end', p['training_end'])]:
        assert old[key] == value
    assert len(t) == p['expected_training_rows'] and t.next_date.lt(p['training_end']).all()
    m = {k: old[k] for k in ['components', 'component_hashes', 'feature_names', 'parameters',
        'rows', 'days', 'last_observation', 'training_start', 'training_end',
        'feature_report_sha256', 'label_report_sha256']}
    score = predict(base.encode(t), m)
    m.update(protocol_sha256=sha(base.PROTOCOL), inputs_protocol_sha256=sha(PROTOCOL),
        implementation_sha256=sha(Path(__file__)), source_root=str(source),
        source_model_sha256=sha(source / 'model_report.json'),
        source_verification_sha256=sha(source / 'model_verification.json'),
        model_family='minimum_of_four_frozen_quarter_models', new_model_fitting_performed=False,
        thresholds=[dict(id=0, training_quantile=.995, threshold=float(np.quantile(score, .995)))],
        training_scores_are_not_out_of_sample=True, new_2026_prices_read=False, no_exit_rules=True)
    root.mkdir(parents=True, exist_ok=True); save_json(root / 'model_report.json', m)
    return {k: v for k, v in m.items() if k not in ['components', 'component_hashes']}


def checked_model():
    p = config(); m = json.loads((base.ROOT / 'model_report.json').read_text())
    source = Path(p['source_root']); old = json.loads((source / 'model_report.json').read_text())
    for key, path in [('protocol_sha256', base.PROTOCOL), ('inputs_protocol_sha256', PROTOCOL),
        ('implementation_sha256', Path(__file__)), ('source_model_sha256', source / 'model_report.json'),
        ('source_verification_sha256', source / 'model_verification.json'),
        ('feature_report_sha256', inputs.ROOT / 'feature_report.json'),
        ('label_report_sha256', base.SOURCE / 'full_label_report.json')]:
        assert m[key] == sha(path)
    for key in ['components', 'component_hashes', 'feature_names', 'parameters', 'rows', 'days',
                'last_observation', 'training_start', 'training_end']:
        assert m[key] == old[key]
    assert m['source_root'] == str(source) and not m['new_model_fitting_performed']
    assert m['model_family'] == 'minimum_of_four_frozen_quarter_models'
    assert m['last_observation'] < p['evaluation_start']
    assert len(m['thresholds']) == 1 and m['thresholds'][0]['training_quantile'] == .995
    assert np.isfinite(m['thresholds'][0]['threshold'])
    return p, m


def native_definitions(m):
    d = [(f'X{i:02d}', f'INTPART(MIN(MAX(100*{n}+10000+0.000001,0),999999))')
         for i, n in enumerate(inputs.EXPRESSIONS, 1)]
    for j, c in enumerate(m['components'], 1):
        names = []
        for i, t in enumerate(c['trees'], 1):
            name = f'QMT{(j-1)*64+i:03d}'; names.append(name)
            d.append((name, base.native_tree(t)))
        d.append((f'QMS{j}', f'({c["bias"]:.17e})+' + '+'.join(names)))
    d.append(('SC', 'MIN(QMS1,MIN(QMS2,MIN(QMS3,QMS4)))'))
    return d


def native_core(m):
    return (inputs.HEADER + '\n'.join(f'{n}:={v};' for n, v in inputs.EXPRESSIONS.items()) + '\n'
        + '\n'.join(f'{n}:={v};' for n, v in native_definitions(m))
        + f'\nCORE:SC>{m["thresholds"][0]["threshold"]:.17e};\n')


def check_tree(node, tree, index=0):
    left = tree['children_left'][index]
    if left < 0:
        assert reuse.number_node(node) == .05 * tree['value'][index]
        return
    assert isinstance(node, ast.Call) and node.func.id == 'IF' and len(node.args) == 3
    test = node.args[0]
    assert isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.LtE)
    assert test.left.id == f'X{tree["feature"][index]+1:02d}'
    assert reuse.number_node(test.comparators[0]) == math.floor(tree['threshold'][index])
    check_tree(node.args[1], tree, left)
    check_tree(node.args[2], tree, tree['children_right'][index])


def verify_native(core, m):
    pairs = re.findall(r'^([A-Z][A-Z0-9]*):=([^;]+);$', core, re.M)
    assert len(pairs) == len({n.casefold() for n, _ in pairs})
    d = dict(pairs)
    for name, expr in native_definitions(m)[:48]:
        assert d[name] == expr
    for j, c in enumerate(m['components'], 1):
        names = []
        for i, t in enumerate(c['trees'], 1):
            name = f'QMT{(j-1)*64+i:03d}'; names.append(name)
            check_tree(ast.parse(d[name], mode='eval').body, t)
        terms = reuse.flatten_addition(ast.parse(d[f'QMS{j}'], mode='eval').body)
        assert reuse.number_node(terms[0]) == c['bias']
        assert [term.id for term in terms[1:]] == names
    assert d['SC'] == 'MIN(QMS1,MIN(QMS2,MIN(QMS3,QMS4)))'
    assert core.endswith(f'CORE:SC>{m["thresholds"][0]["threshold"]:.17e};\n')
    return d


def expression_sql(node):
    """Translate the exported arithmetic, without executing native text."""
    if isinstance(node, (ast.Constant, ast.UnaryOp)):
        return format(reuse.number_node(node), '.17e')
    if isinstance(node, ast.Name):
        assert re.fullmatch(r'(X\d{2}|QMT\d{3}|QMS[1-4]|[A-Z][A-Z0-9]*)', node.id)
        return node.id
    if isinstance(node, ast.BinOp):
        op = {ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/'}[type(node.op)]
        return '(' + expression_sql(node.left) + op + expression_sql(node.right) + ')'
    if isinstance(node, ast.Compare):
        assert len(node.ops) == 1 and isinstance(node.ops[0], ast.LtE)
        return '(' + expression_sql(node.left) + '<=' + expression_sql(node.comparators[0]) + ')'
    assert isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords
    args = [expression_sql(v) for v in node.args]
    if node.func.id == 'IF':
        assert len(args) == 3
        return f'(CASE WHEN {args[0]} THEN {args[1]} ELSE {args[2]} END)'
    fn = {'MIN': 'least', 'MAX': 'greatest', 'INTPART': 'floor'}[node.func.id]
    return fn + '(' + ','.join(args) + ')'


def score_views(c, native):
    sql = lambda expr: expression_sql(ast.parse(expr, mode='eval').body)
    trees = [n for n in native if re.fullmatch(r'QMT\d{3}', n)]
    assert len(trees) == 256
    c.sql('SELECT date,code,' + ','.join(sql(native[n]) + ' AS ' + n for n in trees)
          + ' FROM encoded').create_view('leaves')
    c.sql('SELECT date,code,' + ','.join(sql(native[f'QMS{j}']) + f' AS QMS{j}' for j in range(1, 5))
          + ' FROM leaves').create_view('components')
    c.sql('SELECT date,code,' + sql(native['SC']) + ' AS score FROM components').create_view('rebuilt')


def verify_model():
    p, m = checked_model(); d, t = previous.independent_training()
    assert len(d) == m['rows'] == p['expected_training_rows'] and d.date.nunique() == m['days'] == 241
    assert t.next_date.max() == m['last_observation']
    c = base.conn()
    encoded = d[['date', 'code', *inputs.EXPRESSIONS]].rename(columns={n: f'X{i:02d}' for i, n in enumerate(inputs.EXPRESSIONS, 1)})
    c.register('encoded', encoded)
    score_views(c, verify_native(native_core(m), m))
    rebuilt = c.sql('SELECT * FROM components ORDER BY date,code').df()
    pd.testing.assert_frame_equal(d[['date', 'code']], rebuilt[['date', 'code']], check_exact=True)
    x = base.encode(t)
    for j, component in enumerate(m['components'], 1):
        np.testing.assert_allclose(rebuilt[f'QMS{j}'], base.predict(x, component), rtol=0, atol=2e-12)
    scores = c.sql('SELECT score FROM rebuilt ORDER BY date,code').df().score.to_numpy(); c.close()
    np.testing.assert_allclose(scores, predict(x, m), rtol=0, atol=2e-12)
    np.testing.assert_allclose(np.quantile(scores, .995), m['thresholds'][0]['threshold'], rtol=0, atol=2e-12)
    old = json.loads((Path(p['source_root']) / 'model_verification.json').read_text())
    proof = dict(passed=True, model_report_sha256=sha(base.ROOT / 'model_report.json'), rows=len(d), days=241,
        node_checks=old['node_checks'], node_checks_reused_from_verified_immutable_components=True,
        node_statistics_recomputed=False, source_verification_sha256=m['source_verification_sha256'],
        all_training_keys_targets_weights_encodings_rebuilt=True, four_component_scores_sql_replayed=True,
        minimum_and_training_quantile_independently_rebuilt=True, aggregate_training_scores_are_not_out_of_sample=True,
        new_model_fitting_performed=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(base.ROOT / 'model_verification.json', proof); return proof


def verify_scores():
    p, m = checked_model(); root = base.ROOT
    r = json.loads((root / 'score_report.json').read_text())
    for key, path in [('protocol_sha256', base.PROTOCOL), ('model_report_sha256', root / 'model_report.json'),
        ('feature_report_sha256', inputs.ROOT / 'feature_report.json'), ('scores_sha256', root / 'scores.parquet'),
        ('numeric_score_sha256', root / 'numeric_score.tdx')]:
        assert r[key] == sha(path)
    core = (root / 'numeric_score.tdx').read_text(); assert core == native_core(m)
    d = verify_native(core, m); c = base.conn(); reuse.register_features(c)
    encoded = ','.join(expression_sql(ast.parse(d[f'X{i:02d}'], mode='eval').body) + f'::INT AS X{i:02d}' for i in range(1, 49))
    c.sql('SELECT date,code,' + encoded + ' FROM features WHERE formula_input_valid').create_view('encoded')
    score_views(c, d)
    expected = c.sql('SELECT f.date,f.code,f.half,f.board,f.decision_shares,f.formula_input_valid,r.score '
        'FROM features f LEFT JOIN rebuilt r USING(date,code) ORDER BY date,code').df()
    actual = pd.read_parquet(root / 'scores.parquet')
    pd.testing.assert_frame_equal(actual.drop(columns='score'), expected.drop(columns='score'), check_exact=True)
    np.testing.assert_allclose(actual.score, expected.score, rtol=0, atol=2e-12, equal_nan=True)
    assert actual.score.notna().equals(actual.formula_input_valid)
    cut = m['thresholds'][0]['threshold']
    np.testing.assert_array_equal(actual.score.gt(cut), expected.score.gt(cut))
    reuse.training_sql(c, p); c.register('replayed', expected[['date', 'code', 'score']])
    train = c.sql('SELECT score FROM replayed JOIN training USING(date,code) ORDER BY date,code').df()
    assert len(train) == m['rows']
    np.testing.assert_allclose(np.quantile(train.score, .995), cut, rtol=0, atol=2e-12); c.close()
    proof = dict(passed=True, score_report_sha256=sha(root / 'score_report.json'), rows=len(actual),
        valid=int(actual.formula_input_valid.sum()), all_exported_native_arithmetic_sql_replayed=True,
        all_threshold_flags_and_training_quantile_verified=True,
        max_score_difference=float((actual.score - expected.score).abs().max()),
        software_compilation_verified=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(root / 'score_verification.json', proof); return proof


def analyze():
    joint = json.loads((ROOT / 'joint_selection_freeze.json').read_text())
    assert joint['passed'] and joint['protocol_sha256'] == sha(PROTOCOL) and len(joint['selections']) == 3
    row = next(v for v in joint['selections'] if v['root'] == str(base.ROOT))
    assert row['selection_report_sha256'] == sha(base.ROOT / 'selection_report.json')
    return reuse.evaluation.analyze(base.ROOT, base.PROTOCOL)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['sources', 'model', 'verify_model', 'scores', 'verify_scores', 'freeze', 'verify', 'analyze'])
    p.add_argument('--fold', choices=['2024', 'recent', 'combined'], default='2024')
    a = p.parse_args(); setup(a.fold)
    if a.stage == 'sources':
        checked_sources(); ROOT.mkdir(parents=True, exist_ok=True)
        result = dict(passed=True, protocol_sha256=sha(PROTOCOL), implementation_sha256=sha(Path(__file__)),
                      all_eight_components_and_prior_proofs_bound=True, new_model_fitting_performed=False,
                      new_group_scores_read=False, new_2026_prices_read=False)
        save_json(ROOT / 'source_verification.json', result)
    elif a.fold == 'combined' and a.stage != 'analyze':
        assert a.stage in ['freeze', 'verify']
        result = getattr(linkage, 'combine' if a.stage == 'freeze' else 'verify_combined')()
    elif a.stage in ['scores', 'freeze', 'verify']:
        result = getattr(reuse, a.stage)()
    else:
        result = globals()[a.stage]()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
