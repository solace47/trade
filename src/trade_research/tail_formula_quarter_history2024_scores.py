"""Export and independently replay the frozen chronological 2024 models."""
import argparse
import ast
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as original
from . import tail_formula_polynomial_ridge as arithmetic
from . import tail_formula_quarter_history2024_model as model
from . import tail_formula_quarter_minimum as minimum
from .corporate_cash import save_json, sha

META = ['date', 'code', 'half', 'board', 'decision_shares', 'formula_input_valid']


def native_cores(m):
    full = base.native_core(m['full'], m['thresholds']['full']['threshold'],
                            original.EXPRESSIONS, original.HEADER)
    partial = dict(components=m['components'], thresholds=[m['thresholds']['minimum']])
    lower = minimum.native_core(partial)
    minimum.verify_native(lower, partial)
    sources = [full, lower]
    prefixes, bodies, endings = [], [], []
    for source, marker in zip(sources, ['T01:=', 'QMT001:=']):
        prefix, tail = source.split(marker, 1)
        body, ending = (marker + tail).rsplit('CORE:', 1)
        prefixes.append(prefix); bodies.append(body); endings.append('CORE:' + ending)
    assert prefixes[0] == prefixes[1]
    first = re.sub(r'\bSC\b', 'ORSC', bodies[0])
    second = re.sub(r'\bSC\b', 'MISC', bodies[1])
    both = prefixes[0] + first + second + (
        f"CORE:(ORSC>{m['thresholds']['full']['threshold']:.17e}) AND "
        f"(MISC>{m['thresholds']['minimum']['threshold']:.17e});\n")
    assert prefixes[0] + re.sub(r'\bORSC\b', 'SC', first) + endings[0] == full
    assert prefixes[1] + re.sub(r'\bMISC\b', 'SC', second) + endings[1] == lower
    names = re.findall(r'(?m)^([A-Z][A-Z0-9]*):=', both)
    assert len(names) == len({n.casefold() for n in names})
    assert len(re.findall(r'(?m)^T[0-9]{2}:=', first)) == 64
    assert len(re.findall(r'(?m)^QMT[0-9]{3}:=', second)) == 256
    return dict(full=full, minimum=lower, both=both)


def full_definitions(core, m):
    pairs = re.findall(r'^([A-Z][A-Z0-9]*):=([^;]+);$', core, re.M)
    d = dict(pairs)
    assert len(pairs) == len({n.casefold() for n, _ in pairs})
    for i, name in enumerate(original.EXPRESSIONS, 1):
        assert d[f'X{i:02d}'] == f'INTPART(MIN(MAX(100*{name}+10000+0.000001,0),999999))'
    for i, tree in enumerate(m['full']['trees'], 1):
        minimum.check_tree(ast.parse(d[f'T{i:02d}'], mode='eval').body, tree)
    terms = arithmetic.flatten_addition(ast.parse(d['SC'], mode='eval').body)
    assert arithmetic.number_node(terms[0]) == m['full']['bias']
    assert [term.id for term in terms[1:]] == [f'T{i:02d}' for i in range(1, 65)]
    assert core.endswith('CORE:SC>' + format(m['thresholds']['full']['threshold'], '.17g') + ';\n')
    return d


def checked():
    p, m = model.checked_model()
    proof = json.loads((base.ROOT / 'model_verification.json').read_text())
    assert proof['passed'] and proof['model_report_sha256'] == sha(base.ROOT / 'model_report.json')
    assert proof['all_five_components_node_statistics_independently_rebuilt']
    return p, m


def scores():
    _, m = checked(); root = base.ROOT
    assert not (root / 'score_report.json').exists()
    f = base.feature_inputs(); valid = f.formula_input_valid
    x = base.encode(f.loc[valid]); out = f[META].copy()
    out['full_score'] = np.nan; out['minimum_score'] = np.nan
    out.loc[valid, 'full_score'] = base.predict(x, m['full'])
    out.loc[valid, 'minimum_score'] = minimum.predict(x, m)
    assert all(np.isfinite(out.loc[valid, side + '_score']).all() for side in ['full', 'minimum'])
    out.to_parquet(root / 'scores.parquet', index=False, compression='zstd')
    cores = {}
    for name, text in native_cores(m).items():
        file = root / (name + '_numeric_core.tdx'); file.write_text(text)
        cores[str(file)] = sha(file)
    report = dict(protocol_sha256=sha(base.PROTOCOL), master_protocol_sha256=sha(model.PROTOCOL),
        implementation_sha256=sha(Path(__file__)), model_report_sha256=sha(root / 'model_report.json'),
        model_verification_sha256=sha(root / 'model_verification.json'),
        feature_report_sha256=sha(base.FEATURES / 'feature_report.json'),
        scores_sha256=sha(root / 'scores.parquet'), native_cores=cores, rows=len(out), valid=int(valid.sum()),
        software_compilation_verified=False, new_2024_test_group_outcomes_read=False,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(root / 'score_report.json', report); return report


def verify_scores():
    p, m = checked(); root = base.ROOT
    report = json.loads((root / 'score_report.json').read_text())
    for key, path in [('protocol_sha256', base.PROTOCOL), ('master_protocol_sha256', model.PROTOCOL),
        ('implementation_sha256', Path(__file__)), ('model_report_sha256', root / 'model_report.json'),
        ('model_verification_sha256', root / 'model_verification.json'),
        ('feature_report_sha256', base.FEATURES / 'feature_report.json'), ('scores_sha256', root / 'scores.parquet')]:
        assert report[key] == sha(path)
    texts = native_cores(m)
    for name, text in texts.items():
        path = root / (name + '_numeric_core.tdx')
        assert path.read_text() == text and report['native_cores'][str(path)] == sha(path)
    full = full_definitions(texts['full'], m)
    lower = minimum.verify_native(texts['minimum'], dict(components=m['components'], thresholds=[m['thresholds']['minimum']]))
    c = base.conn(); arithmetic.register_features(c)
    sql = lambda text: minimum.expression_sql(ast.parse(text, mode='eval').body)
    encoded = ','.join(sql(full[f'X{i:02d}']) + f'::INT AS X{i:02d}' for i in range(1, 49))
    c.sql('SELECT date,code,' + encoded + ' FROM features WHERE formula_input_valid').create_view('encoded')
    c.sql('SELECT date,code,' + ','.join(sql(full[f'T{i:02d}']) + f' AS T{i:02d}' for i in range(1, 65))
          + ' FROM encoded').create_view('full_leaves')
    c.sql('SELECT date,code,' + sql(full['SC']) + ' AS full_score FROM full_leaves').create_view('full_rebuilt')
    minimum.score_views(c, lower)
    expected = c.sql('SELECT f.date,f.code,f.half,f.board,f.decision_shares,f.formula_input_valid,'
        'a.full_score,b.score AS minimum_score FROM features f LEFT JOIN full_rebuilt a USING(date,code) '
        'LEFT JOIN rebuilt b USING(date,code) ORDER BY date,code').df()
    actual = pd.read_parquet(root / 'scores.parquet')
    pd.testing.assert_frame_equal(actual[META], expected[META], check_exact=True)
    differences = {}
    arithmetic.training_sql(c, p); c.register('replayed', expected)
    training = c.sql('SELECT r.full_score,r.minimum_score FROM replayed r JOIN training t USING(date,code) ORDER BY date,code').df()
    assert len(training) == m['rows']
    for name in ['full', 'minimum']:
        col = name + '_score'; cut = m['thresholds'][name]['threshold']
        np.testing.assert_allclose(actual[col], expected[col], rtol=0, atol=2e-12, equal_nan=True)
        assert actual[col].notna().equals(actual.formula_input_valid)
        np.testing.assert_array_equal(actual[col].gt(cut), expected[col].gt(cut))
        np.testing.assert_allclose(np.quantile(training[col], .995), cut, rtol=0, atol=2e-12)
        differences[name] = float((actual[col] - expected[col]).abs().max())
    c.close()
    proof = dict(passed=True, score_report_sha256=sha(root / 'score_report.json'), rows=len(actual),
        valid=int(actual.formula_input_valid.sum()), all_320_exported_trees_structurally_verified=True,
        both_parent_native_scores_independently_sql_replayed=True, all_threshold_flags_and_training_quantiles_verified=True,
        both_core_restores_both_parent_cores_exactly=True, max_score_differences=differences,
        software_compilation_verified=False, new_2024_test_group_outcomes_read=False,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(root / 'score_verification.json', proof); return proof


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['scores', 'verify_scores'])
    p.add_argument('--fold', choices=['h1', 'h2'], required=True)
    a = p.parse_args(); model.setup(a.fold)
    print(json.dumps(globals()[a.stage](), ensure_ascii=False, indent=2))
