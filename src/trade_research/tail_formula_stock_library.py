"""Join verified stock-only feature blocks without changing their definitions."""
import argparse
import importlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as previous
from .corporate_cash import save_json, sha

STEM = 'tail_formula_stock_library'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
BLOCKS = ['opening', 'overnight', 'volume_history', 'vwap', 'path_variance',
          'extrema_time', 'minute_pressure', 'history_weight', 'prior_day',
          'history_direction', 'morning_history', 'daily_efficiency',
          'volume_concentration', 'volume_response', 'history_turnover']
TOKEN = re.compile(r'\b[A-Za-z_][A-Za-z_0-9]*\b')
DECL = re.compile(r'(?m)^([A-Z][A-Z0-9]*):=')


def rename(text, mapping):
    return TOKEN.sub(lambda m: mapping.get(m[0], m[0]), text)


def namespace(blocks):
    """Give both colliding features and auxiliary variables disjoint names."""
    expressions = dict(previous.EXPRESSIONS)
    header = previous.HEADER
    mappings = []
    count = 0
    for number, (name, source_expressions, source_header) in enumerate(blocks, 1):
        assert source_header.startswith(previous.HEADER)
        suffix = source_header[len(previous.HEADER):]
        helper_names = DECL.findall(suffix)
        assert len(set(helper_names)) == len(helper_names)
        mapping = {n: f'U{number:02d}{n}' for n in helper_names}
        for feature in source_expressions:
            count += 1
            mapping[feature] = f'X{count:02d}'
        assert len(set(mapping.values())) == len(mapping)
        header += rename(suffix, mapping)
        for feature, expression in source_expressions.items():
            expressions[mapping[feature]] = rename(expression, mapping)
        mappings.append(dict(block=name, aliases=mapping,
                             features={n: mapping[n] for n in source_expressions}))
    names = DECL.findall(header) + list(expressions)
    assert len({n.casefold() for n in names}) == len(names)
    assert max(map(len, names)) <= 16
    return expressions, header, mappings


SOURCES = []
for _name in BLOCKS:
    _module = importlib.import_module('trade_research.tail_formula_' + _name)
    _new = {k: v for k, v in _module.EXPRESSIONS.items() if k not in previous.EXPRESSIONS}
    SOURCES.append(dict(name=_name, root=_module.ROOT, module=Path(_module.__file__),
                        protocol=_module.PROTOCOL, expressions=_new, header=_module.HEADER))
EXPRESSIONS, HEADER, MAPPINGS = namespace([(s['name'], s['expressions'], s['header']) for s in SOURCES])
assert len(EXPRESSIONS) == 82


def source_record(s):
    root = s['root']
    return dict(block=s['name'], root=str(root), columns=list(s['expressions']),
        source_module=str(s['module'].relative_to(Path.cwd())), source_module_sha256=sha(s['module']),
        source_protocol=str(s['protocol']), source_protocol_sha256=sha(s['protocol']),
        feature_report_sha256=sha(root / 'feature_report.json'),
        feature_verification_sha256=sha(root / 'feature_verification.json'))


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    assert p['blocks'] == BLOCKS and p['expected_features'] == 82
    assert p['sources'] == [source_record(s) for s in SOURCES]
    assert p['previous_feature_report_sha256'] == sha(previous.ROOT / 'feature_report.json')
    assert p['previous_feature_verification_sha256'] == sha(previous.ROOT / 'feature_verification.json')
    for root in [previous.ROOT] + [s['root'] for s in SOURCES]:
        report = json.loads((root / 'feature_report.json').read_text())
        proof = json.loads((root / 'feature_verification.json').read_text())
        assert proof['passed'] and proof['feature_report_sha256'] == sha(root / 'feature_report.json')
        assert report['features_sha256'] == sha(root / 'features.parquet')
    return p


def features():
    assert not (ROOT / 'feature_report.json').exists(), 'Do not replace frozen library inputs'
    p = checked_sources()
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    f = old.copy()
    f['prior_formula_input_valid'] = old.formula_input_valid
    coverage = []
    for s, mapping in zip(SOURCES, MAPPINGS):
        columns = ['date', 'code', 'formula_input_valid', *s['expressions']]
        source = pd.read_parquet(s['root'] / 'features.parquet', columns=columns)
        pd.testing.assert_frame_equal(old[['date', 'code']], source[['date', 'code']], check_exact=True)
        assert not source.duplicated(['date', 'code']).any()
        valid_name = 'valid_' + s['name']
        f[valid_name] = source.formula_input_valid
        f['formula_input_valid'] &= source.formula_input_valid
        for original, alias in mapping['features'].items():
            assert alias not in f
            f[alias] = source[original]
        coverage.append(dict(block=s['name'], valid=int(source.formula_input_valid.sum()),
                             cumulative_valid=int(f.formula_input_valid.sum())))
    f['formula_input_valid'] &= np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
    assert f.date.between(p['signal_first'], p['signal_last']).all()
    ROOT.mkdir(parents=True, exist_ok=True)
    f.to_parquet(ROOT / 'features.parquet', index=False, compression='zstd')
    report = dict(protocol_sha256=sha(PROTOCOL), sources=p['sources'],
        previous_feature_report_sha256=p['previous_feature_report_sha256'],
        features_sha256=sha(ROOT / 'features.parquet'), rows=len(f),
        previous_valid=int(old.formula_input_valid.sum()), valid=int(f.formula_input_valid.sum()),
        newly_invalid=int((old.formula_input_valid & ~f.formula_input_valid).sum()),
        block_coverage=coverage, mappings=MAPPINGS, expressions=EXPRESSIONS, native_header=HEADER,
        source_definitions_unchanged=True, target_not_joined=True,
        software_compilation_verified=False, native_source_parity_verified=False,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_report.json', report)
    return {k:v for k,v in report.items() if k not in ['expressions', 'native_header', 'mappings', 'sources']}


def verify_features():
    p = checked_sources(); r = json.loads((ROOT / 'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['sources'] == p['sources']
    assert r['features_sha256'] == sha(ROOT / 'features.parquet')
    assert r['mappings'] == MAPPINGS and r['expressions'] == EXPRESSIONS and r['native_header'] == HEADER
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    f = pd.read_parquet(ROOT / 'features.parquet')
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')], old.drop(columns='formula_input_valid'), check_exact=True)
    pd.testing.assert_series_equal(f.prior_formula_input_valid, old.formula_input_valid, check_exact=True, check_names=False)
    valid = old.formula_input_valid.to_numpy().copy()
    c = base.conn(); c.register('keys', old[['date', 'code']])
    checks = 0
    for s, mapping in zip(SOURCES, MAPPINGS):
        source = pd.read_parquet(s['root'] / 'features.parquet',
            columns=['date', 'code', 'formula_input_valid', *s['expressions']])
        # Project case-sensitive Pandas names before registering in SQL.
        source = source.rename(columns=mapping['features'])
        c.register('source', source)
        expected = c.sql('SELECT source.* FROM keys LEFT JOIN source USING(date,code) ORDER BY keys.date,keys.code').df()
        pd.testing.assert_frame_equal(f[['date', 'code']], expected[['date', 'code']], check_exact=True)
        np.testing.assert_array_equal(f['valid_' + s['name']], expected.formula_input_valid)
        valid &= expected.formula_input_valid.to_numpy()
        for original, alias in mapping['features'].items():
            np.testing.assert_allclose(f[alias], expected[alias], rtol=0, atol=0, equal_nan=True)
            mask = np.isfinite(expected[alias])
            encoded = c.sql(f'SELECT floor(least(greatest(100*{alias}+10000+.000001,0),999999))::BIGINT AS x FROM source WHERE isfinite({alias}) ORDER BY date,code').df().x
            np.testing.assert_array_equal(encoded, np.floor(np.clip(100*f.loc[mask, alias]+10000+.000001, 0, 999999)))
            inverse = {v:k for k,v in mapping['aliases'].items()}
            assert rename(EXPRESSIONS[alias], inverse) == s['expressions'][original]
            checks += len(f)
    c.close()
    valid &= np.isfinite(f[list(EXPRESSIONS)]).all(axis=1).to_numpy()
    np.testing.assert_array_equal(f.formula_input_valid, valid)
    assert len(f) == r['rows'] and int(valid.sum()) == r['valid']
    assert f.isST.eq(0).all() and f.tradestatus.eq(1).all()
    assert not f.code.str[3:].str.startswith(('92', '688', '300', '301')).any()
    proof = dict(passed=True, feature_report_sha256=sha(ROOT / 'feature_report.json'), rows=len(f),
        copied_scalar_checks=checks, all_source_keys_values_and_encodings_rebuilt=True,
        all_48_previous_fields_unchanged=True, exact_input_intersection_rebuilt=True,
        native_renaming_invertible=True, native_namespace_unique=True,
        raw_minute_replay_not_repeated=True, software_compilation_verified=False,
        native_source_parity_verified=False, new_group_outcomes_read=False,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_verification.json', proof)
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['features', 'verify_features'])
    print(json.dumps(globals()[parser.parse_args().stage](), ensure_ascii=False, indent=2))
