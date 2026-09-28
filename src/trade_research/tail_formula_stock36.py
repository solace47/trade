"""Keep stock price/volume and float turnover while removing all index inputs."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as previous
from . import tail_formula_volatility as stock
from .corporate_cash import save_json, sha

STEM = 'tail_formula_stock36'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
EXPRESSIONS = {**stock.EXPRESSIONS, **previous.NEW_EXPRESSIONS}
HEADER = stock.HEADER + 'PSH:=REF(FINANCE(7),B0);\n'


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    for module in [previous, stock]:
        r = json.loads((module.ROOT / 'feature_report.json').read_text())
        v = json.loads((module.ROOT / 'feature_verification.json').read_text())
        assert v['passed'] and v['feature_report_sha256'] == sha(module.ROOT / 'feature_report.json')
        assert r['features_sha256'] == sha(module.ROOT / 'features.parquet')
    assert len(EXPRESSIONS) == p['expected_features'] == 36
    assert p['removed'] == [f'{prefix}{i:02d}' for prefix in ['I', 'J', 'R'] for i in range(1, 5)]
    assert EXPRESSIONS == {k: v for k, v in previous.EXPRESSIONS.items() if k not in p['removed']}
    return p


def features():
    p = checked_sources(); assert not (ROOT / 'feature_report.json').exists()
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    single = pd.read_parquet(stock.ROOT / 'features.parquet', columns=['date', 'code', 'formula_input_valid'])
    pd.testing.assert_frame_equal(old[['date', 'code']], single[['date', 'code']], check_exact=True)
    valid = single.formula_input_valid & old.float_source_valid & np.isfinite(old[list(previous.NEW_EXPRESSIONS)]).all(axis=1)
    f = old.copy(); f['prior_formula_input_valid'] = old.formula_input_valid; f['formula_input_valid'] = valid
    ROOT.mkdir(parents=True, exist_ok=True); f.to_parquet(ROOT / 'features.parquet', index=False, compression='zstd')
    report = dict(protocol_sha256=sha(PROTOCOL), source_hashes=p['source_hashes'],
        features_sha256=sha(ROOT / 'features.parquet'), rows=len(f), valid=int(valid.sum()),
        newly_invalid=int((old.formula_input_valid & ~valid).sum()), newly_valid=int((~old.formula_input_valid & valid).sum()),
        expressions=EXPRESSIONS, native_header=HEADER, no_new_price_extraction=True,
        software_compilation_verified=False, native_FINANCE7_parity_verified=False,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_report.json', report)
    return {k: v for k, v in report.items() if k not in ['expressions', 'native_header', 'source_hashes']}


def verify_features():
    p = checked_sources(); r = json.loads((ROOT / 'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['source_hashes'] == p['source_hashes']
    assert r['features_sha256'] == sha(ROOT / 'features.parquet')
    f = pd.read_parquet(ROOT / 'features.parquet'); old = pd.read_parquet(previous.ROOT / 'features.parquet')
    single = pd.read_parquet(stock.ROOT / 'features.parquet', columns=['date', 'code', 'formula_input_valid', *stock.EXPRESSIONS])
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')], old.drop(columns='formula_input_valid'), check_exact=True)
    pd.testing.assert_frame_equal(f[['date', 'code', *stock.EXPRESSIONS]], single[['date', 'code', *stock.EXPRESSIONS]], check_exact=True)
    c = base.conn(); c.register('old', old); c.register('stock_valid', single[['date', 'code', 'formula_input_valid']].rename(columns={'formula_input_valid': 'stock_valid'}))
    q = c.sql('''SELECT date,code,stock_valid AND float_source_valid AND isfinite(S01) AND isfinite(S02) AND isfinite(S03) AS valid
        FROM old JOIN stock_valid USING(date,code) ORDER BY date,code''').df()
    np.testing.assert_array_equal(f.formula_input_valid, q.valid)
    columns = ','.join(f'floor(least(greatest(100*{name}+10000+.000001,0),999999))::INT AS {name}' for name in EXPRESSIONS)
    encoded = c.sql(f'SELECT {columns} FROM old WHERE formula_input_valid ORDER BY date,code').df(); c.close()
    good = f.formula_input_valid
    unchanged = bool(good.equals(old.formula_input_valid))
    if unchanged:
        actual = np.floor(np.clip(100*f.loc[good, list(EXPRESSIONS)]+10000+.000001, 0, 999999)).astype('int32')
        np.testing.assert_array_equal(actual.to_numpy(), encoded.to_numpy())
    code = HEADER + '\n'.join(f'{k}:={v};' for k, v in EXPRESSIONS.items())
    assert not any(token in code for token in ['INDEXC', 'INSUM(', 'ICP', 'IM20', 'IX14'])
    names = re.findall(r'(?m)^([A-Z][A-Z0-9]*):=', code)
    assert len(names) == len(set(names))
    for removed in p['removed']:
        assert not re.search(r'\b'+removed+r'\b', code)
    assert len(f) == r['rows'] and int(good.sum()) == r['valid']
    proof = dict(passed=True, feature_report_sha256=sha(ROOT / 'feature_report.json'), rows=len(f),
        all_existing_values_and_keys_unchanged=True, independent_33_plus_float_validity_rebuilt=True,
        effective_input_intersection_unchanged=unchanged, projected_36_integer_inputs_sql_rebuilt=unchanged,
        no_direct_or_relative_index_variables_in_numeric_formula=True, all_native_variable_names_unique=True,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_verification.json', proof); return proof


def native():
    checked_sources(); proof = json.loads((ROOT / 'feature_verification.json').read_text())
    assert proof['passed'] and proof['effective_input_intersection_unchanged']
    assert proof['feature_report_sha256'] == sha(ROOT / 'feature_report.json')
    r = dict(passed=True, protocol_sha256=sha(PROTOCOL), feature_report_sha256=sha(ROOT / 'feature_report.json'),
        feature_verification_sha256=sha(ROOT / 'feature_verification.json'),
        previous_individual_input_proofs_reused=True, no_new_individual_input_math=True,
        no_INDEXC_or_index_expression_dependencies=True, software_compilation_verified=False,
        native_FINANCE7_parity_verified=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'native_input_verification.json', r); return r


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['features', 'verify_features', 'native']); args = parser.parse_args()
    print(json.dumps(globals()[args.stage](), ensure_ascii=False, indent=2))
