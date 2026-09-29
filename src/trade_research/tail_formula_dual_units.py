"""Keep both percentage and ATR units, with a same-width constant control."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as previous
from . import tail_formula_unscaled_prices as raw
from .corporate_cash import save_json, sha

STEM = 'tail_formula_dual_units'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
RAW_FIELDS = raw.RAW_FIELDS
CONSTANT_FIELDS = {'Z' + name[1:]: name for name in RAW_FIELDS}
ARM_FIELDS = {'constant': list(CONSTANT_FIELDS), 'dual': list(RAW_FIELDS)}
NEW_EXPRESSIONS = {**RAW_FIELDS, **{name: '0' for name in CONSTANT_FIELDS}}
ALL_EXPRESSIONS = {**previous.EXPRESSIONS, **NEW_EXPRESSIONS}
EXPRESSIONS = ALL_EXPRESSIONS
HEADER = previous.HEADER


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    assert p['arm_fields'] == ARM_FIELDS and p['new_expressions'] == NEW_EXPRESSIONS
    assert p['native_header'] == HEADER and p['fit_columns_per_arm'] == 67
    assert not p['new_2026_prices_allowed']
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    for module in [raw, previous]:
        r = json.loads((module.ROOT / 'feature_report.json').read_text())
        v = json.loads((module.ROOT / 'feature_verification.json').read_text())
        assert v['passed'] and v['feature_report_sha256'] == sha(module.ROOT / 'feature_report.json')
        assert r['features_sha256'] == sha(module.ROOT / 'features.parquet')
    n = json.loads((raw.ROOT / 'native_input_verification.json').read_text())
    assert n['passed'] and n['feature_report_sha256'] == sha(raw.ROOT / 'feature_report.json')
    return p


def features():
    checked_sources(); assert not (ROOT / 'feature_report.json').exists()
    f = pd.read_parquet(raw.ROOT / 'features.parquet')
    for name in CONSTANT_FIELDS:
        f[name] = 0.
    ROOT.mkdir(parents=True, exist_ok=True)
    f.to_parquet(ROOT / 'features.parquet', index=False, compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL), implementation_sha256=sha(Path(__file__)),
        features_sha256=sha(ROOT / 'features.parquet'), rows=len(f), valid=int(f.formula_input_valid.sum()),
        newly_invalid=0, newly_valid=0, expressions=ALL_EXPRESSIONS, native_header=HEADER,
        arm_fields=ARM_FIELDS, fit_columns_per_arm=67, no_new_raw_price_extraction=True,
        software_compilation_verified=False, native_source_parity_verified=False,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_report.json', r)
    return {k: v for k, v in r.items() if k not in ['expressions', 'native_header']}


def verify_features():
    p = checked_sources(); r = json.loads((ROOT / 'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['implementation_sha256'] == sha(Path(__file__))
    assert r['features_sha256'] == sha(ROOT / 'features.parquet')
    old = pd.read_parquet(previous.ROOT / 'features.parquet'); source = pd.read_parquet(raw.ROOT / 'features.parquet')
    got = pd.read_parquet(ROOT / 'features.parquet')
    pd.testing.assert_frame_equal(got[old.columns], old, check_exact=True)
    pd.testing.assert_frame_equal(got[source.columns], source, check_exact=True)
    assert got[list(CONSTANT_FIELDS)].eq(0).all().all()
    c = base.conn(); c.register('old', old)
    expressions = {**{n: n for n in previous.EXPRESSIONS}, **RAW_FIELDS, **{n: '0' for n in CONSTANT_FIELDS}}
    columns = ','.join(f'floor(least(greatest(100*({expr})+10000+.000001,0),999999))::INT AS {n}' for n, expr in expressions.items())
    expected = c.sql('SELECT date,code,' + columns + ' FROM old WHERE formula_input_valid ORDER BY date,code').df()
    c.close(); good = got.formula_input_valid
    pd.testing.assert_frame_equal(got.loc[good, ['date', 'code']].reset_index(drop=True), expected[['date', 'code']], check_exact=True)
    for arm, names in ARM_FIELDS.items():
        columns = [*previous.EXPRESSIONS, *names]
        assert len(columns) == 67
        encoded = np.floor(np.clip(100 * got.loc[good, columns].to_numpy(float) + 10000 + .000001, 0, 999999)).astype('int32')
        np.testing.assert_array_equal(encoded, expected[columns].to_numpy(dtype='int32'))
    assert len(got) == r['rows'] == p['expected_keys'] and int(good.sum()) == r['valid'] == p['expected_valid']
    declarations = re.findall(r'(?m)^([A-Z][A-Z0-9]*):=', HEADER) + list(ALL_EXPRESSIONS)
    assert len(declarations) == len({n.casefold() for n in declarations})
    proof = dict(passed=True, feature_report_sha256=sha(ROOT / 'feature_report.json'), rows=len(got),
        effective_input_intersection_unchanged=True, all_original_and_unscaled_columns_unchanged=True,
        both_67_column_arrays_independently_encoded=True, nineteen_constant_fields_all_zero=True,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_verification.json', proof); return proof


def native():
    checked_sources(); v = json.loads((ROOT / 'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(ROOT / 'feature_report.json')
    assert HEADER == previous.HEADER
    assert all(NEW_EXPRESSIONS[name] == field for name, field in RAW_FIELDS.items())
    assert all(NEW_EXPRESSIONS[name] == '0' for name in CONSTANT_FIELDS)
    proof = dict(passed=True, protocol_sha256=sha(PROTOCOL), feature_report_sha256=sha(ROOT / 'feature_report.json'),
        feature_verification_sha256=sha(ROOT / 'feature_verification.json'),
        previous_normalized_and_raw_alias_proofs_reused=True, only_new_math_is_nineteen_zero_literals=True,
        no_new_raw_price_extraction=True, software_compilation_verified=False, native_source_parity_verified=False,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'native_input_verification.json', proof); return proof


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['features', 'verify_features', 'native'])
    print(json.dumps(globals()[p.parse_args().stage](), ensure_ascii=False, indent=2))
