"""Use only stock price/volume inputs with no index or float-capital dependency."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_volatility as context
from . import tail_formula_float as previous
from .corporate_cash import save_json, sha

STEM = 'tail_formula_price_volume_only'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
REMOVED = [name for name in previous.EXPRESSIONS if name not in context.EXPRESSIONS]
EXPRESSIONS = {**context.EXPRESSIONS, **{name: '0' for name in REMOVED}}
HEADER = context.HEADER


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    assert p['removed'] == REMOVED and p['constant_placeholder'] == 0
    assert p['fit_columns'] == 48 and p['informative_columns'] == 33
    assert p['expressions'] == EXPRESSIONS and p['native_header'] == HEADER
    assert not p['new_2026_prices_allowed']
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    for root in [previous.ROOT, context.ROOT]:
        r = json.loads((root / 'feature_report.json').read_text())
        v = json.loads((root / 'feature_verification.json').read_text())
        assert v['passed'] and v['feature_report_sha256'] == sha(root / 'feature_report.json')
        assert r['features_sha256'] == sha(root / 'features.parquet')
    assert list(EXPRESSIONS) == list(previous.EXPRESSIONS)
    assert list(EXPRESSIONS)[33:] == REMOVED
    assert not any(token in HEADER for token in ['FINANCE', 'PSH', 'INDEXC', 'INSUM('])
    return p


def features():
    checked_sources(); assert not (ROOT / 'feature_report.json').exists()
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    standalone = pd.read_parquet(context.ROOT / 'features.parquet', columns=['date', 'code', 'formula_input_valid'])
    pd.testing.assert_frame_equal(old[['date', 'code']], standalone[['date', 'code']], check_exact=True)
    f = old.copy(); f['prior_formula_input_valid'] = old.formula_input_valid
    f['formula_input_valid'] = standalone.formula_input_valid
    for name in REMOVED: f[name] = 0.
    ROOT.mkdir(parents=True, exist_ok=True)
    f.to_parquet(ROOT / 'features.parquet', index=False, compression='zstd')
    report = dict(protocol_sha256=sha(PROTOCOL), implementation_sha256=sha(Path(__file__)),
        features_sha256=sha(ROOT / 'features.parquet'), rows=len(f), valid=int(f.formula_input_valid.sum()),
        newly_invalid=int((old.formula_input_valid & ~f.formula_input_valid).sum()),
        newly_valid=int((~old.formula_input_valid & f.formula_input_valid).sum()),
        expressions=EXPRESSIONS, native_header=HEADER, removed=REMOVED, informative_columns=33, fit_columns=48,
        no_index_or_float_validity_dependency=True, no_new_raw_price_extraction=True,
        software_compilation_verified=False, native_source_parity_verified=False,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_report.json', report)
    return {k: v for k, v in report.items() if k not in ['expressions', 'native_header']}


def verify_features():
    p = checked_sources(); r = json.loads((ROOT / 'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['implementation_sha256'] == sha(Path(__file__))
    assert r['features_sha256'] == sha(ROOT / 'features.parquet')
    old = pd.read_parquet(previous.ROOT / 'features.parquet'); got = pd.read_parquet(ROOT / 'features.parquet')
    native = pd.read_parquet(context.ROOT / 'features.parquet', columns=['date', 'code', 'formula_input_valid', *context.EXPRESSIONS])
    keep = old.columns.drop(['formula_input_valid', *REMOVED])
    pd.testing.assert_frame_equal(got[keep], old[keep], check_exact=True)
    pd.testing.assert_frame_equal(got[native.columns], native, check_exact=True)
    pd.testing.assert_series_equal(got.prior_formula_input_valid, old.formula_input_valid, check_names=False, check_exact=True)
    assert got[REMOVED].eq(0).all().all()
    c = base.conn(); c.register('public_inputs', native)
    columns = ','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::INT AS {n}' for n in context.EXPRESSIONS)
    constants = ','.join('10000::INT AS ' + name for name in REMOVED)
    expected = c.sql('SELECT date,code,' + columns + ',' + constants +
                     ' FROM public_inputs WHERE formula_input_valid ORDER BY date,code').df(); c.close()
    valid = got.formula_input_valid
    pd.testing.assert_frame_equal(got.loc[valid, ['date', 'code']].reset_index(drop=True), expected[['date', 'code']], check_exact=True)
    encoded = np.floor(np.clip(100*got.loc[valid, list(EXPRESSIONS)].to_numpy(float)+10000+.000001, 0, 999999)).astype('int32')
    np.testing.assert_array_equal(encoded, expected[list(EXPRESSIONS)].to_numpy(dtype='int32'))
    assert r['rows'] == len(got) == p['expected_keys'] and r['valid'] == int(valid.sum()) == p['expected_valid']
    assert r['newly_invalid'] == r['newly_valid'] == 0 and valid.equals(old.formula_input_valid)
    names = re.findall(r'(?m)^([A-Z][A-Z0-9]*):=', HEADER) + list(EXPRESSIONS)
    assert len(names) == len({n.casefold() for n in names})
    assert not any(token in HEADER + ''.join(EXPRESSIONS.values()) for token in ['FINANCE', 'PSH', 'INDEXC', 'INSUM('])
    proof = dict(passed=True, feature_report_sha256=sha(ROOT / 'feature_report.json'), rows=len(got),
        effective_input_intersection_unchanged=True, all_original_nonremoved_values_and_keys_unchanged=True,
        thirty_three_source_columns_and_standalone_validity_exactly_reused=True,
        all_integer_inputs_and_fifteen_constant_placeholders_independently_rebuilt=True,
        no_index_or_float_capital_fields_in_numeric_formula=True,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_verification.json', proof); return proof


def native():
    checked_sources(); v = json.loads((ROOT / 'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(ROOT / 'feature_report.json')
    assert HEADER == context.HEADER and {n: EXPRESSIONS[n] for n in context.EXPRESSIONS} == context.EXPRESSIONS
    assert all(EXPRESSIONS[n] == '0' for n in REMOVED)
    proof = dict(passed=True, protocol_sha256=sha(PROTOCOL), feature_report_sha256=sha(ROOT / 'feature_report.json'),
        feature_verification_sha256=sha(ROOT / 'feature_verification.json'),
        previous_33_input_proofs_and_header_reused=True, new_math_is_fifteen_literal_zeroes_only=True,
        no_INDEXC_FINANCE7_or_PSH_dependency_in_numeric_formula=True, no_new_raw_price_extraction=True,
        software_compilation_verified=False, native_source_parity_verified=False,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'native_input_verification.json', proof); return proof


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['features', 'verify_features', 'native'])
    print(json.dumps(globals()[p.parse_args().stage](), ensure_ascii=False, indent=2))
