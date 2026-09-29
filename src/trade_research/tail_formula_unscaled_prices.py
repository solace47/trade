"""Expose nineteen existing stock price movements in percentage units."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_complete as original
from . import tail_formula_float as previous
from . import tail_formula_volatility as scaled
from .corporate_cash import save_json, sha

STEM = 'tail_formula_unscaled_prices'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
REPLACEMENTS = {'N' + name: 'U' + name for name in scaled.NORMALIZED}
RAW_FIELDS = {'U' + name: name for name in scaled.NORMALIZED}
EXPRESSIONS = {REPLACEMENTS.get(name, name): RAW_FIELDS[REPLACEMENTS[name]]
               if name in REPLACEMENTS else expr for name, expr in previous.EXPRESSIONS.items()}
HEADER = previous.HEADER


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    assert p['replacements'] == REPLACEMENTS and p['raw_fields'] == RAW_FIELDS
    assert p['expressions'] == EXPRESSIONS and p['native_header'] == HEADER
    assert p['changed_columns'] == 19 and p['fit_columns'] == len(EXPRESSIONS) == 48
    assert not p['new_2026_prices_allowed']
    for path, digest in p['source_hashes'].items():
        assert sha(Path(path)) == digest
    for module in [original, previous]:
        root = module.ROOT
        r = json.loads((root / 'feature_report.json').read_text())
        v = json.loads((root / 'feature_verification.json').read_text())
        assert v['passed'] and v['feature_report_sha256'] == sha(root / 'feature_report.json')
        assert r['features_sha256'] == sha(root / 'features.parquet')
    assert list(EXPRESSIONS) == [REPLACEMENTS.get(name, name) for name in previous.EXPRESSIONS]
    return p


def features():
    checked_sources(); assert not (ROOT / 'feature_report.json').exists()
    old = pd.read_parquet(previous.ROOT / 'features.parquet'); f = old.copy()
    for alias, field in RAW_FIELDS.items():
        f[alias] = old[field]
    f['prior_formula_input_valid'] = old.formula_input_valid
    finite = np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
    assert finite.loc[old.formula_input_valid].all()
    ROOT.mkdir(parents=True, exist_ok=True)
    f.to_parquet(ROOT / 'features.parquet', index=False, compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL), implementation_sha256=sha(Path(__file__)),
        features_sha256=sha(ROOT / 'features.parquet'), rows=len(f), valid=int(f.formula_input_valid.sum()),
        newly_invalid=0, newly_valid=0, expressions=EXPRESSIONS, native_header=HEADER,
        raw_fields=RAW_FIELDS, all_old_values_preserved=True, no_new_raw_price_extraction=True,
        V01_and_relative_index_inputs_unchanged=True, input_representation_not_new_information=True,
        software_compilation_verified=False, native_source_parity_verified=False,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_report.json', r)
    return {k: v for k, v in r.items() if k not in ['expressions', 'native_header']}


def verify_features():
    p = checked_sources(); r = json.loads((ROOT / 'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['implementation_sha256'] == sha(Path(__file__))
    assert r['features_sha256'] == sha(ROOT / 'features.parquet')
    old = pd.read_parquet(previous.ROOT / 'features.parquet'); got = pd.read_parquet(ROOT / 'features.parquet')
    raw = pd.read_parquet(original.ROOT / 'features.parquet', columns=['date', 'code', *RAW_FIELDS.values()])
    pd.testing.assert_frame_equal(got[old.columns], old, check_exact=True)
    pd.testing.assert_frame_equal(got[raw.columns], raw, check_exact=True)
    for alias, name in RAW_FIELDS.items():
        pd.testing.assert_series_equal(got[alias], raw[name], check_names=False, check_exact=True)
    good = got.formula_input_valid
    for name in scaled.NORMALIZED:
        np.testing.assert_allclose(got.loc[good, 'N' + name], raw.loc[good, name] / got.loc[good, 'V01'],
                                   rtol=0, atol=2e-12)
    c = base.conn(); c.register('old', old)
    columns = ','.join(f'floor(least(greatest(100*{RAW_FIELDS.get(n,n)}+10000+.000001,0),999999))::INT AS {n}'
                       for n in EXPRESSIONS)
    expected = c.sql('SELECT date,code,' + columns + ' FROM old WHERE formula_input_valid ORDER BY date,code').df()
    c.close()
    encoded = np.floor(np.clip(100 * got.loc[good, list(EXPRESSIONS)].to_numpy(float) + 10000 + .000001, 0, 999999)).astype('int32')
    pd.testing.assert_frame_equal(got.loc[good, ['date', 'code']].reset_index(drop=True), expected[['date', 'code']], check_exact=True)
    np.testing.assert_array_equal(encoded, expected[list(EXPRESSIONS)].to_numpy(dtype='int32'))
    assert len(got) == r['rows'] == p['expected_keys'] and int(good.sum()) == r['valid'] == p['expected_valid']
    declarations = re.findall(r'(?m)^([A-Z][A-Z0-9]*):=', HEADER) + list(EXPRESSIONS)
    assert len(declarations) == len({n.casefold() for n in declarations})
    assert set(RAW_FIELDS.values()).issubset(re.findall(r'(?m)^([A-Z][A-Z0-9]*):=', HEADER))
    assert not set(REPLACEMENTS).intersection(EXPRESSIONS)
    proof = dict(passed=True, feature_report_sha256=sha(ROOT / 'feature_report.json'), rows=len(got),
        effective_input_intersection_unchanged=True, all_original_values_keys_and_validity_unchanged=True,
        nineteen_unscaled_fields_reused_from_independently_verified_32_inputs=True,
        normalized_counterparts_and_V01_relationship_verified=True,
        all_48_integer_inputs_independently_rebuilt=True, native_aliases_defined_once=True,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_verification.json', proof); return proof


def native():
    checked_sources(); v = json.loads((ROOT / 'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(ROOT / 'feature_report.json')
    assert HEADER == previous.HEADER
    assert all(EXPRESSIONS[alias] == field for alias, field in RAW_FIELDS.items())
    proof = dict(passed=True, protocol_sha256=sha(PROTOCOL), feature_report_sha256=sha(ROOT / 'feature_report.json'),
        feature_verification_sha256=sha(ROOT / 'feature_verification.json'),
        original_raw_percentage_and_other_input_proofs_reused=True,
        new_expressions_only_alias_existing_header_variables=True, no_new_raw_price_extraction=True,
        software_compilation_verified=False, native_source_parity_verified=False,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'native_input_verification.json', proof); return proof


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['features', 'verify_features', 'native'])
    print(json.dumps(globals()[p.parse_args().stage](), ensure_ascii=False, indent=2))
