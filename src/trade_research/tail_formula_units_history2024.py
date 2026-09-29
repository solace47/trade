"""Chronological 2024 falsification of the three fixed price-unit representations."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_dual_units as units
from . import tail_formula_float as original
from . import tail_formula_unscaled_prices as raw
from .corporate_cash import save_json, sha

STEM = 'tail_formula_units_history2024'
ROOT = Path('data/research') / STEM
INPUTS = ROOT / 'inputs'
PROTOCOL = Path('config') / (STEM + '_protocol.json')
HISTORY = Path('data/research/tail_formula_quarter_history2024/inputs')
LONG = Path('data/research/tail_formula_long48/inputs')
CONTROL = HISTORY.parent / 'full'
META = ['date', 'code', 'half', 'board', 'decision_shares', 'formula_input_valid']
ARMS = {
    'constant': {**original.EXPRESSIONS, **{n: '0' for n in units.CONSTANT_FIELDS}},
    'raw': raw.EXPRESSIONS,
    'dual': {**original.EXPRESSIONS, **raw.RAW_FIELDS},
}
ALL_EXPRESSIONS = {**original.EXPRESSIONS, **raw.RAW_FIELDS,
                   **{n: '0' for n in units.CONSTANT_FIELDS}}
HEADER = original.HEADER


def checked():
    p = json.loads(PROTOCOL.read_text())
    assert p['arms'] == ARMS and p['native_header'] == HEADER
    assert p['new_2026_prices_allowed'] is False and p['threshold'] == .995
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    for root in [HISTORY, LONG, original.ROOT]:
        r = json.loads((root / 'feature_report.json').read_text())
        v = json.loads((root / 'feature_verification.json').read_text())
        assert v['passed'] and v['feature_report_sha256'] == sha(root / 'feature_report.json')
        assert r['features_sha256'] == sha(root / 'features.parquet')
    return p


def raw_sources():
    cols = ['date', 'code', *raw.RAW_FIELDS.values()]
    frames = [pd.read_parquet(root / 'features.parquet', columns=cols,
        filters=[('date', '>=', first), ('date', '<', end)])
        for root, first, end in [(LONG, '2023-01-01', '2024-01-01'),
                                 (original.ROOT, '2024-01-01', '2025-01-01')]]
    out = pd.concat(frames, ignore_index=True).sort_values(['date', 'code']).reset_index(drop=True)
    assert not out.duplicated(['date', 'code']).any()
    return out


def features():
    checked(); assert not (INPUTS / 'feature_report.json').exists()
    old = pd.read_parquet(HISTORY / 'features.parquet'); source = raw_sources()
    pd.testing.assert_frame_equal(old[['date', 'code']], source[['date', 'code']], check_exact=True)
    f = old.copy()
    for alias, field in raw.RAW_FIELDS.items():
        f[alias] = source[field]
    for name in units.CONSTANT_FIELDS:
        f[name] = 0.
    assert np.isfinite(f.loc[f.formula_input_valid, list(ALL_EXPRESSIONS)]).all().all()
    INPUTS.mkdir(parents=True, exist_ok=True)
    f.to_parquet(INPUTS / 'features.parquet', index=False, compression='zstd')
    report = dict(protocol_sha256=sha(PROTOCOL), implementation_sha256=sha(Path(__file__)),
        features_sha256=sha(INPUTS / 'features.parquet'), rows=len(f), valid=int(f.formula_input_valid.sum()),
        newly_invalid=0, newly_valid=0, expressions=ALL_EXPRESSIONS, native_header=HEADER,
        first=f.date.min(), last=f.date.max(), only_2023_and_2024_values=True,
        prior_2024_decision_changed_for_exploratory_falsification_not_promotion=True,
        no_new_raw_extraction=True, new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(INPUTS / 'feature_report.json', report)
    return {k: v for k, v in report.items() if k not in ['expressions', 'native_header']}


def verify():
    p = checked(); r = json.loads((INPUTS / 'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['implementation_sha256'] == sha(Path(__file__))
    assert r['features_sha256'] == sha(INPUTS / 'features.parquet')
    f = pd.read_parquet(INPUTS / 'features.parquet'); old = pd.read_parquet(HISTORY / 'features.parquet')
    source = raw_sources(); pd.testing.assert_frame_equal(f[old.columns], old, check_exact=True)
    pd.testing.assert_frame_equal(f[['date', 'code']], source[['date', 'code']], check_exact=True)
    good = f.formula_input_valid
    for alias, field in raw.RAW_FIELDS.items():
        pd.testing.assert_series_equal(f[alias], source[field], check_names=False, check_exact=True)
        np.testing.assert_allclose(f.loc[good, 'N' + field], f.loc[good, alias] / f.loc[good, 'V01'],
                                   rtol=0, atol=2e-12)
    assert f[list(units.CONSTANT_FIELDS)].eq(0).all().all()
    c = base.conn(); c.register('old', old); c.register('raw_inputs', source)
    expressions = {**{n: n for n in original.EXPRESSIONS}, **raw.RAW_FIELDS,
                   **{n: '0' for n in units.CONSTANT_FIELDS}}
    cols = ','.join(f'floor(least(greatest(100*({e})+10000+.000001,0),999999))::INT AS {n}'
                    for n, e in expressions.items())
    expected = c.sql('SELECT date,code,' + cols + ' FROM old JOIN raw_inputs USING(date,code) '
                     'WHERE formula_input_valid ORDER BY date,code').df(); c.close()
    pd.testing.assert_frame_equal(f.loc[good, ['date', 'code']].reset_index(drop=True),
                                  expected[['date', 'code']], check_exact=True)
    for arm, expr in ARMS.items():
        encoded = np.floor(np.clip(100 * f.loc[good, list(expr)].to_numpy(float) + 10000 + .000001,
                                    0, 999999)).astype('int32')
        np.testing.assert_array_equal(encoded, expected[list(expr)].to_numpy(dtype='int32'))
    assert len(f) == p['expected_keys'] and int(good.sum()) == p['expected_valid']
    proof = dict(passed=True, feature_report_sha256=sha(INPUTS / 'feature_report.json'), rows=len(f),
        effective_input_intersection_unchanged=True, all_original_48_values_and_metadata_exact=True,
        all_raw_aliases_exact_and_normalized_relationship_verified=True, all_three_integer_arrays_sql_verified=True,
        all_nineteen_constants_zero=True, only_2023_and_2024_values=True,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(INPUTS / 'feature_verification.json', proof)
    native = dict(passed=True, protocol_sha256=sha(PROTOCOL),
        feature_report_sha256=sha(INPUTS / 'feature_report.json'),
        feature_verification_sha256=sha(INPUTS / 'feature_verification.json'),
        original_normalized_header_and_existing_percentage_alias_proofs_reused=True,
        no_new_raw_extraction=True, software_compilation_verified=False, native_source_parity_verified=False,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(INPUTS / 'native_input_verification.json', native)
    return dict(feature=proof, native=native)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('stage', choices=['features', 'verify'])
    print(json.dumps(globals()[p.parse_args().stage](), ensure_ascii=False, indent=2))
