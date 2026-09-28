"""Append five signal-weekday indicators without changing historical price inputs."""
import argparse
from datetime import date
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as previous
from .corporate_cash import save_json, sha

STEM = 'tail_formula_weekday'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
ADDITIONS = {f'WD{i:02d}': f'IF(MOD(DATETODAY(DATE)+3,7)={i},1,0)' for i in range(1, 6)}
EXPRESSIONS = {**previous.EXPRESSIONS, **ADDITIONS}
HEADER = previous.HEADER


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    assert p['expressions'] == ADDITIONS and p['weekdays'] == [1, 2, 3, 4, 5]
    assert p['expected_features'] == 53 and not p['new_2026_prices_allowed']
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    r = json.loads((previous.ROOT / 'feature_report.json').read_text())
    v = json.loads((previous.ROOT / 'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(previous.ROOT / 'feature_report.json')
    assert r['features_sha256'] == sha(previous.ROOT / 'features.parquet')
    return p


def features():
    p = checked_sources()
    assert not (ROOT / 'feature_report.json').exists()
    f = pd.read_parquet(previous.ROOT / 'features.parquet')
    assert not any(name in f for name in ADDITIONS)
    assert f.date.between('2024-01-01', '2025-12-31').all()
    weekday = pd.to_datetime(f.date).dt.dayofweek + 1
    assert weekday.isin(p['weekdays']).all()
    for i, name in enumerate(ADDITIONS, 1):
        f[name] = weekday.eq(i).astype('int8')
    assert len(f) == p['expected_keys'] and int(f.formula_input_valid.sum()) == p['expected_valid']
    ROOT.mkdir(parents=True, exist_ok=True)
    f.to_parquet(ROOT / 'features.parquet', index=False, compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL), source_hashes=p['source_hashes'],
        features_sha256=sha(ROOT / 'features.parquet'), rows=len(f),
        valid=int(f.formula_input_valid.sum()), newly_invalid=0,
        signal_dates=f.date.nunique(), expressions=EXPRESSIONS, native_header=HEADER,
        software_compilation_verified=False, native_source_parity_verified=False,
        original_input_validity_unchanged=True, new_group_outcomes_read=False,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_report.json', r)
    return {k: v for k, v in r.items() if k not in ['expressions', 'native_header']}


def verify_features():
    p = checked_sources()
    r = json.loads((ROOT / 'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['features_sha256'] == sha(ROOT / 'features.parquet')
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    got = pd.read_parquet(ROOT / 'features.parquet')
    pd.testing.assert_frame_equal(got[old.columns], old, check_exact=True)
    c = base.conn()
    c.read_parquet(str(previous.ROOT / 'features.parquet')).create_view('original')
    fields = ','.join(f'CAST(isodow(date::DATE)={i} AS INT) AS WD{i:02d}' for i in range(1, 6))
    rebuilt = c.sql('SELECT date,code,' + fields + ' FROM original ORDER BY date,code').df()
    c.close()
    pd.testing.assert_frame_equal(got[['date', 'code', *ADDITIONS]], rebuilt, check_dtype=False, check_exact=True)
    assert got[list(ADDITIONS)].sum(axis=1).eq(1).all()
    encode = lambda a: np.floor(np.clip(100 * a.astype(float) + 10000 + .000001, 0, 999999))
    np.testing.assert_array_equal(encode(got[list(ADDITIONS)]), 10000 + 100 * rebuilt[list(ADDITIONS)])
    names = re.findall(r'(?m)^([A-Z][A-Z0-9]*):=', HEADER) + list(EXPRESSIONS)
    assert len(names) == len(set(names))
    assert r['expressions'] == EXPRESSIONS and r['native_header'] == HEADER
    assert len(got) == r['rows'] == p['expected_keys']
    assert int(got.formula_input_valid.sum()) == r['valid'] == p['expected_valid']
    proof = dict(passed=True, feature_report_sha256=sha(ROOT / 'feature_report.json'),
        rows=len(got), all_original_columns_and_validity_unchanged=True,
        all_weekday_flags_and_encodings_independently_rebuilt=True,
        effective_input_intersection_unchanged=True, new_group_outcomes_read=False,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_verification.json', proof)
    # Independently interpret the native DATE year code and documented day-zero.
    # This verifies arithmetic, not the full client's price feed or compilation.
    anchor = date(1990, 12, 19)
    assert anchor.weekday() == 2
    dates = sorted(got.date.unique())
    by_day = {}
    for text in dates:
        encoded_date = int(text.replace('-', '')) - 19000000
        native_date = date(encoded_date // 10000 + 1900, encoded_date // 100 % 100, encoded_date % 100)
        assert native_date.isoformat() == text
        days = (native_date - anchor).days
        value = (days + 3) % 7
        assert value == native_date.isoweekday() and 1 <= value <= 5
        by_day[text] = value
    for i, name in enumerate(ADDITIONS, 1):
        expected = got.date.map(by_day).eq(i).astype(int)
        np.testing.assert_array_equal(got[name], expected)
        np.testing.assert_array_equal(encode(got[name]), 10000 + 100 * expected)
    native = dict(passed=True, protocol_sha256=sha(PROTOCOL),
        feature_report_sha256=sha(ROOT / 'feature_report.json'),
        feature_verification_sha256=sha(ROOT / 'feature_verification.json'),
        source_sha256=sha(ROOT / 'source/tdx_calendar.json'),
        signal_dates=len(dates), rows=len(got), original_native_prefix_unchanged=True,
        all_date_decoding_day_differences_modulo_and_flags_verified=True,
        native_calendar_arithmetic_verified=True, software_compilation_verified=False,
        native_source_parity_verified=False, new_group_outcomes_read=False,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'native_input_verification.json', native)
    return dict(features=proof, native=native)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['features', 'verify_features'])
    print(json.dumps(globals()[p.parse_args().stage](), ensure_ascii=False, indent=2))
