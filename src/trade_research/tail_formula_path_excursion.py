"""Past-window ordered close-price drawdown and rebound, never future labels."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as previous
from . import tail_formula_path_variance as source
from .corporate_cash import MINUTES, save_json, sha

STEM = 'tail_formula_path_excursion'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
ADDED_HEADER = 'PEB:=BARSLAST(TIME=1420)+1;\nPED:=1-C/HHV(C,PEB);\nPEU:=C/LLV(C,PEB)-1;\n'
ADDED_EXPRESSIONS = {'PE01': '100*VALUEWHEN(TIME=1449,HHV(PED,30))/V01',
                     'PE02': '100*VALUEWHEN(TIME=1449,HHV(PEU,30))/V01'}
HEADER = previous.HEADER + ADDED_HEADER
EXPRESSIONS = {**previous.EXPRESSIONS, **ADDED_EXPRESSIONS}
WINDOW_COLUMNS = ['date', 'code', 'pv_bars', 'pv_clocks', 'pv_good_bars', *source.PRICE_COLUMNS]


def ordered_excursions(prices, atr):
    assert prices.ndim == 2 and prices.shape[1] == 30 and len(prices) == len(atr)
    cents = np.rint(prices * 100)
    with np.errstate(divide='ignore', invalid='ignore'):
        down = (1 - cents / np.maximum.accumulate(cents, axis=1)).max(axis=1)
        up = (cents / np.minimum.accumulate(cents, axis=1) - 1).max(axis=1)
        return np.column_stack([100 * down / atr, 100 * up / atr])


def native_values(clocks, cents, atr):
    """Scalar BARSLAST and rolling HHV/LLV semantics with VALUEWHEN capture."""
    anchor = None
    downs, ups = [], []
    answer = None
    for i, clock in enumerate(clocks):
        if clock == 1420:
            anchor = i
        if anchor is None:
            downs.append(float('nan'))
            ups.append(float('nan'))
            continue
        lookback = i - anchor + 1
        prefix = cents[i-lookback+1:i+1]
        downs.append(1 - cents[i] / max(prefix))
        ups.append(cents[i] / min(prefix) - 1)
        if clock == 1449:
            assert lookback == 30
            answer = [100*max(downs[-30:])/atr, 100*max(ups[-30:])/atr]
    assert answer is not None
    return answer


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    assert p['prices'] == 30 and p['expected_features'] == 50
    assert not p['new_2026_prices_allowed']
    assert p['native_header'] == ADDED_HEADER and p['expressions'] == ADDED_EXPRESSIONS
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    for root in [previous.ROOT, source.ROOT]:
        r = json.loads((root/'feature_report.json').read_text())
        v = json.loads((root/'feature_verification.json').read_text())
        assert v['passed'] and v['feature_report_sha256'] == sha(root/'feature_report.json')
        assert r['features_sha256'] == sha(root/'features.parquet')
    r = json.loads((source.ROOT/'feature_report.json').read_text())
    assert r['window_report_sha256'] == sha(source.ROOT/'window_report.json')
    return p


def features():
    p = checked_sources()
    assert not (ROOT/'feature_report.json').exists()
    old = pd.read_parquet(previous.ROOT/'features.parquet')
    d = pd.read_parquet(source.ROOT/'features.parquet', columns=WINDOW_COLUMNS)
    pd.testing.assert_frame_equal(old[['date', 'code']], d[['date', 'code']], check_exact=True)
    prices = d[source.PRICE_COLUMNS].to_numpy(float)
    valid = (d.pv_bars.eq(30) & d.pv_clocks.eq(30) & d.pv_good_bars.eq(30)
             & np.isfinite(prices).all(axis=1) & (prices > 0).all(axis=1)
             & np.isfinite(old.V01) & old.V01.gt(0))
    values = ordered_excursions(prices, old.V01.to_numpy(float))
    f = old.copy()
    for i, name in enumerate(ADDED_EXPRESSIONS):
        f[name] = pd.Series(values[:, i]).where(valid)
    f['path_excursion_valid'] = valid
    f['prior_formula_input_valid'] = old.formula_input_valid
    f['formula_input_valid'] &= valid
    assert len(f) == p['expected_keys'] and int(f.formula_input_valid.sum()) == p['expected_valid']
    assert f.date.between(p['signal_first'], p['signal_last']).all()
    ROOT.mkdir(parents=True, exist_ok=True)
    f.to_parquet(ROOT/'features.parquet', index=False, compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL), source_hashes=p['source_hashes'],
        features_sha256=sha(ROOT/'features.parquet'), rows=len(f), valid=int(f.formula_input_valid.sum()),
        newly_invalid=int((old.formula_input_valid & ~f.formula_input_valid).sum()),
        expressions=EXPRESSIONS, native_header=HEADER, software_compilation_verified=False,
        native_source_parity_verified=False, new_group_outcomes_read=False,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT/'feature_report.json', r)
    return {k: v for k, v in r.items() if k not in ['expressions', 'native_header']}


def verify_features():
    checked_sources()
    r = json.loads((ROOT/'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['features_sha256'] == sha(ROOT/'features.parquet')
    old = pd.read_parquet(previous.ROOT/'features.parquet')
    got = pd.read_parquet(ROOT/'features.parquet')
    pd.testing.assert_frame_equal(got[old.columns.drop('formula_input_valid')],
                                  old.drop(columns='formula_input_valid'), check_exact=True)
    w = json.loads((source.ROOT/'window_report.json').read_text())
    for file, digest in w['parts_sha256'].items():
        assert sha(Path(file)) == digest
    d = pd.concat([pd.read_parquet(file, columns=WINDOW_COLUMNS) for file in w['parts_sha256']], ignore_index=True)
    c = base.conn()
    c.register('raw_windows', d)
    c.register('old', old[['date', 'code', 'V01', 'formula_input_valid']])
    aliases = ','.join(f'try_cast(round(100*{name}) AS BIGINT) AS k{i}' for i, name in enumerate(source.PRICE_COLUMNS))
    good = ' AND '.join(f'isfinite({name}) AND {name}>0' for name in source.PRICE_COLUMNS)
    c.execute(f'''CREATE TEMP TABLE windows AS SELECT date,code,V01,{aliases},
        coalesce(pv_bars=30 AND pv_clocks=30 AND pv_good_bars=30 AND {good}
        AND isfinite(V01) AND V01>0,false) AS valid,formula_input_valid AS old_valid
        FROM old LEFT JOIN raw_windows USING(date,code)''')
    # All 435 strictly ordered pairs; zero is the 30 diagonal pairs.
    down = ','.join(f'1.0-k{j}::DOUBLE/k{i}' for j in range(1, 30) for i in range(j))
    up = ','.join(f'k{j}::DOUBLE/k{i}-1.0' for j in range(1, 30) for i in range(j))
    expected = c.sql(f'''SELECT date,code,valid,old_valid AND valid AS formula_input_valid,
        CASE WHEN valid THEN 100*greatest(0.0,{down})/V01 END AS PE01,
        CASE WHEN valid THEN 100*greatest(0.0,{up})/V01 END AS PE02
        FROM windows ORDER BY date,code''').df()
    c.close()
    pd.testing.assert_frame_equal(got[['date', 'code']], expected[['date', 'code']], check_exact=True)
    np.testing.assert_array_equal(got.path_excursion_valid, expected.valid)
    np.testing.assert_array_equal(got.formula_input_valid, expected.formula_input_valid)
    for name in ADDED_EXPRESSIONS:
        np.testing.assert_allclose(got[name], expected[name], rtol=0, atol=2e-12, equal_nan=True)
        valid = got.formula_input_valid
        encode = lambda values: np.floor(np.clip(100*values+10000+.000001, 0, 999999))
        np.testing.assert_array_equal(encode(got.loc[valid, name]), encode(expected.loc[valid, name]))
    names = re.findall(r'(?m)^([A-Z][A-Z0-9]*):=', HEADER) + list(EXPRESSIONS)
    assert len(names) == len(set(names))
    assert r['expressions'] == EXPRESSIONS and r['native_header'] == HEADER
    proof = dict(passed=True, feature_report_sha256=sha(ROOT/'feature_report.json'), rows=len(got),
        all_435_ordered_pairs_and_diagonal_zero_independently_rebuilt=True,
        original48_values_and_all_effective_integer_encodings_verified=True,
        effective_input_intersection_unchanged=bool(got.formula_input_valid.equals(old.formula_input_valid)),
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT/'feature_verification.json', proof)
    return proof


def native():
    checked_sources()
    v = json.loads((ROOT/'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(ROOT/'feature_report.json')
    f = pd.read_parquet(ROOT/'features.parquet')
    f = f.loc[f.formula_input_valid].copy()
    f['probe_hash'] = [hashlib.sha256((d+'|'+c+'|path-excursion-v1').encode()).hexdigest()
                       for d, c in zip(f.date, f.code)]
    selected = f.sort_values('probe_hash').groupby('half', sort=True).head(8)
    assert len(selected) == 32
    manifest = json.loads((source.ROOT/'window_report.json').read_text())['source_sha256']
    cases = []
    for row in selected.itertuples():
        path = MINUTES/row.code[:2].upper()/(row.code[3:]+'.parquet')
        assert sha(path) == manifest[str(path)]
        start, end = [pd.Timestamp(row.date+' '+t) for t in ['14:20', '14:49']]
        raw = pd.read_parquet(path, columns=['timestamp', 'close'],
            filters=[('timestamp', '>=', start), ('timestamp', '<=', end)]).sort_values('timestamp')
        assert raw.timestamp.tolist() == pd.date_range(start, end, freq='min').tolist()
        cents = [int(round(x*100)) for x in raw.close]
        assert all(abs(x-y/100) <= .0001 for x, y in zip(raw.close, cents))
        clocks = list(range(1420, 1450))
        values = native_values(clocks, cents, row.V01)
        np.testing.assert_allclose(values, [row.PE01, row.PE02], rtol=0, atol=2e-12)
        perturbed = native_values([1418, 1419]+clocks+[1450, 1451],
                                 [100000000, 1]+cents+[1, 100000000], row.V01)
        np.testing.assert_allclose(perturbed, values, rtol=0, atol=0)
        cases.append(dict(date=row.date, code=row.code, source_sha256=manifest[str(path)],
                          raw_bars=30, down=values[0], up=values[1]))
    r = dict(passed=True, protocol_sha256=sha(PROTOCOL), feature_report_sha256=sha(ROOT/'feature_report.json'),
        feature_verification_sha256=sha(ROOT/'feature_verification.json'), fixed_samples=32, raw_bars=960,
        samples=cases, original_minute_prices_and_native_rolling_semantics_verified=True,
        before_window_and_after_cutoff_perturbations=32,
        software_compilation_verified=False, native_source_parity_verified=False,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT/'native_input_verification.json', r)
    return {k: v for k, v in r.items() if k != 'samples'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['features', 'verify_features', 'native'])
    print(json.dumps(globals()[parser.parse_args().stage](), ensure_ascii=False, indent=2))
