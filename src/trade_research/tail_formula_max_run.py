"""Longest strict up/down positive-volume run inside the fixed late window."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as previous
from . import tail_formula_volume_response as cached
from .corporate_cash import MINUTES, save_json, sha

STEM = 'tail_formula_max_run'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')


def native_parts():
    header = previous.HEADER
    additions = {}
    for side, op, output in [('U', '>', 'LR01'), ('D', '<', 'LR02')]:
        names = []
        for i in range(29):
            name = f'RR{side}{i:02d}'
            offset = 28 - i
            condition = f'REF(C,{offset}){op}REF(C,{offset+1}) AND REF(V,{offset})>0'
            value = '1' if not names else names[-1] + '+1'
            header += f'{name}:=IF({condition},{value},0);\n'
            names.append(name)
        maximum = names[0]
        for name in names[1:]:
            maximum = f'MAX({maximum},{name})'
        additions[output] = f'VALUEWHEN(TIME=1449,{maximum})'
    return header, {**previous.EXPRESSIONS, **additions}


HEADER, EXPRESSIONS = native_parts()


def longest_runs(prices, volume):
    assert prices.ndim == volume.ndim == 2 and prices.shape[1] == 30 and volume.shape[1] == 29
    assert len(prices) == len(volume)
    cents = np.rint(prices * 100)
    answer = []
    for flags in [(cents[:, 1:] > cents[:, :-1]) & (volume > 0),
                  (cents[:, 1:] < cents[:, :-1]) & (volume > 0)]:
        current = np.zeros(len(prices), dtype='int32')
        best = current.copy()
        for column in flags.T:
            current = np.where(column, current + 1, 0)
            best = np.maximum(best, current)
        answer.append(best)
    return np.column_stack(answer)


def native_run_values(close, volume):
    """Interpret the generated REF/IF assignments, including their actual offsets."""
    assert len(close) == 30 and len(volume) == 29
    values = {}
    pattern = (r'(RR[UD]\d{2}):=IF\(REF\(C,(\d+)\)([<>])REF\(C,(\d+)\) '
               r'AND REF\(V,(\d+)\)>0,(1|RR[UD]\d{2}\+1),0\);')
    for line in HEADER[len(previous.HEADER):].splitlines():
        match = re.fullmatch(pattern, line)
        assert match is not None
        name, now, op, prior, vol, update = match.groups()
        left, right = close[-1-int(now)], close[-1-int(prior)]
        condition = (left > right if op == '>' else left < right) and volume[-1-int(vol)] > 0
        values[name] = (1 if update == '1' else values[update[:-2]] + 1) if condition else 0
    assert len(values) == 58
    answer = []
    for name in ['LR01', 'LR02']:
        expression = EXPRESSIONS[name]
        prefix = 'VALUEWHEN(TIME=1449,'
        assert expression.startswith(prefix) and expression.endswith(')')
        tokens = iter(re.findall(r'[A-Z][A-Z0-9]*|[(),]', expression[len(prefix):-1]))

        def parse():
            token = next(tokens)
            if token != 'MAX':
                return values[token]
            assert next(tokens) == '('
            left = parse()
            assert next(tokens) == ','
            right = parse()
            assert next(tokens) == ')'
            return max(left, right)

        answer.append(parse())
        assert next(tokens, None) is None
    return answer


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    assert p['transitions'] == 29 and p['expected_features'] == 50
    assert not p['new_2026_prices_allowed']
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    cached.checked_sources()
    r = json.loads((cached.ROOT / 'feature_report.json').read_text())
    v = json.loads((cached.ROOT / 'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(cached.ROOT / 'feature_report.json')
    assert r['features_sha256'] == sha(cached.ROOT / 'features.parquet')
    return p


def windows(old):
    d = cached.cached_windows(old[['date', 'code']])
    prices = d[cached.price_source.PRICE_COLUMNS].to_numpy(float)
    volume = d[cached.volume_source.COLUMNS['v']].to_numpy(float)
    np.testing.assert_allclose(prices[:, 1:], d[cached.volume_source.COLUMNS['c']].to_numpy(float),
                               rtol=0, atol=0, equal_nan=True)
    return d, prices, volume


def features():
    p = checked_sources()
    assert not (ROOT / 'feature_report.json').exists()
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    d, prices, volume = windows(old)
    source = pd.read_parquet(cached.ROOT / 'features.parquet', columns=['date', 'code', 'volume_response_valid'])
    pd.testing.assert_frame_equal(old[['date', 'code']], source[['date', 'code']], check_exact=True)
    valid = source.volume_response_valid
    values = longest_runs(prices, volume)
    f = old.copy()
    f['LR01'] = pd.Series(values[:, 0]).where(valid)
    f['LR02'] = pd.Series(values[:, 1]).where(valid)
    f['max_run_valid'] = valid
    f['prior_formula_input_valid'] = old.formula_input_valid
    f['formula_input_valid'] &= valid
    assert len(f) == p['expected_keys'] and int(f.formula_input_valid.sum()) == p['expected_valid']
    assert f.date.between(p['signal_first'], p['signal_last']).all()
    ROOT.mkdir(parents=True, exist_ok=True)
    f.to_parquet(ROOT / 'features.parquet', index=False, compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL), source_hashes=p['source_hashes'],
        features_sha256=sha(ROOT / 'features.parquet'), rows=len(f), valid=int(f.formula_input_valid.sum()),
        newly_invalid=int((old.formula_input_valid & ~f.formula_input_valid).sum()),
        expressions=EXPRESSIONS, native_header=HEADER, software_compilation_verified=False,
        native_source_parity_verified=False, new_group_outcomes_read=False, new_2026_prices_read=False,
        no_exit_rules=True)
    save_json(ROOT / 'feature_report.json', r)
    return {k: v for k, v in r.items() if k not in ['expressions', 'native_header']}


def verify_features():
    p = checked_sources()
    r = json.loads((ROOT / 'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['features_sha256'] == sha(ROOT / 'features.parquet')
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    got = pd.read_parquet(ROOT / 'features.parquet')
    pd.testing.assert_frame_equal(got[old.columns.drop('formula_input_valid')],
                                  old.drop(columns='formula_input_valid'), check_exact=True)
    d, prices, volume = windows(old)
    valid = (d.pv_bars.eq(30) & d.pv_clocks.eq(30) & d.pv_good_bars.eq(30)
        & d.mp_bars.eq(29) & d.mp_clocks.eq(29) & d.mp_good_bars.eq(29)
        & np.isfinite(prices).all(axis=1) & (prices > 0).all(axis=1)
        & np.isfinite(volume).all(axis=1) & (volume >= 0).all(axis=1)
        & (volume == np.floor(volume)).all(axis=1) & (volume.sum(axis=1) > 0) & old.V01.gt(0))
    np.testing.assert_array_equal(got.max_run_valid, valid)
    np.testing.assert_array_equal(got.formula_input_valid, old.formula_input_valid & valid)
    c = base.conn()
    c.register('source', d)
    bits = []
    for name, op in [('up_bits', '>'), ('down_bits', '<')]:
        terms = [f'(CASE WHEN try_cast(round(100*pv_c{i:02d}) AS BIGINT){op}'
                 f'try_cast(round(100*pv_c{i-1:02d}) AS BIGINT) AND mp_v{i:02d}>0 '
                 f'THEN {2**(i-21)}::BIGINT ELSE 0::BIGINT END)' for i in range(21, 50)]
        bits.append('(' + '+'.join(terms) + ') AS ' + name)
    c.execute('CREATE TEMP TABLE runs AS SELECT date,code,' + ','.join(bits) +
              ',0::INT AS LR01,0::INT AS LR02 FROM source')
    for _ in range(29):
        c.execute('''UPDATE runs SET LR01=LR01+CAST(up_bits>0 AS INT),
            LR02=LR02+CAST(down_bits>0 AS INT),up_bits=up_bits & (up_bits << 1),
            down_bits=down_bits & (down_bits << 1)''')
    expected = c.sql('SELECT date,code,LR01,LR02 FROM runs ORDER BY date,code').df()
    assert c.sql('SELECT max(up_bits),max(down_bits) FROM runs').fetchone() == (0, 0)
    c.close()
    pd.testing.assert_frame_equal(expected[['date', 'code']], got[['date', 'code']], check_exact=True)
    for name in ['LR01', 'LR02']:
        value = expected[name].where(valid)
        np.testing.assert_allclose(got[name], value, rtol=0, atol=0, equal_nan=True)
        np.testing.assert_array_equal(np.floor(np.clip(100*got.loc[valid, name]+10000+.000001, 0, 999999)),
                                      10000 + 100*expected.loc[valid, name])
    names = re.findall(r'(?m)^([A-Z][A-Z0-9]*):=', HEADER) + list(EXPRESSIONS)
    assert len(names) == len(set(names))
    assert r['expressions'] == EXPRESSIONS and r['native_header'] == HEADER
    assert len(got) == r['rows'] and int(got.formula_input_valid.sum()) == r['valid']
    proof = dict(passed=True, feature_report_sha256=sha(ROOT / 'feature_report.json'), rows=len(got),
        all_integer_price_directions_positive_volumes_bitmaps_and_run_lengths_rebuilt=True,
        all_original48_values_validity_and_integer_encodings_verified=True,
        effective_input_intersection_unchanged=bool(got.formula_input_valid.equals(old.formula_input_valid)),
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_verification.json', proof)
    return proof


def native():
    checked_sources()
    proof = json.loads((ROOT / 'feature_verification.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256'] == sha(ROOT / 'feature_report.json')
    f = pd.read_parquet(ROOT / 'features.parquet')
    f = f.loc[f.formula_input_valid].copy()
    f['probe_hash'] = [hashlib.sha256((d+'|'+c+'|max-run-v1').encode()).hexdigest() for d, c in zip(f.date, f.code)]
    selected = f.sort_values('probe_hash').groupby('half', sort=True).head(8)
    assert len(selected) == 32
    price_manifest = json.loads((cached.price_source.ROOT / 'window_report.json').read_text())['source_sha256']
    volume_manifest = json.loads((cached.volume_source.ROOT / 'window_report.json').read_text())['source_sha256']
    cases = []
    for row in selected.itertuples():
        path = MINUTES / row.code[:2].upper() / (row.code[3:] + '.parquet')
        assert sha(path) == price_manifest[str(path)] == volume_manifest[str(path)]
        start, end = [pd.Timestamp(row.date + ' ' + time) for time in ['14:20', '14:49']]
        raw = pd.read_parquet(path, columns=['timestamp', 'close', 'volume'],
            filters=[('timestamp', '>=', start), ('timestamp', '<=', end)]).sort_values('timestamp')
        assert raw.timestamp.tolist() == pd.date_range(start, end, freq='min').tolist()
        cents = [int(round(x*100)) for x in raw.close]
        assert all(abs(x-y/100) <= .0001 for x, y in zip(raw.close, cents))
        results = native_run_values(cents, raw.volume.iloc[1:].tolist())
        assert results == [row.LR01, row.LR02]
        cases.append(dict(date=row.date, code=row.code, source_sha256=price_manifest[str(path)],
                          raw_bars=30, up=results[0], down=results[1]))
    r = dict(passed=True, protocol_sha256=sha(PROTOCOL),
        feature_report_sha256=sha(ROOT / 'feature_report.json'),
        feature_verification_sha256=sha(ROOT / 'feature_verification.json'),
        fixed_samples=32, raw_bars=960, samples=cases,
        all_bounded_native_assignments_and_raw_run_lengths_verified=True,
        software_compilation_verified=False, native_source_parity_verified=False,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'native_input_verification.json', r)
    return {k: v for k, v in r.items() if k != 'samples'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['features', 'verify_features', 'native'])
    print(json.dumps(globals()[parser.parse_args().stage](), ensure_ascii=False, indent=2))
