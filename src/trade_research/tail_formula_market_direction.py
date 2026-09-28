"""Separate excess stock moves during rising and falling index minutes."""
import argparse
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as previous
from . import tail_formula_path_variance as stock
from .corporate_cash import MINUTES, save_json, sha

STEM = 'tail_formula_market_direction'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
INDEX_REPORT = Path('data/research/tail_formula_context/source_report.json')
PROBES = Path('data/research/tail_formula_volume_response/native_input_verification.json')
POINTS = [f'ib_p{i}' for i in range(199, 228)]
NEW_EXPRESSIONS = {'DU01': 'DUD/V01', 'DU02': 'DUU/V01'}
EXPRESSIONS = {**previous.EXPRESSIONS, **NEW_EXPRESSIONS}
EXTRA_HEADER = ('DUCR:=100*LN(C/REF(C,1));\nDUIR:=100*LN(INDEXC/REF(INDEXC,1));\n'
    'DUD:=VALUEWHEN(TIME=1448,SUM(IF(DUIR<0,DUCR-DUIR,0),28));\n'
    'DUU:=VALUEWHEN(TIME=1448,SUM(IF(DUIR>0,DUCR-DUIR,0),28));\n')
HEADER = previous.HEADER + EXTRA_HEADER
native_core = base.native_core


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    for folder in [previous.ROOT, stock.ROOT]:
        r = json.loads((folder / 'feature_report.json').read_text())
        v = json.loads((folder / 'feature_verification.json').read_text())
        assert v['passed'] and v['feature_report_sha256'] == sha(folder / 'feature_report.json')
        assert r['features_sha256'] == sha(folder / 'features.parquet')
    assert p['return_bars'] == 28 and p['index_first_sequence'] == 199 and p['index_last_sequence'] == 227
    assert p['expressions'] == NEW_EXPRESSIONS and p['new_fields'] == list(NEW_EXPRESSIONS)
    assert not p['new_2026_prices_allowed']
    return p


def source_sessions():
    report = json.loads(INDEX_REPORT.read_text())
    rows = []
    seen = set()
    for item in report['sessions']:
        date, code = item['date'], item['symbol']
        assert '2024-01-01' <= date <= '2025-12-30' and code in ['sh.000001', 'sz.399001']
        assert (date, code) not in seen
        seen.add((date, code))
        if 'path' in item:
            path = Path(item['path'])
            assert sha(path) == item['sha256']
            data = json.loads(path.read_text())[:228]
        else:
            data = []
        rows.append((date, code, data))
    return rows


def index_windows():
    result = []
    for date, code, data in source_sessions():
        valid = len(data) == 228 and all(x['sequence'] == i and np.isfinite(x['price_raw'])
            and x['price_raw'] > 0 and x['price_raw'] == math.floor(x['price_raw']) for i, x in enumerate(data))
        item = dict(date=date, index_code=code, ib_index_valid=valid)
        item.update({name: data[i]['price_raw']/100 if valid else np.nan for i, name in zip(range(199, 228), POINTS)})
        result.append(item)
    return pd.DataFrame(result).sort_values(['date', 'index_code']).reset_index(drop=True)


def stock_windows(keys):
    r = json.loads((stock.ROOT / 'window_report.json').read_text())
    names = ['date', 'code', 'pv_bars', 'pv_clocks', *stock.PRICE_COLUMNS[:-1]]
    pieces = []
    for file, digest in r['parts_sha256'].items():
        assert sha(Path(file)) == digest
        pieces.append(pd.read_parquet(file, columns=names))
    raw = pd.concat(pieces, ignore_index=True)
    assert not raw.duplicated(['date', 'code']).any()
    return keys.merge(raw, on=['date', 'code'], how='left', validate='one_to_one').sort_values(['date', 'code']).reset_index(drop=True)


def measure(prices, indices, atr):
    prices, indices, atr = np.asarray(prices, float), np.asarray(indices, float), np.asarray(atr, float)
    assert prices.ndim == indices.ndim == 2 and prices.shape == indices.shape and prices.shape[1] == 29
    assert atr.shape == (len(prices),)
    valid = np.isfinite(prices).all(axis=1) & (prices > 0).all(axis=1)
    valid &= np.isfinite(indices).all(axis=1) & (indices > 0).all(axis=1) & np.isfinite(atr) & (atr > 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        x = 100 * np.log(indices[:, 1:]/indices[:, :-1])
        y = 100 * np.log(prices[:, 1:]/prices[:, :-1])
        down, up = x < 0, x > 0
        downside = np.where(down, y-x, 0).sum(axis=1)
        upside = np.where(up, y-x, 0).sum(axis=1)
        d, u = downside/atr, upside/atr
    valid &= np.isfinite(d) & np.isfinite(u)
    return dict(valid=valid, du_down_sum=downside, du_up_sum=upside,
        du_down_count=down.sum(axis=1), du_up_count=up.sum(axis=1),
        DU01=np.where(valid, d, np.nan), DU02=np.where(valid, u, np.nan))


def features():
    p = checked_sources()
    assert not (ROOT / 'feature_report.json').exists()
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    indices = index_windows()
    d = stock_windows(old[['date', 'code']])
    d = d.merge(old[['date', 'code', 'index_code', 'V01']], on=['date', 'code'], how='left', validate='one_to_one')
    d = d.merge(indices, on=['date', 'index_code'], how='left', validate='many_to_one')
    pd.testing.assert_frame_equal(d[['date', 'code']], old[['date', 'code']], check_exact=True)
    pieces = []
    for start in range(0, len(d), 50000):
        part = d.iloc[start:start+50000]
        values = measure(part[stock.PRICE_COLUMNS[:-1]], part[POINTS], part.V01)
        pieces.append(pd.DataFrame(values))
    values = pd.concat(pieces, ignore_index=True)
    # The cached pivot has all thirty labels; no 14:49 price enters these features.
    valid = values.valid & d.pv_bars.eq(30) & d.pv_clocks.eq(30) & d.ib_index_valid.fillna(False).astype(bool)
    f = old.copy()
    for name in values.columns.drop('valid'):
        f[name] = values[name]
    for name in NEW_EXPRESSIONS:
        f[name] = f[name].where(valid)
    f['market_direction_valid'] = valid
    f['prior_formula_input_valid'] = old.formula_input_valid
    f['formula_input_valid'] &= valid
    active = f.formula_input_valid
    assert f.date.between(p['signal_first'], p['signal_last']).all()
    ROOT.mkdir(parents=True, exist_ok=True)
    indices.to_parquet(ROOT / 'index_windows.parquet', index=False, compression='zstd')
    f.to_parquet(ROOT / 'features.parquet', index=False, compression='zstd')
    report = dict(protocol_sha256=sha(PROTOCOL), source_hashes=p['source_hashes'],
        features_sha256=sha(ROOT / 'features.parquet'), index_windows_sha256=sha(ROOT / 'index_windows.parquet'),
        rows=len(f), valid=int(active.sum()), newly_invalid=int((old.formula_input_valid & ~active).sum()),
        index_sessions=len(indices), invalid_index_sessions=int((~indices.ib_index_valid).sum()),
        valid_no_down_index=int((active & f.du_down_count.eq(0)).sum()),
        valid_no_up_index=int((active & f.du_up_count.eq(0)).sum()),
        expressions=EXPRESSIONS, native_header=HEADER, software_compilation_verified=False,
        native_INDEXC_parity_verified=False, new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_report.json', report)
    return {k: v for k, v in report.items() if k not in ['source_hashes', 'expressions', 'native_header']}


def verify_features():
    checked_sources()
    r = json.loads((ROOT / 'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['features_sha256'] == sha(ROOT / 'features.parquet')
    assert r['index_windows_sha256'] == sha(ROOT / 'index_windows.parquet')
    old, f = [pd.read_parquet(folder / 'features.parquet') for folder in [previous.ROOT, ROOT]]
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')], old.drop(columns='formula_input_valid'), check_exact=True)
    pd.testing.assert_series_equal(f.prior_formula_input_valid, old.formula_input_valid, check_exact=True, check_names=False)
    sessions = source_sessions()
    identity = pd.DataFrame([dict(date=d, index_code=i) for d, i, _ in sessions])
    records = pd.DataFrame([dict(date=d, index_code=i, sequence=z['sequence'], price=z['price_raw'])
        for d, i, rows in sessions for z in rows])
    c = base.conn()
    c.register('identity', identity)
    c.register('records', records)
    points = ','.join(f'max(CASE WHEN sequence={i} THEN price::DOUBLE/100 END) AS {name}' for i, name in zip(range(199, 228), POINTS))
    indexed = c.sql('''SELECT date,index_code,count(sequence)=228 AND count(DISTINCT sequence)=228
        AND min(sequence)=0 AND max(sequence)=227 AND bool_and(coalesce(isfinite(price) AND price>0 AND price=floor(price),false)) AS ib_index_valid,
        ''' + points + ''' FROM identity LEFT JOIN records USING(date,index_code)
        GROUP BY date,index_code ORDER BY date,index_code''').df()
    indexed['ib_index_valid'] = indexed.ib_index_valid.fillna(False).astype(bool)
    indexed.loc[~indexed.ib_index_valid, POINTS] = np.nan
    actual = pd.read_parquet(ROOT / 'index_windows.parquet')
    pd.testing.assert_frame_equal(actual, indexed, check_exact=True)
    d = stock_windows(old[['date', 'code']]).merge(old[['date', 'code', 'index_code', 'V01']], on=['date', 'code'], validate='one_to_one')
    d = d.merge(indexed, on=['date', 'index_code'], how='left', validate='many_to_one')
    c.register('cached', d)
    atoms = []
    for minute in range(21, 49):
        current = 199 + minute - 20
        atoms.append(f'struct_pack(s0:=pv_c{minute-1:02d},s1:=pv_c{minute:02d},i0:=ib_p{current-1},i1:=ib_p{current})')
    c.sql('SELECT date,code,unnest([' + ','.join(atoms) + ']) AS z FROM cached').create_view('atoms')
    c.sql('''SELECT date,code,coalesce(isfinite(z.s0) AND isfinite(z.s1) AND isfinite(z.i0) AND isfinite(z.i1)
        AND least(z.s0,z.s1,z.i0,z.i1)>0,false) AS good,
        CASE WHEN good THEN 100*ln(z.i1/z.i0) END AS x,
        CASE WHEN good THEN 100*ln(z.s1/z.s0) END AS y,
        z.i1<z.i0 AS down,z.i1>z.i0 AS up FROM atoms''').create_view('returns')
    moments = c.sql('''SELECT date,code,bool_and(good) AS good,
        sum(CASE WHEN down THEN y-x ELSE 0 END) AS du_down_sum,
        sum(CASE WHEN up THEN y-x ELSE 0 END) AS du_up_sum,
        sum(down::INT) AS du_down_count,sum(up::INT) AS du_up_count
        FROM returns GROUP BY date,code ORDER BY date,code''').df()
    c.register('moments', moments)
    expected = c.sql('''SELECT date,code,coalesce(good AND pv_bars=30 AND pv_clocks=30 AND ib_index_valid AND isfinite(V01) AND V01>0,false) AS valid,
        CASE WHEN valid THEN du_down_sum/V01 END AS DU01,
        CASE WHEN valid THEN du_up_sum/V01 END AS DU02
        FROM moments JOIN cached USING(date,code) ORDER BY date,code''').df()
    c.close()
    pd.testing.assert_frame_equal(f[['date', 'code']], expected[['date', 'code']], check_exact=True)
    np.testing.assert_array_equal(f.market_direction_valid, expected.valid)
    final = old.formula_input_valid & expected.valid
    np.testing.assert_array_equal(f.formula_input_valid, final)
    for name in ['du_down_sum', 'du_up_sum', 'du_down_count', 'du_up_count']:
        np.testing.assert_allclose(f.loc[expected.valid, name], moments.loc[expected.valid, name], rtol=0, atol=2e-10)
    for name in NEW_EXPRESSIONS:
        np.testing.assert_allclose(f[name], expected[name], rtol=0, atol=2e-8, equal_nan=True)
        np.testing.assert_array_equal(np.floor(100*f.loc[final, name]+10000+.000001).clip(0, 999999),
            np.floor(100*expected.loc[final, name]+10000+.000001).clip(0, 999999))
    names = re.findall(r'(?m)^([A-Z][A-Z0-9]*):=', HEADER) + list(EXPRESSIONS)
    assert len(names) == len({name.casefold() for name in names})
    assert r['expressions'] == EXPRESSIONS and r['native_header'] == HEADER
    assert r['rows'] == len(f) and r['valid'] == int(final.sum())
    assert r['newly_invalid'] == int((old.formula_input_valid & ~final).sum())
    assert r['index_sessions'] == len(indexed) and r['invalid_index_sessions'] == int((~indexed.ib_index_valid).sum())
    for name, column in [('valid_no_down_index', 'du_down_count'), ('valid_no_up_index', 'du_up_count')]:
        assert r[name] == int((final & moments[column].eq(0)).sum())
    result = dict(passed=True, feature_report_sha256=sha(ROOT / 'feature_report.json'), rows=len(f), valid=int(final.sum()),
        all_28_sign_conditioned_excess_returns_encodings_and_validities_sql_rebuilt=True,
        all_original_48_values_and_keys_unchanged=True, effective_input_intersection_unchanged=bool(final.equals(old.formula_input_valid)),
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_verification.json', result)
    return result


def native_value(prices, indices, atr, outside=1000000.):
    assert len(prices) == len(indices) == 29
    env = dict(C=np.r_[outside, prices, outside, outside], INDEXC=np.r_[outside, indices, outside, outside],
        TIME=np.r_[1419, np.arange(1420, 1449), 1449, 1450], V01=float(atr), IF=np.where, LN=np.log, MAX=np.maximum, SQRT=np.sqrt,
        REF=lambda x, n: pd.Series(x).shift(int(n)).to_numpy(),
        SUM=lambda x, n: pd.Series(x).rolling(int(n), min_periods=int(n)).apply(math.fsum, raw=True).to_numpy(),
        VALUEWHEN=lambda mask, values: np.asarray(values)[np.flatnonzero(mask)[-1]])
    with np.errstate(divide='ignore', invalid='ignore'):
        for line in EXTRA_HEADER.splitlines():
            name, expr = line.rstrip(';').split(':=')
            expr = re.sub(r'(?<![<>=!])=(?!=)', '==', expr)
            env[name] = eval(expr, {'__builtins__': {}}, env)
        return {name: float(eval(expr, {'__builtins__': {}}, env)) for name, expr in NEW_EXPRESSIONS.items()}


def native():
    checked_sources()
    proof = json.loads((ROOT / 'feature_verification.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256'] == sha(ROOT / 'feature_report.json')
    old = json.loads(PROBES.read_text())
    assert old['passed']
    indices = {(date, code): data for date, code, data in source_sessions()}
    f = pd.read_parquet(ROOT / 'features.parquet').set_index(['date', 'code'])
    samples = []
    for sample in old['samples']:
        date, code = sample['date'], sample['code']
        path = MINUTES / code[:2].upper() / (code[3:] + '.parquet')
        assert sha(path) == sample['source_sha256'] and '2024-01-01' <= date <= '2025-12-30'
        raw = pd.read_parquet(path, columns=['timestamp', 'close'],
            filters=[('timestamp', '>=', pd.Timestamp(date+' 14:20')), ('timestamp', '<=', pd.Timestamp(date+' 14:48'))]).sort_values('timestamp')
        assert raw.timestamp.dt.strftime('%H:%M').tolist() == [f'14:{i}' for i in range(20, 49)]
        prices = raw.close.to_numpy(float)
        assert (np.abs(prices-prices.round(2)) <= .0001).all()
        prices = prices.round(2)
        row = f.loc[(date, code)]
        assert row.formula_input_valid
        ix = [z['price_raw']/100 for z in indices[(date, row.index_code)][199:228]]
        result = native_value(prices, ix, row.V01)
        changed = native_value(prices, ix, row.V01, .01)
        for name in NEW_EXPRESSIONS:
            np.testing.assert_allclose(result[name], row[name], rtol=0, atol=2e-8)
            assert result[name] == changed[name]
            assert np.clip(np.floor(100*result[name]+10000+.000001), 0, 999999) == np.clip(np.floor(100*row[name]+10000+.000001), 0, 999999)
        samples.append(dict(date=date, code=code, index_code=row.index_code, source_sha256=sample['source_sha256'], minutes=29, **result))
    assert len(samples) == 32
    result = dict(passed=True, protocol_sha256=sha(PROTOCOL), feature_report_sha256=sha(ROOT / 'feature_report.json'),
        feature_verification_sha256=sha(ROOT / 'feature_verification.json'), index_source_sha256=sha(INDEX_REPORT), samples=samples,
        stock_raw_minutes=928, actual_ref_log_if_sum_and_1448_anchor_rebuilt=True,
        current_1449_and_later_quotes_do_not_change_new_inputs=True,
        software_compilation_verified=False, native_INDEXC_parity_verified=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'native_input_verification.json', result)
    return {k: v for k, v in result.items() if k != 'samples'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['features', 'verify_features', 'native'])
    print(json.dumps(globals()[parser.parse_args().stage](), ensure_ascii=False, indent=2))
