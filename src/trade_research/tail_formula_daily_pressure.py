"""Volume-weighted close location over strictly previous complete stock days."""
import argparse
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import tail_formula_additive as base
from . import tail_formula_float as previous
from .corporate_cash import DAILY, MINUTES, save_json, sha
from .turnover_reference import CALENDAR

STEM = 'tail_formula_daily_pressure'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
DAILY_REPORT = Path('data/research/tail_formula_1000/feature_report.json')
KEYS = Path('data/research/tail_formula_complete/native_history_probe_keys.json')
MANIFEST = Path('data/research/economic_winner/input_manifest.json')
SOURCE_BOUNDARY = Path('config/tail_formula_daily_pressure_source_boundary.json')
NEW_EXPRESSIONS = {'CP01': '100*CPW/MAX(CPV,0.000000000001)'}
EXPRESSIONS = {**previous.EXPRESSIONS, **NEW_EXPRESSIONS}
HEADER = previous.HEADER + '''CPH:=INTPART(HHV(H,B0)*100+0.5);
CPL:=INTPART(LLV(L,B0)*100+0.5);
CPC:=INTPART(C*100+0.5);
CPD:=SUM(V,B0);
CPP:=(2*CPC-CPH-CPL)/MAX(CPH-CPL,1);
'''
HEADER += ''.join(f'CPD{i:02d}:=REF(CPD,B{i-1});\nCPP{i:02d}:=REF(CPP,B{i-1});\n' for i in range(1, 21))
HEADER += 'CPV:=' + '+'.join(f'CPD{i:02d}' for i in range(1, 21)) + ';\n'
HEADER += 'CPW:=' + '+'.join(f'CPD{i:02d}*CPP{i:02d}' for i in range(1, 21)) + ';\n'
ORIGINAL_NATIVE_CORE = base.native_core


def native_core(*args, **kwargs):
    text = ORIGINAL_NATIVE_CORE(*args, **kwargs)
    assert text.count('CORE:SC>') == 1
    return text.replace('CORE:SC>', 'CORE:CPV>0 AND SC>')


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    report = json.loads((previous.ROOT / 'feature_report.json').read_text())
    proof = json.loads((previous.ROOT / 'feature_verification.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256'] == sha(previous.ROOT / 'feature_report.json')
    assert report['features_sha256'] == sha(previous.ROOT / 'features.parquet')
    sources = json.loads(DAILY_REPORT.read_text())['source_sha256']
    for file, digest in sources.items():
        assert sha(Path(file)) == digest
    assert p['history_days'] == 20 and p['new_fields'] == list(NEW_EXPRESSIONS)
    assert (p['history_first'], p['history_last']) == ('2023-06-01', '2025-12-30')
    assert not p['new_2026_prices_allowed']
    if (ROOT / 'native_input_verification.json').exists():
        native_proof = json.loads((ROOT / 'native_input_verification.json').read_text())
        assert native_proof['source_boundary_protocol_sha256'] == sha(SOURCE_BOUNDARY)
    return p, list(sources)


def daily_parts(high, low, close, volume, adjustflag):
    prices = np.column_stack([high, low, close]).astype(float)
    cents = np.floor(prices * 100 + .5)
    volume = np.asarray(volume, float)
    good = (np.isfinite(prices).all(axis=1) & (prices > 0).all(axis=1)
        & (np.abs(prices - cents / 100) <= .0001).all(axis=1)
        & (cents[:, 0] >= cents[:, 2]) & (cents[:, 2] >= cents[:, 1])
        & np.isfinite(volume) & (volume >= 0) & (volume == np.floor(volume))
        & (np.asarray(adjustflag) == 3))
    with np.errstate(invalid='ignore', divide='ignore'):
        position = (2 * cents[:, 2] - cents[:, 0] - cents[:, 1]) / np.maximum(cents[:, 0] - cents[:, 1], 1)
    return good, position


def history_for_stock(d):
    """Independent explicit-date windows; bad stock days remain in the window."""
    d = d.sort_values('date').reset_index(drop=True)
    assert not d.date.duplicated().any()
    good, position = daily_parts(d.high, d.low, d.close, d.volume, d.adjustflag)
    out = d[['date', 'code']].copy()
    n = len(d)
    out['cp_rows20'] = np.minimum(np.arange(n), 20)
    out['cp_good20'] = pd.Series(good.astype(int)).rolling(20, min_periods=1).sum().shift()
    out['cp_volume20'] = d.volume.astype(float).rolling(20, min_periods=1).sum().shift()
    # Sum each window independently, without rolling subtraction of old values.
    atoms = position * d.volume.to_numpy(float)
    out['cp_weighted20'] = [float(np.nansum(atoms[max(0, i-20):i])) if i else np.nan for i in range(n)]
    out['cp_first_date'] = [d.date.iloc[max(0, i-20)] if i else None for i in range(n)]
    out['cp_last_date'] = d.date.shift()
    flat = d.high.astype(float).round(2).eq(d.low.astype(float).round(2))
    out['cp_flat_days20'] = flat.astype(int).rolling(20, min_periods=1).sum().shift()
    changed = d.preclose.astype(float).sub(d.close.astype(float).round(2).shift()).abs().gt(.005)
    out['cp_reference_breaks20'] = changed.astype(int).rolling(20, min_periods=1).sum().shift()
    return out


def features():
    p, files = checked_sources()
    assert not (ROOT / 'feature_report.json').exists()
    c = base.conn()
    c.read_parquet(files).create_view('daily')
    hist = c.sql('''WITH active AS (
        SELECT date,code,high::DOUBLE AS rh,low::DOUBLE AS rl,close::DOUBLE AS rc,
            round(high::DOUBLE*100) AS hc,round(low::DOUBLE*100) AS lc,round(close::DOUBLE*100) AS cc,
            volume::DOUBLE AS vol,preclose::DOUBLE AS pre,adjustflag::DOUBLE AS adj
        FROM daily WHERE tradestatus=1 AND date BETWEEN '2023-06-01' AND '2025-12-30'),
        lagged AS (SELECT *,lag(cc) OVER(PARTITION BY code ORDER BY date) AS pc FROM active),
        atoms AS (SELECT *,coalesce(isfinite(rh) AND isfinite(rl) AND isfinite(rc) AND least(rh,rl,rc)>0
            AND abs(rh-hc/100)<=.0001 AND abs(rl-lc/100)<=.0001 AND abs(rc-cc/100)<=.0001
            AND hc>=cc AND cc>=lc AND adj=3 AND isfinite(vol) AND vol>=0 AND vol=floor(vol),false) AS good,
            (2*cc-hc-lc)/greatest(hc-lc,1)*vol AS weighted,
            coalesce(abs(pre-pc/100)>.005,false) AS changed FROM lagged)
        SELECT date,code,count(*) OVER w AS cp_rows20,sum(good::INT) OVER w AS cp_good20,
            sum(vol) OVER w AS cp_volume20,sum(weighted) OVER w AS cp_weighted20,
            min(date) OVER w AS cp_first_date,max(date) OVER w AS cp_last_date,
            sum((hc=lc)::INT) OVER w AS cp_flat_days20,sum(changed::INT) OVER w AS cp_reference_breaks20
        FROM atoms WINDOW w AS(PARTITION BY code ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING)
        ORDER BY date,code''').df()
    c.close()
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    f = old.merge(hist, on=['date', 'code'], how='left', validate='one_to_one').sort_values(['date', 'code']).reset_index(drop=True)
    valid = (f.cp_rows20.eq(20) & f.cp_good20.eq(20) & f.cp_volume20.gt(0)
        & f.cp_last_date.lt(f.date) & np.isfinite(f.cp_weighted20))
    f['CP01'] = (100 * f.cp_weighted20 / f.cp_volume20).where(valid)
    f['daily_pressure_valid'] = valid
    f['prior_formula_input_valid'] = f.formula_input_valid
    f['formula_input_valid'] &= valid & np.isfinite(f.CP01)
    cal = pd.read_parquet(CALENDAR)
    days = sorted(cal.loc[cal.is_trading_day.eq('1') & cal.calendar_date.between('2023-06-01', '2025-12-30'), 'calendar_date'])
    ranks = {day: i for i, day in enumerate(days)}
    f['cp_market_span'] = f.cp_last_date.map(ranks) - f.cp_first_date.map(ranks) + 1
    f['cp_last_gap'] = f.date.map(ranks) - f.cp_last_date.map(ranks)
    ROOT.mkdir(parents=True, exist_ok=True)
    hist.to_parquet(ROOT / 'history.parquet', index=False, compression='zstd')
    f.to_parquet(ROOT / 'features.parquet', index=False, compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL), source_hashes=p['source_hashes'],
        history_sha256=sha(ROOT / 'history.parquet'), features_sha256=sha(ROOT / 'features.parquet'),
        calendar_sha256=sha(CALENDAR), rows=len(f), valid=int(f.formula_input_valid.sum()),
        newly_invalid=int((old.formula_input_valid & ~f.formula_input_valid).sum()),
        valid_with_reference_breaks=int((f.formula_input_valid & f.cp_reference_breaks20.gt(0)).sum()),
        valid_with_stock_day_gaps=int((f.formula_input_valid & (f.cp_market_span.gt(20) | f.cp_last_gap.gt(1))).sum()),
        expressions=EXPRESSIONS, native_header=HEADER, native_core_gate='CPV>0',
        native_source_parity_verified=False, software_compilation_verified=False,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_report.json', r)
    return {k: v for k, v in r.items() if k not in ['source_hashes', 'expressions', 'native_header']}


def verify_features():
    p, files = checked_sources()
    r = json.loads((ROOT / 'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['features_sha256'] == sha(ROOT / 'features.parquet')
    assert r['history_sha256'] == sha(ROOT / 'history.parquet') and r['calendar_sha256'] == sha(CALENDAR)
    pieces = []
    for file in files:
        d = pd.read_parquet(file, columns=['date', 'code', 'high', 'low', 'close', 'preclose', 'volume', 'adjustflag', 'tradestatus'],
            filters=[('date', '>=', '2023-06-01'), ('date', '<=', '2025-12-30')])
        d = d.loc[d.tradestatus.eq(1)]
        if not d.empty:
            pieces.append(history_for_stock(d))
    expected = pd.concat(pieces, ignore_index=True).sort_values(['date', 'code']).reset_index(drop=True)
    actual = pd.read_parquet(ROOT / 'history.parquet')
    pd.testing.assert_frame_equal(actual, expected[actual.columns], check_dtype=False, rtol=3e-13, atol=2e-4)
    for name in actual.columns.drop('cp_weighted20'):
        pd.testing.assert_series_equal(actual[name], expected[name], check_dtype=False, check_exact=True)
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    f = pd.read_parquet(ROOT / 'features.parquet')
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')], old.drop(columns='formula_input_valid'), check_exact=True)
    pd.testing.assert_series_equal(f.prior_formula_input_valid, old.formula_input_valid, check_names=False, check_exact=True)
    e = old[['date', 'code']].merge(expected, on=['date', 'code'], how='left', validate='one_to_one')
    valid = (e.cp_rows20.eq(20) & e.cp_good20.eq(20) & e.cp_volume20.gt(0)
        & e.cp_last_date.lt(e.date) & np.isfinite(e.cp_weighted20))
    cp = (100 * e.cp_weighted20 / e.cp_volume20).where(valid)
    np.testing.assert_array_equal(f.daily_pressure_valid, valid)
    np.testing.assert_allclose(f.CP01, cp, rtol=0, atol=2e-10, equal_nan=True)
    final = old.formula_input_valid & valid
    np.testing.assert_array_equal(f.formula_input_valid, final)
    assert f.loc[final, 'CP01'].between(-100-1e-10, 100+1e-10).all()
    encode = lambda x: np.floor(np.clip(100 * x + 10000 + .000001, 0, 999999)).astype('int32')
    np.testing.assert_array_equal(encode(f.loc[final, 'CP01']), encode(cp.loc[final]))
    # Source window dates, including suspended-stock gaps, use the same calendar.
    cal = pd.read_parquet(CALENDAR)
    days = sorted(cal.loc[cal.is_trading_day.eq('1') & cal.calendar_date.between('2023-06-01', '2025-12-30'), 'calendar_date'])
    rank = {day: i for i, day in enumerate(days)}
    np.testing.assert_allclose(f.cp_market_span, e.cp_last_date.map(rank)-e.cp_first_date.map(rank)+1, equal_nan=True)
    np.testing.assert_allclose(f.cp_last_gap, e.date.map(rank)-e.cp_last_date.map(rank), equal_nan=True)
    assert int(final.sum()) == r['valid'] and int((old.formula_input_valid & ~final).sum()) == r['newly_invalid']
    assert r['valid_with_reference_breaks'] == int((final & e.cp_reference_breaks20.gt(0)).sum())
    assert r['valid_with_stock_day_gaps'] == int((final & (f.cp_market_span.gt(20) | f.cp_last_gap.gt(1))).sum())
    names = re.findall(r'(?m)^([A-Z][A-Z0-9]*):=', HEADER) + list(EXPRESSIONS)
    assert len(names) == len({n.casefold() for n in names})
    assert r['expressions'] == EXPRESSIONS and r['native_header'] == HEADER
    v = dict(passed=True, feature_report_sha256=sha(ROOT / 'feature_report.json'), rows=len(f),
        valid=int(final.sum()), all_source_windows_and_close_locations_independently_rebuilt=True,
        all_old_values_keys_dates_validity_and_new_encodings_checked=True,
        effective_input_intersection_unchanged=bool(final.equals(old.formula_input_valid)),
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_verification.json', v)
    return v


def native_value(high, low, close, volume):
    env = {'MAX': max}
    for i, (h, l, c, v) in enumerate(zip(high, low, close, volume), 1):
        h, l, c = [math.floor(float(x) * 100 + .5) for x in [h, l, c]]
        env[f'CPP{i:02d}'] = (2*c-h-l) / max(h-l, 1)
        env[f'CPD{i:02d}'] = float(v)
    for line in HEADER[HEADER.index('CPV:='):].strip().split(';'):
        if line.strip():
            name, expression = line.strip().split(':=')
            env[name] = eval(expression, {'__builtins__': {}}, env)
    assert env['CPV'] > 0
    return eval(NEW_EXPRESSIONS['CP01'], {'__builtins__': {}}, env)


def native():
    checked_sources()
    boundary = json.loads(SOURCE_BOUNDARY.read_text())
    assert boundary['original_protocol_sha256'] == sha(PROTOCOL)
    assert boundary['initial_failure_sha256'] == sha(ROOT / 'native_initial_failure.json')
    assert boundary['diagnostic_sha256'] == sha(ROOT / 'native_source_diagnostic.json')
    diagnostic = json.loads((ROOT / 'native_source_diagnostic.json').read_text())
    diagnosed = {(x['date'], x['code']): x for x in diagnostic['cases']}
    proof = json.loads((ROOT / 'feature_verification.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256'] == sha(ROOT / 'feature_report.json')
    keys = json.loads(KEYS.read_text())['sampled_keys']
    assert len(keys) == 32
    hashes = json.loads(MANIFEST.read_text())['source_sha256']
    daily_hashes = json.loads(DAILY_REPORT.read_text())['source_sha256']
    values = pd.read_parquet(ROOT / 'features.parquet', columns=['date', 'code', 'CP01', 'daily_pressure_valid', 'cp_first_date', 'cp_last_date']).set_index(['date', 'code'])
    cases = []
    for key in keys:
        day, code = key['date'], key['code']
        row = values.loc[(day, code)]
        assert row.daily_pressure_valid and '2024-01-01' <= day <= '2025-12-30'
        path = DAILY / (code.replace('.', '_') + '.parquet')
        assert sha(path) == daily_hashes[str(path)]
        d = pd.read_parquet(path, columns=['date', 'high', 'low', 'close', 'volume', 'tradestatus', 'adjustflag'],
            filters=[('date', '>=', '2023-06-01'), ('date', '<', day)])
        d = d.loc[d.tradestatus.eq(1)].sort_values('date').tail(20).set_index('date')
        assert len(d) == 20 and d.adjustflag.eq(3).all()
        assert (d.index[0], d.index[-1]) == (row.cp_first_date, row.cp_last_date)
        raw_path = MINUTES / code[:2].upper() / (code[3:] + '.parquet')
        assert sha(raw_path) == hashes[str(raw_path)]
        raw = pq.read_table(raw_path, columns=['timestamp', 'high', 'low', 'close', 'volume'],
            filters=[('timestamp', '>=', pd.Timestamp(d.index[0]).to_pydatetime()),
                     ('timestamp', '<', pd.Timestamp(day).to_pydatetime())]).to_pandas().sort_values('timestamp')
        raw['date'] = raw.timestamp.dt.strftime('%Y-%m-%d')
        raw[['high', 'low', 'close']] = raw[['high', 'low', 'close']].round(2).astype(float)
        agg = raw.groupby('date').agg(high=('high', 'max'), low=('low', 'min'), close=('close', 'last'),
            volume=('volume', 'sum'), bars=('timestamp', 'size'))
        active = agg.loc[d.index]
        assert active.bars.eq(241).all()
        np.testing.assert_allclose(active[['high', 'low', 'close']], d[['high', 'low', 'close']].astype(float), rtol=0, atol=1e-8)
        volume_delta = active.volume.astype(float) - d.volume.astype(float)
        volume_differences = [dict(history_date=dt, volume=float(v)) for dt, v in volume_delta.items() if v != 0]
        assert volume_differences == diagnosed[(day, code)]['source_differences']
        # Read the twenty frozen prior days; zero-volume suspension placeholders
        # cannot silently become additional stock-history observations.
        omitted = agg.loc[~agg.index.isin(d.index)]
        assert omitted.volume.eq(0).all()
        reverse = active.iloc[::-1]
        result = native_value(reverse.high, reverse.low, reverse.close, reverse.volume / 100)
        shares_result = native_value(reverse.high, reverse.low, reverse.close, reverse.volume)
        daily_reverse = d.iloc[::-1]
        daily_result = native_value(daily_reverse.high, daily_reverse.low, daily_reverse.close, daily_reverse.volume)
        np.testing.assert_allclose(result, shares_result, rtol=0, atol=2e-10)
        np.testing.assert_allclose(daily_result, row.CP01, rtol=0, atol=2e-10)
        assert math.floor(100*result+10000+.000001) == math.floor(100*row.CP01+10000+.000001)
        cases.append(dict(date=day, code=code, first=d.index[0], last=d.index[-1], days=20,
            raw_minute_rows=len(active)*241, omitted_zero_volume_days=len(omitted),
            raw_minute_value=result, daily_value=daily_result, raw_daily_volume_differences=volume_differences,
            encoded_equal=True))
    report = dict(passed=True, protocol_sha256=sha(PROTOCOL),
        source_boundary_protocol_sha256=sha(SOURCE_BOUNDARY), initial_failure_sha256=sha(ROOT / 'native_initial_failure.json'),
        source_diagnostic_sha256=sha(ROOT / 'native_source_diagnostic.json'),
        feature_report_sha256=sha(ROOT / 'feature_report.json'), feature_verification_sha256=sha(ROOT / 'feature_verification.json'),
        prior_frozen_keys_sha256=sha(KEYS), cases=cases, samples=len(cases),
        complete_prior_day_high_low_close_and_new_native_expression_verified=True,
        raw_daily_volume_all_equal=all(not c['raw_daily_volume_differences'] for c in cases),
        raw_daily_volume_mismatch_days=sum(len(c['raw_daily_volume_differences']) for c in cases),
        raw_daily_max_input_difference=max(abs(c['raw_minute_value']-c['daily_value']) for c in cases),
        all_32_probe_encodings_identical_not_full_source_parity=True,
        shares_and_lots_identical=True, native_source_parity_verified=False, software_compilation_verified=False,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'native_input_verification.json', report)
    return {k: v for k, v in report.items() if k != 'cases'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['features', 'verify_features', 'native'])
    print(json.dumps(globals()[parser.parse_args().stage](), ensure_ascii=False, indent=2))
