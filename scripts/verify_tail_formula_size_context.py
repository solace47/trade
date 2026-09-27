"""Rebuild the two size-context inputs from wire bytes and strict stock-date ASOF."""
import json
from pathlib import Path
import re
import struct

import numpy as np
import pandas as pd

from verify_index_minute_probe import replay
from trade_research import tail_formula_size_context as study
from trade_research.corporate_cash import save_json, sha

ROOT = study.ROOT


def main():
    files = study.checked_sources(); r = json.loads((ROOT / 'feature_report.json').read_text())
    s = json.loads((ROOT / 'source_report.json').read_text()); cfg = json.loads(study.PROTOCOL.read_text())
    for key, path in [('protocol_sha256', study.PROTOCOL), ('source_report_sha256', ROOT / 'source_report.json'),
                      ('index_points_sha256', ROOT / 'index_points.parquet'), ('source_audit_sha256', ROOT / 'source_audit.parquet'),
                      ('features_sha256', ROOT / 'features.parquet'), ('calendar_sha256', study.CALENDAR)]:
        assert r[key] == sha(path)
    frames = []
    for name, digest in s['daily_raw_sha256'].items():
        path = Path(name); assert sha(path) == digest
        raw = json.loads(path.read_text())
        assert raw['start'] == cfg['history_first'] and raw['end'] == cfg['history_last']
        d = pd.DataFrame(raw['records']); assert d.code.eq(raw['symbol']).all()
        for field in ['open', 'high', 'low', 'close']:
            d[field] = d[field].astype(float)
        prices = d[['open', 'high', 'low', 'close']]
        assert np.isfinite(prices).all().all() and prices.gt(0).all().all()
        assert d.high.ge(prices.max(axis=1)).all() and d.low.le(prices.min(axis=1)).all()
        frames.append(d)
    daily = pd.concat(frames).sort_values(['code', 'date']).reset_index(drop=True)
    pd.testing.assert_frame_equal(daily, pd.read_parquet(ROOT / 'indices.parquet'), check_exact=True)
    calendar = pd.read_parquet(study.CALENDAR)
    days = sorted(calendar.loc[calendar.is_trading_day.eq('1') &
        calendar.calendar_date.between(cfg['history_first'], cfg['history_last']), 'calendar_date'])
    for code in cfg['symbols']:
        assert daily.loc[daily.code.eq(code), 'date'].tolist() == days
    indexed = daily.set_index(['code', 'date'])
    points = []; audit = []; perturbations = 0; wire_count = 0; price_conflicts = 0
    for item in s['sessions']:
        code, day = item['symbol'], item['date']; assert '2024-01-01' <= day <= '2025-12-30'
        for attempt in item['attempts']:
            for name, digest in attempt['wire_sha256'].items():
                assert sha(Path(name)) == digest
        if 'path' in item:
            path = Path(item['path']); assert sha(path) == item['sha256']
            packet = Path(item['request_path']).read_bytes()
            assert packet[:12] == bytes.fromhex('0c01300001010d000d00b40f')
            requested_day, market, symbol = struct.unpack('<IB6s', packet[12:])
            assert (requested_day, market, symbol.decode()) == (int(day.replace('-', '')), 1, code[3:])
            rows = replay(Path(item['response_path']).read_bytes())
            assert rows == json.loads(path.read_text()) and len(rows) == item['rows']; wire_count += 1
        else:
            rows = []
        valid = len(rows) >= 228 and all(x['sequence'] == i and x['price_raw'] > 0 for i, x in enumerate(rows[:228]))
        points.append(dict(date=day, index_code=code, p48=rows[227]['price_raw'] / 100 if valid else np.nan,
            p20=rows[199]['price_raw'] / 100 if valid else np.nan, prefix_valid=valid))
        d = indexed.loc[code, day]; values = np.array([x['price_raw'] / 100 for x in rows])
        inside = bool((values > 0).all() and (values >= d.low - .0051).all() and (values <= d.high + .0051).all())
        match_close = bool(len(values) == 240 and abs(values[-1] - d.close) <= .0051)
        full = len(values) == 240 and inside and match_close
        price_conflicts += int(bool(len(values)) and (not inside or (len(values) == 240 and not match_close)))
        audit.append(dict(date=day, index_code=code, rows=len(rows), full_day_source_valid=full,
            last_minus_daily_close=float(values[-1] - d.close) if len(values) else None))
        if valid:
            changed = [dict(x) for x in rows]
            for x in changed[228:]:
                x['sequence'] = -1; x['price_raw'] = -999999
            assert study.prefix_points(rows) == study.prefix_points(changed)
            perturbations += 1
    # Actual price conflicts require investigation, not outcome-driven date deletion.
    assert price_conflicts == 0
    points = pd.DataFrame(points).sort_values(['date', 'index_code']).reset_index(drop=True)
    audit = pd.DataFrame(audit).sort_values(['date', 'index_code']).reset_index(drop=True)
    pd.testing.assert_frame_equal(points, pd.read_parquet(ROOT / 'index_points.parquet'), check_exact=True)
    pd.testing.assert_frame_equal(audit, pd.read_parquet(ROOT / 'source_audit.parquet'), check_exact=True)
    old = pd.read_parquet(study.previous.ROOT / 'features.parquet'); actual = pd.read_parquet(ROOT / 'features.parquet')
    assert {(x.index_code, x.date) for x in points.itertuples()} == {(code, day) for code in cfg['symbols'] for day in old.date.unique()}
    pd.testing.assert_frame_equal(actual[old.columns.drop('formula_input_valid')], old.drop(columns='formula_input_valid'), check_exact=True)
    assert actual.prior_formula_input_valid.equals(old.formula_input_valid)
    c = study.base.conn(); c.register('keys', old[['date', 'code']]); c.register('indices', daily)
    c.read_parquet(files).create_view('stock_daily')
    prior = c.sql('''WITH traded AS(SELECT date,code FROM stock_daily WHERE tradestatus=1
        AND date BETWEEN '2023-06-01' AND '2025-12-30'), p AS(
        SELECT k.date,k.code,t.date AS sc_prior_date FROM keys k
        ASOF LEFT JOIN traded t ON k.code=t.code AND k.date>t.date)
        SELECT p.date,p.code,p.sc_prior_date,s.close AS sc_small_prior,l.close AS sc_large_prior
        FROM p LEFT JOIN indices s ON s.code='sh.000852' AND s.date=p.sc_prior_date
        LEFT JOIN indices l ON l.code='sh.000300' AND l.date=p.sc_prior_date ORDER BY p.date,p.code''').df()
    c.close()
    f = prior
    for name, code in [('small', 'sh.000852'), ('large', 'sh.000300')]:
        q = points.loc[points.index_code.eq(code)].drop(columns='index_code').rename(
            columns={k: f'sc_{name}_{k}' for k in ['p48', 'p20', 'prefix_valid']})
        f = f.merge(q, on='date', how='left', validate='many_to_one')
    pd.testing.assert_frame_equal(actual[f.columns], f, check_dtype=False, rtol=0, atol=2e-12)
    values = f[['sc_small_prior', 'sc_large_prior', 'sc_small_p48', 'sc_small_p20', 'sc_large_p48', 'sc_large_p20']]
    valid = (np.isfinite(values).all(axis=1) & values.gt(0).all(axis=1) & f.sc_prior_date.lt(f.date)
        & f.sc_small_prefix_valid.fillna(False) & f.sc_large_prefix_valid.fillna(False))
    # Cross-products provide an algebraically independent evaluation.
    one = 100 * (f.sc_small_p48 * f.sc_large_prior - f.sc_large_p48 * f.sc_small_prior) / (f.sc_small_prior * f.sc_large_prior)
    two = 100 * (f.sc_small_p48 * f.sc_large_p20 - f.sc_large_p48 * f.sc_small_p20) / (f.sc_small_p20 * f.sc_large_p20)
    expected = pd.DataFrame(dict(SZ01=one.where(valid), SZ02=two.where(valid)))
    np.testing.assert_allclose(actual[list(expected)], expected, rtol=0, atol=2e-12, equal_nan=True)
    final = old.formula_input_valid & valid & np.isfinite(expected).all(axis=1)
    assert actual.size_context_valid.equals(valid) and actual.formula_input_valid.equals(final)
    for q in [actual.loc[final, list(expected)], expected.loc[final]]:
        assert np.isfinite(q).all().all()
    np.testing.assert_array_equal(np.floor(np.clip(actual.loc[final, list(expected)] * 100 + 10000 + .000001, 0, 999999)),
        np.floor(np.clip(expected.loc[final] * 100 + 10000 + .000001, 0, 999999)))
    ranks = {day: i for i, day in enumerate(days)}
    gaps = f.date.map(ranks) - f.sc_prior_date.map(ranks)
    np.testing.assert_allclose(actual.sc_prior_gap, gaps, rtol=0, atol=0, equal_nan=True)
    assert r['rows'] == len(actual) and r['valid'] == int(final.sum())
    assert r['newly_invalid'] == int((old.formula_input_valid & ~final).sum())
    assert r['valid_with_prior_stock_day_gaps'] == int((final & gaps.gt(1)).sum())
    assert r['full_day_source_conflicts'] == int((~audit.full_day_source_valid).sum())
    assert r['invalid_prefix_sessions'] == int((~points.prefix_valid).sum())
    assert r['expressions'] == study.EXPRESSIONS and r['native_header'] == study.HEADER
    names = re.findall(r'(?m)^([A-Za-z][A-Za-z0-9]*):=', study.HEADER) + list(study.EXPRESSIONS)
    assert len(names) == len(set(names))
    for prefix, symbol in [('SM', 'SH000852'), ('LG', 'SH000300')]:
        assert f'{prefix}P:=REF("{symbol}$CLOSE",B0);' in study.HEADER
        for minute in ['1420', '1448']:
            assert f'{prefix}{minute[2:]}:=VALUEWHEN(TIME={minute},"{symbol}$CLOSE");' in study.HEADER
    variables = dict(SM48=f.sc_small_p48, SMP=f.sc_small_prior, SM20=f.sc_small_p20,
        LG48=f.sc_large_p48, LGP=f.sc_large_prior, LG20=f.sc_large_p20)
    for name, expression in study.NEW_EXPRESSIONS.items():
        # Only the repository's two fixed arithmetic strings, no external expressions.
        value = eval(expression, {'__builtins__': {}}, variables).where(valid)
        np.testing.assert_allclose(value, expected[name], rtol=0, atol=2e-12, equal_nan=True)
    proof = dict(passed=True, feature_report_sha256=sha(ROOT / 'feature_report.json'),
        rows=len(actual), valid=int(final.sum()), sessions=len(points), original_wire_replays=wire_count,
        all_stock_prior_dates_and_cross_symbol_values_independently_rebuilt=True,
        original_48_inputs_unchanged=True, all_integer_encodings_and_validity_verified=True,
        after_cutoff_perturbation_checks=perturbations, native_expression_arithmetic_verified=True,
        native_source_price_parity_verified=False, full_day_source_audit_used_for_selection=False,
        software_compilation_verified=False, new_selection_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_verification.json', proof); return proof


if __name__ == '__main__':
    print(json.dumps(main(), ensure_ascii=False, indent=2))
