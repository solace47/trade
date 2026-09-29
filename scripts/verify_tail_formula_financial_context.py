"""Rebuild the two financial-context inputs from wire bytes and strict stock-date ASOF."""
import json
from pathlib import Path
import re
import struct

import numpy as np
import pandas as pd

from verify_index_minute_probe import replay
from trade_research import tail_formula_financial_context as study
from trade_research.corporate_cash import save_json, sha

ROOT = study.ROOT


def verify_daily_responses(source, cfg):
    from urllib.parse import parse_qs, urlsplit
    actual = pd.read_parquet(ROOT / 'indices.parquet')
    rebuilt = []
    assert len(source['daily_blocks']) == 2
    for block in source['daily_blocks']:
        code = block['code']; url = urlsplit(block['url']); query = parse_qs(url.query)
        assert code in cfg['symbols']
        assert url.scheme + '://' + url.netloc + url.path == cfg['daily_endpoint']
        assert query == dict(_var=['kline_dayqfq'], param=[f"{code.replace('.', '')},day,{cfg['history_first']},{cfg['history_last']},640,qfq"], r=['0.8205512681390605'])
        assert 1 <= len(block['attempts']) <= 2
        a = next(v for v in block['attempts'] if v['status'] == 'nonempty')
        for attempt in block['attempts']:
            if 'path' in attempt: assert sha(Path(attempt['path'])) == attempt['sha256']
        raw = Path(a['path']).read_text(); payload = json.loads(raw[raw.index('=')+1:])
        assert payload['code'] == 0
        bars = payload['data'][code.replace('.', '')]['day']
        dates = [r[0] for r in bars]
        assert dates == sorted(set(dates)) and dates[-1] == cfg['history_last']
        for b in bars:
            if cfg['history_first'] <= b[0] <= cfg['history_last']:
                rebuilt.append(dict(date=b[0], code=code, open=float(b[1]), high=float(b[3]), low=float(b[4]), close=float(b[2])))
    expected = pd.DataFrame(rebuilt).sort_values(['code', 'date']).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual[expected.columns], expected, check_exact=True)
    lookup = expected.set_index(['code', 'date'])
    probe = json.loads(Path('data/research/tail_formula_financial_daily_probe/source_report.json').read_text())
    for item in probe['sessions']:
        a = next(v for v in item['attempts'] if v['status'] == 'nonempty')
        assert sha(Path(a['path'])) == a['sha256']
        b = json.loads(Path(a['path']).read_bytes())['data']['klines'][0].split(',')
        value = lookup.loc[(item['code'], item['date']), ['open', 'close', 'high', 'low']].to_numpy(float)
        np.testing.assert_array_equal(np.rint(value*100), np.rint(np.array(b[1:], float)*100))


def main():
    files = study.checked_sources(); r = json.loads((ROOT / 'feature_report.json').read_text())
    s = json.loads((ROOT / 'source_report.json').read_text()); cfg = json.loads(study.PROTOCOL.read_text())
    for key, path in [('protocol_sha256', study.PROTOCOL), ('source_report_sha256', ROOT / 'source_report.json'),
                      ('index_points_sha256', ROOT / 'index_points.parquet'), ('source_audit_sha256', ROOT / 'source_audit.parquet'),
                      ('features_sha256', ROOT / 'features.parquet'), ('calendar_sha256', study.CALENDAR)]:
        assert r[key] == sha(path)
    verify_daily_responses(s, cfg)
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
            assert (requested_day, market, symbol.decode()) == (int(day.replace('-', '')), int(code.startswith('sh.')), code[3:])
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
        SELECT k.date,k.code,t.date AS fi_prior_date FROM keys k
        ASOF LEFT JOIN traded t ON k.code=t.code AND k.date>t.date)
        SELECT p.date,p.code,p.fi_prior_date,s.close AS fi_broker_prior,l.close AS fi_bank_prior
        FROM p LEFT JOIN indices s ON s.code='sz.399975' AND s.date=p.fi_prior_date
        LEFT JOIN indices l ON l.code='sz.399986' AND l.date=p.fi_prior_date ORDER BY p.date,p.code''').df()
    c.close()
    f = prior
    for name, code in [('broker', 'sz.399975'), ('bank', 'sz.399986')]:
        q = points.loc[points.index_code.eq(code)].drop(columns='index_code').rename(
            columns={k: f'fi_{name}_{k}' for k in ['p48', 'p20', 'prefix_valid']})
        f = f.merge(q, on='date', how='left', validate='many_to_one')
    pd.testing.assert_frame_equal(actual[f.columns], f, check_dtype=False, rtol=0, atol=2e-12)
    values = f[['fi_broker_prior', 'fi_bank_prior', 'fi_broker_p48', 'fi_broker_p20', 'fi_bank_p48', 'fi_bank_p20']]
    valid = (np.isfinite(values).all(axis=1) & values.gt(0).all(axis=1) & f.fi_prior_date.lt(f.date)
        & f.fi_broker_prefix_valid.fillna(False) & f.fi_bank_prefix_valid.fillna(False))
    # Cross-products provide an algebraically independent evaluation.
    one = 100 * (f.fi_broker_p48 * f.fi_bank_prior - f.fi_bank_p48 * f.fi_broker_prior) / (f.fi_broker_prior * f.fi_bank_prior)
    two = 100 * (f.fi_broker_p48 * f.fi_bank_p20 - f.fi_bank_p48 * f.fi_broker_p20) / (f.fi_broker_p20 * f.fi_bank_p20)
    expected = pd.DataFrame(dict(FI01=one.where(valid), FI02=two.where(valid)))
    np.testing.assert_allclose(actual[list(expected)], expected, rtol=0, atol=2e-12, equal_nan=True)
    final = old.formula_input_valid & valid & np.isfinite(expected).all(axis=1)
    assert actual.financial_context_valid.equals(valid) and actual.formula_input_valid.equals(final)
    for q in [actual.loc[final, list(expected)], expected.loc[final]]:
        assert np.isfinite(q).all().all()
    np.testing.assert_array_equal(np.floor(np.clip(actual.loc[final, list(expected)] * 100 + 10000 + .000001, 0, 999999)),
        np.floor(np.clip(expected.loc[final] * 100 + 10000 + .000001, 0, 999999)))
    ranks = {day: i for i, day in enumerate(days)}
    gaps = f.date.map(ranks) - f.fi_prior_date.map(ranks)
    np.testing.assert_allclose(actual.fi_prior_gap, gaps, rtol=0, atol=0, equal_nan=True)
    assert r['rows'] == len(actual) and r['valid'] == int(final.sum())
    assert r['newly_invalid'] == int((old.formula_input_valid & ~final).sum())
    assert r['valid_with_prior_stock_day_gaps'] == int((final & gaps.gt(1)).sum())
    assert r['full_day_source_conflicts'] == int((~audit.full_day_source_valid).sum())
    assert r['invalid_prefix_sessions'] == int((~points.prefix_valid).sum())
    assert r['expressions'] == study.EXPRESSIONS and r['native_header'] == study.HEADER
    names = re.findall(r'(?m)^([A-Za-z][A-Za-z0-9]*):=', study.HEADER) + list(study.EXPRESSIONS)
    assert len(names) == len({n.casefold() for n in names})
    for prefix, symbol in [('FB', 'SZ399975'), ('FK', 'SZ399986')]:
        assert f'{prefix}P:=REF("{symbol}$CLOSE",B0);' in study.HEADER
        for minute in ['1420', '1448']:
            assert f'{prefix}{minute[2:]}:=VALUEWHEN(TIME={minute},"{symbol}$CLOSE");' in study.HEADER
    variables = dict(FB48=f.fi_broker_p48, FBP=f.fi_broker_prior, FB20=f.fi_broker_p20,
        FK48=f.fi_bank_p48, FKP=f.fi_bank_prior, FK20=f.fi_bank_p20)
    for name, expression in study.NEW_EXPRESSIONS.items():
        # Only the repository's two fixed arithmetic strings, no external expressions.
        value = eval(expression, {'__builtins__': {}}, variables).where(valid)
        np.testing.assert_allclose(value, expected[name], rtol=0, atol=2e-12, equal_nan=True)
    proof = dict(passed=True, feature_report_sha256=sha(ROOT / 'feature_report.json'),
        rows=len(actual), valid=int(final.sum()), effective_input_intersection_unchanged=bool(final.equals(old.formula_input_valid)), sessions=len(points), original_wire_replays=wire_count,
        all_stock_prior_dates_and_cross_symbol_values_independently_rebuilt=True,
        original_48_inputs_unchanged=True, all_integer_encodings_and_validity_verified=True,
        after_cutoff_perturbation_checks=perturbations, native_expression_arithmetic_verified=True,
        native_source_price_parity_verified=False, full_day_source_audit_used_for_selection=False,
        software_compilation_verified=False, new_selection_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_verification.json', proof); return proof


if __name__ == '__main__':
    print(json.dumps(main(), ensure_ascii=False, indent=2))
