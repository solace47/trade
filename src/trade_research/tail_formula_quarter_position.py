"""Strictly prior 60-stock-day closing-price position for next-morning research."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as previous
from .corporate_cash import MINUTES, save_json, sha
from .turnover_reference import CALENDAR

STEM = 'tail_formula_quarter_position'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
DAILY_REPORT = Path('data/research/tail_formula_1000/feature_report.json')
MINUTE_MANIFEST = Path('data/research/economic_winner/input_manifest.json')
NATIVE_PROTOCOL = Path('config') / (STEM + '_native_protocol.json')
NEW_EXPRESSIONS = {'QP01': '100*(Q/QPM60-1)/V01',
                   'QP02': '100*(Q-QPL60)/MAX(QPH60-QPL60,0.01)'}
EXPRESSIONS = {**previous.EXPRESSIONS, **NEW_EXPRESSIONS}
HEADER = previous.HEADER
for i in range(21, 60):
    HEADER += f'B{i}:=B{i-1}+REF(BARSLAST(DD)+1,B{i-1});\n'
for i in range(22, 61):
    HEADER += f'DCP{i}:=REF(C,B{i-1});\n'
HEADER += 'QPM60:=(' + '+'.join(f'DCP{i}' for i in range(1, 61)) + ')/60;\n'
for direction, op in [('H', 'MAX'), ('L', 'MIN')]:
    HEADER += f'QP{direction}01:=DCP1;\n'
    for i in range(2, 61):
        HEADER += f'QP{direction}{i:02d}:={op}(QP{direction}{i-1:02d},DCP{i});\n'
native_core = base.native_core


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    assert p['history_first'] == '2023-01-01' and p['history_stock_days'] == 60
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    report = json.loads((previous.ROOT / 'feature_report.json').read_text())
    proof = json.loads((previous.ROOT / 'feature_verification.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256'] == sha(previous.ROOT / 'feature_report.json')
    assert report['features_sha256'] == sha(previous.ROOT / 'features.parquet')
    sources = json.loads(DAILY_REPORT.read_text())['source_sha256']
    for file, digest in sources.items():
        assert sha(Path(file)) == digest
    return p, sources


def position(price, volatility, total_cents, lo_cents, hi_cents):
    mean = np.asarray(total_cents, dtype=float) / 6000
    low = np.asarray(lo_cents, dtype=float) / 100
    high = np.asarray(hi_cents, dtype=float) / 100
    return 100*(price/mean-1)/volatility, 100*(price-low)/np.maximum(high-low, .01)


def trading_minute_projection(raw, daily, signal_date):
    """Project a padded vendor grid onto the predeclared active-day convention.

    This is a conditional replay convention, not proof of client data parity.
    Unexplained, missing, or traded extra dates remain errors.
    """
    assert not daily.date.duplicated().any()
    expected = set(daily.loc[daily.tradestatus.eq(1), 'date']) | {signal_date}
    dates = raw.timestamp.dt.strftime('%Y-%m-%d')
    assert expected <= set(dates), 'Missing active-day minutes cannot be discarded'
    removed = []
    for day in sorted(set(dates)-expected):
        state = daily.loc[daily.date.eq(day)]
        assert len(state) == 1 and state.tradestatus.iloc[0] == 0, 'Unexplained extra date'
        bars = raw.loc[dates.eq(day)]
        assert bars[['close', 'volume', 'turnover']].notna().to_numpy().all(), 'Missing halt values cannot establish padding'
        assert pd.notna(state.close.iloc[0]) and np.isfinite(float(state.close.iloc[0]))
        assert bars.volume.eq(0).all() and bars.turnover.eq(0).all(), 'Traded records cannot be treated as halt padding'
        assert np.isfinite(bars.close).all() and bars.close.sub(state.close.iloc[0]).abs().le(.0001).all()
        removed.append(dict(date=day, rows=len(bars), daily_status=0, all_volume_and_turnover_zero=True))
    projected = raw.loc[dates.isin(expected)].reset_index(drop=True)
    return projected, removed


def features():
    assert not (ROOT / 'feature_report.json').exists(), 'Do not replace fixed quarter inputs'
    p, sources = checked_sources()
    c = base.conn(); c.read_parquet(list(sources)).create_view('daily')
    hist = c.sql('''WITH a AS(SELECT date,code,close::DOUBLE AS raw_cl,
        try_cast(floor(close::DOUBLE*100+.5) AS BIGINT) AS cents,preclose::DOUBLE AS pc,adjustflag::DOUBLE AS adj
        FROM daily WHERE tradestatus=1 AND date BETWEEN '2023-01-01' AND '2025-12-30'),
        b AS(SELECT *,coalesce(isfinite(raw_cl) AND raw_cl>0 AND abs(raw_cl-cents/100.)<=.0001 AND adj=3,false) AS good,
        coalesce(abs(pc-lag(cents) OVER(PARTITION BY code ORDER BY date)/100.)>.005,false) AS reference_break FROM a),
        h AS(SELECT date,code,count(*) OVER w AS qp_rows,count(*) FILTER(WHERE good) OVER w AS qp_good,
        sum(cents) OVER w AS qp_sum,min(cents) OVER w AS qp_low,max(cents) OVER w AS qp_high,
        min(date) OVER w AS qp_first_date,max(date) OVER w AS qp_last_date,
        count(*) FILTER(WHERE reference_break) OVER w AS qp_reference_breaks FROM b
        WINDOW w AS(PARTITION BY code ORDER BY date ROWS BETWEEN 60 PRECEDING AND 1 PRECEDING))
        SELECT * FROM h WHERE date>='2024-01-01' ORDER BY date,code''').df(); c.close()
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    f = old.merge(hist, on=['date', 'code'], how='left', validate='one_to_one').sort_values(['date', 'code']).reset_index(drop=True)
    valid = (f.qp_rows.eq(60) & f.qp_good.eq(60) & f.qp_last_date.lt(f.date)
             & f.qp_sum.gt(0) & f.qp_low.gt(0) & f.qp_high.ge(f.qp_low))
    f['quarter_position_valid'] = valid
    for name, value in zip(NEW_EXPRESSIONS, position(f.price_1449, f.V01, f.qp_sum, f.qp_low, f.qp_high)):
        f[name] = pd.Series(value).where(valid)
    f['prior_formula_input_valid'] = f.formula_input_valid
    f['formula_input_valid'] &= valid & np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
    cal = pd.read_parquet(CALENDAR)
    days = sorted(cal.loc[cal.is_trading_day.eq('1') & cal.calendar_date.between('2023-01-01', '2025-12-30'), 'calendar_date'])
    rank = {d:i for i,d in enumerate(days)}
    f['qp_market_span'] = f.qp_last_date.map(rank)-f.qp_first_date.map(rank)+1
    f['qp_last_gap'] = f.date.map(rank)-f.qp_last_date.map(rank)
    ROOT.mkdir(parents=True, exist_ok=True)
    hist.to_parquet(ROOT / 'history.parquet', index=False, compression='zstd')
    f.to_parquet(ROOT / 'features.parquet', index=False, compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL), source_hashes=p['source_hashes'], calendar_sha256=sha(CALENDAR),
        history_sha256=sha(ROOT / 'history.parquet'), features_sha256=sha(ROOT / 'features.parquet'),
        rows=len(f), valid=int(f.formula_input_valid.sum()), previous_valid=int(old.formula_input_valid.sum()),
        newly_invalid=int((old.formula_input_valid & ~f.formula_input_valid).sum()),
        valid_with_reference_breaks=int((f.formula_input_valid & f.qp_reference_breaks.gt(0)).sum()),
        valid_with_stock_day_gaps=int((f.formula_input_valid & (f.qp_market_span.gt(60) | f.qp_last_gap.gt(1))).sum()),
        first_history_date=f.qp_first_date.min(), last_history_date=f.qp_last_date.max(),
        expressions=EXPRESSIONS, native_header=HEADER, price_history_is_raw_unadjusted=True,
        software_compilation_verified=False, native_source_parity_verified=False,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_report.json', r)
    return {k:v for k,v in r.items() if k not in ['expressions', 'native_header']}


def verify_features():
    p, sources = checked_sources(); r = json.loads((ROOT / 'feature_report.json').read_text())
    for key, path in [('protocol_sha256', PROTOCOL), ('features_sha256', ROOT / 'features.parquet'),
                      ('history_sha256', ROOT / 'history.parquet'), ('calendar_sha256', CALENDAR)]:
        assert r[key] == sha(path)
    rebuilt = []
    for file in sources:
        d = pd.read_parquet(file, columns=['date', 'code', 'close', 'preclose', 'tradestatus', 'adjustflag'],
            filters=[('date', '>=', '2023-01-01'), ('date', '<=', '2025-12-30')])
        d = d.loc[d.tradestatus.eq(1)].sort_values('date').reset_index(drop=True)
        if d.empty:
            continue
        assert not d.date.duplicated().any()
        cents = np.floor(d.close*100+.5).where(np.isfinite(d.close))
        good = np.isfinite(d.close) & d.close.gt(0) & d.close.sub(cents/100).abs().le(.0001) & d.adjustflag.eq(3)
        o = d[['date', 'code']].copy()
        o['qp_rows'] = np.minimum(np.arange(len(d)), 60)
        o['qp_good'] = good.astype(int).rolling(60, min_periods=1).sum().shift().fillna(0)
        o['qp_sum'] = cents.rolling(60, min_periods=1).sum().shift()
        o['qp_low'] = cents.rolling(60, min_periods=1).min().shift()
        o['qp_high'] = cents.rolling(60, min_periods=1).max().shift()
        o['qp_first_date'] = d.date.shift(60).fillna(d.date.iloc[0]); o.loc[0, 'qp_first_date'] = None
        o['qp_last_date'] = d.date.shift()
        o['qp_reference_breaks'] = d.preclose.sub(cents.shift()/100).abs().gt(.005).astype(int).rolling(60, min_periods=1).sum().shift().fillna(0)
        rebuilt.append(o.loc[o.date.ge('2024-01-01')])
    hist = pd.concat(rebuilt, ignore_index=True).sort_values(['date', 'code']).reset_index(drop=True)
    actual = pd.read_parquet(ROOT / 'history.parquet')
    pd.testing.assert_frame_equal(actual, hist[actual.columns], check_dtype=False, check_exact=True)
    old = pd.read_parquet(previous.ROOT / 'features.parquet'); f = pd.read_parquet(ROOT / 'features.parquet')
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')], old.drop(columns='formula_input_valid'), check_exact=True)
    e = old[['date', 'code']].merge(hist, on=['date', 'code'], how='left', validate='one_to_one')
    pd.testing.assert_frame_equal(f[hist.columns], e[hist.columns], check_dtype=False, check_exact=True)
    c = base.conn(); c.register('history', e)
    c.register('visible', old[['date', 'code', 'price_1449', 'V01', 'formula_input_valid']])
    expected = c.sql('''SELECT date,code,coalesce(qp_rows=60 AND qp_good=60 AND qp_last_date<date
        AND qp_sum>0 AND qp_low>0 AND qp_high>=qp_low,false) AS valid,
        CASE WHEN valid THEN 100*(price_1449*6000/qp_sum-1)/V01 END AS QP01,
        CASE WHEN valid THEN 100*(100*price_1449-qp_low)/greatest(qp_high-qp_low,1) END AS QP02
        FROM visible JOIN history USING(date,code) ORDER BY date,code''').df(); c.close()
    np.testing.assert_array_equal(f.quarter_position_valid, expected.valid)
    for name in NEW_EXPRESSIONS:
        np.testing.assert_allclose(f[name], expected[name], rtol=0, atol=2e-9, equal_nan=True)
        v = np.isfinite(expected[name])
        np.testing.assert_array_equal(np.floor(np.clip(100*f.loc[v,name]+10000+.000001,0,999999)),
                                      np.floor(np.clip(100*expected.loc[v,name]+10000+.000001,0,999999)))
    np.testing.assert_array_equal(f.formula_input_valid, old.formula_input_valid & expected.valid & np.isfinite(expected[list(NEW_EXPRESSIONS)]).all(axis=1))
    assert int(f.formula_input_valid.sum()) == r['valid'] and len(f) == r['rows']
    proof = dict(passed=True, feature_report_sha256=sha(ROOT / 'feature_report.json'), rows=len(f),
        independent_pandas_60_day_windows_and_sql_values_verified=True,
        all_prior_48_values_unchanged=True, all_keys_validity_and_integer_encodings_verified=True,
        effective_input_intersection_unchanged=bool(np.array_equal(f.formula_input_valid, old.formula_input_valid)),
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_verification.json', proof); return proof


def native():
    p, _ = checked_sources(); proof = json.loads((ROOT / 'feature_verification.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256'] == sha(ROOT / 'feature_report.json')
    amendment = json.loads(NATIVE_PROTOCOL.read_text())
    assert amendment['inputs_protocol_sha256'] == sha(PROTOCOL)
    for file, digest in amendment['evidence'].items():
        assert sha(Path(file)) == digest
    f = pd.read_parquet(ROOT / 'features.parquet'); selected = f.loc[f.formula_input_valid].copy()
    selected['sample_hash'] = [hashlib.sha256((d+'|'+s+'|quarter-position-v1').encode()).hexdigest() for d,s in zip(selected.date,selected.code)]
    sample = selected.sort_values('sample_hash').groupby('half', sort=True).head(8).sort_values(['date', 'code'])
    sources = json.loads(MINUTE_MANIFEST.read_text())['source_sha256']; receipts = []
    from .corporate_cash import DAILY
    for row in sample.itertuples():
        daily_all = pd.read_parquet(DAILY / (row.code.replace('.', '_')+'.parquet'), columns=['date', 'close', 'tradestatus'],
            filters=[('date', '>=', row.qp_first_date), ('date', '<', row.date)])
        daily = daily_all.loc[daily_all.tradestatus.eq(1)].sort_values('date'); assert len(daily) == 60
        path = MINUTES / row.code[:2].upper() / (row.code[3:]+'.parquet'); assert sha(path) == sources[str(path)]
        raw = pd.read_parquet(path, columns=['timestamp', 'close', 'volume', 'turnover'], filters=[
            ('timestamp', '>=', pd.Timestamp(row.qp_first_date)), ('timestamp', '<=', pd.Timestamp(row.date+' 14:49'))]).sort_values('timestamp').reset_index(drop=True)
        vendor_rows = len(raw)
        raw, removed = trading_minute_projection(raw, daily_all, row.date)
        dates = raw.timestamp.dt.strftime('%Y-%m-%d'); assert dates.drop_duplicates().tolist() == daily.date.tolist()+[row.date]
        assert not raw.timestamp.duplicated().any()
        last = raw.groupby(dates, sort=True).tail(1)
        assert last.timestamp.dt.strftime('%H:%M').tolist() == ['15:00']*60+['14:49']
        closes = np.floor(last.close.to_numpy(float)*100+.5)/100
        np.testing.assert_allclose(closes[:-1], daily.close, rtol=0, atol=2e-12)
        assert closes[-1] == row.price_1449
        # B0 counts current bars; recursively adding each preceding day's
        # size must select exactly DCP1...DCP60, irrespective of day lengths.
        sizes = raw.groupby(dates, sort=True).size().to_numpy()[::-1]
        offsets = np.cumsum(sizes)[:60]
        refs = np.floor(raw.close.to_numpy(float)[len(raw)-1-offsets]*100+.5)/100
        np.testing.assert_array_equal(refs, closes[:-1][::-1])
        ma = sum(refs)/60; high = max(refs); low = min(refs)
        values = [100*(closes[-1]/ma-1)/row.V01, 100*(closes[-1]-low)/max(high-low,.01)]
        for name, value in zip(NEW_EXPRESSIONS, values):
            np.testing.assert_allclose(value, getattr(row,name), rtol=0, atol=2e-9)
            assert np.floor(np.clip(100*value+10000+.000001,0,999999)) == np.floor(np.clip(100*getattr(row,name)+10000+.000001,0,999999))
        receipts.append(dict(date=row.date, code=row.code, first=row.qp_first_date, raw_minutes=len(raw),
            vendor_rows=vendor_rows, removed_halt_padding=removed, source_sha256=sources[str(path)]))
    assert len(receipts) == 32
    r = dict(passed=True, feature_report_sha256=sha(ROOT / 'feature_report.json'),
        feature_verification_sha256=sha(ROOT / 'feature_verification.json'), samples=receipts,
        raw_minutes=sum(x['raw_minutes'] for x in receipts), vendor_rows=sum(x['vendor_rows'] for x in receipts),
        native_protocol_sha256=sha(NATIVE_PROTOCOL), conditional_active_day_projection=True,
        all_60_native_stock_day_boundaries_replayed=True,
        software_history_depth_verified=False, software_compilation_verified=False,
        native_source_parity_verified=False, new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'native_input_verification.json', r); return r


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['features', 'verify_features', 'native'])
    print(json.dumps(globals()[parser.parse_args().stage](), ensure_ascii=False, indent=2))
