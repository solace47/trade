"""Signed net movement relative to the path of prior daily closing prices."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as previous
from .corporate_cash import DAILY, MINUTES, save_json, sha
from .turnover_reference import CALENDAR

STEM = 'tail_formula_daily_efficiency'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
COMBINED_PROTOCOL = Path('config') / (STEM + '_combined_protocol.json')
DAILY_REPORT = Path('data/research/tail_formula_1000/feature_report.json')
PRICE_COLUMNS = [f'de_c{i:02d}' for i in range(1, 22)]
NEW_EXPRESSIONS = {
    'DE05': 'IF(DE5D>0,100*(DCP1-DCP6)/DE5D,0)',
    'DE20': 'IF(DE20D>0,100*(DCP1-DCP21)/DE20D,0)'}
EXPRESSIONS = {**previous.EXPRESSIONS, **NEW_EXPRESSIONS}
HEADER = previous.HEADER
for n in [5, 20]:
    HEADER += f'DE{n}D:=' + '+'.join(f'ABS(DCP{i}-DCP{i+1})' for i in range(1, n+1)) + ';\n'
native_core = base.native_core


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    r = json.loads((previous.ROOT / 'feature_report.json').read_text())
    v = json.loads((previous.ROOT / 'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(previous.ROOT / 'feature_report.json')
    assert r['features_sha256'] == sha(previous.ROOT / 'features.parquet')
    for file, digest in json.loads(DAILY_REPORT.read_text())['source_sha256'].items():
        assert sha(Path(file)) == digest
    assert p['history_first'] == '2023-06-01' and p['windows'] == [5, 20]
    assert not p['new_2026_prices_allowed']
    return p


def efficiency(prices, n):
    cents = np.rint(np.asarray(prices, dtype=float)[:, :n+1] * 100)
    changes = cents[:, :-1] - cents[:, 1:]
    distance = np.abs(changes).sum(axis=1)
    values = np.divide(100*changes.sum(axis=1), distance, out=np.zeros(len(prices)), where=distance>0)
    return np.where(np.isfinite(cents).all(axis=1), values, np.nan)


def features():
    assert not (ROOT / 'feature_report.json').exists(), 'Do not replace frozen daily-path inputs'
    p = checked_sources(); files = list(json.loads(DAILY_REPORT.read_text())['source_sha256'])
    c = base.conn(); c.read_parquet(files).create_view('daily')
    lagged = ','.join(f'lag(cl,{i}) OVER w AS de_c{i:02d}' for i in range(1, 22))
    hist = c.sql(f'''WITH active AS(SELECT date,code,close::DOUBLE AS raw_cl,round(close::DOUBLE,2) AS cl,
        preclose::DOUBLE AS pc,adjustflag::DOUBLE AS adj FROM daily
        WHERE tradestatus=1 AND date BETWEEN '2023-06-01' AND '2025-12-30'),
        atoms AS(SELECT *,coalesce(isfinite(raw_cl) AND raw_cl>0 AND abs(raw_cl-cl)<=.0001 AND adj=3,false) AS good,
            coalesce(abs(pc-lag(cl) OVER(PARTITION BY code ORDER BY date))>.005,false) AS reference_break FROM active),
        histories AS(SELECT date,code,{lagged},count(*) OVER h AS de_rows,sum(good::INT) OVER h AS de_good,
            min(date) OVER h AS de_first_date,max(date) OVER h AS de_last_date,
            sum(reference_break::INT) OVER h20 AS de_reference_breaks
            FROM atoms WINDOW w AS(PARTITION BY code ORDER BY date),
            h AS(PARTITION BY code ORDER BY date ROWS BETWEEN 21 PRECEDING AND 1 PRECEDING),
            h20 AS(PARTITION BY code ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING))
        SELECT * FROM histories WHERE date>='2024-01-01' ORDER BY date,code''').df(); c.close()
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    assert not ({n.casefold() for n in NEW_EXPRESSIONS} & {n.casefold() for n in old.columns})
    f = old.merge(hist, on=['date', 'code'], how='left', validate='one_to_one').sort_values(['date', 'code']).reset_index(drop=True)
    prices = f[PRICE_COLUMNS].to_numpy(float)
    valid = (f.de_rows.eq(21) & f.de_good.eq(21) & f.de_last_date.lt(f.date)
        & np.isfinite(prices).all(axis=1) & (prices > 0).all(axis=1))
    f['daily_efficiency_valid'] = valid
    for n in [5, 20]:
        f[f'DE{n:02d}'] = pd.Series(efficiency(prices, n)).where(valid)
    f['prior_formula_input_valid'] = f.formula_input_valid
    f['formula_input_valid'] &= valid & np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
    calendar = pd.read_parquet(CALENDAR)
    dates = sorted(calendar.loc[calendar.is_trading_day.eq('1') & calendar.calendar_date.between('2023-06-01', '2025-12-30'), 'calendar_date'])
    ranks = {d:i for i,d in enumerate(dates)}
    f['de_market_span'] = f.de_last_date.map(ranks)-f.de_first_date.map(ranks)+1
    f['de_last_gap'] = f.date.map(ranks)-f.de_last_date.map(ranks)
    ROOT.mkdir(parents=True, exist_ok=True); hist.to_parquet(ROOT / 'history.parquet', index=False, compression='zstd')
    f.to_parquet(ROOT / 'features.parquet', index=False, compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL), source_hashes=p['source_hashes'], calendar_sha256=sha(CALENDAR),
        history_sha256=sha(ROOT / 'history.parquet'), features_sha256=sha(ROOT / 'features.parquet'),
        rows=len(f), valid=int(f.formula_input_valid.sum()), previous_valid=int(old.formula_input_valid.sum()),
        newly_invalid=int((old.formula_input_valid & ~f.formula_input_valid).sum()),
        valid_with_reference_breaks=int((f.formula_input_valid & f.de_reference_breaks.gt(0)).sum()),
        valid_with_stock_day_gaps=int((f.formula_input_valid & (f.de_market_span.gt(21) | f.de_last_gap.gt(1))).sum()),
        flat_20_day_valid=int((f.formula_input_valid & (np.ptp(prices, axis=1)==0)).sum()),
        first_history_date=f.de_first_date.dropna().min(), last_history_date=f.de_last_date.dropna().max(),
        expressions=EXPRESSIONS, native_header=HEADER, raw_unadjusted_price_path=True,
        software_compilation_verified=False, native_source_parity_verified=False,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_report.json', r); return {k:v for k,v in r.items() if k not in ['expressions', 'native_header']}


def verify_features():
    p = checked_sources(); r = json.loads((ROOT / 'feature_report.json').read_text())
    for key, path in [('protocol_sha256', PROTOCOL), ('features_sha256', ROOT / 'features.parquet'),
                      ('history_sha256', ROOT / 'history.parquet'), ('calendar_sha256', CALENDAR)]:
        assert r[key] == sha(path)
    rebuilt = []
    for file in json.loads(DAILY_REPORT.read_text())['source_sha256']:
        d = pd.read_parquet(file, columns=['date', 'code', 'close', 'preclose', 'tradestatus', 'adjustflag'],
            filters=[('date', '>=', '2023-06-01'), ('date', '<=', '2025-12-30')])
        d = d.loc[d.tradestatus.eq(1)].sort_values('date').reset_index(drop=True)
        if d.empty:
            continue
        assert not d.date.duplicated().any()
        q = np.floor(d.close.astype(float)*100+.5)/100
        good = np.isfinite(d.close) & d.close.gt(0) & d.close.sub(q).abs().le(.0001) & d.adjustflag.eq(3)
        out = d[['date', 'code']].copy()
        for i in range(1, 22):
            out[f'de_c{i:02d}'] = q.shift(i)
        out['de_rows'] = np.minimum(np.arange(len(d)), 21)
        out['de_good'] = good.astype(int).rolling(21, min_periods=1).sum().shift()
        out['de_first_date'] = d.date.shift(21).fillna(d.date.iloc[0]); out.loc[0, 'de_first_date'] = None
        out['de_last_date'] = d.date.shift()
        out['de_reference_breaks'] = d.preclose.sub(q.shift()).abs().gt(.005).astype(int).rolling(20, min_periods=1).sum().shift()
        rebuilt.append(out.loc[out.date.ge('2024-01-01')])
    hist = pd.concat(rebuilt, ignore_index=True).sort_values(['date', 'code']).reset_index(drop=True)
    actual = pd.read_parquet(ROOT / 'history.parquet')
    pd.testing.assert_frame_equal(actual, hist[actual.columns], check_dtype=False, rtol=0, atol=2e-12)
    old = pd.read_parquet(previous.ROOT / 'features.parquet'); f = pd.read_parquet(ROOT / 'features.parquet')
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')], old.drop(columns='formula_input_valid'), check_exact=True)
    pd.testing.assert_series_equal(f.prior_formula_input_valid, old.formula_input_valid, check_exact=True, check_names=False)
    e = old[['date', 'code']].merge(hist, on=['date', 'code'], how='left', validate='one_to_one')
    pd.testing.assert_frame_equal(f[hist.columns], e[hist.columns], check_dtype=False, rtol=0, atol=2e-12)
    prices = e[PRICE_COLUMNS]; valid = (e.de_rows.eq(21) & e.de_good.eq(21) & e.de_last_date.lt(e.date)
        & np.isfinite(prices).all(axis=1) & prices.gt(0).all(axis=1))
    expected = pd.DataFrame(index=e.index)
    for n in [5, 20]:
        changes = [np.rint(100*e[f'de_c{i:02d}'])-np.rint(100*e[f'de_c{i+1:02d}']) for i in range(1, n+1)]
        positive = sum(change.clip(lower=0) for change in changes)
        negative = sum(-change.clip(upper=0) for change in changes)
        den = positive+negative
        expected[f'DE{n:02d}'] = ((100*(positive-negative)/den.where(den.ne(0))).mask(den.eq(0), 0)).where(valid)
    pd.testing.assert_frame_equal(f[list(expected)], expected, check_dtype=False, rtol=0, atol=2e-12)
    pd.testing.assert_series_equal(f.daily_efficiency_valid, valid, check_names=False, check_exact=True)
    final = old.formula_input_valid & valid & np.isfinite(expected).all(axis=1)
    np.testing.assert_array_equal(f.formula_input_valid, final)
    assert int(final.sum()) == r['valid'] and int((old.formula_input_valid & ~final).sum()) == r['newly_invalid']
    assert expected.loc[valid].abs().le(100+1e-12).all().all()
    np.testing.assert_array_equal(np.floor(100*f.loc[final, list(expected)]+10000+.000001),
                                  np.floor(100*expected.loc[final]+10000+.000001))
    calendar = pd.read_parquet(CALENDAR)
    dates = sorted(calendar.loc[calendar.is_trading_day.eq('1') & calendar.calendar_date.between('2023-06-01', '2025-12-30'), 'calendar_date'])
    ranks = {d:i for i,d in enumerate(dates)}
    np.testing.assert_allclose(f.de_market_span, e.de_last_date.map(ranks)-e.de_first_date.map(ranks)+1, rtol=0, atol=0, equal_nan=True)
    np.testing.assert_allclose(f.de_last_gap, e.date.map(ranks)-e.de_last_date.map(ranks), rtol=0, atol=0, equal_nan=True)
    assert r['valid_with_reference_breaks'] == int((final & e.de_reference_breaks.gt(0)).sum())
    assert r['valid_with_stock_day_gaps'] == int((final & (f.de_market_span.gt(21) | f.de_last_gap.gt(1))).sum())
    assert r['expressions'] == EXPRESSIONS and r['native_header'] == HEADER
    names = re.findall(r'(?m)^([A-Z][A-Z0-9]*):=', HEADER)+list(EXPRESSIONS)
    assert len(names) == len({name.casefold() for name in names})
    proof = dict(passed=True, feature_report_sha256=sha(ROOT / 'feature_report.json'), rows=len(f), history_rows=len(hist),
        all_21_prior_stock_days_independently_rebuilt=True, all_inputs_validity_and_integer_encodings_rebuilt=True,
        all_previous_48_inputs_and_keys_unchanged=True, new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_verification.json', proof); return proof


def native():
    checked_sources(); proof = json.loads((ROOT / 'feature_verification.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256'] == sha(ROOT / 'feature_report.json')
    f = pd.read_parquet(ROOT / 'features.parquet'); f = f.loc[f.formula_input_valid].copy()
    f['identity'] = [hashlib.sha256((d+'|'+s+'|daily-efficiency-native-v1').encode()).hexdigest() for d,s in zip(f.date,f.code)]
    sample = f.sort_values('identity').groupby('half', sort=True).head(8).sort_values(['date', 'code'])
    assert len(sample) == 32
    manifest = Path('data/research/economic_winner/input_manifest.json')
    hashes = json.loads(manifest.read_text())['source_sha256']; cases = []; total = 0
    for row in sample.itertuples():
        d = pd.read_parquet(DAILY / (row.code.replace('.', '_')+'.parquet'), columns=['date', 'close', 'tradestatus'],
            filters=[('date', '>=', '2023-06-01'), ('date', '<', row.date)])
        days = d.loc[d.tradestatus.eq(1)].sort_values('date').tail(21); assert len(days) == 21
        path = MINUTES / row.code[:2].upper() / (row.code[3:]+'.parquet'); assert sha(path) == hashes[str(path)]
        q = pd.read_parquet(path, columns=['timestamp', 'close'], filters=[
            ('timestamp', '>=', pd.Timestamp(days.date.iloc[0])), ('timestamp', '<=', pd.Timestamp(row.date+' 14:49'))])
        q['date'] = q.timestamp.dt.strftime('%Y-%m-%d'); q = q.loc[q.date.isin(days.date.tolist()+[row.date])].sort_values('timestamp')
        assert q.timestamp.max() == pd.Timestamp(row.date+' 14:49') and not q.timestamp.duplicated().any()
        ends = q.groupby('date', sort=True).tail(1).set_index('date')
        assert q.loc[q.date.lt(row.date)].groupby('date').size().eq(241).all()
        close = [round(float(ends.loc[day, 'close']), 2) for day in days.date.iloc[::-1]]
        np.testing.assert_allclose(close, [getattr(row, name) for name in PRICE_COLUMNS], rtol=0, atol=2e-12)
        values = []
        for n in [5, 20]:
            distance = sum(abs(close[i]-close[i+1]) for i in range(n))
            values.append(100*(close[0]-close[n])/distance if distance else 0.)
        np.testing.assert_allclose(values, [row.DE05, row.DE20], rtol=0, atol=2e-10)
        np.testing.assert_array_equal(np.floor(100*np.asarray(values)+10000+.000001),
                                      np.floor(100*np.asarray([row.DE05,row.DE20])+10000+.000001))
        total += len(q); cases.append(dict(date=row.date, code=row.code, minutes=len(q), first_history_date=days.date.iloc[0], source_sha256=hashes[str(path)]))
    r = dict(passed=True, feature_report_sha256=sha(ROOT / 'feature_report.json'),
        feature_verification_sha256=sha(ROOT / 'feature_verification.json'), samples=cases, raw_minutes=total,
        prior_stock_day_final_minute_closes_and_scalar_native_expressions_rebuilt=True,
        software_compilation_verified=False, native_source_parity_verified=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'native_input_verification.json', r); return r


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('stage', choices=['features', 'verify_features', 'native'])
    a = p.parse_args(); print(json.dumps(globals()[a.stage](), ensure_ascii=False, indent=2))
