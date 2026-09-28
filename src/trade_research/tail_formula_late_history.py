"""Strictly prior stock-day 14:49-to-close movement, from verified caches."""
import argparse
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as previous
from .corporate_cash import save_json, sha
from .turnover_reference import CALENDAR

STEM = 'tail_formula_late_history'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
DAILY_REPORT = Path('data/research/tail_formula_1000/feature_report.json')
WINDOWS = Path('data/research/tail_formula_morning_history')
SCHEDULE = Path('data/research/tail_formula_volume_history')
DAILY_PROOF = Path('data/research/tail_formula_daily_efficiency/native_input_verification.json')
HEADER = previous.HEADER + ''.join(
    f'LQG{i}:=INTPART(DCP{i}*100+0.5)/MAX(INTPART(REF(Q,B{i-1})*100+0.5),1)-1;\n'
    for i in range(1, 21))
NEW_EXPRESSIONS = {'LQ01': '100*(' + '+'.join(f'LQG{i}' for i in range(1, 21)) + ')/20/V01'}
EXPRESSIONS = {**previous.EXPRESSIONS, **NEW_EXPRESSIONS}


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    for folder in [previous.ROOT, WINDOWS]:
        name = 'feature' if folder == previous.ROOT else 'window'
        r = json.loads((folder / (name + '_report.json')).read_text())
        v = json.loads((folder / (name + '_verification.json')).read_text())
        assert v['passed'] and v[name + '_report_sha256'] == sha(folder / (name + '_report.json'))
        key = 'features' if name == 'feature' else 'windows'
        assert r[key + '_sha256'] == sha(folder / (key + '.parquet'))
    m = json.loads((SCHEDULE / 'input_manifest.json').read_text())
    assert m['stock_days_sha256'] == sha(SCHEDULE / 'stock_days.parquet')
    assert m['daily_source_sha256'] == json.loads(DAILY_REPORT.read_text())['source_sha256']
    for file, digest in m['daily_source_sha256'].items():
        assert sha(Path(file)) == digest
    assert sha(CALENDAR) == '25d81e17d9e32f25fc7f61cda244f2ea2df818e6a1c1669d0e8bfcf1c1696400'
    assert p['history_days'] == 20 and p['new_fields'] == list(NEW_EXPRESSIONS)
    assert (p['history_first'], p['history_last']) == ('2023-06-01', '2025-12-30')
    assert not p['new_2026_prices_allowed']
    return p


def prior_tail_mean(tails, closes, valid, n=20):
    """One stock's strict prior n observations; missing observations never skipped."""
    tails = np.floor(np.asarray(tails, dtype=float) * 100 + .5)
    closes = np.floor(np.asarray(closes, dtype=float) * 100 + .5)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(tails) & np.isfinite(closes) & (tails > 0) & (closes > 0)
    values = np.full(len(tails), np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        atoms = (closes - tails) / tails
    for i in range(n, len(values)):
        if valid[i-n:i].all():
            values[i] = math.fsum(atoms[i-n:i]) / n
    return values


def market_ranks():
    f = pd.read_parquet(CALENDAR)
    days = sorted(f.loc[f.is_trading_day.eq('1') & f.calendar_date.between('2023-06-01', '2025-12-30'), 'calendar_date'])
    return {day: i for i, day in enumerate(days)}


def features():
    p = checked_sources()
    assert not (ROOT / 'feature_report.json').exists(), 'Do not replace frozen inputs'
    c = base.conn()
    c.read_parquet(list(json.loads(DAILY_REPORT.read_text())['source_sha256'])).create_view('daily')
    c.read_parquet(str(WINDOWS / 'windows.parquet')).create_view('tails')
    active = c.sql('''SELECT date,code,close::DOUBLE AS cl,adjustflag::DOUBLE AS adj
        FROM daily WHERE tradestatus=1 AND date BETWEEN '2023-06-01' AND '2025-12-30'
        ORDER BY code,date''').df()
    schedule = pd.read_parquet(SCHEDULE / 'stock_days.parquet', columns=['date', 'code']).sort_values(['code', 'date']).reset_index(drop=True)
    pd.testing.assert_frame_equal(active[['date', 'code']], schedule, check_dtype=False, check_exact=True)
    active['market_position'] = active.date.map(market_ranks())
    assert active.market_position.notna().all()
    c.register('active', active)
    hist = c.sql('''WITH a AS(SELECT d.*,t.price_1449 AS q,t.tail_valid,
        coalesce(t.tail_valid AND isfinite(t.price_1449) AND t.price_1449>0
            AND isfinite(d.cl) AND d.cl>0 AND abs(d.cl-round(d.cl,2))<=.0001 AND d.adj=3,false) AS good,
        (round(d.cl*100)-round(t.price_1449*100))/round(t.price_1449*100) AS atom
        FROM active d LEFT JOIN tails t USING(date,code)),
        h AS(SELECT date,code,count(*) OVER w AS lq_rows,sum(good::INT) OVER w AS lq_good,
            min(date) OVER w AS lq_start,max(date) OVER w AS lq_end,
            min(market_position) OVER w AS lq_start_position,max(market_position) OVER w AS lq_end_position,
            CASE WHEN count(*) OVER w=20 AND sum(good::INT) OVER w=20
                THEN avg(atom) OVER w ELSE NULL END AS lq_mean
        FROM a WINDOW w AS(PARTITION BY code ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING))
        SELECT * FROM h WHERE date>='2024-01-01' ORDER BY date,code''').df()
    c.close()
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    f = old.merge(hist, on=['date', 'code'], how='left', validate='one_to_one').sort_values(['date', 'code']).reset_index(drop=True)
    valid = (f.lq_rows.eq(20) & f.lq_good.eq(20) & f.lq_end.lt(f.date)
             & np.isfinite(f[['lq_mean', 'V01']]).all(axis=1) & f.V01.gt(0))
    f['late_history_valid'] = valid
    f['prior_formula_input_valid'] = f.formula_input_valid
    f['LQ01'] = (100 * f.lq_mean / f.V01).where(valid)
    f['formula_input_valid'] &= valid & np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
    f['lq_market_span'] = f.lq_end_position - f.lq_start_position + 1
    f['lq_last_gap'] = f.date.map(market_ranks()) - f.lq_end_position
    ROOT.mkdir(parents=True, exist_ok=True)
    hist.to_parquet(ROOT / 'history.parquet', index=False, compression='zstd')
    f.to_parquet(ROOT / 'features.parquet', index=False, compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL), source_hashes=p['source_hashes'],
             features_sha256=sha(ROOT / 'features.parquet'), history_sha256=sha(ROOT / 'history.parquet'),
             rows=len(f), valid=int(f.formula_input_valid.sum()), previous_valid=int(old.formula_input_valid.sum()),
             newly_invalid=int((old.formula_input_valid & ~f.formula_input_valid).sum()),
             valid_with_stock_day_gaps=int((f.formula_input_valid & (f.lq_market_span.gt(20) | f.lq_last_gap.gt(1))).sum()),
             first_history_date=f.lq_start.dropna().min(), last_history_date=f.lq_end.dropna().max(),
             expressions=EXPRESSIONS, native_header=HEADER,
             native_source_parity_verified=False, software_compilation_verified=False,
             new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_report.json', r)
    return {k: v for k, v in r.items() if k not in ['source_hashes', 'expressions', 'native_header']}


def verify_features():
    checked_sources()
    r = json.loads((ROOT / 'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['features_sha256'] == sha(ROOT / 'features.parquet')
    assert r['history_sha256'] == sha(ROOT / 'history.parquet')
    tails = pd.read_parquet(WINDOWS / 'windows.parquet', columns=['date', 'code', 'price_1449', 'tail_valid'])
    by_code = {code: d for code, d in tails.groupby('code', sort=False)}
    ranks = market_ranks()
    pieces = []
    for file in json.loads(DAILY_REPORT.read_text())['source_sha256']:
        d = pd.read_parquet(file, columns=['date', 'code', 'close', 'adjustflag', 'tradestatus'],
                            filters=[('date', '>=', '2023-06-01'), ('date', '<=', '2025-12-30')])
        d = d.loc[d.tradestatus.eq(1)].sort_values('date').reset_index(drop=True)
        if d.empty:
            continue
        assert not d.date.duplicated().any()
        d = d.merge(by_code[d.code.iloc[0]], on=['date', 'code'], how='left', validate='one_to_one')
        cl = d.close.astype(float)
        cents = np.floor(cl * 100 + .5)
        good = (d.tail_valid.eq(True) & np.isfinite(d.price_1449) & d.price_1449.gt(0)
                & np.isfinite(cl) & cl.gt(0) & cl.sub(cents / 100).abs().le(.0001) & d.adjustflag.eq(3))
        q = d[['date', 'code']].copy()
        q['lq_rows'] = np.minimum(np.arange(len(d)), 20)
        q['lq_good'] = good.astype(int).rolling(20, min_periods=1).sum().shift()
        q['lq_start'] = d.date.shift(20).fillna(d.date.iloc[0]); q.loc[0, 'lq_start'] = None
        q['lq_end'] = d.date.shift()
        q['lq_start_position'] = q.lq_start.map(ranks)
        q['lq_end_position'] = q.lq_end.map(ranks)
        q['lq_mean'] = prior_tail_mean(d.price_1449, cl, good)
        pieces.append(q.loc[q.date.ge('2024-01-01')])
    h = pd.concat(pieces, ignore_index=True).sort_values(['date', 'code']).reset_index(drop=True)
    actual = pd.read_parquet(ROOT / 'history.parquet')
    # DuckDB emits nullable integers for empty prior windows; compare missing
    # calendar positions in the same floating representation as the rebuild.
    for field in ['lq_start_position', 'lq_end_position']:
        actual[field] = actual[field].astype(float)
    pd.testing.assert_frame_equal(actual, h[actual.columns], check_dtype=False, rtol=0, atol=3e-14)
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    f = pd.read_parquet(ROOT / 'features.parquet')
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')], old.drop(columns='formula_input_valid'), check_exact=True)
    pd.testing.assert_series_equal(f.prior_formula_input_valid, old.formula_input_valid, check_names=False, check_exact=True)
    e = old[['date', 'code']].merge(h, on=['date', 'code'], how='left', validate='one_to_one')
    valid = (e.lq_rows.eq(20) & e.lq_good.eq(20) & e.lq_end.lt(e.date)
             & np.isfinite(e.lq_mean) & np.isfinite(old.V01) & old.V01.gt(0))
    value = (100 * e.lq_mean / old.V01).where(valid)
    np.testing.assert_allclose(f.LQ01, value, rtol=0, atol=2e-10, equal_nan=True)
    np.testing.assert_array_equal(f.late_history_valid, valid)
    final = old.formula_input_valid & valid & np.isfinite(value)
    np.testing.assert_array_equal(f.formula_input_valid, final)
    for field, expected in [('lq_market_span', e.lq_end_position - e.lq_start_position + 1),
                            ('lq_last_gap', e.date.map(ranks) - e.lq_end_position)]:
        np.testing.assert_allclose(f[field], expected, rtol=0, atol=0, equal_nan=True)
    assert r['valid'] == int(final.sum()) and r['newly_invalid'] == int((old.formula_input_valid & ~final).sum())
    np.testing.assert_array_equal(np.floor(np.clip(100 * f.loc[final, 'LQ01'] + 10000 + .000001, 0, 999999)),
                                  np.floor(np.clip(100 * value[final] + 10000 + .000001, 0, 999999)))
    names = re.findall(r'(?m)^([A-Z][A-Z0-9]*):=', HEADER) + list(EXPRESSIONS)
    assert len(names) == len({n.casefold() for n in names})
    assert r['expressions'] == EXPRESSIONS and r['native_header'] == HEADER
    proof = dict(passed=True, feature_report_sha256=sha(ROOT / 'feature_report.json'), rows=len(f), valid=int(final.sum()),
                 all_prior_dates_values_validity_and_integer_encodings_rebuilt=True,
                 all_original_48_values_unchanged=True,
                 effective_input_intersection_unchanged=bool(np.array_equal(final, old.formula_input_valid)),
                 new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_verification.json', proof)
    return proof


def native():
    checked_sources()
    v = json.loads((ROOT / 'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(ROOT / 'feature_report.json')
    old = json.loads(DAILY_PROOF.read_text())
    assert old['passed'] and len(old['samples']) == 32
    f = pd.read_parquet(ROOT / 'features.parquet', columns=['date', 'code', 'formula_input_valid', 'LQ01', 'V01']).set_index(['date', 'code'])
    c = base.conn()
    c.read_parquet(list(json.loads(DAILY_REPORT.read_text())['source_sha256'])).create_view('daily')
    c.read_parquet(str(WINDOWS / 'windows.parquet')).create_view('tails')
    cases = []
    for case in old['samples']:
        row = f.loc[(case['date'], case['code'])]
        d = c.execute('''SELECT d.date,d.close::DOUBLE AS cl,t.price_1449 AS q,t.tail_valid
            FROM daily d LEFT JOIN tails t USING(date,code)
            WHERE d.code=? AND d.date>=? AND d.date<? AND d.tradestatus=1 ORDER BY d.date''',
            [case['code'], case['first_history_date'], case['date']]).df()
        assert len(d) == 21
        d = d.iloc[1:]
        record = dict(date=case['date'], code=case['code'], first_date=d.date.iloc[0], last_date=d.date.iloc[-1],
                      source_days=20, formula_input_valid=bool(row.formula_input_valid))
        if row.formula_input_valid:
            assert d.tail_valid.all()
            atoms = [math.floor(cl * 100 + .5) / max(math.floor(q * 100 + .5), 1) - 1 for cl, q in zip(d.cl, d.q)]
            value = 100 * math.fsum(atoms) / 20 / row.V01
            np.testing.assert_allclose(value, row.LQ01, rtol=0, atol=2e-10)
            assert math.floor(100 * value + 10000 + .000001) == math.floor(100 * row.LQ01 + 10000 + .000001)
            record['LQ01'] = value
        cases.append(record)
    c.close()
    proof = dict(passed=True, protocol_sha256=sha(PROTOCOL), feature_report_sha256=sha(ROOT / 'feature_report.json'),
                 feature_verification_sha256=sha(ROOT / 'feature_verification.json'),
                 reused_daily_raw_proof_sha256=sha(DAILY_PROOF),
                 reused_all_tail_window_proof_sha256=sha(WINDOWS / 'window_verification.json'),
                 cases=cases, original_raw_minutes_not_reextracted=True, native_scalar_algebra_and_encodings_rebuilt=True,
                 software_compilation_verified=False, native_source_parity_verified=False,
                 new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'native_input_verification.json', proof)
    return {k: v for k, v in proof.items() if k != 'cases'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['features', 'verify_features', 'native'])
    print(json.dumps(globals()[p.parse_args().stage](), ensure_ascii=False, indent=2))
