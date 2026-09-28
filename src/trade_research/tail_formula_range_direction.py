"""Direction of strictly prior daily range migration, not closing-price drift."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as previous
from .corporate_cash import MINUTES, save_json, sha
from .turnover_reference import CALENDAR

STEM = 'tail_formula_range_direction'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
DAILY_REPORT = Path('data/research/tail_formula_1000/feature_report.json')
PROBES = Path('data/research/tail_formula_daily_efficiency/native_input_verification.json')
NEW_EXPRESSIONS = {'HM01': '100*(HMSU-HMSD)/MAX(HMSU+HMSD,1)', 'HM02': '100*(HMSU+HMSD)/MAX(HMST,1)'}
EXPRESSIONS = {**previous.EXPRESSIONS, **NEW_EXPRESSIONS}
EXTRA_HEADER = 'HMH:=INTPART(HHV(H,B0)*100+0.5);\nHML:=INTPART(LLV(L,B0)*100+0.5);\n'
for i in range(1, 22):
    EXTRA_HEADER += f'HMH{i}:=REF(HMH,B{i-1});\nHML{i}:=REF(HML,B{i-1});\n'
for i in range(1, 21):
    EXTRA_HEADER += f'HMU{i}:=MAX(HMH{i}-HMH{i+1},0);\nHMD{i}:=MAX(HML{i+1}-HML{i},0);\n'
    EXTRA_HEADER += f'HMUP{i}:=IF(HMU{i}>HMD{i},HMU{i},0);\nHMDN{i}:=IF(HMD{i}>HMU{i},HMD{i},0);\n'
    EXTRA_HEADER += f'HMT{i}:=MAX(HMH{i}-HML{i},MAX(ABS(HMH{i}-INTPART(DCP{i+1}*100+0.5)),ABS(HML{i}-INTPART(DCP{i+1}*100+0.5))));\n'
for name, prefix in [('HMSU', 'HMUP'), ('HMSD', 'HMDN'), ('HMST', 'HMT')]:
    EXTRA_HEADER += name + ':=' + '+'.join(f'{prefix}{i}' for i in range(1, 21)) + ';\n'
HEADER = previous.HEADER + EXTRA_HEADER
native_core = base.native_core


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    r = json.loads((previous.ROOT / 'feature_report.json').read_text())
    v = json.loads((previous.ROOT / 'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(previous.ROOT / 'feature_report.json')
    assert r['features_sha256'] == sha(previous.ROOT / 'features.parquet')
    sources = json.loads(DAILY_REPORT.read_text())['source_sha256']
    for file, digest in sources.items():
        assert sha(Path(file)) == digest
    assert p['history_days'] == 20 and p['new_fields'] == list(NEW_EXPRESSIONS) and not p['new_2026_prices_allowed']
    return p, sources


def history_arrays(high, low, close, adjustflag):
    """Each output excludes the current row, and requires the full preceding 21."""
    prices = np.column_stack([high, low, close]).astype(float)
    cents = np.rint(prices*100)
    good = np.isfinite(prices).all(axis=1) & (prices > 0).all(axis=1) & (np.abs(prices-cents/100) <= .0001).all(axis=1)
    good &= (cents[:, 0] >= cents[:, 2]) & (cents[:, 2] >= cents[:, 1]) & (np.asarray(adjustflag) == 3)
    n = len(prices)
    valid = np.zeros(n, bool)
    values = np.full((n, 2), np.nan)
    totals = np.full((n, 3), np.nan)
    for i in range(21, n):
        block = cents[i-21:i]
        h, l, c = block.T
        up, down = h[1:]-h[:-1], l[:-1]-l[1:]
        u = np.where((up > 0) & (up > down), up, 0)
        d = np.where((down > 0) & (down > up), down, 0)
        tr = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
        totals[i] = [u.sum(), d.sum(), tr.sum()]
        valid[i] = good[i-21:i].all()
        if valid[i]:
            su, sd, st = totals[i]
            values[i] = [100*(su-sd)/max(su+sd, 1), 100*(su+sd)/max(st, 1)]
    return valid, values, totals


def features():
    p, sources = checked_sources()
    assert not (ROOT / 'feature_report.json').exists()
    c = base.conn()
    c.read_parquet(list(sources)).create_view('daily')
    hist = c.sql('''WITH d AS(SELECT date,code,high::DOUBLE AS rh,low::DOUBLE AS rl,close::DOUBLE AS rc,
        preclose::DOUBLE AS pc,adjustflag::DOUBLE AS adj,round(high::DOUBLE*100) AS h,
        round(low::DOUBLE*100) AS l,round(close::DOUBLE*100) AS cl FROM daily
        WHERE tradestatus=1 AND date BETWEEN '2023-06-01' AND '2025-12-30'),
        g AS(SELECT *,coalesce(isfinite(rh) AND isfinite(rl) AND isfinite(rc) AND least(rh,rl,rc)>0
        AND abs(rh-h/100)<=.0001 AND abs(rl-l/100)<=.0001 AND abs(rc-cl/100)<=.0001
        AND h>=cl AND cl>=l AND adj=3,false) AS good,
        h-lag(h) OVER w AS up,lag(l) OVER w-l AS down,lag(cl) OVER w AS prior_close
        FROM d WINDOW w AS(PARTITION BY code ORDER BY date)),
        atoms AS(SELECT *,CASE WHEN up>0 AND up>down THEN up ELSE 0 END AS u,
        CASE WHEN down>0 AND down>up THEN down ELSE 0 END AS dn,
        greatest(h-l,abs(h-prior_close),abs(l-prior_close)) AS tr,
        coalesce(abs(pc-prior_close/100)>.005,false) AS reference_break FROM g),
        h AS(SELECT date,code,sum(u) OVER w20 AS hm_up,sum(dn) OVER w20 AS hm_down,sum(tr) OVER w20 AS hm_tr,
        count(*) OVER w21 AS hm_rows,sum(good::INT) OVER w21 AS hm_good,
        min(date) OVER w21 AS hm_first_date,max(date) OVER w21 AS hm_last_date,
        sum(reference_break::INT) OVER w20 AS hm_reference_breaks FROM atoms
        WINDOW w20 AS(PARTITION BY code ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING),
        w21 AS(PARTITION BY code ORDER BY date ROWS BETWEEN 21 PRECEDING AND 1 PRECEDING))
        SELECT * FROM h WHERE date>='2024-01-01' ORDER BY date,code''').df()
    c.close()
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    f = old.merge(hist, on=['date', 'code'], how='left', validate='one_to_one').sort_values(['date', 'code']).reset_index(drop=True)
    valid = f.hm_rows.eq(21) & f.hm_good.eq(21) & f.hm_last_date.lt(f.date)
    valid &= np.isfinite(f[['hm_up', 'hm_down', 'hm_tr']]).all(axis=1)
    f['range_direction_valid'] = valid
    f['HM01'] = (100*(f.hm_up-f.hm_down)/np.maximum(f.hm_up+f.hm_down, 1)).where(valid)
    f['HM02'] = (100*(f.hm_up+f.hm_down)/np.maximum(f.hm_tr, 1)).where(valid)
    f['prior_formula_input_valid'] = f.formula_input_valid
    f['formula_input_valid'] &= valid
    cal = pd.read_parquet(CALENDAR)
    dates = sorted(cal.loc[cal.is_trading_day.eq('1'), 'calendar_date'])
    ranks = {d: i for i, d in enumerate(dates)}
    f['hm_source_gap'] = f.date.map(ranks)-f.hm_last_date.map(ranks)
    f['hm_source_span'] = f.hm_last_date.map(ranks)-f.hm_first_date.map(ranks)+1
    ROOT.mkdir(parents=True, exist_ok=True)
    hist.to_parquet(ROOT / 'history.parquet', index=False, compression='zstd')
    f.to_parquet(ROOT / 'features.parquet', index=False, compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL), history_sha256=sha(ROOT / 'history.parquet'), features_sha256=sha(ROOT / 'features.parquet'),
        rows=len(f), valid=int(f.formula_input_valid.sum()), newly_invalid=int((old.formula_input_valid & ~valid).sum()),
        valid_reference_breaks=int((f.formula_input_valid & f.hm_reference_breaks.gt(0)).sum()),
        valid_stock_day_gaps=int((f.formula_input_valid & (f.hm_source_gap.gt(1) | f.hm_source_span.gt(21))).sum()),
        zero_direction_windows=int((f.formula_input_valid & f.hm_up.add(f.hm_down).eq(0)).sum()),
        expressions=EXPRESSIONS, native_header=HEADER, calendar_sha256=sha(CALENDAR),
        native_source_parity_verified=False, software_compilation_verified=False,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_report.json', r)
    return {k: v for k, v in r.items() if k not in ['expressions', 'native_header']}


def verify_features():
    p, sources = checked_sources()
    r = json.loads((ROOT / 'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['features_sha256'] == sha(ROOT / 'features.parquet')
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    f = pd.read_parquet(ROOT / 'features.parquet')
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')], old.drop(columns='formula_input_valid'), check_exact=True)
    pd.testing.assert_series_equal(f.prior_formula_input_valid, old.formula_input_valid, check_exact=True, check_names=False)
    pieces = []
    for file in sources:
        d = pd.read_parquet(file, columns=['date', 'code', 'high', 'low', 'close', 'preclose', 'adjustflag', 'tradestatus'],
            filters=[('date', '>=', p['history_first']), ('date', '<=', p['history_last'])])
        d = d.loc[d.tradestatus.eq(1)].sort_values('date').reset_index(drop=True)
        if d.empty:
            continue
        assert not d.date.duplicated().any()
        valid, values, sums = history_arrays(d.high, d.low, d.close, d.adjustflag)
        out = d[['date', 'code']].copy()
        out['good'], out['HM01'], out['HM02'] = valid, values[:, 0], values[:, 1]
        out[['hm_up', 'hm_down', 'hm_tr']] = sums
        out['hm_first_date'], out['hm_last_date'] = d.date.shift(21), d.date.shift(1)
        out['hm_reference_breaks'] = d.preclose.astype(float).sub(d.close.astype(float).round(2).shift()).abs().gt(.005).astype(int).rolling(20, min_periods=20).sum().shift()
        pieces.append(out.loc[out.date.ge('2024-01-01')])
    e = old[['date', 'code']].merge(pd.concat(pieces, ignore_index=True), on=['date', 'code'], how='left', validate='one_to_one')
    good = e.good.fillna(False).astype(bool)
    np.testing.assert_array_equal(f.range_direction_valid, good)
    for name in ['HM01', 'HM02']:
        np.testing.assert_allclose(f[name], e[name], rtol=0, atol=2e-12, equal_nan=True)
    # Incomplete windows keep their partial diagnostic totals, never valid features.
    for name in ['hm_up', 'hm_down', 'hm_tr', 'hm_reference_breaks']:
        np.testing.assert_allclose(f.loc[good, name], e.loc[good, name], rtol=0, atol=0)
    for name in ['hm_first_date', 'hm_last_date']:
        pd.testing.assert_series_equal(f.loc[good, name], e.loc[good, name], check_exact=True)
    final = old.formula_input_valid & good
    np.testing.assert_array_equal(f.formula_input_valid, final)
    assert f.loc[final, 'HM01'].between(-100, 100).all() and f.loc[final, 'HM02'].between(0, 100).all()
    enc = lambda a: np.floor(np.clip(100*np.asarray(a)+10000+.000001, 0, 999999)).astype('int32')
    np.testing.assert_array_equal(enc(f.loc[final, list(NEW_EXPRESSIONS)]), enc(e.loc[final, list(NEW_EXPRESSIONS)]))
    assert r['valid'] == int(final.sum()) and r['newly_invalid'] == int((old.formula_input_valid & ~final).sum())
    assert r['valid_reference_breaks'] == int((final & e.hm_reference_breaks.gt(0)).sum())
    assert r['calendar_sha256'] == sha(CALENDAR)
    c = base.conn()
    e['final'] = final
    c.register('expected', e[['date', 'code', 'hm_first_date', 'hm_last_date', 'final']])
    gaps = c.sql(f'''WITH dates AS(SELECT calendar_date,row_number() OVER(ORDER BY calendar_date) AS n
        FROM read_parquet('{CALENDAR}') WHERE is_trading_day='1')
        SELECT e.date,e.code,a.n-b.n AS gap,b.n-z.n+1 AS span FROM expected e
        LEFT JOIN dates a ON a.calendar_date=e.date LEFT JOIN dates b ON b.calendar_date=e.hm_last_date
        LEFT JOIN dates z ON z.calendar_date=e.hm_first_date ORDER BY date,code''').df()
    c.close()
    np.testing.assert_allclose(f.loc[final, 'hm_source_gap'], gaps.loc[final, 'gap'], rtol=0, atol=0)
    np.testing.assert_allclose(f.loc[final, 'hm_source_span'], gaps.loc[final, 'span'], rtol=0, atol=0)
    assert r['valid_stock_day_gaps'] == int((final & (gaps.gap.gt(1) | gaps.span.gt(21))).sum())
    assert r['zero_direction_windows'] == int((final & e.hm_up.add(e.hm_down).eq(0)).sum())
    assert r['expressions'] == EXPRESSIONS and r['native_header'] == HEADER
    names = re.findall(r'(?m)^([A-Z][A-Z0-9]*):=', HEADER) + list(EXPRESSIONS)
    assert len(names) == len({n.casefold() for n in names})
    proof = dict(passed=True, feature_report_sha256=sha(ROOT / 'feature_report.json'), rows=len(f),
        all_21_day_windows_from_independent_raw_arrays=True, all_directions_ranges_validity_dates_and_encodings_rebuilt=True,
        all_original_48_values_and_quality_unchanged=True, effective_input_intersection_unchanged=r['newly_invalid'] == 0,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_verification.json', proof)
    return proof


def native_values(high, low, close, dates):
    n = len(close)
    first = np.maximum.accumulate(np.where(np.r_[True, dates[1:] != dates[:-1]], np.arange(n), 0))
    b0 = np.arange(n)-first+1
    def ref(x, offsets):
        offsets = np.broadcast_to(offsets, n)
        good = np.isfinite(offsets) & (offsets >= 0) & (offsets <= np.arange(n))
        ix = np.where(good, np.arange(n)-offsets, 0).astype(int)
        return np.where(good, np.asarray(x)[ix], np.nan)
    def extreme(x, lens, maximum):
        np.testing.assert_array_equal(lens, b0)
        out = np.empty(n)
        for start, end in zip(np.flatnonzero(b0 == 1), np.r_[np.flatnonzero(b0 == 1)[1:], n]):
            out[start:end] = (np.maximum.accumulate if maximum else np.minimum.accumulate)(x[start:end])
        return out
    env = dict(H=np.asarray(high), L=np.asarray(low), C=np.asarray(close), B0=b0, REF=ref,
        HHV=lambda x, k: extreme(x, k, True), LLV=lambda x, k: extreme(x, k, False),
        INTPART=np.trunc, IF=np.where, MAX=np.maximum, ABS=np.abs)
    for i in range(1, 21):
        env[f'B{i}'] = env[f'B{i-1}'] + ref(b0, env[f'B{i-1}'])
    for i in range(1, 22):
        env[f'DCP{i}'] = ref(env['C'], env[f'B{i-1}'])
    for statement in EXTRA_HEADER.strip().split(';'):
        if statement.strip():
            key, expr = statement.strip().split(':=')
            env[key] = eval(expr, {'__builtins__': {}}, env)
    return [float(eval(expr, {'__builtins__': {}}, env)[-1]) for expr in NEW_EXPRESSIONS.values()]


def native():
    p, sources = checked_sources()
    v = json.loads((ROOT / 'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(ROOT / 'feature_report.json')
    samples = json.loads(PROBES.read_text())['samples']
    assert len(samples) == 32
    f = pd.read_parquet(ROOT / 'features.parquet').set_index(['date', 'code'])
    daily = {Path(file).stem.replace('_', '.'): file for file in sources}
    cases, total = [], 0
    for sample in samples:
        day, code = sample['date'], sample['code']
        row = f.loc[(day, code)]
        d = pd.read_parquet(daily[code], columns=['date', 'high', 'low', 'close', 'tradestatus'],
            filters=[('date', '>=', p['history_first']), ('date', '<', day)])
        d = d.loc[d.tradestatus.eq(1)].sort_values('date').tail(21)
        assert len(d) == 21 and d.date.iloc[0] == row.hm_first_date and d.date.iloc[-1] == row.hm_last_date
        path = MINUTES / code[:2].upper() / (code[3:] + '.parquet')
        assert sha(path) == sample['source_sha256']
        q = pd.read_parquet(path, columns=['timestamp', 'high', 'low', 'close'], filters=[
            ('timestamp', '>=', pd.Timestamp(d.date.iloc[0])), ('timestamp', '<=', pd.Timestamp(day + ' 14:49'))])
        q['date'] = q.timestamp.dt.strftime('%Y-%m-%d')
        q = q.loc[q.date.isin(d.date.tolist() + [day])].sort_values('timestamp')
        assert q.timestamp.is_unique and q.timestamp.max() == pd.Timestamp(day + ' 14:49')
        prior = q.loc[q.date.lt(day)]
        assert prior.groupby('date').size().eq(241).all()
        rebuilt = prior.groupby('date').agg(high=('high', 'max'), low=('low', 'min'), close=('close', 'last'))
        np.testing.assert_allclose(rebuilt.to_numpy().round(2), d.set_index('date')[['high', 'low', 'close']].to_numpy(), rtol=0, atol=.0001)
        values = native_values(q.high.to_numpy(), q.low.to_numpy(), q.close.to_numpy(), q.date.to_numpy())
        changed = q[['high', 'low', 'close']].to_numpy().copy()
        changed[q.date.eq(day)] = [100000., .01, 2.]
        again = native_values(changed[:, 0], changed[:, 1], changed[:, 2], q.date.to_numpy())
        np.testing.assert_array_equal(values, again)
        if row.formula_input_valid:
            np.testing.assert_allclose(values, [row.HM01, row.HM02], rtol=0, atol=2e-12)
            encode = lambda a: np.floor(np.clip(100*np.asarray(a)+10000+.000001, 0, 999999))
            np.testing.assert_array_equal(encode(values), encode([row.HM01, row.HM02]))
        cases.append(dict(date=day, code=code, first_source=d.date.iloc[0], last_source=d.date.iloc[-1],
            minutes=len(q), valid=bool(row.formula_input_valid), source_sha256=sample['source_sha256']))
        total += len(q)
    proof = dict(passed=True, protocol_sha256=sha(PROTOCOL), feature_report_sha256=sha(ROOT / 'feature_report.json'),
        feature_verification_sha256=sha(ROOT / 'feature_verification.json'), samples=cases, raw_minutes=total,
        all_672_source_days_prices_and_actual_native_expressions_verified=True, current_day_perturbation_cannot_change_prior_inputs=True,
        native_source_parity_verified=False, software_compilation_verified=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'native_input_verification.json', proof)
    return {k: v for k, v in proof.items() if k != 'samples'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['features', 'verify_features', 'native'])
    print(json.dumps(globals()[p.parse_args().stage](), ensure_ascii=False, indent=2))
