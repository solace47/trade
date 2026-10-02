"""Input-only pilot of time-contained, cost-adjusted historical quotations."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from trade_research import tail_formula_additive as numeric
from trade_research import tail_formula_baseline as baseline
from trade_research.research_io import check_runtime, check_sources, save_json, sha

ROOT = Path('data/research/tail_formula_cost_history_probe')
PROTOCOL = Path('config/tail_formula_cost_history_probe_protocol.json')


def aggregate(raw):
    c = numeric.conn()
    c.register('raw', raw)
    out = c.sql('''WITH b AS(SELECT *,coalesce(isfinite(open) AND isfinite(high) AND isfinite(low)
        AND isfinite(close) AND least(open,high,low,close)>0
        AND high+.0001>=greatest(open,close,low) AND low-.0001<=least(open,close)
        AND abs(open-round(open,2))<=.0001 AND abs(high-round(high,2))<=.0001
        AND abs(low-round(low,2))<=.0001 AND abs(close-round(close,2))<=.0001
        AND isfinite(volume) AND volume>=0 AND volume=floor(volume)
        AND isfinite(amount) AND amount>=0 AND (volume=0)=(amount=0)
        AND (volume=0 OR amount/volume BETWEEN low-.0101 AND high+.0101),false) AS good FROM raw),
        r AS(SELECT *,min(close) OVER w AS three_close,count(*) OVER w AS n,
        count(*) FILTER(WHERE good AND volume>0) OVER w AS three_active,
        min(timestamp) OVER w AS first_stamp FROM b WINDOW w AS(PARTITION BY date,code
        ORDER BY timestamp ROWS BETWEEN 2 PRECEDING AND CURRENT ROW))
        SELECT date,code,
        count(*) FILTER(WHERE clock=1449 AND good) AS q_count,
        count(*) FILTER(WHERE clock=1449) AS q_rows,
        max(close) FILTER(WHERE clock=1449) AS q,
        count(*) FILTER(WHERE clock BETWEEN 1452 AND 1455) AS entry_rows,
        count(DISTINCT clock) FILTER(WHERE clock BETWEEN 1452 AND 1455) AS entry_clocks,
        count(*) FILTER(WHERE clock BETWEEN 1452 AND 1455 AND good AND volume>0) AS entry_good,
        max(high) FILTER(WHERE clock BETWEEN 1452 AND 1455) AS entry_high,
        max(volume) FILTER(WHERE clock BETWEEN 1452 AND 1455) AS entry_capacity,
        count(*) FILTER(WHERE clock BETWEEN 931 AND 959) AS morning_rows,
        count(DISTINCT clock) FILTER(WHERE clock BETWEEN 931 AND 959) AS morning_clocks,
        count(*) FILTER(WHERE clock BETWEEN 931 AND 959 AND good) AS morning_good,
        max(close) FILTER(WHERE clock=959 AND good AND volume>0) AS morning_close,
        max(three_close) FILTER(WHERE clock BETWEEN 933 AND 959 AND n=3
        AND three_active=3 AND timestamp-first_stamp=INTERVAL 2 MINUTE) AS sustained
        FROM r GROUP BY date,code ORDER BY code,date''').df()
    c.close()
    # Separate vector arithmetic and grouped aggregations, with no SQL predicate
    # reused. A broken or missing day remains in the subsequent stock calendar.
    f = raw.copy()
    prices = f[['open','high','low','close']]
    f['good'] = (np.isfinite(prices).all(axis=1) & prices.gt(0).all(axis=1)
        & f.high.ge(prices.max(axis=1)-.0001) & f.low.le(prices.min(axis=1)+.0001)
        & (prices-prices.round(2)).abs().le(.0001).all(axis=1)
        & np.isfinite(f.volume) & f.volume.ge(0) & f.volume.eq(np.floor(f.volume))
        & np.isfinite(f.amount) & f.amount.ge(0) & f.volume.eq(0).eq(f.amount.eq(0))
        & (f.volume.eq(0) | (f.amount/f.volume).between(f.low-.0101,f.high+.0101)))
    group = f.groupby(['code','date'], sort=False)
    prior1, prior2 = group.shift(1), group.shift(2)
    f['three'] = np.minimum.reduce([f.close.to_numpy(),prior1.close.to_numpy(),prior2.close.to_numpy()])
    active = f.good & f.volume.gt(0)
    trio = active & prior1.good.eq(True) & prior1.volume.gt(0) & prior2.good.eq(True) & prior2.volume.gt(0)
    f['three'] = f.three.where(trio & f.timestamp.sub(prior2.timestamp).eq(pd.Timedelta(minutes=2))
                               & f.clock.between(933,959))
    key = ['date','code']
    q = f.loc[f.clock.eq(1449)].groupby(key).agg(q_count=('good','sum'),q_rows=('good','size'),q=('close','max'))
    e = f.loc[f.clock.between(1452,1455)].copy(); e['active_good'] = e.good & e.volume.gt(0)
    e = e.groupby(key).agg(entry_rows=('clock','size'),entry_clocks=('clock','nunique'),
        entry_good=('active_good','sum'),entry_high=('high','max'),entry_capacity=('volume','max'))
    m = f.loc[f.clock.between(931,959)].copy()
    m['point'] = m.close.where(m.clock.eq(959) & m.good & m.volume.gt(0))
    m = m.groupby(key).agg(morning_rows=('clock','size'),morning_clocks=('clock','nunique'),
        morning_good=('good','sum'),morning_close=('point','max'),sustained=('three','max'))
    expected = f[key].drop_duplicates().set_index(key).join(q).join(e).join(m).reset_index()
    counts = ['q_count','q_rows','entry_rows','entry_clocks','entry_good','morning_rows','morning_clocks','morning_good']
    expected[counts] = expected[counts].fillna(0)
    expected = expected.sort_values(['code','date']).reset_index(drop=True)
    for frame in [out, expected]:
        complete = frame.morning_rows.eq(29) & frame.morning_clocks.eq(29) & frame.morning_good.eq(29)
        frame['sustained'] = frame.sustained.where(complete)
    pd.testing.assert_frame_equal(out, expected[out.columns], check_dtype=False, atol=1e-10, rtol=0)
    return out


def history(windows, calendar):
    f = calendar.merge(windows, on=['date','code'], how='left', validate='one_to_one').sort_values(['code','date']).reset_index(drop=True)
    f['shares'] = np.floor(20000/(100*f.q))*100
    f['entry_valid'] = (f.q_count.eq(1) & f.q_rows.eq(1) & f.shares.ge(100)
        & f.entry_rows.eq(4) & f.entry_clocks.eq(4) & f.entry_good.eq(4)
        & f.entry_capacity.ge(10*f.shares))
    e = f.entry_high + np.maximum(.0015*f.entry_high,.005)
    value = f.shares*e
    f['cash'] = value + np.maximum(.0003*value,5) + .00001*value
    previous = f.groupby('code', sort=False)[['date','shares','cash','entry_valid']].shift(1)
    good = (previous.entry_valid.eq(True) & previous.date.lt(f.date) & f.morning_rows.eq(29)
        & f.morning_clocks.eq(29) & f.morning_good.eq(29) & f.morning_close.notna())
    def mark(price):
        sale = previous.shares*(price-np.maximum(.0015*price,.005))
        # Date-specific stamp tax; the brokerage commission remains an explicit
        # research assumption, not a claim about the user's actual account.
        levy = np.where(f.date.ge('2023-08-28'),.00051,.00101)
        return ((sale-np.maximum(.0003*sale,5)-levy*sale)/previous.cash-1).where(good)
    f['proxy_mark'] = 100*mark(f.morning_close)
    f['proxy_positive'] = mark(f.sustained).gt(0).astype(float).where(good)
    f['atom_valid'] = good
    rolling = f.groupby('code',sort=False)[['proxy_mark','proxy_positive']].rolling(20,min_periods=20).mean().reset_index(level=0,drop=True)
    f['mean20'] = rolling.proxy_mark
    f['frequency20'] = 100*rolling.proxy_positive
    f['history_valid'] = f.mean20.notna() & f.frequency20.notna()
    f['reference_date'] = previous.date
    return f


def probe():
    check_runtime()
    p = json.loads(PROTOCOL.read_text())
    assert subprocess.check_output(['git','show',f'HEAD:{PROTOCOL}']) == PROTOCOL.read_bytes()
    check_sources(p['source_hashes'])
    assert not (ROOT/'pilot_verified.json').exists()
    ROOT.mkdir(exist_ok=True)
    sources, calendars = {}, []
    raw_files = []
    for code in p['pilot_codes']:
        daily = Path(p['daily_sources'][code]['path']); minute = Path(p['minute_sources'][code]['path'])
        for path, digest in [(daily,p['daily_sources'][code]['sha256']),(minute,p['minute_sources'][code]['sha256'])]:
            assert sha(path) == digest; sources[str(path)] = digest
        d = pd.read_parquet(daily,columns=['date','code','tradestatus'],
            filters=[('date','>=',p['history_start']),('date','<=',p['signal_last'])])
        d = d.loc[d.tradestatus.eq(1),['date','code']]
        assert d.code.eq(code).all() and not d.date.duplicated().any()
        calendars.append(d); raw_files.append(str(minute))
    calendar = pd.concat(calendars,ignore_index=True)
    c = numeric.conn(); c.read_parquet(raw_files,filename=True).create_view('source')
    raw = c.execute('''SELECT filename,lower(exchange)||'.'||symbol AS code,strftime(timestamp,'%Y-%m-%d') AS date,
        timestamp,(extract(hour FROM timestamp)*100+extract(minute FROM timestamp))::INT AS clock,
        open::DOUBLE AS open,high::DOUBLE AS high,low::DOUBLE AS low,close::DOUBLE AS close,
        volume::DOUBLE AS volume,turnover::DOUBLE AS amount FROM source
        WHERE timestamp>=?::DATE AND timestamp<?::DATE+INTERVAL 1 DAY
        AND (strftime(timestamp,'%H%M') BETWEEN '0931' AND '0959'
          OR strftime(timestamp,'%H%M') IN ('1449','1452','1453','1454','1455'))
        ORDER BY code,date,timestamp''',[p['history_start'],p['signal_last']]).df(); c.close()
    assert set(raw.code).issubset(p['pilot_codes']) and raw.date.le('2025-12-30').all()
    expected_identity = {p['minute_sources'][code]['path']:code for code in p['pilot_codes']}
    assert raw.code.eq(raw.filename.map(expected_identity)).all()
    raw = raw.drop(columns='filename')
    assert raw.timestamp.eq(raw.timestamp.dt.floor('min')).all()
    windows = aggregate(raw)
    h = history(windows,calendar)
    # Destroy today's cost anchors (which enter tomorrow's atom) and all later
    # quotations. Today's captured morning and every earlier input must remain
    # identical, including validity.
    point = p['mutation_date']
    changed = windows.copy(); mask = changed.date.ge(point)
    changed.loc[mask,['q','entry_high','entry_capacity']] = 999999
    changed.loc[changed.date.gt(point),['morning_close','sustained']] = .01
    altered = history(changed,calendar)
    columns = ['date','code','proxy_mark','proxy_positive','atom_valid','mean20','frequency20','history_valid','reference_date']
    pd.testing.assert_frame_equal(h.loc[h.date.le(point),columns].reset_index(drop=True),
        altered.loc[altered.date.le(point),columns].reset_index(drop=True),check_exact=True)
    windows.to_parquet(ROOT/'pilot_windows.parquet',index=False,compression='zstd')
    h.to_parquet(ROOT/'pilot_history.parquet',index=False,compression='zstd')
    current = baseline.original(); current = current.loc[current.code.isin(p['pilot_codes'])]
    projected = current[['date','code','half','formula_input_valid','V01']].merge(
        h[['date','code','history_valid','mean20','frequency20']],on=['date','code'],how='left',validate='one_to_one')
    valid = projected.formula_input_valid & projected.history_valid.eq(True)
    save_json(ROOT/'pilot_verified.json',dict(passed=True,protocol_sha256=sha(PROTOCOL),source_hashes=sources,
        outputs={str(ROOT/n):sha(ROOT/n) for n in ['pilot_windows.parquet','pilot_history.parquet']},
        pilot_codes=p['pilot_codes'],raw_minutes=len(raw),calendar_rows=len(calendar),
        current_keys=len(projected),original_valid=int(projected.formula_input_valid.sum()),
        new_valid=int(valid.sum()),input_valid_by_half=projected.assign(valid=valid).groupby('half').agg(
            keys=('code','size'),original_valid=('formula_input_valid','sum'),new_valid=('valid','sum')).reset_index().to_dict('records'),
        all_raw_windows_independently_SQL_and_Pandas_rebuilt=True,all_file_security_identities_verified=True,
        current_and_future_tail_mutation_invariance=True,
        no_cached_outcome_or_whole_period_quality_flags_used=True,new_model_fits=0,
        strategy_economic_results_read=False,new_2026_prices_read=False,
        quote_cost_proxy_not_realized_returns_or_queue_fill=True,full_population_verified=False,native_client_parity_verified=False))
    return dict(pilot_sha256=sha(ROOT/'pilot_verified.json'),new_fits=0,original_valid=int(projected.formula_input_valid.sum()),new_valid=int(valid.sum()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['probe'])
    print(json.dumps(probe(),ensure_ascii=False))
