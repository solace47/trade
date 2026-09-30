"""Independently rebuild expanded-pool windows, entry states and cash labels."""
import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from trade_research import tail_formula_pool_scope_labels as producer
from trade_research.corporate_cash import MINUTES, save_json, sha
from trade_research.tail_formula_1000_analysis import BUY_COLUMNS
from trade_research.turnover_reference import CALENDAR

KEYS = producer.KEYS


def equal(a, b, exact=False):
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True),
        check_dtype=False, check_exact=exact, rtol=0, atol=2e-10)


def context(scope):
    p, manifest, keys = producer.checked_keys(scope)
    root = producer.directory(scope)
    c = duckdb.connect(); c.execute('SET threads=4'); c.execute("SET memory_limit='4GB'")
    cutoff = p['training_observation_end_exclusive'] if scope == 'training' else '2026-01-01'
    wanted = c.execute('''WITH days AS(SELECT calendar_date AS date,
        lead(calendar_date) OVER(ORDER BY calendar_date) AS next_date
        FROM read_parquet(?) WHERE is_trading_day='1'
        AND calendar_date BETWEEN '2024-01-01' AND '2025-12-31')
        SELECT u.*,d.next_date FROM read_parquet(?) u JOIN days d USING(date)
        WHERE d.next_date<? ORDER BY date,code''',
        [str(CALENDAR), str(producer.inputs.OUT/'universe.parquet'), cutoff]).df()
    equal(keys, wanted, exact=True)
    assert keys.necessary_tradeable.all() and keys.board.eq('main').all()
    assert keys.isST.eq(0).all() and keys.tradestatus.eq(1).all()
    assert not keys.code.str[3:].str.startswith(('92', '688', '300', '301')).any()
    extra = keys.loc[~keys.original_pool].reset_index(drop=True)
    equal(extra, pd.read_parquet(root/'extra_keys.parquet'), exact=True)
    c.register('keys', extra[KEYS])
    old_buy = c.execute('SELECT '+','.join('o.'+n for n in BUY_COLUMNS)+
        ' FROM read_parquet(?) o JOIN keys USING(date,code,next_date) ORDER BY date,code',
        [str(producer.OLD_BUY/'labels.parquet')]).df()
    equal(old_buy, pd.read_parquet(root/'entry_original.parquet'), exact=True)
    assert not old_buy.necessary_tradeable.any()
    assert keys.next_date.gt(keys.date).all() and keys.next_date.lt(cutoff).all()
    assert len(keys) == manifest['rows'] and len(extra) == manifest['extra_rows']
    return c, root, keys, extra, old_buy


def aggregate(raw):
    """Pandas implementation; producer uses SQL windows and filtered aggregates."""
    raw = raw.sort_values(['date', 'code', 'timestamp']).reset_index(drop=True).copy()
    assert not raw.duplicated(['date', 'code', 'timestamp']).any()
    assert raw.clock.eq(raw.timestamp.dt.strftime('%H:%M')).all()
    assert raw.next_date.eq(raw.timestamp.dt.strftime('%Y-%m-%d')).all()
    assert raw.clock.between('09:31', '10:00').all()
    p = raw[['open', 'high', 'low', 'close']].astype(float)
    v, a, t = raw.volume, raw.amount, raw.timestamp
    good = (np.isfinite(p).all(axis=1) & p.gt(0).all(axis=1) & np.isfinite(v) & np.isfinite(a)
        & p.high.add(.0001).ge(p.max(axis=1)) & p.low.sub(.0001).le(p.min(axis=1))
        & (p-p.round(2)).abs().le(.0001).all(axis=1) & v.ge(0) & a.ge(0)
        & v.eq(0).eq(a.eq(0)) & (v.eq(0) | (a/v).between(p.low-.0101, p.high+.0101))
        & t.eq(t.dt.floor('min')))
    raw['good'] = good; raw['active'] = good & v.gt(0)
    raw['before'] = raw.clock.le('09:59'); raw['before_good'] = good & raw.before
    raw['before_active'] = raw.active & raw.before
    same = raw.date.eq(raw.date.shift(2)) & raw.code.eq(raw.code.shift(2))
    three = (same & raw.active.astype(int).rolling(3, min_periods=3).sum().eq(3)
        & t.sub(t.shift(2)).eq(pd.Timedelta(minutes=2)) & raw.before)
    raw['sustained'] = raw.close.rolling(3, min_periods=3).min().where(three)
    raw['active_close'] = raw.close.where(raw.before_active)
    raw['active_low'] = raw.low.where(raw.before_active)
    raw['at0959'] = raw.close.where(raw.before_active & raw.clock.eq('09:59'))
    raw['clock29'] = raw.clock.where(raw.before)
    return raw.groupby(KEYS).agg(bars30=('timestamp', 'size'), labels30=('clock', 'nunique'),
        good30=('good', 'sum'), bars29=('before', 'sum'), labels29=('clock29', 'nunique'),
        good29=('before_good', 'sum'), active_minutes=('before_active', 'sum'),
        max_close=('active_close', 'max'), sustained_close=('sustained', 'max'),
        min_low=('active_low', 'min'), price_0959=('at0959', 'max')).reset_index()


def windows(scope):
    c, root, keys, extra, entry = context(scope)
    r = json.loads((root/'raw_report.json').read_text())
    report = json.loads((root/'window_report.json').read_text())
    for name, file in [('protocol_sha256', producer.PROTOCOL),
        ('input_manifest_sha256', root/'input_manifest.json'),
        ('raw_report_sha256', root/'raw_report.json'), ('observations_sha256', root/'observations.parquet')]:
        assert report[name] == sha(file), name
    assert r['input_manifest_sha256'] == sha(root/'input_manifest.json')
    manifest = json.loads((root/'input_manifest.json').read_text())
    frames = []; raw_rows = 0
    for file, digest in r['parts_sha256'].items():
        assert sha(Path(file)) == digest
        if file not in manifest['reused_parts_sha256']:
            receipt = json.loads(Path(file).with_suffix('.json').read_text())
            assert receipt['sha256'] == digest and receipt['extractor_sha256'] == r['extractor_sha256']
            assert receipt['input_manifest_sha256'] == sha(root/'input_manifest.json')
        b = pd.read_parquet(file, columns=producer.RAW); raw_rows += len(b)
        assert b.merge(extra[KEYS], how='left', indicator=True)._merge.eq('both').all()
        assert b.next_date.lt('2025-07-01' if scope == 'training' else '2026-01-01').all()
        frames.append(aggregate(b))
    combined = pd.concat(frames, ignore_index=True)
    assert not combined.duplicated(KEYS).any(), 'A key was split or duplicated across raw parts'
    assert raw_rows == r['raw_rows']
    expected = extra[KEYS].merge(combined, how='left', validate='one_to_one')
    expected['source_valid'] = expected.bars30.eq(30) & expected.labels30.eq(30) & expected.good30.eq(30)
    expected['source_valid_0959'] = expected.bars29.eq(29) & expected.labels29.eq(29) & expected.good29.eq(29)
    actual = pd.read_parquet(root/'observations.parquet')
    equal(actual, expected[actual.columns], exact=True)
    assert len(actual) == report['rows'] and actual.loc[actual.source_valid, 'source_valid_0959'].all()
    actual['half'] = extra.half
    actual['hash'] = [hashlib.sha256(('pool-scope-window-v1|'+d+'|'+code).encode()).hexdigest()
        for d, code in zip(actual.date, actual.code)]
    samples = actual.sort_values('hash').groupby(['half', 'source_valid']).head(8)
    sources = json.loads(producer.inputs.cached.MINUTE_MANIFEST.read_text())['source_sha256']
    c.read_parquet(list(r['parts_sha256'])).create_view('raw')
    e = entry.set_index(['date', 'code']); checks = 0
    for s in samples.itertuples():
        path = MINUTES/s.code[:2].upper()/(s.code[3:]+'.parquet'); assert sha(path) == sources[str(path)]
        for day, start, end in [(s.next_date, '09:31', '10:01'), (s.date, '14:52', '14:56')]:
            b = pq.read_table(path, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'],
                filters=[('timestamp', '>=', pd.Timestamp(day+' '+start).to_pydatetime()),
                         ('timestamp', '<', pd.Timestamp(day+' '+end).to_pydatetime())]).to_pandas()
            b = b.sort_values('timestamp').reset_index(drop=True); checks += len(b)
            if start == '09:31':
                saved = c.execute('SELECT timestamp,open,high,low,close,volume,amount FROM raw WHERE date=? AND code=? ORDER BY timestamp', [s.date, s.code]).df()
                equal(saved, b.rename(columns={'turnover': 'amount'}), exact=True)
            else:
                value = b.turnover.astype(float).sum()/b.volume.astype(float).sum() if b.volume.sum()>0 else np.nan
                for name, value in [('entry_bars', len(b)), ('entry_labels', b.timestamp.dt.strftime('%H:%M').nunique()),
                    ('entry_volume', b.volume.astype(float).sum()), ('entry_vwap', value),
                    ('entry_low', b.loc[b.volume.gt(0), 'low'].min()), ('entry_high', b.loc[b.volume.gt(0), 'high'].max())]:
                    np.testing.assert_allclose(e.loc[(s.date, s.code), name], value, atol=2e-10, rtol=0, equal_nan=True)
    c.close()
    proof = dict(passed=True, window_report_sha256=sha(root/'window_report.json'), scope=scope,
        rows=len(extra), raw_rows=raw_rows, all_aggregates_independently_rebuilt=True,
        complete_pool_keys_and_strict_next_dates_checked=True, old_buy_sources_rejoined=True,
        fixed_source_windows=len(samples)*2, original_minutes_rechecked=checks,
        sample_keys=samples[['date', 'code']].to_dict('records'), original_30bar_quality_guard_retained=True,
        economic_observation_end='09:59', no_evaluation_group_statistics_computed=True,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(root/'window_verification.json', proof)
    return {k: v for k, v in proof.items() if k != 'sample_keys'}


def labels(scope):
    c, root, keys, extra, entry = context(scope)
    report = json.loads((root/'full_label_report.json').read_text())
    proof = json.loads((root/'window_verification.json').read_text())
    assert proof['passed'] and proof['window_report_sha256'] == sha(root/'window_report.json')
    for name, file in [('protocol_sha256', producer.PROTOCOL), ('input_manifest_sha256', root/'input_manifest.json'),
        ('window_verification_sha256', root/'window_verification.json'), ('labels_sha256', root/'full_labels.parquet'),
        ('added_labels_sha256', root/'added_labels.parquet')]:
        assert report[name] == sha(file), name
    got = pd.read_parquet(root/'added_labels.parquet')
    obs = pd.read_parquet(root/'observations.parquet')
    keep_obs = [x for x in obs if x in got]
    unchanged = [x for x in BUY_COLUMNS if x not in ['necessary_tradeable', 'entry_fill_status', 'entry_recorded', 'entry_queue_unknown']]
    equal(got[unchanged], entry[unchanged], exact=True)
    equal(got[keep_obs], obs[keep_obs], exact=True)
    equal(got[KEYS+['necessary_tradeable']], extra[KEYS+['necessary_tradeable']], exact=True)
    entry['necessary_tradeable'] = True
    source = entry.merge(obs, on=KEYS, validate='one_to_one'); c.register('source', source)
    c.execute('''CREATE VIEW facts AS SELECT *,
        CASE WHEN NOT necessary_tradeable THEN 'not_submitted'
             WHEN NOT coalesce(isfinite(entry_vwap) AND entry_vwap>0 AND entry_volume>0,false) THEN 'no_liquidity'
             WHEN NOT coalesce(entry_volume*.1>=decision_shares,false) THEN 'volume_cap'
             WHEN entry_vwap*1.0005>=upper_limit-.005 THEN 'estimated_upper_limit' ELSE 'filled' END AS fill,
        coalesce(isfinite(next_preclose) AND next_preclose>0 AND abs(next_preclose-round(next_preclose,2))<=.0001,false) AS valid_ref,
        coalesce(isfinite(day_close) AND day_close>0 AND abs(day_close-round(day_close,2))<=.0001,false) AS valid_close,
        NOT entry_source_valid OR period_entry_bad_day OR period_bad_symbol AS entry_source_unknown FROM source''')
    c.execute('''CREATE VIEW checked AS SELECT *,fill='filled' AS recorded,
        NOT entry_source_unknown AND fill<>'filled' AS known_no_trade,
        NOT catalog_covered OR action_exposure OR NOT valid_ref OR NOT valid_close OR abs(next_preclose-day_close)>.005 AS corporate_unknown,
        coalesce(valid_ref AND next_trade_status=1 AND next_isST IN (0,1) AND next_adjustflag=3,false) AS next_daily_valid,
        fill='filled' AND (NOT entry_bounds_valid OR round(entry_high,2)>=upper_limit) AS queue_unknown FROM facts''')
    c.execute('''CREATE VIEW states AS SELECT *,CASE WHEN entry_source_unknown THEN 'entry_source_unknown'
        WHEN known_no_trade THEN 'no_trade' WHEN queue_unknown THEN 'entry_queue_unknown'
        WHEN corporate_unknown THEN 'corporate_unknown' WHEN NOT next_daily_valid THEN 'next_daily_unknown'
        WHEN NOT source_valid THEN 'morning_source_unknown' ELSE 'known' END AS observation_status FROM checked''')
    expected = c.sql('''SELECT date,code,corporate_unknown,next_daily_valid,entry_source_unknown,known_no_trade,observation_status,
        recorded AS entry_recorded,queue_unknown AS entry_queue_unknown,fill AS entry_fill_status FROM states ORDER BY date,code''').df()
    equal(got[expected.columns], expected, exact=True); count = len(got)*(len(expected.columns)-2)
    for bps in [5, 15]:
        c.execute(f'''CREATE OR REPLACE VIEW cash AS WITH b AS(SELECT *,entry_vwap+greatest(entry_vwap*{bps}/10000.,.005) AS buy_price FROM states),
            v AS(SELECT *,decision_shares*buy_price AS buy_value FROM b)
            SELECT *,buy_value+greatest(buy_value*.0003,5)+buy_value*.00001 AS buy_cash,
                recorded AND buy_price>=upper_limit-.005 AS stress,
                observation_status='known' AND NOT(recorded AND buy_price>=upper_limit-.005) AS known FROM v''')
        ex = c.sql('''SELECT date,code,buy_cash,stress,known,known AND NOT period_exit_bad_day AS sensitive_known,
            NOT known AND NOT known_no_trade AS unknown FROM cash ORDER BY date,code''').df()
        ex = ex.rename(columns={n: f'{n}{bps}' for n in ['buy_cash', 'known', 'sensitive_known', 'unknown']}).rename(columns={'stress': f'entry_stress_unknown{bps}'})
        equal(got[ex.columns], ex); count += len(got)*5
        for name, price in [('sustained', 'sustained_close'), ('any_close', 'max_close'), ('mark_0959', 'price_0959'), ('adverse', 'min_low')]:
            ex = c.sql(f'''WITH v AS(SELECT *,decision_shares*({price}-greatest({price}*{bps}/10000.,.005)) AS mark_value FROM cash)
                SELECT date,code,CASE WHEN known THEN (mark_value-greatest(mark_value*.0003,5)-mark_value*.00051)/buy_cash-1 END AS value
                FROM v ORDER BY date,code''').df()
            np.testing.assert_allclose(got[f'{name}_return{bps}'], ex.value, atol=2e-10, rtol=0, equal_nan=True); count += len(got)
            if name in ['sustained', 'any_close']:
                target = f'opportunity{bps}' if name == 'sustained' else f'any_opportunity{bps}'
                np.testing.assert_array_equal(got[target], np.where(got[f'known{bps}'], ex.value.gt(0).astype(float), np.nan)); count += len(got)
                if name == 'sustained':
                    np.testing.assert_array_equal(got[f'one_percent{bps}'], np.where(got[f'known{bps}'], ex.value.ge(.01).astype(float), np.nan)); count += len(got)
    full = pd.read_parquet(root/'full_labels.parquet')
    old_keys = keys.loc[keys.original_pool, KEYS]; c.register('old_keys', old_keys)
    old = c.execute('SELECT o.* FROM read_parquet(?) o JOIN old_keys USING(date,code,next_date) ORDER BY date,code',
        [str(producer.original.ROOT/'full_labels.parquet')]).df()
    equal(full.loc[keys.original_pool], old, exact=True)
    combined = pd.concat([old, got], ignore_index=True).sort_values(['date', 'code'])
    equal(full, combined, exact=True); equal(full[KEYS], keys[KEYS], exact=True)
    assert len(full) == report['rows'] and len(got) == report['extra_rows']
    assert not full.duplicated(['date', 'code']).any()
    c.close()
    result = dict(passed=True, label_report_sha256=sha(root/'full_label_report.json'), scope=scope,
        rows=len(full), extra_rows=len(got), numeric_and_status_checks=count,
        original_labels_and_unknowns_exactly_unchanged=True, all_sources_rejoined=True,
        all_complete_pool_keys_covered=True, raw_buy_status_rebuilt=True,
        economic_observation_end='09:59', original_30bar_quality_guard_retained=True,
        last_observation=full.next_date.max(), old_tail_exit_outcomes_not_read=True,
        no_evaluation_group_statistics_computed=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(root/'full_label_verification.json', result); return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('stage', choices=['windows', 'labels'])
    p.add_argument('--scope', required=True, choices=['training', 'evaluation']); a = p.parse_args()
    print(json.dumps(globals()[a.stage](a.scope), ensure_ascii=False, indent=2))
