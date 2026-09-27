"""Rebuild lower-liquidity observations and cash without producer classification."""
import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from trade_research.corporate_cash import MINUTES, save_json, sha
from trade_research.tail_formula_1000_analysis import BUY_COLUMNS, OBS_COLUMNS
from trade_research.tail_formula_liquidity_inputs import ROOT, OUT, PROTOCOL
from trade_research.turnover_reference import CALENDAR
from verify_tail_formula_1000_observations import rebuild

LABELS = ROOT/'labels'
OLD_BUY = Path('data/research/economic_winner/period_quality')
OLD_FULL = Path('data/research/tail_formula_1000')


def equal(a, b, exact=False):
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True),
        check_dtype=False, check_exact=exact, rtol=0, atol=2e-10)


def context():
    m = json.loads((LABELS/'input_manifest.json').read_text())
    for key, path in [('protocol_sha256', PROTOCOL), ('calendar_sha256', CALENDAR),
                      ('source_manifest_sha256', OUT/'source_manifest.json'),
                      ('keys_sha256', LABELS/'keys.parquet'),
                      ('entry_original_sha256', LABELS/'entry_original.parquet'),
                      ('old_buy_label_report_sha256', OLD_BUY/'label_report.json'),
                      ('old_buy_label_verification_sha256', OLD_BUY/'label_verification.json'),
                      ('original_full_label_report_sha256', OLD_FULL/'full_label_report.json'),
                      ('original_full_label_verification_sha256', OLD_FULL/'full_label_verification.json')]:
        assert m[key] == sha(path), key
    for kind, digest in m['selection_report_sha256'].items():
        r = json.loads((ROOT/kind/'selection_report.json').read_text())
        v = json.loads((ROOT/kind/'selection_verification.json').read_text())
        assert digest == sha(ROOT/kind/'selection_report.json') == v['selection_report_sha256'] and v['passed']
        assert r['selection_sha256'] == sha(ROOT/kind/'selection.parquet')
    for root, report, proof, data in [(OLD_BUY, 'label_report.json', 'label_verification.json', 'labels.parquet'),
                                    (OLD_FULL, 'full_label_report.json', 'full_label_verification.json', 'full_labels.parquet')]:
        r = json.loads((root/report).read_text()); v = json.loads((root/proof).read_text())
        assert v['passed'] and v['label_report_sha256'] == sha(root/report)
        assert r['labels_sha256'] == sha(root/data)
    keys = pd.read_parquet(LABELS/'keys.parquet')
    chosen = pd.read_parquet(ROOT/'combined/selection.parquet')
    dates = chosen.loc[chosen.selected, 'date'].unique()
    universe = pd.read_parquet(OUT/'universe.parquet')
    wanted = universe.loc[universe.date.isin(dates)].copy()
    cal = pd.read_parquet(CALENDAR)
    days = cal.loc[cal.is_trading_day.eq('1') & cal.calendar_date.between('2025-01-01', '2025-12-31'), 'calendar_date'].sort_values().to_numpy()
    ix = np.searchsorted(days, wanted.date.to_numpy())
    assert (days[ix] == wanted.date.to_numpy()).all()
    wanted['next_date'] = days[ix+1]
    equal(keys, wanted, exact=True)
    assert keys.necessary_tradeable.all() and keys.board.eq('main').all()
    assert keys.isST.eq(0).all() and keys.tradestatus.eq(1).all()
    assert not keys.code.str[3:].str.startswith(('92', '688', '300', '301')).any()
    assert keys.date.ge('2025-01-01').all() and keys.next_date.le('2025-12-31').all()
    c = duckdb.connect(); c.execute('SET threads=4'); c.execute("SET memory_limit='4GB'")
    c.register('keys', keys[['date', 'code']])
    original = c.execute('SELECT '+','.join('o.'+x for x in BUY_COLUMNS)+
        ' FROM read_parquet(?) o JOIN keys USING(date,code) ORDER BY date,code', [str(OLD_BUY/'labels.parquet')]).df()
    equal(pd.read_parquet(LABELS/'entry_original.parquet'), original, exact=True)
    assert len(original) == len(keys) and not original.necessary_tradeable.any()
    cols = ['date', 'code', 'next_date', 'half', 'board', 'decision_shares', 'price_1449', 'preclose', 'upper_limit']
    equal(original[cols], keys[cols], exact=True)
    return c, keys, original


def windows():
    c, keys, entry = context()
    report = json.loads((LABELS/'window_report.json').read_text())
    r = json.loads((LABELS/'raw_report.json').read_text())
    assert report['protocol_sha256'] == sha(PROTOCOL)
    assert report['input_manifest_sha256'] == r['input_manifest_sha256'] == sha(LABELS/'input_manifest.json')
    assert report['raw_report_sha256'] == sha(LABELS/'raw_report.json')
    assert r['extractor_sha256'] == sha(Path('src/trade_research/tail_formula_liquidity_labels.py'))
    assert report['observations_sha256'] == sha(LABELS/'observations.parquet')
    frames = []
    for path, digest in r['parts_sha256'].items():
        assert sha(Path(path)) == digest
        receipt = json.loads(Path(path).with_suffix('.json').read_text())
        assert receipt['sha256'] == digest and receipt['extractor_sha256'] == r['extractor_sha256']
        frames.append(pd.read_parquet(path))
    raw = pd.concat(frames, ignore_index=True).sort_values(['date', 'code', 'timestamp']).reset_index(drop=True)
    assert len(raw) == r['raw_rows'] and not raw.duplicated(['date', 'code', 'timestamp']).any()
    assert raw.clock.between('09:31', '10:00').all() and raw.clock.eq(raw.timestamp.dt.strftime('%H:%M')).all()
    assert raw.next_date.eq(raw.timestamp.dt.strftime('%Y-%m-%d')).all()
    assert raw.merge(keys[['date', 'code', 'next_date']], how='left', indicator=True)._merge.eq('both').all()
    p = raw[['open', 'high', 'low', 'close']].astype(float)
    v, a, t = raw.volume, raw.amount, raw.timestamp
    good = (np.isfinite(p).all(axis=1) & p.gt(0).all(axis=1) & np.isfinite(v) & np.isfinite(a)
        & p.high.add(.0001).ge(p.max(axis=1)) & p.low.sub(.0001).le(p.min(axis=1))
        & (p-p.round(2)).abs().le(.0001).all(axis=1) & v.ge(0) & a.ge(0) & v.eq(0).eq(a.eq(0))
        & (v.eq(0) | (a/v).between(p.low-.0101, p.high+.0101)) & t.eq(t.dt.floor('min')))
    raw['good'] = good; raw['active'] = good & v.gt(0)
    same = raw.date.eq(raw.date.shift(2)) & raw.code.eq(raw.code.shift(2))
    three = same & raw.active.astype(int).rolling(3, min_periods=3).sum().eq(3) & t.sub(t.shift(2)).eq(pd.Timedelta(minutes=2))
    raw['sustained'] = raw.close.rolling(3, min_periods=3).min().where(three)
    raw['active_close'] = raw.close.where(raw.active); raw['active_low'] = raw.low.where(raw.active)
    raw['at1000'] = raw.close.where(raw.active & raw.clock.eq('10:00'))
    aggregated = raw.groupby(['date', 'code', 'next_date']).agg(bars=('timestamp', 'size'), labels=('clock', 'nunique'),
        valid_bars=('good', 'sum'), active_minutes=('active', 'sum'), max_close=('active_close', 'max'),
        sustained_close=('sustained', 'max'), min_low=('active_low', 'min'), price_1000=('at1000', 'max')).reset_index()
    expected = keys[['date', 'code', 'next_date']].merge(aggregated, how='left', validate='one_to_one')
    expected['source_valid'] = expected.bars.eq(30) & expected.labels.eq(30) & expected.valid_bars.eq(30)
    actual = pd.read_parquet(LABELS/'observations.parquet')
    equal(actual, expected[actual.columns], exact=True)
    assert len(actual) == report['rows'] and actual.source_valid.sum() == report['complete']
    actual['half'] = keys.half
    actual['hash'] = [hashlib.sha256(('liquidity-window-v1|'+d+'|'+code).encode()).hexdigest() for d, code in zip(actual.date, actual.code)]
    samples = actual.sort_values('hash').groupby(['half', 'source_valid']).head(16)
    sources = json.loads((OUT/'source_manifest.json').read_text())['minute_sha256']
    raw_index = raw.set_index(['date', 'code']); entry_index = entry.set_index(['date', 'code'])
    checks = 0
    for s in samples.itertuples():
        path = MINUTES/s.code[:2].upper()/(s.code[3:]+'.parquet')
        assert sha(path) == sources[str(path)]
        for day, start, end in [(s.next_date, '09:31', '10:01'), (s.date, '14:52', '14:56')]:
            b = pq.read_table(path, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'],
                filters=[('timestamp', '>=', pd.Timestamp(day+' '+start).to_pydatetime()),
                         ('timestamp', '<', pd.Timestamp(day+' '+end).to_pydatetime())]).to_pandas().sort_values('timestamp').reset_index(drop=True)
            checks += len(b)
            if start == '09:31':
                saved = raw_index.loc[[(s.date, s.code)], ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'amount']]
                equal(saved, b.rename(columns={'turnover': 'amount'}), exact=True)
                rebuilt = rebuild(b)
                for name, value in rebuilt.items():
                    np.testing.assert_allclose(getattr(s, name), value, atol=0, rtol=0, equal_nan=True)
            else:
                e = entry_index.loc[(s.date, s.code)]
                value = b.turnover.astype(float).sum()/b.volume.astype(float).sum() if b.volume.sum()>0 else np.nan
                for name, value in [('entry_bars', len(b)), ('entry_labels', b.timestamp.dt.strftime('%H:%M').nunique()),
                                    ('entry_volume', b.volume.astype(float).sum()), ('entry_vwap', value),
                                    ('entry_low', b.loc[b.volume.gt(0), 'low'].min()), ('entry_high', b.loc[b.volume.gt(0), 'high'].max())]:
                    np.testing.assert_allclose(e[name], value, atol=2e-10, rtol=0, equal_nan=True)
    c.close()
    proof = dict(passed=True, window_report_sha256=sha(LABELS/'window_report.json'), rows=len(keys), raw_rows=len(raw),
        all_morning_aggregates_independently_rebuilt=True, old_buy_sources_rejoined=True,
        frozen_signal_date_pool_complete=True, strict_next_market_dates_checked=True,
        fixed_source_windows=len(samples)*2, original_minutes_rechecked=checks,
        sample_keys=samples[['date', 'code']].to_dict('records'), new_2026_prices_read=False, no_exit_rules=True)
    save_json(LABELS/'window_verification.json', proof)
    return {k: v for k, v in proof.items() if k != 'sample_keys'}


def labels():
    c, keys, original = context()
    report = json.loads((LABELS/'full_label_report.json').read_text())
    proof = json.loads((LABELS/'window_verification.json').read_text())
    assert proof['passed'] and proof['window_report_sha256'] == sha(LABELS/'window_report.json')
    for key, path in [('protocol_sha256', PROTOCOL), ('input_manifest_sha256', LABELS/'input_manifest.json'),
                      ('window_verification_sha256', LABELS/'window_verification.json'),
                      ('classified_inputs_sha256', LABELS/'classified_inputs.parquet'),
                      ('added_labels_sha256', LABELS/'added_labels.parquet'), ('labels_sha256', LABELS/'full_labels.parquet'),
                      ('original_full_labels_sha256', OLD_FULL/'full_labels.parquet')]:
        assert report[key] == sha(path), key
    got = pd.read_parquet(LABELS/'added_labels.parquet')
    obs = pd.read_parquet(LABELS/'observations.parquet', columns=OBS_COLUMNS)
    unchanged = [x for x in BUY_COLUMNS if x not in ['necessary_tradeable', 'entry_fill_status', 'entry_recorded', 'entry_queue_unknown']]
    equal(got[unchanged], original[unchanged], exact=True); equal(got[OBS_COLUMNS], obs, exact=True)
    equal(got[['date', 'code', 'necessary_tradeable']], keys[['date', 'code', 'necessary_tradeable']], exact=True)
    original['necessary_tradeable'] = True
    source = original.merge(obs, on=['date', 'code', 'next_date'], validate='one_to_one')
    c.register('source', source)
    c.execute('''CREATE VIEW facts AS SELECT *,
        CASE WHEN NOT necessary_tradeable THEN 'not_submitted'
             WHEN NOT coalesce(isfinite(entry_vwap) AND entry_vwap>0 AND entry_volume>0,false) THEN 'no_liquidity'
             WHEN NOT coalesce(entry_volume*.1>=decision_shares,false) THEN 'volume_cap'
             WHEN entry_vwap*1.0005>=upper_limit-.005 THEN 'estimated_upper_limit' ELSE 'filled' END AS fill,
        coalesce(isfinite(next_preclose) AND next_preclose>0 AND abs(next_preclose-round(next_preclose,2))<=.0001,false) AS valid_ref,
        coalesce(isfinite(day_close) AND day_close>0 AND abs(day_close-round(day_close,2))<=.0001,false) AS valid_close,
        NOT entry_source_valid OR period_entry_bad_day OR period_bad_symbol AS entry_source_unknown FROM source''')
    c.execute('''CREATE VIEW checked AS SELECT *, fill='filled' AS recorded,
        NOT entry_source_unknown AND fill<>'filled' AS known_no_trade,
        NOT catalog_covered OR action_exposure OR NOT valid_ref OR NOT valid_close OR abs(next_preclose-day_close)>.005 AS corporate_unknown,
        coalesce(valid_ref AND next_trade_status=1 AND next_isST IN (0,1) AND next_adjustflag=3,false) AS next_daily_valid,
        fill='filled' AND (NOT entry_bounds_valid OR round(entry_high,2)>=upper_limit) AS queue_unknown FROM facts''')
    c.execute('''CREATE VIEW states AS SELECT *, CASE WHEN entry_source_unknown THEN 'entry_source_unknown'
        WHEN known_no_trade THEN 'no_trade' WHEN queue_unknown THEN 'entry_queue_unknown'
        WHEN corporate_unknown THEN 'corporate_unknown' WHEN NOT next_daily_valid THEN 'next_daily_unknown'
        WHEN NOT source_valid THEN 'morning_source_unknown' ELSE 'known' END AS observation_status FROM checked''')
    expected = c.sql('''SELECT date,code,corporate_unknown,next_daily_valid,entry_source_unknown,known_no_trade,observation_status,
        recorded AS entry_recorded,queue_unknown AS entry_queue_unknown,fill AS entry_fill_status FROM states ORDER BY date,code''').df()
    equal(got[expected.columns], expected, exact=True)
    count = len(got)*(len(expected.columns)-2)
    inputs = pd.read_parquet(LABELS/'classified_inputs.parquet')
    equal(got[inputs.columns], inputs, exact=True)
    for bps in [5, 15]:
        c.execute(f'''CREATE OR REPLACE VIEW cash AS WITH b AS (SELECT *,entry_vwap+greatest(entry_vwap*{bps}/10000.,.005) AS buy_price FROM states),
            v AS (SELECT *,decision_shares*buy_price AS buy_value FROM b)
            SELECT *,buy_value+greatest(buy_value*.0003,5)+buy_value*.00001 AS buy_cash,
                recorded AND buy_price>=upper_limit-.005 AS stress,
                observation_status='known' AND NOT (recorded AND buy_price>=upper_limit-.005) AS known FROM v''')
        ex = c.sql('SELECT date,code,buy_cash,stress,known,known AND NOT period_exit_bad_day AS sensitive_known,NOT known AND NOT known_no_trade AS unknown FROM cash ORDER BY date,code').df()
        ex = ex.rename(columns={k: f'{k}{bps}' for k in ['buy_cash', 'known', 'unknown', 'sensitive_known']}).rename(columns={'stress': f'entry_stress_unknown{bps}'})
        equal(got[ex.columns], ex); count += len(got)*5
        for name, price in [('sustained', 'sustained_close'), ('any_close', 'max_close'), ('mark_1000', 'price_1000'), ('adverse', 'min_low')]:
            ex = c.sql(f'''WITH v AS (SELECT *,decision_shares*({price}-greatest({price}*{bps}/10000.,.005)) AS mark_value FROM cash)
                SELECT date,code,CASE WHEN known THEN (mark_value-greatest(mark_value*.0003,5)-mark_value*.00051)/buy_cash-1 END AS value
                FROM v ORDER BY date,code''').df()
            np.testing.assert_allclose(got[f'{name}_return{bps}'], ex.value, atol=2e-10, rtol=0, equal_nan=True); count += len(got)
            if name in ['sustained', 'any_close']:
                target = f'opportunity{bps}' if name == 'sustained' else f'any_opportunity{bps}'
                np.testing.assert_array_equal(got[target], np.where(got[f'known{bps}'], ex.value.gt(0).astype(float), np.nan)); count += len(got)
                if name == 'sustained':
                    np.testing.assert_array_equal(got[f'one_percent{bps}'], np.where(got[f'known{bps}'], ex.value.ge(.01).astype(float), np.nan)); count += len(got)
    full = pd.read_parquet(LABELS/'full_labels.parquet'); old = pd.read_parquet(OLD_FULL/'full_labels.parquet')
    expected = pd.concat([old, got], ignore_index=True).sort_values(['date', 'code'])
    equal(full, expected, exact=True)
    assert len(full) == report['rows'] and len(got) == report['added_rows'] and len(old) == report['original_rows']
    assert not full.duplicated(['date', 'code']).any()
    for kind in ['added', 'combined']:
        chosen = pd.read_parquet(ROOT/kind/'selection.parquet')
        dates = chosen.loc[chosen.selected, 'date'].unique()
        equal(full.loc[full.date.isin(dates), ['date', 'code']], chosen.loc[chosen.date.isin(dates), ['date', 'code']], exact=True)
        assert chosen.loc[chosen.selected, ['date', 'code']].merge(full[['date', 'code']], how='left', indicator=True)._merge.eq('both').all()
    c.close()
    result = dict(passed=True, label_report_sha256=sha(LABELS/'full_label_report.json'), rows=len(full), added_rows=len(got),
        numeric_and_status_checks=count, original_labels_exactly_unchanged=True, all_sources_rejoined=True,
        all_selected_and_same_date_pool_keys_covered=True, raw_buy_status_rebuilt=True,
        old_tail_outcomes_not_read=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(LABELS/'full_label_verification.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('stage', choices=['windows', 'labels'])
    print(json.dumps(globals()[p.parse_args().stage](), ensure_ascii=False, indent=2))
