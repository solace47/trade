"""Bounded training labels and post-freeze full labels for an expanded pool.

Old morning labels remain byte-for-byte values. Added rows reuse declared
morning caches before extracting missing windows, and reauthorize the same
buy simulator. No old tail-exit outcome is projected.
"""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from . import tail_formula_pool_scope_inputs as inputs
from . import tail_formula_additive as base
from . import tail_formula_before1000 as original
from .corporate_cash import MINUTES, save_json, sha
from .tail_formula_1000_analysis import BUY_COLUMNS
from .turnover_reference import CALENDAR

PROTOCOL = Path('config/tail_formula_pool_scope_labels_protocol.json')
OLD_BUY = Path('data/research/economic_winner/period_quality')
OLD_LOW = Path('data/research/tail_formula_liquidity/labels')
KEYS = ['date', 'code', 'next_date']
RAW = [*KEYS, 'timestamp', 'clock', 'open', 'high', 'low', 'close', 'volume', 'amount']


def directory(scope):
    assert scope in ['training', 'evaluation']
    return inputs.ROOT / (scope + '_labels')


def checked(scope):
    p = json.loads(PROTOCOL.read_text()); inputs.checked_base()
    assert p['training_observation_end_exclusive'] == '2025-07-01'
    assert not p['new_2026_prices_allowed'] and p['window_end'] == '09:59'
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest, file
    r = json.loads((inputs.OUT / 'feature_report.json').read_text())
    v = json.loads((inputs.OUT / 'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(inputs.OUT / 'feature_report.json')
    assert r['features_sha256'] == sha(inputs.OUT / 'features.parquet')
    for root, report, proof, data in [(OLD_BUY, 'label_report.json', 'label_verification.json', 'labels.parquet'),
        (original.ROOT, 'full_label_report.json', 'full_label_verification.json', 'full_labels.parquet')]:
        r = json.loads((root / report).read_text()); v = json.loads((root / proof).read_text())
        assert v['passed'] and v['label_report_sha256'] == sha(root / report)
        assert r['labels_sha256'] == sha(root / data)
    if scope == 'evaluation':
        file = inputs.ROOT / 'joint_selection_freeze.json'; joint = json.loads(file.read_text())
        committed = subprocess.run(['git', 'show', 'HEAD:docs/selection-formula.md'],
            capture_output=True, text=True, check=True).stdout
        assert joint['passed'] and sha(file) in committed
        for file, digest in joint['source_hashes'].items():
            assert sha(Path(file)) == digest
    return p


def wanted(scope):
    p = checked(scope); f = pd.read_parquet(inputs.OUT / 'universe.parquet')
    cal = pd.read_parquet(CALENDAR)
    days = sorted(cal.loc[cal.is_trading_day.eq('1') &
        cal.calendar_date.between('2024-01-01', '2025-12-31'), 'calendar_date'])
    f['next_date'] = f.date.map(dict(zip(days[:-1], days[1:])))
    assert f.next_date.gt(f.date).all() and f.next_date.lt('2026-01-01').all()
    if scope == 'training':
        f = f.loc[f.next_date.lt(p['training_observation_end_exclusive'])]
    return f.reset_index(drop=True)


def cache_sources(scope):
    sources = []
    if scope == 'evaluation':
        root = directory('training'); r = json.loads((root / 'raw_report.json').read_text())
        v = json.loads((root / 'window_verification.json').read_text())
        assert v['passed'] and v['window_report_sha256'] == sha(root / 'window_report.json')
        sources.append((root, root / 'extra_keys.parquet', r['parts_sha256']))
    r = json.loads((OLD_LOW / 'raw_report.json').read_text())
    v = json.loads((OLD_LOW / 'window_verification.json').read_text())
    assert v['passed'] and v['window_report_sha256'] == sha(OLD_LOW / 'window_report.json')
    assert json.loads((OLD_LOW / 'window_report.json').read_text())['raw_report_sha256'] == sha(OLD_LOW / 'raw_report.json')
    sources.append((OLD_LOW, OLD_LOW / 'keys.parquet', r['parts_sha256']))
    return sources


def prepare(scope):
    p = checked(scope); root = directory(scope); assert not (root / 'input_manifest.json').exists()
    root.mkdir(parents=True, exist_ok=True); f = wanted(scope)
    extra = f.loc[~f.original_pool].reset_index(drop=True)
    assert extra.necessary_tradeable.all() and not extra.historical_necessary_tradeable.any()
    for name, frame in [('keys', f), ('extra_keys', extra)]:
        frame.to_parquet(root / (name + '.parquet'), index=False, compression='zstd')
    c = base.conn(); c.register('keys', extra[KEYS])
    entry = c.execute('SELECT ' + ','.join('o.' + x for x in BUY_COLUMNS) +
        ' FROM read_parquet(?) o JOIN keys USING(date,code,next_date) ORDER BY date,code',
        [str(OLD_BUY / 'labels.parquet')]).df(); c.close()
    assert len(entry) == len(extra) and not entry.necessary_tradeable.any()
    cols = [*KEYS, 'half', 'board', 'decision_shares', 'price_1449', 'preclose', 'upper_limit']
    pd.testing.assert_frame_equal(entry[cols], extra[cols], check_exact=True)
    entry.to_parquet(root / 'entry_original.parquet', index=False, compression='zstd')
    remaining = extra[KEYS].copy(); reused = {}; cache_receipts = {}; reused_keys = 0
    for i, (parent, keyfile, parts) in enumerate(cache_sources(scope)):
        known = pd.read_parquet(keyfile, columns=KEYS)
        matched = remaining.merge(known, on=KEYS, how='inner', validate='one_to_one')
        for file, digest in parts.items():
            assert sha(Path(file)) == digest
        cache_receipts[str(parent / 'raw_report.json')] = sha(parent / 'raw_report.json')
        cache_receipts[str(parent / 'window_verification.json')] = sha(parent / 'window_verification.json')
        cache_receipts[str(keyfile)] = sha(keyfile)
        if not len(matched):
            continue
        c = base.conn(); c.register('keys', matched); c.read_parquet(list(parts)).create_view('raw')
        raw = c.sql('SELECT ' + ','.join('r.' + n for n in RAW) +
            ' FROM raw r JOIN keys USING(date,code,next_date) ORDER BY r.date,r.code,r.timestamp').df(); c.close()
        assert not raw.duplicated(['date', 'code', 'timestamp']).any()
        path = root / f'reused_raw_{i}.parquet'; raw.to_parquet(path, index=False, compression='zstd')
        reused[str(path)] = sha(path); reused_keys += len(matched)
        remaining = remaining.merge(matched.assign(cached=True), on=KEYS, how='left', validate='one_to_one')
        remaining = remaining.loc[~remaining.cached.eq(True), KEYS].reset_index(drop=True)
    remaining.to_parquet(root / 'missing_keys.parquet', index=False, compression='zstd')
    m = dict(protocol_sha256=sha(PROTOCOL), scope=scope,
        keys_sha256=sha(root / 'keys.parquet'), extra_keys_sha256=sha(root / 'extra_keys.parquet'),
        missing_keys_sha256=sha(root / 'missing_keys.parquet'), entry_original_sha256=sha(root / 'entry_original.parquet'),
        feature_verification_sha256=sha(inputs.OUT / 'feature_verification.json'), calendar_sha256=sha(CALENDAR),
        cache_receipts=cache_receipts, reused_parts_sha256=reused,
        rows=len(f), extra_rows=len(extra), reused_extra_keys=reused_keys, missing_extra_keys=len(remaining),
        last_observation=f.next_date.max(), no_old_tail_exit_fields_projected=True,
        no_evaluation_group_statistics_computed=True, new_2026_prices_read=False, no_exit_rules=True)
    assert reused_keys + len(remaining) == len(extra)
    save_json(root / 'input_manifest.json', m); return {k: v for k, v in m.items() if k not in ['cache_receipts', 'reused_parts_sha256']}


def checked_keys(scope):
    p = checked(scope); root = directory(scope); m = json.loads((root / 'input_manifest.json').read_text())
    assert m['protocol_sha256'] == sha(PROTOCOL) and m['scope'] == scope
    for name in ['keys', 'extra_keys', 'missing_keys', 'entry_original']:
        assert m[name + '_sha256'] == sha(root / (name + '.parquet'))
    for file, digest in {**m['cache_receipts'], **m['reused_parts_sha256']}.items():
        assert sha(Path(file)) == digest
    f = pd.read_parquet(root / 'keys.parquet')
    if scope == 'training':
        assert f.next_date.lt(p['training_observation_end_exclusive']).all()
    assert f.date.ge('2024-01-01').all() and f.next_date.lt('2026-01-01').all()
    return p, m, f


def raw(scope):
    p, m, _ = checked_keys(scope); root = directory(scope); assert not (root / 'raw_report.json').exists()
    keys = pd.read_parquet(root / 'missing_keys.parquet'); codes = sorted(keys.code.unique())
    folder = root / 'raw_parts'; folder.mkdir(exist_ok=True)
    hashes = json.loads(inputs.cached.MINUTE_MANIFEST.read_text())['source_sha256']
    parts = dict(m['reused_parts_sha256'])
    end = p['training_observation_end_exclusive'] if scope == 'training' else '2026-01-01'
    for offset in range(0, len(codes), 64):
        subset = codes[offset:offset + 64]; path = folder / f'part_{offset // 64:03d}.parquet'; receipt = path.with_suffix('.json')
        identity = dict(codes=subset, input_manifest_sha256=sha(root / 'input_manifest.json'), extractor_sha256=sha(Path(__file__)))
        if receipt.exists():
            meta = json.loads(receipt.read_text())
            assert all(meta[k] == v for k, v in identity.items()) and meta['sha256'] == sha(path)
        else:
            files = [MINUTES / code[:2].upper() / (code[3:] + '.parquet') for code in subset]
            for file in files:
                assert sha(file) == hashes[str(file)]
            c = base.conn(); c.read_parquet([str(file) for file in files]).create_view('raw')
            c.register('keys', keys.loc[keys.code.isin(subset)])
            f = c.execute('''WITH s AS(SELECT lower(exchange)||'.'||symbol AS code,
                strftime(timestamp,'%Y-%m-%d') AS next_date,timestamp,strftime(timestamp,'%H:%M') AS clock,
                open::DOUBLE AS open,high::DOUBLE AS high,low::DOUBLE AS low,close::DOUBLE AS close,
                volume::DOUBLE AS volume,turnover::DOUBLE AS amount FROM raw
                WHERE timestamp>=TIMESTAMP '2024-01-01' AND timestamp<CAST(? AS TIMESTAMP)
                AND strftime(timestamp,'%H:%M') BETWEEN '09:31' AND '10:00')
                SELECT k.date,s.* FROM s JOIN keys k USING(code,next_date) ORDER BY date,code,timestamp''', [end]).df(); c.close()
            assert not f.duplicated(['date', 'code', 'timestamp']).any()
            f.to_parquet(path, index=False, compression='zstd'); meta = dict(**identity, sha256=sha(path), rows=len(f)); save_json(receipt, meta)
        parts[str(path)] = meta['sha256']
        print(json.dumps(dict(codes=offset + len(subset), total_codes=len(codes))), flush=True)
    r = dict(protocol_sha256=sha(PROTOCOL), input_manifest_sha256=sha(root / 'input_manifest.json'),
        extractor_sha256=sha(Path(__file__)), parts_sha256=parts,
        raw_rows=sum(len(pd.read_parquet(file, columns=['date'])) for file in parts),
        scope=scope, last_observation=m['last_observation'], no_evaluation_group_statistics_computed=True,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(root / 'raw_report.json', r); return {k: v for k, v in r.items() if k != 'parts_sha256'}


def windows(scope):
    _, _, _ = checked_keys(scope); root = directory(scope); assert not (root / 'window_report.json').exists()
    r = json.loads((root / 'raw_report.json').read_text()); assert r['input_manifest_sha256'] == sha(root / 'input_manifest.json')
    for file, digest in r['parts_sha256'].items():
        assert sha(Path(file)) == digest
    keys = pd.read_parquet(root / 'extra_keys.parquet', columns=KEYS)
    c = base.conn(); c.read_parquet(list(r['parts_sha256'])).create_view('raw'); c.register('keys', keys)
    f = c.sql('''WITH b AS(SELECT *,coalesce(timestamp=date_trunc('minute',timestamp)
        AND isfinite(open) AND isfinite(high) AND isfinite(low) AND isfinite(close)
        AND isfinite(volume) AND isfinite(amount) AND least(open,high,low,close)>0
        AND high+.0001>=greatest(open,close,low) AND low-.0001<=least(open,close)
        AND abs(open-round(open,2))<=.0001 AND abs(high-round(high,2))<=.0001
        AND abs(low-round(low,2))<=.0001 AND abs(close-round(close,2))<=.0001
        AND volume>=0 AND amount>=0 AND (volume=0)=(amount=0)
        AND (volume=0 OR amount/volume BETWEEN low-.0101 AND high+.0101),false) AS valid FROM raw),
        rolling AS(SELECT *,min(close) OVER three AS low_three,count(*) OVER three AS n_three,
        count(*) FILTER(WHERE valid AND volume>0) OVER three AS active_three,min(timestamp) OVER three AS first_three
        FROM b WINDOW three AS(PARTITION BY date,code ORDER BY timestamp ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)),
        a AS(SELECT date,code,next_date,count(*) AS bars30,count(DISTINCT clock) AS labels30,
        count(*) FILTER(WHERE valid) AS good30,
        count(*) FILTER(WHERE clock<='09:59') AS bars29,count(DISTINCT clock) FILTER(WHERE clock<='09:59') AS labels29,
        count(*) FILTER(WHERE clock<='09:59' AND valid) AS good29,
        count(*) FILTER(WHERE clock<='09:59' AND valid AND volume>0) AS active_minutes,
        max(close) FILTER(WHERE clock<='09:59' AND valid AND volume>0) AS max_close,
        max(low_three) FILTER(WHERE clock<='09:59' AND n_three=3 AND active_three=3
            AND timestamp-first_three=INTERVAL 2 MINUTE) AS sustained_close,
        min(low) FILTER(WHERE clock<='09:59' AND valid AND volume>0) AS min_low,
        max(close) FILTER(WHERE clock='09:59' AND valid AND volume>0) AS price_0959
        FROM rolling GROUP BY date,code,next_date)
        SELECT k.*,a.* EXCLUDE(date,code,next_date),
        coalesce(bars30=30 AND labels30=30 AND good30=30,false) AS source_valid,
        coalesce(bars29=29 AND labels29=29 AND good29=29,false) AS source_valid_0959
        FROM keys k LEFT JOIN a USING(date,code,next_date) ORDER BY date,code''').df(); c.close()
    assert f.loc[f.source_valid, 'source_valid_0959'].all()
    f.to_parquet(root / 'observations.parquet', index=False, compression='zstd')
    report = dict(protocol_sha256=sha(PROTOCOL), input_manifest_sha256=sha(root / 'input_manifest.json'),
        raw_report_sha256=sha(root / 'raw_report.json'), observations_sha256=sha(root / 'observations.parquet'),
        scope=scope, rows=len(f), original_30bar_quality_guard_retained=True,
        economic_observation_end='09:59', no_evaluation_group_statistics_computed=True,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(root / 'window_report.json', report); return report


def classify(entry, obs):
    e = entry.copy(); e['necessary_tradeable'] = True
    liquid = np.isfinite(e.entry_vwap) & e.entry_vwap.gt(0) & e.entry_volume.gt(0)
    e['entry_fill_status'] = np.select([~liquid, ~e.entry_volume.mul(.1).ge(e.decision_shares),
        e.entry_vwap.mul(1.0005).ge(e.upper_limit-.005)], ['no_liquidity', 'volume_cap', 'estimated_upper_limit'], default='filled')
    e['entry_recorded'] = e.entry_fill_status.eq('filled')
    e['entry_queue_unknown'] = e.entry_recorded & (~e.entry_bounds_valid | e.entry_high.round(2).ge(e.upper_limit))
    r = obs.merge(e, on=KEYS, validate='one_to_one').sort_values(['date', 'code']).reset_index(drop=True)
    valid_ref = np.isfinite(r.next_preclose) & r.next_preclose.gt(0) & (r.next_preclose-r.next_preclose.round(2)).abs().le(.0001)
    valid_close = np.isfinite(r.day_close) & r.day_close.gt(0) & (r.day_close-r.day_close.round(2)).abs().le(.0001)
    r['corporate_unknown'] = ~r.catalog_covered | r.action_exposure | ~valid_ref | ~valid_close | (r.next_preclose-r.day_close).abs().gt(.005)
    r['next_daily_valid'] = valid_ref & r.next_trade_status.eq(1) & r.next_isST.isin([0, 1]) & r.next_adjustflag.eq(3)
    r['entry_source_unknown'] = ~r.entry_source_valid | r.period_entry_bad_day | r.period_bad_symbol
    r['known_no_trade'] = ~r.entry_source_unknown & ~r.entry_recorded
    r['observation_status'] = np.select([r.entry_source_unknown, r.known_no_trade, r.entry_queue_unknown,
        r.corporate_unknown, ~r.next_daily_valid, ~r.source_valid],
        ['entry_source_unknown', 'no_trade', 'entry_queue_unknown', 'corporate_unknown', 'next_daily_unknown', 'morning_source_unknown'], default='known')
    for bps in [5, 15]:
        buy = r.entry_vwap + np.maximum(r.entry_vwap*bps/10000, .005); value = r.decision_shares*buy
        r[f'buy_cash{bps}'] = value + np.maximum(5, value*.0003) + value*.00001
        r[f'entry_stress_unknown{bps}'] = r.entry_recorded & buy.ge(r.upper_limit-.005)
        known = r.observation_status.eq('known') & ~r[f'entry_stress_unknown{bps}']; r[f'known{bps}'] = known
        for name, price in [('sustained', r.sustained_close), ('any_close', r.max_close), ('mark_0959', r.price_0959), ('adverse', r.min_low)]:
            r[f'{name}_return{bps}'] = original.net_mark(price, r.decision_shares, r[f'buy_cash{bps}'], bps).where(known)
        r[f'opportunity{bps}'] = r[f'sustained_return{bps}'].gt(0).astype(float).where(known)
        r[f'any_opportunity{bps}'] = r[f'any_close_return{bps}'].gt(0).astype(float).where(known)
        r[f'one_percent{bps}'] = r[f'sustained_return{bps}'].ge(.01).astype(float).where(known)
        r[f'unknown{bps}'] = ~known & ~r.known_no_trade; r[f'sensitive_known{bps}'] = known & ~r.period_exit_bad_day
    return r


def labels(scope):
    _, _, keys = checked_keys(scope); root = directory(scope); assert not (root / 'full_label_report.json').exists()
    v = json.loads((root / 'window_verification.json').read_text())
    assert v['passed'] and v['window_report_sha256'] == sha(root / 'window_report.json')
    c = base.conn(); c.register('keys', keys.loc[keys.original_pool, KEYS])
    old = c.execute('SELECT o.* FROM read_parquet(?) o JOIN keys USING(date,code,next_date) ORDER BY date,code',
        [str(original.ROOT / 'full_labels.parquet')]).df(); c.close()
    obs = pd.read_parquet(root / 'observations.parquet'); entry = pd.read_parquet(root / 'entry_original.parquet')
    extra = classify(entry, obs)[old.columns]
    extra.to_parquet(root / 'added_labels.parquet', index=False, compression='zstd')
    full = pd.concat([old, extra], ignore_index=True).sort_values(['date', 'code']).reset_index(drop=True)
    pd.testing.assert_frame_equal(full[KEYS], keys[KEYS], check_exact=True)
    pd.testing.assert_frame_equal(full.loc[keys.original_pool].reset_index(drop=True), old.reset_index(drop=True), check_exact=True)
    full.to_parquet(root / 'full_labels.parquet', index=False, compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL), input_manifest_sha256=sha(root / 'input_manifest.json'),
        window_verification_sha256=sha(root / 'window_verification.json'),
        labels_sha256=sha(root / 'full_labels.parquet'), added_labels_sha256=sha(root / 'added_labels.parquet'),
        rows=len(full), extra_rows=len(extra), scope=scope, last_observation=full.next_date.max(),
        original_labels_values_and_unknowns_preserved=True, buy_windows_reused_and_fill_status_recomputed=True,
        reference_label='09:59', no_old_tail_exit_fields_projected=True,
        no_evaluation_group_statistics_computed=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(root / 'full_label_report.json', r); return r


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('stage', choices=['prepare', 'raw', 'windows', 'labels'])
    p.add_argument('--scope', choices=['training', 'evaluation'], required=True); a = p.parse_args()
    print(json.dumps(globals()[a.stage](a.scope), ensure_ascii=False, indent=2))
