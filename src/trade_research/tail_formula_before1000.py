"""Preserve old selections while excluding the ambiguous 10:00 minute label."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_1000 as original
from .corporate_cash import MINUTES, save_json, sha

ROOT = Path('data/research/tail_formula_before1000')
PROTOCOL = Path('config/tail_formula_before1000_protocol.json')
MANIFEST = Path('data/research/economic_winner/input_manifest.json')
KEYS = ['date', 'code', 'next_date']
OLD_OBS = ['bars', 'labels', 'valid_bars', 'active_minutes', 'max_close',
           'sustained_close', 'min_low', 'price_1000', 'source_valid']
NEW_OBS = ['bars_0959', 'labels_0959', 'valid_bars_0959', 'active_minutes_0959',
           'max_close_0959', 'sustained_close_0959', 'min_low_0959',
           'price_0959', 'source_valid_0959']


def checked_protocol():
    p = json.loads(PROTOCOL.read_text())
    for path, digest in p['source_hashes'].items():
        assert sha(Path(path)) == digest, path
    assert sha(MANIFEST) == p['raw_manifest_sha256']
    for name in ['observation', 'full_label']:
        proof = json.loads((original.ROOT / (name + '_verification.json')).read_text())
        assert proof['passed']
    for root, source in p['selections'].items():
        for name in ['selection_report', 'analysis_report']:
            assert sha(Path('data/research') / root / (name + '.json')) == source[name + '_sha256']
        assert sha(Path('data/research') / root / 'selection.parquet') == source['selection_sha256']
    return p


def observations():
    if (ROOT / 'observation_report.json').exists():
        raise ValueError('Do not replace frozen boundary observations')
    checked_protocol()
    keys = pd.read_parquet(original.ROOT / 'observation_keys.parquet')
    assert keys.date.ge('2024-01-01').all() and keys.next_date.lt('2026-01-01').all()
    hashes = json.loads(MANIFEST.read_text())['source_sha256']
    codes = sorted(keys.code.unique())
    folder = ROOT / 'observation_parts'; folder.mkdir(parents=True, exist_ok=True)
    parts = {}; extractor = sha(Path(__file__))
    for offset in range(0, len(codes), 64):
        subset = codes[offset:offset + 64]
        path = folder / f'part_{offset // 64:03d}.parquet'; receipt = path.with_suffix('.json')
        if receipt.exists():
            meta = json.loads(receipt.read_text())
            assert meta['protocol_sha256'] == sha(PROTOCOL) and meta['extractor_sha256'] == extractor
            assert meta['codes'] == subset and meta['sha256'] == sha(path)
        else:
            files = [MINUTES / code[:2].upper() / (code[3:] + '.parquet') for code in subset]
            for file in files:
                assert sha(file) == hashes[str(file)]
            c = original.connection()
            c.read_parquet([str(x) for x in files]).create_view('raw')
            c.register('keys', keys.loc[keys.code.isin(subset), KEYS])
            rows = c.sql('''WITH s AS (
                SELECT lower(exchange)||'.'||symbol AS code,
                    strftime(timestamp,'%Y-%m-%d') AS next_date,timestamp,
                    strftime(timestamp,'%H:%M') AS clock,
                    (extract(hour FROM timestamp)*60+extract(minute FROM timestamp)-571)::INTEGER AS pos,
                    open::DOUBLE AS open,high::DOUBLE AS high,low::DOUBLE AS low,
                    close::DOUBLE AS close,volume::DOUBLE AS volume,turnover::DOUBLE AS amount
                FROM raw WHERE timestamp>=TIMESTAMP '2024-01-01' AND timestamp<TIMESTAMP '2026-01-01'
                    AND strftime(timestamp,'%H:%M') BETWEEN '09:31' AND '10:00'),
                b AS (SELECT k.date,s.*,coalesce(timestamp=date_trunc('minute',timestamp)
                    AND isfinite(open) AND isfinite(high) AND isfinite(low) AND isfinite(close)
                    AND isfinite(volume) AND isfinite(amount) AND least(open,high,low,close)>0
                    AND high+.0001>=greatest(open,close,low) AND low-.0001<=least(open,close)
                    AND abs(open-round(open,2))<=.0001 AND abs(high-round(high,2))<=.0001
                    AND abs(low-round(low,2))<=.0001 AND abs(close-round(close,2))<=.0001
                    AND volume>=0 AND amount>=0 AND (volume=0)=(amount=0)
                    AND (volume=0 OR amount/volume BETWEEN low-.0101 AND high+.0101),false) AS valid
                    FROM s JOIN keys k USING(next_date,code)),
                rolling AS (SELECT *,min(close) OVER three AS low_three,count(*) OVER three AS n_three,
                    count(*) FILTER(WHERE valid AND volume>0) OVER three AS active_three,
                    min(timestamp) OVER three AS first_three
                    FROM b WINDOW three AS(PARTITION BY date,code ORDER BY timestamp ROWS BETWEEN 2 PRECEDING AND CURRENT ROW))
                SELECT date,code,next_date,count(*) AS bars,count(DISTINCT clock) AS labels,
                    count(*) FILTER(WHERE valid) AS valid_bars,
                    count(*) FILTER(WHERE valid AND volume>0) AS active_minutes,
                    max(close) FILTER(WHERE valid AND volume>0) AS max_close,
                    max(low_three) FILTER(WHERE n_three=3 AND active_three=3 AND timestamp-first_three=INTERVAL 2 MINUTE) AS sustained_close,
                    min(low) FILTER(WHERE valid AND volume>0) AS min_low,
                    max(close) FILTER(WHERE clock='10:00' AND valid AND volume>0) AS price_1000,
                    bars=30 AND labels=30 AND valid_bars=30 AS source_valid,
                    count(*) FILTER(WHERE clock<='09:59') AS bars_0959,
                    count(DISTINCT clock) FILTER(WHERE clock<='09:59') AS labels_0959,
                    count(*) FILTER(WHERE clock<='09:59' AND valid) AS valid_bars_0959,
                    count(*) FILTER(WHERE clock<='09:59' AND valid AND volume>0) AS active_minutes_0959,
                    max(close) FILTER(WHERE clock<='09:59' AND valid AND volume>0) AS max_close_0959,
                    max(low_three) FILTER(WHERE clock<='09:59' AND n_three=3 AND active_three=3
                        AND timestamp-first_three=INTERVAL 2 MINUTE) AS sustained_close_0959,
                    min(low) FILTER(WHERE clock<='09:59' AND valid AND volume>0) AS min_low_0959,
                    max(close) FILTER(WHERE clock='09:59' AND valid AND volume>0) AS price_0959,
                    bars_0959=29 AND labels_0959=29 AND valid_bars_0959=29 AS source_valid_0959,
                    list(close ORDER BY timestamp) AS close_values,list(low ORDER BY timestamp) AS low_values,
                    bit_or(CASE WHEN valid THEN 1::BIGINT << pos ELSE 0::BIGINT END) AS valid_mask,
                    bit_or(CASE WHEN valid AND volume>0 THEN 1::BIGINT << pos ELSE 0::BIGINT END) AS active_mask
                FROM rolling GROUP BY date,code,next_date ORDER BY date,code''').df()
            c.close()
            assert not rows.duplicated(['date', 'code']).any()
            rows.to_parquet(path, index=False, compression='zstd')
            meta = dict(protocol_sha256=sha(PROTOCOL), extractor_sha256=extractor,
                        codes=subset, rows=len(rows), sha256=sha(path))
            save_json(receipt, meta)
        parts[str(path)] = meta['sha256']
        print(json.dumps(dict(codes=offset + len(subset), total_codes=len(codes))), flush=True)
    c = original.connection(); c.read_parquet(list(parts)).create_view('parts'); c.register('keys', keys[KEYS])
    rows = c.sql('''SELECT k.*,p.* EXCLUDE(date,code,next_date,close_values,low_values,valid_mask,active_mask)
        FROM keys k LEFT JOIN parts p USING(date,code,next_date) ORDER BY date,code''').df(); c.close()
    rows['source_valid'] = rows.source_valid.fillna(False)
    rows['source_valid_0959'] = rows.source_valid_0959.fillna(False)
    old = pd.read_parquet(original.ROOT / 'observations.parquet').sort_values(['date', 'code']).reset_index(drop=True)
    pd.testing.assert_frame_equal(rows[KEYS + OLD_OBS], old[KEYS + OLD_OBS], check_dtype=False, check_exact=True)
    assert rows.loc[rows.source_valid, 'source_valid_0959'].all()
    rows.to_parquet(ROOT / 'observations.parquet', index=False, compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL), extractor_sha256=extractor, parts_sha256=parts,
             observations_sha256=sha(ROOT / 'observations.parquet'), rows=len(rows),
             all_original_30bar_aggregates_exact=True, original_complete=int(rows.source_valid.sum()),
             new_complete=int(rows.source_valid_0959.sum()), original_unknowns_will_not_be_recovered=True,
             last_observation=rows.next_date.max(), new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'observation_report.json', r)
    return {k: v for k, v in r.items() if k != 'parts_sha256'}


def net_mark(prices, shares, cash, bps):
    value = shares * (prices - np.maximum(prices * bps / 10000, .005))
    return (value - np.maximum(value * .0003, 5) - value * .00051) / cash - 1


def labels():
    if (ROOT / 'full_label_report.json').exists():
        raise ValueError('Do not replace conservative labels')
    checked_protocol()
    proof = json.loads((ROOT / 'observation_verification.json').read_text())
    assert proof['passed'] and proof['observation_report_sha256'] == sha(ROOT / 'observation_report.json')
    old = pd.read_parquet(original.ROOT / 'full_labels.parquet').sort_values(['date', 'code']).reset_index(drop=True)
    obs = pd.read_parquet(ROOT / 'observations.parquet')
    pd.testing.assert_frame_equal(old[KEYS], obs[KEYS], check_exact=True)
    rows = old.copy()
    for new, source in [('active_minutes', 'active_minutes_0959'), ('max_close', 'max_close_0959'),
                        ('sustained_close', 'sustained_close_0959'), ('min_low', 'min_low_0959')]:
        rows[new] = obs[source]
    # No column aliases suggesting this reference price is at 10:00.
    rows = rows.drop(columns=['price_1000', 'mark_1000_return5', 'mark_1000_return15'])
    rows['price_0959'] = obs.price_0959
    rows['source_valid_0959'] = obs.source_valid_0959
    for bps in [5, 15]:
        known = old[f'known{bps}']
        assert rows.loc[known, 'source_valid_0959'].all()
        for name, price in [('sustained', rows.sustained_close), ('any_close', rows.max_close),
                            ('mark_0959', rows.price_0959), ('adverse', rows.min_low)]:
            rows[f'{name}_return{bps}'] = net_mark(price, rows.decision_shares, rows[f'buy_cash{bps}'], bps).where(known)
        rows[f'opportunity{bps}'] = rows[f'sustained_return{bps}'].gt(0).astype(float).where(known)
        rows[f'any_opportunity{bps}'] = rows[f'any_close_return{bps}'].gt(0).astype(float).where(known)
        rows[f'one_percent{bps}'] = rows[f'sustained_return{bps}'].ge(.01).astype(float).where(known)
        assert (rows.loc[known, f'opportunity{bps}'] <= old.loc[known, f'opportunity{bps}']).all()
        assert (rows.loc[known, f'one_percent{bps}'] <= old.loc[known, f'one_percent{bps}']).all()
    rows.to_parquet(ROOT / 'full_labels.parquet', index=False, compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL), rows=len(rows), labels_sha256=sha(ROOT / 'full_labels.parquet'),
             old_label_report_sha256=sha(original.ROOT / 'full_label_report.json'),
             observation_verification_sha256=sha(ROOT / 'observation_verification.json'),
             window_labels=['09:31', '09:59'], window_bars=29, mark_reference_label='09:59',
             all_original_statuses_preserved=True, no_reclassification_of_old_unknowns=True,
             new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'full_label_report.json', r)
    return r


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['observations', 'labels'])
    print(json.dumps(globals()[parser.parse_args().stage](), ensure_ascii=False, indent=2))
