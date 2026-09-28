"""Independent array, source and cash checks for the conservative morning window."""
import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from trade_research import tail_formula_before1000 as study
from trade_research.corporate_cash import MINUTES, save_json, sha


def array_observations(closes, lows, active, size):
    """Calculate a prefix using arrays, without SQL rolling-window operations."""
    x = closes[:, :size]; lo = lows[:, :size]; ok = active[:, :size]
    triples = np.minimum(np.minimum(x[:, :-2], x[:, 1:-1]), x[:, 2:])
    valid_triples = ok[:, :-2] & ok[:, 1:-1] & ok[:, 2:]
    sustained = np.where(valid_triples, triples, -np.inf).max(axis=1)
    maximum = np.where(ok, x, -np.inf).max(axis=1)
    minimum = np.where(ok, lo, np.inf).min(axis=1)
    for values in [sustained, maximum, minimum]:
        values[~np.isfinite(values)] = np.nan
    return dict(active_minutes=ok.sum(axis=1), sustained_close=sustained,
                max_close=maximum, min_low=minimum, price=np.where(ok[:, -1], x[:, -1], np.nan))


def observations():
    study.checked_protocol()
    report = json.loads((study.ROOT / 'observation_report.json').read_text())
    assert report['protocol_sha256'] == sha(study.PROTOCOL)
    assert report['observations_sha256'] == sha(study.ROOT / 'observations.parquet')
    total = 0; raw_samples = []; samples = []
    for path, digest in report['parts_sha256'].items():
        assert sha(Path(path)) == digest
        f = pd.read_parquet(path)
        f = f.loc[f.source_valid].reset_index(drop=True)
        if not len(f):
            continue
        assert f.bars.eq(30).all() and f.labels.eq(30).all() and f.valid_bars.eq(30).all()
        assert f.valid_mask.eq((1 << 30) - 1).all()
        x = np.stack(f.close_values); lo = np.stack(f.low_values)
        assert x.shape == lo.shape == (len(f), 30)
        active = (f.active_mask.to_numpy(dtype='int64')[:, None] & (1 << np.arange(30))) != 0
        for size, suffix, price in [(30, '', 'price_1000'), (29, '_0959', 'price_0959')]:
            ex = array_observations(x, lo, active, size)
            for name, values in ex.items():
                column = price if name == 'price' else name + suffix
                np.testing.assert_allclose(f[column], values, rtol=0, atol=0, equal_nan=True)
            if size == 29:
                assert f.bars_0959.eq(29).all() and f.labels_0959.eq(29).all()
                assert f.valid_bars_0959.eq(29).all() and f.source_valid_0959.all()
        f['sample_hash'] = [hashlib.sha256(('boundary-raw-v1' + d + c).encode()).hexdigest()
                            for d, c in zip(f.date, f.code)]
        f['half'] = f.date.str[:4] + np.where(f.date.str[5:7].le('06'), 'H1', 'H2')
        samples.append(f.sort_values('sample_hash').groupby('half').head(8))
        total += len(f)
    assert total == report['original_complete']
    cases = pd.concat(samples, ignore_index=True).sort_values('sample_hash').groupby('half').head(8)
    assert len(cases) == 32
    for row in cases.itertuples():
        path = MINUTES / row.code[:2].upper() / (row.code[3:] + '.parquet')
        raw = pd.read_parquet(path, filters=[('timestamp', '>=', pd.Timestamp(row.next_date + ' 09:31')),
                                            ('timestamp', '<=', pd.Timestamp(row.next_date + ' 10:00'))])
        raw = raw.sort_values('timestamp').reset_index(drop=True)
        assert raw.timestamp.tolist() == pd.date_range(row.next_date + ' 09:31', periods=30, freq='min').tolist()
        np.testing.assert_array_equal(raw.close.to_numpy(float), row.close_values)
        np.testing.assert_array_equal(raw.low.to_numpy(float), row.low_values)
        prices = raw[['open', 'high', 'low', 'close']].to_numpy(float)
        volume = raw.volume.to_numpy(float); amount = raw.turnover.to_numpy(float)
        valid = np.isfinite(prices).all(axis=1) & (prices > 0).all(axis=1)
        valid &= (np.abs(prices - np.round(prices, 2)) <= .0001).all(axis=1)
        valid &= prices[:, 1] + .0001 >= prices.max(axis=1)
        valid &= prices[:, 2] - .0001 <= prices.min(axis=1)
        valid &= np.isfinite(volume) & np.isfinite(amount) & (volume >= 0) & (amount >= 0)
        valid &= (volume == 0) == (amount == 0)
        vwap = np.divide(amount, volume, out=np.zeros(len(raw)), where=volume > 0)
        valid &= (volume == 0) | ((vwap >= prices[:, 2] - .0101) & (vwap <= prices[:, 1] + .0101))
        assert valid.all()
        mask = sum(1 << i for i, positive in enumerate(volume > 0) if positive)
        assert mask == row.active_mask
        raw_samples.append(dict(date=row.date, code=row.code, next_date=row.next_date, bars=len(raw)))
    r = dict(passed=True, observation_report_sha256=sha(study.ROOT / 'observation_report.json'),
             verifier_sha256=sha(Path(__file__)), all_complete_windows_checked=total,
             all_29_and_30_prefix_prices_independently_rebuilt=True, raw_samples=raw_samples,
             sample_bars=sum(r['bars'] for r in raw_samples), new_2026_prices_read=False, no_exit_rules=True)
    save_json(study.ROOT / 'observation_verification.json', r)
    return r


def labels():
    report = json.loads((study.ROOT / 'full_label_report.json').read_text())
    assert report['protocol_sha256'] == sha(study.PROTOCOL)
    assert report['labels_sha256'] == sha(study.ROOT / 'full_labels.parquet')
    got = pd.read_parquet(study.ROOT / 'full_labels.parquet')
    old = pd.read_parquet(study.original.ROOT / 'full_labels.parquet').sort_values(['date', 'code']).reset_index(drop=True)
    changed = {'active_minutes', 'max_close', 'sustained_close', 'min_low', 'price_1000'}
    for bps in [5, 15]:
        changed.update(f'{name}{bps}' for name in ['sustained_return', 'any_close_return', 'mark_1000_return',
                                                   'adverse_return', 'opportunity', 'any_opportunity', 'one_percent'])
    unchanged = [name for name in old.columns if name not in changed]
    pd.testing.assert_frame_equal(got[unchanged], old[unchanged], check_exact=True)
    obs = pd.read_parquet(study.ROOT / 'observations.parquet')
    pd.testing.assert_frame_equal(got[study.KEYS], obs[study.KEYS], check_exact=True)
    for name in ['active_minutes', 'max_close', 'sustained_close', 'min_low']:
        np.testing.assert_allclose(got[name], obs[name + '_0959'], atol=0, rtol=0, equal_nan=True)
    np.testing.assert_allclose(got.price_0959, obs.price_0959, atol=0, rtol=0, equal_nan=True)
    c = duckdb.connect(); c.register('g', got); count = len(got) * len(unchanged)
    for bps in [5, 15]:
        for metric, price in [('sustained', 'sustained_close'), ('any_close', 'max_close'),
                              ('mark_0959', 'price_0959'), ('adverse', 'min_low')]:
            ex = c.sql(f'''WITH v AS(SELECT *,decision_shares*({price}-greatest(.005,{price}*{bps}/10000.)) AS mark FROM g)
                SELECT CASE WHEN known{bps} THEN (mark-CASE WHEN mark*.0003>5 THEN mark*.0003 ELSE 5 END
                    -mark*.00051)/buy_cash{bps}-1 END AS value FROM v''').df().value
            np.testing.assert_allclose(got[f'{metric}_return{bps}'], ex, rtol=0, atol=2e-10, equal_nan=True)
            count += len(got)
        for label, value, threshold, inclusive in [
                ('opportunity', 'sustained', 0, False), ('any_opportunity', 'any_close', 0, False),
                ('one_percent', 'sustained', .01, True)]:
            values = got[f'{value}_return{bps}']
            expected = np.where(got[f'known{bps}'], values.ge(threshold) if inclusive else values.gt(threshold), np.nan)
            np.testing.assert_array_equal(got[f'{label}{bps}'], expected)
            assert (got.loc[got[f'known{bps}'], f'{label}{bps}'] <= old.loc[old[f'known{bps}'], f'{label}{bps}']).all()
            count += len(got)
    c.close()
    assert not any('1000' in name for name in got.columns)
    r = dict(passed=True, label_report_sha256=sha(study.ROOT / 'full_label_report.json'), rows=len(got),
             verifier_sha256=sha(Path(__file__)), unchanged_original_columns=len(unchanged), scalar_checks=count,
             all_old_statuses_and_buy_cash_exact=True, window_end='09:59',
             no_old_unknown_recovered=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(study.ROOT / 'full_label_verification.json', r)
    return r


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['observations', 'labels'])
    print(json.dumps(globals()[parser.parse_args().stage](), ensure_ascii=False, indent=2))
