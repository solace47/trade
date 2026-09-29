"""Restore 2023 training labels to the frozen 09:59 boundary from cached minutes."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_before1000 as modern_labels
from . import tail_formula_float as modern_features
from .corporate_cash import save_json, sha

STEM = 'tail_formula_quarter_history2024'
PROTOCOL = Path('config') / (STEM + '_protocol.json')
ROOT = Path('data/research') / STEM / 'inputs'
HISTORY = ROOT / 'history_2023'
OLD = Path('data/research/tail_formula_long48/labels')
EARLIER = Path('data/research/tail_formula_long48/inputs')
KEYS = ['date', 'code', 'next_date']
FIELDS = ['bars', 'labels', 'valid_bars', 'active_minutes', 'max_close', 'sustained_close', 'min_low', 'price_1000', 'source_valid']
FEATURE_COLUMNS = ['date', 'code', 'half', 'board', 'decision_shares', 'formula_input_valid', *modern_features.EXPRESSIONS]
LABEL_COLUMNS = ['date', 'code', 'next_date', 'known15', 'opportunity15', 'known_no_trade', 'adverse_return15']


def bounded(file, year, columns=None):
    return pd.read_parquet(file, columns=columns, filters=[('date', '>=', f'{year}-01-01'), ('date', '<', f'{year+1}-01-01')])


def checked():
    p = json.loads(PROTOCOL.read_text())
    for file, digest in p['references'].items():
        assert sha(Path(file)) == digest
    raw = json.loads((OLD / 'raw_report.json').read_text())
    assert raw['parts_sha256'] == p['historical_raw_parts_sha256']
    for file, digest in raw['parts_sha256'].items():
        assert sha(Path(file)) == digest
    for root, report, proof, key in [
        (OLD, 'full_label_report.json', 'full_label_verification.json', 'label_report_sha256'),
        (OLD, 'window_report.json', 'window_verification.json', 'window_report_sha256'),
        (EARLIER, 'feature_report.json', 'feature_verification.json', 'feature_report_sha256'),
        (modern_features.ROOT, 'feature_report.json', 'feature_verification.json', 'feature_report_sha256'),
        (modern_labels.ROOT, 'full_label_report.json', 'full_label_verification.json', 'label_report_sha256')]:
        v = json.loads((root / proof).read_text())
        assert v['passed'] and v[key] == sha(root / report)
    tax = json.loads((OLD.parent / 'tax_verification.json').read_text())
    assert tax['passed'] and tax['classification_source_sha256'] == sha(Path(__file__).with_name('tail_formula_long48_labels.py'))
    assert p['years']['training_historical_signal_first'] == '2023-01-01'
    assert p['years']['training_historical_signal_end'] == '2024-01-01' and not p['new_2026_prices_allowed']
    return p


def observations():
    p = checked(); assert not (HISTORY / 'observation_report.json').exists()
    HISTORY.mkdir(parents=True, exist_ok=True)
    keys = bounded(OLD / 'full_labels.parquet', 2023, KEYS)
    c = base.conn(); c.read_parquet(list(p['historical_raw_parts_sha256'])).create_view('raw')
    c.register('keys', keys)
    c.execute('''CREATE VIEW bars AS SELECT *,coalesce(
        timestamp=date_trunc('minute',timestamp) AND isfinite(open) AND isfinite(high) AND isfinite(low)
        AND isfinite(close) AND isfinite(volume) AND isfinite(amount) AND least(open,high,low,close)>0
        AND high+.0001>=greatest(open,close,low) AND low-.0001<=least(open,close)
        AND volume>=0 AND amount>=0 AND (volume=0)=(amount=0)
        AND (volume=0 OR amount/volume BETWEEN low-.0101 AND high+.0101)
        AND abs(open-round(open,2))<=.0001 AND abs(high-round(high,2))<=.0001
        AND abs(low-round(low,2))<=.0001 AND abs(close-round(close,2))<=.0001,false) AS valid
        FROM raw WHERE kind='morning' AND date>='2023-01-01' AND date<'2024-01-01'
        AND clock BETWEEN '09:31' AND '10:00' ''')
    c.execute('''CREATE VIEW rolling AS SELECT *,min(close) OVER three AS low_three,
        count(*) OVER three AS n_three,count(*) FILTER(WHERE valid AND volume>0) OVER three AS active_three,
        min(timestamp) OVER three AS first_three FROM bars
        WINDOW three AS(PARTITION BY date,code ORDER BY timestamp ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)''')
    frames = []
    for end, count, price in [('10:00', 30, 'price_1000'), ('09:59', 29, 'price_0959')]:
        f = c.sql(f'''WITH a AS(SELECT date,code,count(*) AS bars,count(DISTINCT clock) AS labels,
            count(*) FILTER(WHERE valid) AS valid_bars,count(*) FILTER(WHERE valid AND volume>0) AS active_minutes,
            max(close) FILTER(WHERE valid AND volume>0) AS max_close,
            max(low_three) FILTER(WHERE n_three=3 AND active_three=3 AND timestamp-first_three=INTERVAL 2 MINUTE) AS sustained_close,
            min(low) FILTER(WHERE valid AND volume>0) AS min_low,
            max(close) FILTER(WHERE clock='{end}' AND valid AND volume>0) AS {price}
            FROM rolling WHERE clock<='{end}' GROUP BY date,code)
            SELECT k.*,a.* EXCLUDE(date,code),coalesce(bars={count} AND labels={count} AND valid_bars={count},false) AS source_valid
            FROM keys k LEFT JOIN a USING(date,code) ORDER BY date,code''').df()
        frames.append(f)
    c.close()
    old = bounded(OLD / 'morning_windows.parquet', 2023)
    pd.testing.assert_frame_equal(frames[0], old, check_dtype=False, check_exact=True)
    assert frames[1].loc[old.source_valid, 'source_valid'].all()
    frames[1].to_parquet(HISTORY / 'observations.parquet', index=False, compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL), implementation_sha256=sha(Path(__file__)),
        raw_report_sha256=sha(OLD / 'raw_report.json'), old_window_verification_sha256=sha(OLD / 'window_verification.json'),
        observations_sha256=sha(HISTORY / 'observations.parquet'), rows=len(keys),
        old_30_bar_aggregates_exactly_restored=True, original_unknowns_not_recovered=True,
        first_signal=keys.date.min(), last_signal=keys.date.max(), last_observation=keys.next_date.max(),
        training_outcomes_only=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(HISTORY / 'observation_report.json', r); return r


def pandas_morning(r, end):
    """Independent vectorized rolling reconstruction, including missing minutes."""
    z = r.loc[r.clock.le(end)].sort_values(['date', 'code', 'timestamp']).reset_index(drop=True).copy()
    a = z[['open', 'high', 'low', 'close', 'volume', 'amount']].to_numpy(dtype=float)
    o, h, lo, cl, vol, amt = a.T
    price = amt / np.where(vol > 0, vol, np.nan)
    good = (np.isfinite(a).all(axis=1) & (a[:, :4].min(axis=1) > 0)
        & (h + .0001 >= np.maximum.reduce([o, lo, cl])) & (lo - .0001 <= np.minimum(o, cl))
        & (vol >= 0) & (amt >= 0) & ((vol == 0) == (amt == 0))
        & ((vol == 0) | ((price >= lo - .0101) & (price <= h + .0101)))
        & (np.abs(a[:, :4] - np.rint(a[:, :4] * 100) / 100) <= .0001).all(axis=1)
        & z.timestamp.eq(z.timestamp.dt.floor('min')))
    z['valid'] = good; z['active'] = good & (vol > 0)
    same = z.date.eq(z.date.shift(2)) & z.code.eq(z.code.shift(2))
    triple = same & z.active.astype(int).rolling(3, min_periods=3).sum().eq(3) & z.timestamp.sub(z.timestamp.shift(2)).eq(pd.Timedelta(minutes=2))
    z['three'] = z.close.rolling(3, min_periods=3).min().where(triple)
    z['active_close'] = z.close.where(z.active); z['active_low'] = z.low.where(z.active)
    z['reference'] = z.close.where(z.active & z.clock.eq(end))
    g = z.groupby(['date', 'code']).agg(bars=('timestamp', 'size'), labels=('clock', 'nunique'),
        valid_bars=('valid', 'sum'), active_minutes=('active', 'sum'), max_close=('active_close', 'max'),
        sustained_close=('three', 'max'), min_low=('active_low', 'min'), reference=('reference', 'max')).reset_index()
    return g.rename(columns={'reference': 'price_1000' if end == '10:00' else 'price_0959'})


def verify_observations():
    p = checked(); r = json.loads((HISTORY / 'observation_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['implementation_sha256'] == sha(Path(__file__))
    assert r['observations_sha256'] == sha(HISTORY / 'observations.parquet')
    parts = {'10:00': [], '09:59': []}; raw_count = 0
    for file in p['historical_raw_parts_sha256']:
        raw = pd.read_parquet(file, filters=[('date', '>=', '2023-01-01'), ('date', '<', '2024-01-01'), ('kind', '=', 'morning')])
        if raw.empty:
            continue
        raw_count += len(raw)
        assert not raw.duplicated(['date', 'code', 'timestamp']).any()
        assert raw.clock.between('09:31', '10:00').all()
        assert raw.clock.eq(raw.timestamp.dt.strftime('%H:%M')).all()
        assert raw.source_date.eq(raw.timestamp.dt.strftime('%Y-%m-%d')).all()
        assert raw.source_date.gt(raw.date).all() and raw.source_date.le('2024-01-02').all()
        for end in parts:
            parts[end].append(pandas_morning(raw, end))
    keys = bounded(OLD / 'full_labels.parquet', 2023, KEYS)
    for end, count, path in [('10:00', 30, OLD / 'morning_windows.parquet'), ('09:59', 29, HISTORY / 'observations.parquet')]:
        f = keys.merge(pd.concat(parts[end], ignore_index=True), on=['date', 'code'], how='left', validate='one_to_one')
        f['source_valid'] = f.bars.eq(count) & f.labels.eq(count) & f.valid_bars.eq(count)
        expected = bounded(path, 2023)
        pd.testing.assert_frame_equal(expected, f[expected.columns], check_dtype=False, check_exact=True)
    proof = dict(passed=True, observation_report_sha256=sha(HISTORY / 'observation_report.json'), rows=len(keys),
        cached_raw_morning_rows=raw_count, all_old30_and_new29_aggregates_independently_rebuilt=True,
        old_raw_source_and_fee_proofs_reused=True, training_outcomes_only=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(HISTORY / 'observation_verification.json', proof); return proof


def historical_net_mark(prices, shares, cash, bps, next_date):
    assert next_date.ge('2023-01-01').all() and next_date.le('2024-01-02').all()
    value = shares * (prices - np.maximum(prices * bps / 10000, .005))
    taxes = np.where(next_date.lt('2023-08-28'), .001, .0005) + .00001
    return (value - np.maximum(value * .0003, 5) - value * taxes) / cash - 1


def labels():
    checked(); assert not (HISTORY / 'full_label_report.json').exists()
    proof = json.loads((HISTORY / 'observation_verification.json').read_text())
    assert proof['passed'] and proof['observation_report_sha256'] == sha(HISTORY / 'observation_report.json')
    old = bounded(OLD / 'full_labels.parquet', 2023); obs = pd.read_parquet(HISTORY / 'observations.parquet')
    pd.testing.assert_frame_equal(old[KEYS], obs[KEYS], check_exact=True)
    f = old.drop(columns=['price_1000', 'mark_1000_return5', 'mark_1000_return15']).copy()
    for name in ['active_minutes', 'max_close', 'sustained_close', 'min_low', 'price_0959']:
        f[name] = obs[name]
    f['source_valid_0959'] = obs.source_valid
    changes = {}
    for bps in [5, 15]:
        known = old[f'known{bps}']; assert f.loc[known, 'source_valid_0959'].all()
        for name, price in [('sustained', f.sustained_close), ('any_close', f.max_close), ('mark_0959', f.price_0959), ('adverse', f.min_low)]:
            f[f'{name}_return{bps}'] = historical_net_mark(price, f.decision_shares, f[f'buy_cash{bps}'], bps, f.next_date).where(known)
        f[f'opportunity{bps}'] = f[f'sustained_return{bps}'].gt(0).astype(float).where(known)
        f[f'any_opportunity{bps}'] = f[f'any_close_return{bps}'].gt(0).astype(float).where(known)
        f[f'one_percent{bps}'] = f[f'sustained_return{bps}'].ge(.01).astype(float).where(known)
        for field in ['opportunity', 'one_percent']:
            assert (f.loc[known, f'{field}{bps}'] <= old.loc[known, f'{field}{bps}']).all()
        changes[str(bps)] = int((known & f[f'opportunity{bps}'].ne(old[f'opportunity{bps}'])).sum())
    f.to_parquet(HISTORY / 'full_labels.parquet', index=False, compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL), labels_sha256=sha(HISTORY / 'full_labels.parquet'),
        old_label_report_sha256=sha(OLD / 'full_label_report.json'), observation_verification_sha256=sha(HISTORY / 'observation_verification.json'),
        tax_verification_sha256=sha(OLD.parent / 'tax_verification.json'), rows=len(f), known15=int(f.known15.sum()),
        old_unknowns_and_all_entry_cash_unchanged=True, window_start='09:31', window_end='09:59',
        changed_training_binary_labels=changes, training_outcomes_only=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(HISTORY / 'full_label_report.json', r); return r


def verify_labels():
    checked(); r = json.loads((HISTORY / 'full_label_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['labels_sha256'] == sha(HISTORY / 'full_labels.parquet')
    assert r['old_label_report_sha256'] == sha(OLD / 'full_label_report.json')
    assert r['observation_verification_sha256'] == sha(HISTORY / 'observation_verification.json')
    old = bounded(OLD / 'full_labels.parquet', 2023)
    got = pd.read_parquet(HISTORY / 'full_labels.parquet'); obs = pd.read_parquet(HISTORY / 'observations.parquet')
    updated = ['active_minutes', 'max_close', 'sustained_close', 'min_low', 'price_1000']
    updated += [f'{stem}{bps}' for bps in [5, 15] for stem in [
        'sustained_return', 'any_close_return', 'mark_1000_return', 'adverse_return',
        'opportunity', 'any_opportunity', 'one_percent']]
    unchanged = [name for name in old if name not in updated]
    pd.testing.assert_frame_equal(got[unchanged], old[unchanged], check_exact=True)
    for name in ['active_minutes', 'max_close', 'sustained_close', 'min_low', 'price_0959']:
        pd.testing.assert_series_equal(got[name], obs[name], check_exact=True)
    pd.testing.assert_series_equal(got.source_valid_0959, obs.source_valid, check_names=False, check_exact=True)
    c = base.conn()
    c.register('source', got[['date', 'code', 'next_date', 'decision_shares', 'known5', 'known15',
                              'buy_cash5', 'buy_cash15', 'sustained_close', 'max_close', 'price_0959', 'min_low']])
    maximum = 0.; comparisons = 0
    for bps in [5, 15]:
        for name, price in [('sustained', 'sustained_close'), ('any_close', 'max_close'), ('mark_0959', 'price_0959'), ('adverse', 'min_low')]:
            values = c.sql(f'''WITH v AS(SELECT *,decision_shares*({price}-greatest({price}*{bps}/1e4,5e-3)) AS amount FROM source)
                SELECT CASE WHEN known{bps} THEN (amount-greatest(amount*3e-4,5e0)
                    -amount*(1e-5+CASE WHEN next_date<'2023-08-28' THEN 1e-3 ELSE 5e-4 END))/buy_cash{bps}-1e0 END AS value
                FROM v ORDER BY date,code''').df().value
            actual = got[f'{name}_return{bps}']
            np.testing.assert_allclose(actual, values, rtol=0, atol=2e-12, equal_nan=True)
            maximum = max(maximum, float((actual - values).abs().max())); comparisons += len(got)
            if name in ['sustained', 'any_close']:
                field = 'opportunity' if name == 'sustained' else 'any_opportunity'
                expected = values.gt(0).astype(float).where(got[f'known{bps}'])
                pd.testing.assert_series_equal(got[f'{field}{bps}'], expected, check_names=False, check_exact=True)
            if name == 'sustained':
                expected = values.ge(.01).astype(float).where(got[f'known{bps}'])
                pd.testing.assert_series_equal(got[f'one_percent{bps}'], expected, check_names=False, check_exact=True)
    c.close()
    v = dict(passed=True, label_report_sha256=sha(HISTORY / 'full_label_report.json'), rows=len(got),
        cash_mark_scalar_checks=comparisons, maximum_cash_mark_difference=maximum,
        every_original_nonobservation_field_exact=True, all_old_unknown_and_no_trade_states_retained=True,
        dated_fees_and_all_binary_labels_sql_rebuilt=True, training_outcomes_only=True,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(HISTORY / 'full_label_verification.json', v); return v


def assemble():
    checked(); assert not (ROOT / 'feature_report.json').exists() and not (ROOT / 'full_label_report.json').exists()
    proof = json.loads((HISTORY / 'full_label_verification.json').read_text())
    assert proof['passed'] and proof['label_report_sha256'] == sha(HISTORY / 'full_label_report.json')
    result = {}
    for kind, columns, sources, file, report, hash_key in [
        ('features', FEATURE_COLUMNS, [EARLIER, modern_features.ROOT], 'features.parquet', 'feature_report.json', 'features_sha256'),
        ('labels', LABEL_COLUMNS, [HISTORY, modern_labels.ROOT], 'full_labels.parquet', 'full_label_report.json', 'labels_sha256')]:
        frames = [bounded(folder / file, year, columns) for folder, year in zip(sources, [2023, 2024])]
        out = pd.concat(frames, ignore_index=True).sort_values(['date', 'code']).reset_index(drop=True)
        assert not out.duplicated(['date', 'code']).any() and out.date.between('2023-01-01', '2024-12-31').all()
        for year, frame in zip([2023, 2024], frames):
            subset = out.loc[out.date.str.startswith(str(year))].reset_index(drop=True)
            pd.testing.assert_frame_equal(subset, frame, check_dtype=False, check_exact=True)
        out.to_parquet(ROOT / file, index=False, compression='zstd')
        r = dict(protocol_sha256=sha(PROTOCOL), sources=[str(folder / file) for folder in sources],
            source_sha256={str(folder / file): sha(folder / file) for folder in sources}, columns=columns,
            rows=len(out), first=out.date.min(), last=out.date.max(), only_2023_and_2024_values=True,
            new_2026_prices_read=False, no_exit_rules=True)
        r[hash_key] = sha(ROOT / file)
        if kind == 'features':
            r.update(expressions=modern_features.EXPRESSIONS, native_header=modern_features.HEADER,
                     valid=int(out.formula_input_valid.sum()))
        else:
            r.update(training_projection_only=True, evaluation_uses_verified_original_2024_labels=True,
                     historical_label_verification_sha256=sha(HISTORY / 'full_label_verification.json'))
        save_json(ROOT / report, r); result[kind] = sha(ROOT / report)
    return result


def verify_assembly():
    import pyarrow.parquet as pq
    checked(); c = base.conn(); results = {}
    for file, report, proof, key, digest_key in [
        ('features.parquet', 'feature_report.json', 'feature_verification.json', 'feature_report_sha256', 'features_sha256'),
        ('full_labels.parquet', 'full_label_report.json', 'full_label_verification.json', 'label_report_sha256', 'labels_sha256')]:
        r = json.loads((ROOT / report).read_text())
        assert r['protocol_sha256'] == sha(PROTOCOL) and r[digest_key] == sha(ROOT / file)
        c.read_parquet(str(ROOT / file)).create_view('assembled', replace=True)
        for year, path in zip([2023, 2024], r['sources']):
            assert r['source_sha256'][path] == sha(Path(path))
            c.register('projected', pq.read_table(path, columns=r['columns'],
                filters=[('date', '>=', f'{year}-01-01'), ('date', '<', f'{year+1}-01-01')]))
            left = f"SELECT * FROM assembled WHERE date>='{year}-01-01' AND date<'{year+1}-01-01'"
            right = 'SELECT * FROM projected'
            for a, b in [(left, right), (right, left)]:
                assert c.sql(f'SELECT count(*) FROM (({a}) EXCEPT ALL ({b}))').fetchone()[0] == 0
            c.unregister('projected')
        counts = c.sql('SELECT count(*),count(DISTINCT (date,code)),min(date),max(date) FROM assembled').fetchone()
        assert counts == (r['rows'], r['rows'], r['first'], r['last'])
        v = dict(passed=True, **{key: sha(ROOT / report)}, rows=r['rows'],
            all_2023_and_2024_source_projected_values_independently_unchanged=True,
            all_keys_unique=True, source_2025_rows_excluded=True, new_2026_prices_read=False, no_exit_rules=True)
        save_json(ROOT / proof, v); results[proof] = sha(ROOT / proof)
    c.close(); return results


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['observations', 'verify_observations', 'labels', 'verify_labels', 'assemble', 'verify_assembly'])
    print(json.dumps(globals()[p.parse_args().stage](), ensure_ascii=False, indent=2))
