"""Distance to the close of the largest prior raw close-ratio day."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_daily_efficiency as history
from . import tail_formula_daily_extremes as extremes
from . import tail_formula_float as previous
from . import tail_formula_path_variance as minute_source
from .corporate_cash import DAILY, MINUTES, save_json, sha

STEM = 'tail_formula_event_anchor'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
HISTORY_COLUMNS = extremes.HISTORY_COLUMNS


def extra_header():
    lines = [f'PAC{i:02d}:=INTPART(100*DCP{i}+0.5);' for i in range(1, 22)]
    lines += ['PAN20:=PAC20;', 'PAD20:=PAC21;']
    for i in range(19, 0, -1):
        condition = f'PAC{i:02d}*PAD{i+1:02d}>=PAN{i+1:02d}*PAC{i+1:02d}'
        lines += [f'PAN{i:02d}:=IF({condition},PAC{i:02d},PAN{i+1:02d});',
                  f'PAD{i:02d}:=IF({condition},PAC{i+1:02d},PAD{i+1:02d});']
    return '\n'.join(lines) + '\n'


EXTRA_HEADER = extra_header()
NEW_EXPRESSIONS = {'PA01': '100*(INTPART(100*Q+0.5)/PAN01-1)/V01'}
EXPRESSIONS = {**previous.EXPRESSIONS, **NEW_EXPRESSIONS}
HEADER = previous.HEADER + EXTRA_HEADER


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    assert p['history_returns'] == 20 and p['history_prices'] == 21 and p['tie_policy'] == 'most_recent'
    assert p['expressions'] == NEW_EXPRESSIONS and p['native_header'] == EXTRA_HEADER
    assert p['maximum_verified_price_cents'] == 1000000 and not p['new_2026_prices_allowed']
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    for root in [previous.ROOT, history.ROOT]:
        r = json.loads((root / 'feature_report.json').read_text())
        v = json.loads((root / 'feature_verification.json').read_text())
        assert v['passed'] and v['feature_report_sha256'] == sha(root / 'feature_report.json')
        assert r['features_sha256'] == sha(root / 'features.parquet')
    return p


def anchor(prices, current, atr):
    prices = np.asarray(prices, float); current = np.asarray(current, float); atr = np.asarray(atr, float)
    assert prices.ndim == 2 and prices.shape[1] == 21 and current.shape == atr.shape == (len(prices),)
    valid = (np.isfinite(prices).all(axis=1) & (prices > 0).all(axis=1)
             & np.isfinite(current) & (current > 0) & np.isfinite(atr) & (atr > 0))
    cents = np.floor(np.where(valid[:, None], prices, 0)*100+.5).astype('int64')
    quote = np.floor(np.where(valid, current, 0)*100+.5).astype('int64')
    assert np.all(cents[valid] <= 1000000) and np.all(quote[valid] <= 1000000)
    numerator = cents[:, 19].copy(); denominator = cents[:, 20].copy()
    lag = np.full(len(prices), 20, dtype='int64')
    for i in range(18, -1, -1):
        replace = cents[:, i]*denominator >= numerator*cents[:, i+1]
        numerator = np.where(replace, cents[:, i], numerator)
        denominator = np.where(replace, cents[:, i+1], denominator)
        lag = np.where(replace, i+1, lag)
    value = 100*(quote / np.maximum(numerator, 1)-1) / np.where(valid, atr, 1)
    return np.where(valid, value, np.nan), np.where(valid, numerator, 0), np.where(valid, lag, 0), valid


def features():
    checked_sources(); assert not (ROOT / 'feature_report.json').exists()
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    raw = pd.read_parquet(history.ROOT / 'features.parquet', columns=HISTORY_COLUMNS)
    f = old.merge(raw, on=['date', 'code'], how='left', validate='one_to_one').sort_values(['date', 'code']).reset_index(drop=True)
    value, price, lag, usable = anchor(f[history.PRICE_COLUMNS].to_numpy(), f.price_1449.to_numpy(), f.V01.to_numpy())
    f['event_anchor_valid'] = f.daily_efficiency_valid.eq(True) & usable
    f['PA01'] = pd.Series(value).where(f.event_anchor_valid)
    f['anchor_price_cents'] = price; f['anchor_stock_days_ago'] = lag
    f['prior_formula_input_valid'] = f.formula_input_valid
    f['formula_input_valid'] &= f.event_anchor_valid
    ROOT.mkdir(parents=True, exist_ok=True); f.to_parquet(ROOT / 'features.parquet', index=False, compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL), implementation_sha256=sha(Path(__file__)),
        features_sha256=sha(ROOT / 'features.parquet'), rows=len(f), valid=int(f.formula_input_valid.sum()),
        newly_invalid=int((old.formula_input_valid & ~f.formula_input_valid).sum()),
        valid_with_reference_breaks=int((f.formula_input_valid & f.de_reference_breaks.gt(0)).sum()),
        valid_with_stock_day_gaps=int((f.formula_input_valid & (f.de_market_span.gt(21) | f.de_last_gap.gt(1))).sum()),
        expressions=EXPRESSIONS, native_header=HEADER, raw_unadjusted_price_path=True,
        maximum_cross_product_bound=1000000**2, anchor_is_not_investor_cost_or_official_limitup=True,
        software_compilation_verified=False, native_source_parity_verified=False,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_report.json', r)
    return {k: v for k, v in r.items() if k not in ['expressions', 'native_header']}


def verify_features():
    p = checked_sources(); r = json.loads((ROOT / 'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['implementation_sha256'] == sha(Path(__file__))
    assert r['features_sha256'] == sha(ROOT / 'features.parquet')
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    raw = pd.read_parquet(history.ROOT / 'features.parquet', columns=HISTORY_COLUMNS)
    got = pd.read_parquet(ROOT / 'features.parquet')
    c = base.conn(); c.register('raw', raw); c.register('old', old[['date', 'code', 'price_1449', 'V01']])
    current = 'floor(100*price_1449+.5)::BIGINT'
    cents = [f'floor(100*de_c{i:02d}+.5)::BIGINT' for i in range(1, 22)]
    good = ' AND '.join(f'isfinite({n}) AND {n}>0' for n in [*history.PRICE_COLUMNS, 'price_1449', 'V01'])
    c.sql(f'SELECT date,code,coalesce(daily_efficiency_valid AND {good},false) AS valid,{current} AS quote,V01 '
          'FROM raw JOIN old USING(date,code)').create_view('eligible')
    c.sql('SELECT date,code,unnest([' + ','.join(cents[:-1]) + ']) AS price,'
          'unnest([' + ','.join(cents[1:]) + ']) AS previous,unnest(range(1,21)) AS lag FROM raw').create_view('atoms')
    # Independent long-form ratio ordering, then a complete integer rational
    # dominance check ensures the chosen day never wins due to division error.
    c.sql('SELECT * FROM atoms WHERE price>0 AND previous>0 QUALIFY row_number() OVER('
          'PARTITION BY date,code ORDER BY price::DOUBLE/previous DESC,lag)=1').create_view('chosen')
    expected = c.sql('SELECT e.date,e.code,e.valid,p.price AS anchor_price_cents,p.lag AS anchor_stock_days_ago,'
        'CASE WHEN e.valid THEN 100*(e.quote::DOUBLE/p.price-1)/e.V01 END AS PA01 '
        'FROM eligible e LEFT JOIN chosen p USING(date,code) ORDER BY date,code').df()
    assert c.sql('SELECT count(*) FROM chosen p JOIN atoms a USING(date,code) JOIN eligible e USING(date,code) '
        'WHERE e.valid AND (a.price*p.previous>p.price*a.previous OR '
        '(a.price*p.previous=p.price*a.previous AND a.lag<p.lag))').fetchone()[0] == 0
    c.close()
    pd.testing.assert_frame_equal(got[old.columns.drop('formula_input_valid')], old.drop(columns='formula_input_valid'), check_exact=True)
    pd.testing.assert_frame_equal(got[raw.columns], raw, check_exact=True)
    pd.testing.assert_series_equal(got.prior_formula_input_valid, old.formula_input_valid, check_names=False, check_exact=True)
    np.testing.assert_array_equal(got.event_anchor_valid, expected.valid)
    for name in ['anchor_price_cents', 'anchor_stock_days_ago']:
        np.testing.assert_array_equal(got.loc[expected.valid, name], expected.loc[expected.valid, name])
    np.testing.assert_allclose(got.PA01, expected.PA01, rtol=0, atol=2e-10, equal_nan=True)
    valid = old.formula_input_valid & expected.valid
    np.testing.assert_array_equal(got.formula_input_valid, valid)
    encode = lambda a: np.floor(np.clip(100*a+10000+.000001, 0, 999999)).astype('int32')
    np.testing.assert_array_equal(encode(got.loc[valid, 'PA01']), encode(expected.loc[valid, 'PA01']))
    assert r['rows'] == len(got) == p['expected_keys'] and r['valid'] == int(valid.sum()) == p['expected_valid']
    assert r['newly_invalid'] == int((old.formula_input_valid & ~valid).sum()) == 0
    for key, mask in [('valid_with_reference_breaks', raw.de_reference_breaks.gt(0)),
                      ('valid_with_stock_day_gaps', raw.de_market_span.gt(21) | raw.de_last_gap.gt(1))]:
        assert r[key] == int((valid & mask).sum())
    names = re.findall(r'(?m)^([A-Z][A-Z0-9]*):=', HEADER) + list(EXPRESSIONS)
    assert len(names) == len({n.casefold() for n in names})
    assert r['expressions'] == EXPRESSIONS and r['native_header'] == HEADER
    proof = dict(passed=True, feature_report_sha256=sha(ROOT / 'feature_report.json'), rows=len(got),
        effective_input_intersection_unchanged=bool(valid.equals(old.formula_input_valid)),
        independent_long_form_ordering_and_integer_cross_product_dominance_checked=True,
        all_anchor_prices_nearest_ties_distances_and_encodings_rebuilt=True,
        all_old_values_keys_and_validity_unchanged=True, history_gaps_and_reference_breaks_preserved=True,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_verification.json', proof); return proof


def native_value(prices, current, atr):
    assert len(prices) == 21
    env = {f'DCP{i}': float(price) for i, price in enumerate(prices, 1)}
    env.update(Q=float(current), V01=float(atr), INTPART=np.floor, IF=lambda yes, a, b: a if yes else b)
    for statement in EXTRA_HEADER.strip().split(';'):
        if statement:
            name, expr = statement.strip().split(':='); env[name] = eval(expr, {'__builtins__': {}}, env)
    return float(eval(NEW_EXPRESSIONS['PA01'], {'__builtins__': {}}, env)), int(env['PAN01'])


def native():
    checked_sources(); v = json.loads((ROOT / 'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(ROOT / 'feature_report.json')
    source = history.ROOT / 'native_input_verification.json'; samples = json.loads(source.read_text())
    assert samples['passed'] and samples['feature_report_sha256'] == sha(history.ROOT / 'feature_report.json')
    assert samples['feature_verification_sha256'] == sha(history.ROOT / 'feature_verification.json')
    daily_hashes = json.loads(history.DAILY_REPORT.read_text())['source_sha256']
    minute_hashes = json.loads((minute_source.ROOT / 'window_report.json').read_text())['source_sha256']
    f = pd.read_parquet(ROOT / 'features.parquet').set_index(['date', 'code']); receipts = []
    for sample in samples['samples']:
        date, code = sample['date'], sample['code']; assert '2024-01-01' <= date <= '2025-12-30'
        path = DAILY / (code.replace('.', '_') + '.parquet'); assert sha(path) == daily_hashes[str(path)]
        d = pd.read_parquet(path, columns=['date', 'close', 'tradestatus', 'adjustflag'],
                            filters=[('date', '>=', '2023-06-01'), ('date', '<', date)])
        days = d.loc[d.tradestatus.eq(1)].sort_values('date').tail(21)
        assert len(days) == 21 and days.adjustflag.eq(3).all() and days.date.max() < date
        assert days.date.iloc[0] == sample['first_history_date']
        values = days.close.iloc[::-1].to_numpy(float).round(2); row = f.loc[(date, code)]
        path = MINUTES / code[:2].upper() / (code[3:] + '.parquet')
        assert sha(path) == minute_hashes[str(path)] == sample['source_sha256']
        current = pd.read_parquet(path, columns=['timestamp', 'close'], filters=[('timestamp', '==', pd.Timestamp(date+' 14:49'))])
        assert len(current) == 1 and row.formula_input_valid
        quote = round(float(current.close.iloc[0]), 2)
        np.testing.assert_allclose(quote, row.price_1449, rtol=0, atol=.0001)
        assert np.floor(quote*100+.5) == np.floor(row.price_1449*100+.5)
        np.testing.assert_array_equal(values, row[history.PRICE_COLUMNS].to_numpy(float))
        value, price = native_value(values, quote, row.V01)
        np.testing.assert_allclose(value, row.PA01, rtol=0, atol=2e-10)
        assert price == row.anchor_price_cents
        assert native_value(values, quote*2, row.V01)[1] == price
        encode = lambda a: np.floor(np.clip(100*a+10000+.000001, 0, 999999))
        assert encode(value) == encode(row.PA01)
        receipts.append(dict(date=date, code=code, prior_stock_days=21, quote_label='14:49',
                             first_history_date=days.date.min(), last_history_date=days.date.max()))
    assert len(receipts) == 32
    proof = dict(passed=True, protocol_sha256=sha(PROTOCOL), feature_report_sha256=sha(ROOT / 'feature_report.json'),
        feature_verification_sha256=sha(ROOT / 'feature_verification.json'), fixed_sample_source_sha256=sha(source),
        samples=receipts, raw_daily_rows=672, raw_1449_quotes=32,
        generated_integer_tournament_and_scalar_distance_rebuilt=True, current_price_cannot_change_history_anchor=True,
        prior_daily_close_to_final_minute_parity_reused=True, software_compilation_verified=False, native_source_parity_verified=False,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'native_input_verification.json', proof)
    return {k: v for k, v in proof.items() if k != 'samples'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['features', 'verify_features', 'native'])
    print(json.dumps(globals()[p.parse_args().stage](), ensure_ascii=False, indent=2))
