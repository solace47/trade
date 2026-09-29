"""A fixed history-only routing state: prior close above its 20-day mean."""
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
from .corporate_cash import DAILY, save_json, sha

STEM = 'tail_formula_prior_trend'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
PRICE_COLUMNS = history.PRICE_COLUMNS[:20]
EXTRA_HEADER = ''.join(f'PTC{i:02d}:=INTPART(100*DCP{i}+0.5);\n' for i in range(1, 21))
EXTRA_HEADER += 'PTS20:=' + '+'.join(f'PTC{i:02d}' for i in range(1, 21)) + ';\n'
NEW_EXPRESSIONS = {'PT01': 'IF(20*PTC01>PTS20,1,0)'}
EXPRESSIONS = {**previous.EXPRESSIONS, **NEW_EXPRESSIONS}
HEADER = previous.HEADER + EXTRA_HEADER


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    assert p['prior_stock_days'] == 20 and p['tie_state'] == 'lower'
    assert p['expressions'] == NEW_EXPRESSIONS and p['native_header'] == EXTRA_HEADER
    assert not p['new_2026_prices_allowed']
    for file, digest in p['references'].items():
        assert sha(Path(file)) == digest
    for root in [previous.ROOT, history.ROOT]:
        r = json.loads((root / 'feature_report.json').read_text())
        v = json.loads((root / 'feature_verification.json').read_text())
        assert v['passed'] and v['feature_report_sha256'] == sha(root / 'feature_report.json')
        assert r['features_sha256'] == sha(root / 'features.parquet')
    return p


def prior_state(prices):
    values = np.asarray(prices, float)
    assert values.ndim == 2 and values.shape[1] == 20
    valid = np.isfinite(values).all(axis=1) & (values > 0).all(axis=1)
    cents = np.floor(100*np.where(valid[:, None], values, 0)+.5).astype('int64')
    assert (cents[valid] <= 1000000).all()
    difference = 20*cents[:, 0] - cents.sum(axis=1)
    return np.where(valid, (difference > 0).astype(float), np.nan), difference, valid


def features():
    checked_sources(); assert not (ROOT / 'feature_report.json').exists()
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    raw = pd.read_parquet(history.ROOT / 'features.parquet', columns=extremes.HISTORY_COLUMNS)
    f = old.merge(raw, on=['date', 'code'], how='left', validate='one_to_one').sort_values(['date', 'code']).reset_index(drop=True)
    state, difference, usable = prior_state(f[PRICE_COLUMNS].to_numpy())
    valid = f.daily_efficiency_valid.eq(True) & usable
    f['PT01'] = pd.Series(state).where(valid); f['prior_trend_valid'] = valid
    f['prior_trend_integer_difference'] = difference
    f['prior_formula_input_valid'] = f.formula_input_valid
    f['formula_input_valid'] &= valid
    ROOT.mkdir(parents=True, exist_ok=True)
    f.to_parquet(ROOT / 'features.parquet', index=False, compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL), implementation_sha256=sha(Path(__file__)),
        features_sha256=sha(ROOT / 'features.parquet'), rows=len(f), valid=int(f.formula_input_valid.sum()),
        newly_invalid=int((old.formula_input_valid & ~f.formula_input_valid).sum()),
        upper_valid=int((f.formula_input_valid & f.PT01.eq(1)).sum()),
        lower_valid=int((f.formula_input_valid & f.PT01.eq(0)).sum()),
        exact_ties_valid=int((f.formula_input_valid & f.prior_trend_integer_difference.eq(0)).sum()),
        valid_with_reference_breaks=int((f.formula_input_valid & f.de_reference_breaks.gt(0)).sum()),
        valid_with_stock_day_gaps=int((f.formula_input_valid & (f.de_market_span.gt(21) | f.de_last_gap.gt(1))).sum()),
        expressions=EXPRESSIONS, native_header=HEADER, unadjusted_prior_stock_days=True,
        software_compilation_verified=False, native_source_parity_verified=False,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_report.json', r)
    return {k: v for k, v in r.items() if k not in ['expressions', 'native_header']}


def verify_features():
    p = checked_sources(); r = json.loads((ROOT / 'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['implementation_sha256'] == sha(Path(__file__))
    assert r['features_sha256'] == sha(ROOT / 'features.parquet')
    old = pd.read_parquet(previous.ROOT / 'features.parquet'); got = pd.read_parquet(ROOT / 'features.parquet')
    raw = pd.read_parquet(history.ROOT / 'features.parquet', columns=extremes.HISTORY_COLUMNS)
    c = base.conn(); c.register('raw', raw)
    good = ' AND '.join(f'isfinite({n}) AND {n}>0' for n in PRICE_COLUMNS)
    terms = [f'floor(100*{n}+.5)::BIGINT' for n in PRICE_COLUMNS]
    difference = '20*(' + terms[0] + ')-(' + '+'.join(terms) + ')'
    expected = c.sql('SELECT date,code,coalesce(daily_efficiency_valid AND ' + good + ',false) AS valid,'
        + difference + ' AS delta,CASE WHEN valid THEN CAST(delta>0 AS DOUBLE) END AS PT01 '
        'FROM raw ORDER BY date,code').df(); c.close()
    pd.testing.assert_frame_equal(got[old.columns.drop('formula_input_valid')], old.drop(columns='formula_input_valid'), check_exact=True)
    pd.testing.assert_frame_equal(got[raw.columns], raw, check_exact=True)
    np.testing.assert_array_equal(got.prior_formula_input_valid, old.formula_input_valid)
    np.testing.assert_array_equal(got.prior_trend_valid, expected.valid)
    np.testing.assert_array_equal(got.loc[expected.valid, 'prior_trend_integer_difference'], expected.loc[expected.valid, 'delta'])
    np.testing.assert_allclose(got.PT01, expected.PT01, rtol=0, atol=0, equal_nan=True)
    valid = old.formula_input_valid & expected.valid
    np.testing.assert_array_equal(got.formula_input_valid, valid)
    assert got.loc[valid, 'PT01'].isin([0, 1]).all()
    encode = lambda x: np.floor(np.clip(100*x+10000+.000001, 0, 999999)).astype('int32')
    np.testing.assert_array_equal(encode(got.loc[valid, 'PT01']), encode(expected.loc[valid, 'PT01']))
    assert r['rows'] == len(got) == p['expected_keys'] and r['valid'] == int(valid.sum()) == p['expected_valid']
    assert r['newly_invalid'] == int((old.formula_input_valid & ~valid).sum()) == 0
    for key, mask in [('upper_valid', expected.PT01.eq(1)), ('lower_valid', expected.PT01.eq(0)),
                      ('exact_ties_valid', expected.delta.eq(0)), ('valid_with_reference_breaks', raw.de_reference_breaks.gt(0)),
                      ('valid_with_stock_day_gaps', raw.de_market_span.gt(21) | raw.de_last_gap.gt(1))]:
        assert r[key] == int((valid & mask).sum())
    names = re.findall(r'(?m)^([A-Z][A-Z0-9]*):=', HEADER) + list(EXPRESSIONS)
    assert len(names) == len({n.casefold() for n in names}) and r['expressions'] == EXPRESSIONS and r['native_header'] == HEADER
    proof = dict(passed=True, feature_report_sha256=sha(ROOT / 'feature_report.json'), rows=len(got),
        effective_input_intersection_unchanged=bool(valid.equals(old.formula_input_valid)),
        all_twenty_integer_prices_history_states_ties_and_encodings_independently_rebuilt=True,
        all_old_keys_values_validity_reference_breaks_and_gaps_preserved=True,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_verification.json', proof); return proof


def native_value(prices, current=1000000.):
    assert len(prices) == 21
    env = {f'DCP{i}': float(value) for i, value in enumerate(prices, 1)}
    env.update(Q=float(current), INTPART=np.floor, IF=lambda yes, a, b: a if yes else b)
    for statement in EXTRA_HEADER.strip().split(';'):
        if statement:
            name, expr = statement.strip().split(':='); env[name] = eval(expr, {'__builtins__': {}}, env)
    return float(eval(NEW_EXPRESSIONS['PT01'], {'__builtins__': {}}, env))


def native():
    checked_sources(); v = json.loads((ROOT / 'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(ROOT / 'feature_report.json')
    source = history.ROOT / 'native_input_verification.json'; samples = json.loads(source.read_text())
    assert samples['passed'] and samples['feature_report_sha256'] == sha(history.ROOT / 'feature_report.json')
    assert samples['feature_verification_sha256'] == sha(history.ROOT / 'feature_verification.json')
    hashes = json.loads(history.DAILY_REPORT.read_text())['source_sha256']
    f = pd.read_parquet(ROOT / 'features.parquet').set_index(['date', 'code']); receipts = []
    for sample in samples['samples']:
        date, code = sample['date'], sample['code']; assert '2024-01-01' <= date <= '2025-12-30'
        path = DAILY / (code.replace('.', '_') + '.parquet'); assert sha(path) == hashes[str(path)]
        daily = pd.read_parquet(path, columns=['date', 'close', 'tradestatus', 'adjustflag'],
            filters=[('date', '>=', '2023-06-01'), ('date', '<', date)])
        days = daily.loc[daily.tradestatus.eq(1)].sort_values('date').tail(21)
        assert len(days) == 21 and days.adjustflag.eq(3).all() and days.date.max() < date
        assert days.date.iloc[0] == sample['first_history_date']
        values = days.close.iloc[::-1].to_numpy(float).round(2); row = f.loc[(date, code)]
        assert row.formula_input_valid
        np.testing.assert_array_equal(values, row[history.PRICE_COLUMNS].to_numpy(float))
        value = native_value(values); outside = values.copy(); outside[-1] *= 100
        assert value == row.PT01 == native_value(values, .01) == native_value(outside)
        receipts.append(dict(date=date, code=code, prior_stock_days_used=20, prior_stock_days_checked=21,
                             first_history_date=days.date.min(), last_history_date=days.date.max()))
    assert len(receipts) == 32
    proof = dict(passed=True, protocol_sha256=sha(PROTOCOL), feature_report_sha256=sha(ROOT / 'feature_report.json'),
        feature_verification_sha256=sha(ROOT / 'feature_verification.json'), fixed_sample_source_sha256=sha(source),
        samples=receipts, raw_daily_rows=672, used_input_daily_rows=640,
        current_quote_and_21st_prior_close_do_not_change_state=True,
        prior_daily_close_to_final_minute_parity_reused=True, software_compilation_verified=False, native_source_parity_verified=False,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'native_input_verification.json', proof)
    return {k: v for k, v in proof.items() if k != 'samples'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['features', 'verify_features', 'native'])
    print(json.dumps(globals()[p.parse_args().stage](), ensure_ascii=False, indent=2))
