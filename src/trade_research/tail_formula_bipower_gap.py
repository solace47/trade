"""One bounded adjacent absolute-return variation input, with fixed 29 bars."""
import argparse
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as previous
from . import tail_formula_path_variance as prices
from .corporate_cash import MINUTES, save_json, sha

STEM = 'tail_formula_bipower_gap'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
COEFFICIENT = math.pi / 2 * 29 / 28
EXTRA_HEADER = 'BPR:=100*LN(C/REF(C,1));\nBPQ:=SUM(BPR*BPR,29);\nBPA:=SUM(ABS(BPR)*ABS(REF(BPR,1)),28);\n'
NEW_EXPRESSIONS = {'BP01': f'VALUEWHEN(TIME=1449,IF(BPQ>0,100*MAX(1-{COEFFICIENT:.17e}*BPA/MAX(BPQ,0.000000000001),0),0))'}
EXPRESSIONS = {**previous.EXPRESSIONS, **NEW_EXPRESSIONS}
HEADER = previous.HEADER + EXTRA_HEADER


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    assert p['return_bars'] == 29 and p['adjacent_products'] == 28 and not p['new_2026_prices_allowed']
    assert p['expressions'] == NEW_EXPRESSIONS and p['native_header'] == EXTRA_HEADER
    assert p['finite_sample_multiplier'] == 29 / 28 and p['normal_absolute_moment_inverse_square'] == math.pi / 2
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    for root in [previous.ROOT, prices.ROOT]:
        r = json.loads((root / 'feature_report.json').read_text())
        v = json.loads((root / 'feature_verification.json').read_text())
        assert v['passed'] and v['feature_report_sha256'] == sha(root / 'feature_report.json')
        assert r['features_sha256'] == sha(root / 'features.parquet')
    return p


def bipower_gap(returns):
    assert returns.ndim == 2 and returns.shape[1] == 29
    total = np.sum(returns * returns, axis=1)
    adjacent = np.sum(np.abs(returns[:, 1:]) * np.abs(returns[:, :-1]), axis=1)
    # The native denominator is guarded, but valid non-flat observations must
    # never activate that floor. Keep exactly zero windows at their defined 0.
    ratio = COEFFICIENT * adjacent / np.maximum(total, 1e-12)
    return np.where(total > 0, 100 * np.maximum(1 - ratio, 0), 0), total, adjacent


def features():
    checked_sources(); assert not (ROOT / 'feature_report.json').exists()
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    raw = pd.read_parquet(prices.ROOT / 'features.parquet', columns=['date', 'code', 'path_variance_valid', *prices.PRICE_COLUMNS])
    f = old.merge(raw, on=['date', 'code'], how='left', validate='one_to_one').sort_values(['date', 'code']).reset_index(drop=True)
    values = f[prices.PRICE_COLUMNS].to_numpy(float)
    valid = f.path_variance_valid.eq(True) & np.isfinite(values).all(axis=1) & (values > 0).all(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        returns = 100 * np.log(values[:, 1:] / values[:, :-1])
    value, total, adjacent = bipower_gap(returns)
    f['bp_square_sum'] = total; f['bp_adjacent_sum'] = adjacent
    f['BP01'] = pd.Series(value).where(valid)
    f['bipower_input_valid'] = valid & np.isfinite(f.BP01)
    f['prior_formula_input_valid'] = f.formula_input_valid
    f['formula_input_valid'] &= f.bipower_input_valid
    assert not (f.formula_input_valid & f.bp_square_sum.gt(0) & f.bp_square_sum.lt(1e-12)).any()
    ROOT.mkdir(parents=True, exist_ok=True)
    f.to_parquet(ROOT / 'features.parquet', index=False, compression='zstd')
    report = dict(protocol_sha256=sha(PROTOCOL), implementation_sha256=sha(Path(__file__)),
        features_sha256=sha(ROOT / 'features.parquet'), rows=len(f), valid=int(f.formula_input_valid.sum()),
        newly_invalid=int((old.formula_input_valid & ~f.formula_input_valid).sum()),
        flat_valid=int((f.formula_input_valid & f.bp_square_sum.eq(0)).sum()),
        positive_gap_valid=int((f.formula_input_valid & f.BP01.gt(0)).sum()),
        nonzero_native_floor_active_valid=0, expressions=EXPRESSIONS, native_header=HEADER,
        software_compilation_verified=False, native_source_parity_verified=False,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_report.json', report)
    return {k: v for k, v in report.items() if k not in ['expressions', 'native_header']}


def verify_features():
    p = checked_sources(); r = json.loads((ROOT / 'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['implementation_sha256'] == sha(Path(__file__))
    assert r['features_sha256'] == sha(ROOT / 'features.parquet')
    old = pd.read_parquet(previous.ROOT / 'features.parquet'); got = pd.read_parquet(ROOT / 'features.parquet')
    raw = pd.read_parquet(prices.ROOT / 'features.parquet', columns=['date', 'code', 'path_variance_valid', *prices.PRICE_COLUMNS])
    c = base.conn(); c.register('raw', raw)
    changes = [f'(CASE WHEN pv_c{i:02d}>0 AND pv_c{i-1:02d}>0 THEN 100*ln(pv_c{i:02d}/pv_c{i-1:02d}) END) AS R{i}' for i in range(21, 50)]
    c.sql('SELECT date,code,' + ','.join(changes) + ' FROM raw').create_view('returns')
    square = '+'.join(f'R{i}*R{i}' for i in range(21, 50))
    adjacent = '+'.join(f'abs(R{i}*R{i-1})' for i in range(22, 50))
    c.sql(f'SELECT date,code,{square} AS bp_square_sum,{adjacent} AS bp_adjacent_sum FROM returns').create_view('stats')
    good = ' AND '.join(f'isfinite({n}) AND {n}>0' for n in prices.PRICE_COLUMNS)
    expected = c.sql(f'''SELECT date,code,coalesce(path_variance_valid AND {good},false) AS valid,
        bp_square_sum,bp_adjacent_sum,CASE WHEN valid THEN CASE WHEN bp_square_sum=0 THEN 0 ELSE
        100*greatest(1-{COEFFICIENT:.17e}*bp_adjacent_sum/bp_square_sum,0) END END AS BP01
        FROM raw JOIN stats USING(date,code) ORDER BY date,code''').df(); c.close()
    pd.testing.assert_frame_equal(got[old.columns.drop('formula_input_valid')], old.drop(columns='formula_input_valid'), check_exact=True)
    pd.testing.assert_frame_equal(got[raw.columns], raw, check_exact=True)
    pd.testing.assert_series_equal(got.prior_formula_input_valid, old.formula_input_valid, check_names=False, check_exact=True)
    np.testing.assert_array_equal(got.bipower_input_valid, expected.valid)
    for name in ['BP01', 'bp_square_sum', 'bp_adjacent_sum']:
        np.testing.assert_allclose(got.loc[expected.valid, name], expected.loc[expected.valid, name], rtol=0, atol=2e-10)
    assert got.loc[~expected.valid, 'BP01'].isna().all()
    valid = old.formula_input_valid & expected.valid
    np.testing.assert_array_equal(got.formula_input_valid, valid)
    assert got.loc[valid, 'BP01'].between(0, 100).all()
    encode = lambda x: np.floor(np.clip(100*x+10000+.000001, 0, 999999)).astype('int32')
    np.testing.assert_array_equal(encode(got.loc[valid, 'BP01']), encode(expected.loc[valid, 'BP01']))
    assert not (valid & expected.bp_square_sum.gt(0) & expected.bp_square_sum.lt(1e-12)).any()
    assert r['flat_valid'] == int((valid & expected.bp_square_sum.eq(0)).sum())
    assert r['positive_gap_valid'] == int((valid & expected.BP01.gt(0)).sum())
    assert r['rows'] == len(got) == p['expected_keys'] and r['valid'] == int(valid.sum()) == p['expected_valid']
    assert r['newly_invalid'] == int((old.formula_input_valid & ~valid).sum()) == 0
    names = re.findall(r'(?m)^([A-Z][A-Z0-9]*):=', HEADER) + list(EXPRESSIONS)
    assert len(names) == len({n.casefold() for n in names}) and r['expressions'] == EXPRESSIONS and r['native_header'] == HEADER
    proof = dict(passed=True, feature_report_sha256=sha(ROOT / 'feature_report.json'), rows=len(got),
        effective_input_intersection_unchanged=bool(valid.equals(old.formula_input_valid)),
        all_29_log_returns_28_adjacent_products_squares_gaps_and_encodings_independently_rebuilt=True,
        all_old_values_keys_and_validity_unchanged=True, flat_cases_and_denominator_floor_checked=True,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_verification.json', proof); return proof


def native_value(values, outside=1000000.):
    env = dict(C=np.r_[outside, outside, values, outside, outside], TIME=np.r_[1418, 1419, np.arange(1420, 1450), 1450, 1451])
    env.update(REF=lambda x, n: pd.Series(x).shift(int(n)).to_numpy(), LN=np.log, MAX=np.maximum, IF=np.where, ABS=np.abs,
        SUM=lambda x, n: pd.Series(x).rolling(int(n), min_periods=int(n)).apply(math.fsum, raw=True).to_numpy(),
        VALUEWHEN=lambda condition, values: np.asarray(values)[np.flatnonzero(condition)[-1]])
    for statement in EXTRA_HEADER.strip().split(';'):
        if statement:
            name, expr = statement.strip().split(':='); env[name] = eval(expr, {'__builtins__': {}}, env)
    return float(eval(NEW_EXPRESSIONS['BP01'].replace('TIME=1449', 'TIME==1449'), {'__builtins__': {}}, env))


def native():
    checked_sources(); v = json.loads((ROOT / 'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(ROOT / 'feature_report.json')
    source = Path('data/research/tail_formula_serial_price/native_input_verification.json')
    samples = json.loads(source.read_text())
    assert samples['passed'] and samples['fixed_samples'] == 32
    assert samples['feature_report_sha256'] == sha(source.parent / 'feature_report.json')
    f = pd.read_parquet(ROOT / 'features.parquet').set_index(['date', 'code'])
    hashes = json.loads((prices.ROOT / 'window_report.json').read_text())['source_sha256']; receipts = []
    for sample in samples['checks']:
        date, code = sample['date'], sample['code']; assert '2024-01-01' <= date <= '2025-12-30'
        path = MINUTES / code[:2].upper() / (code[3:] + '.parquet')
        assert sha(path) == hashes[str(path)] == sample['source_sha256']
        minute = pd.read_parquet(path, columns=['timestamp', 'close'], filters=[
            ('timestamp', '>=', pd.Timestamp(date+' 14:20')), ('timestamp', '<=', pd.Timestamp(date+' 14:49'))]).sort_values('timestamp')
        assert minute.timestamp.dt.strftime('%H:%M').tolist() == [f'14:{i}' for i in range(20, 50)]
        raw = minute.close.to_numpy(float).round(2); row = f.loc[(date, code)]; value = native_value(raw)
        assert row.formula_input_valid
        np.testing.assert_array_equal(raw, row[prices.PRICE_COLUMNS].to_numpy(float))
        np.testing.assert_allclose(value, row.BP01, rtol=0, atol=2e-10)
        encode = lambda x: np.floor(np.clip(100*x+10000+.000001, 0, 999999))
        assert encode(value) == encode(row.BP01) and value == native_value(raw, .01)
        receipts.append(dict(date=date, code=code, source_sha256=sample['source_sha256'], minutes=30))
    assert len(receipts) == 32
    proof = dict(passed=True, protocol_sha256=sha(PROTOCOL), feature_report_sha256=sha(ROOT / 'feature_report.json'),
        feature_verification_sha256=sha(ROOT / 'feature_verification.json'), fixed_sample_source_sha256=sha(source),
        samples=receipts, raw_minutes=960, actual_generated_ref_abs_sum_and_valuewhen_expressions_rebuilt=True,
        outside_window_extreme_prices_no_effect=True, software_compilation_verified=False, native_source_parity_verified=False,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'native_input_verification.json', proof)
    return {k: v for k, v in proof.items() if k != 'samples'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['features', 'verify_features', 'native'])
    print(json.dumps(globals()[p.parse_args().stage](), ensure_ascii=False, indent=2))
