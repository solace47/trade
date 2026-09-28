"""Four fixed orthogonal coefficients of the visible 30-minute price path."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as previous
from . import tail_formula_path_variance as cached
from . import tail_formula_volume_response as probes
from .corporate_cash import MINUTES, save_json, sha

STEM = 'tail_formula_price_curve'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
WEIGHTS = np.asarray(json.loads(PROTOCOL.read_text())['primitive_integer_weights'], dtype='int64')
MAXIMUM = np.max(np.abs(WEIGHTS), axis=1)
SQUARES = (WEIGHTS * WEIGHTS).sum(axis=1)
NEW_EXPRESSIONS = {}
EXTRA_HEADER = ''
for i in range(30):
    EXTRA_HEADER += f'PCR{i}:=INTPART(100*VALUEWHEN(TIME=1449,REF(C,{29-i}))+0.5)/100;\n'
    EXTRA_HEADER += f'PCL{i}:=100*LN(PCR{i}/PCR0);\n'
for degree, weights in enumerate(WEIGHTS, start=1):
    terms = '+'.join(f'({int(w)})*PCL{i}' for i, w in enumerate(weights) if w)
    NEW_EXPRESSIONS[f'PC0{degree}'] = f'({terms})*{MAXIMUM[degree-1]}/{SQUARES[degree-1]}/VP20'
EXPRESSIONS = {**previous.EXPRESSIONS, **NEW_EXPRESSIONS}
HEADER = previous.HEADER + EXTRA_HEADER
native_core = base.native_core


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    for path, digest in p['source_hashes'].items():
        assert sha(Path(path)) == digest
    for folder in [previous.ROOT, cached.ROOT]:
        r = json.loads((folder / 'feature_report.json').read_text())
        v = json.loads((folder / 'feature_verification.json').read_text())
        assert v['passed'] and v['feature_report_sha256'] == sha(folder / 'feature_report.json')
        assert r['features_sha256'] == sha(folder / 'features.parquet')
    r = json.loads((cached.ROOT / 'feature_report.json').read_text())
    assert r['window_report_sha256'] == sha(cached.ROOT / 'window_report.json')
    np.testing.assert_array_equal(WEIGHTS, p['primitive_integer_weights'])
    assert WEIGHTS.shape == (4, 30) and (WEIGHTS.sum(axis=1) == 0).all()
    np.testing.assert_array_equal(WEIGHTS @ WEIGHTS.T, np.diag(SQUARES))
    assert p['new_fields'] == list(NEW_EXPRESSIONS) and p['expected_features'] == len(EXPRESSIONS)
    assert not p['new_2026_prices_allowed']
    return p


def cached_windows(keys):
    r = json.loads((cached.ROOT / 'window_report.json').read_text())
    for file, digest in r['parts_sha256'].items():
        assert sha(Path(file)) == digest
    names = ['date', 'code', 'pv_bars', 'pv_clocks', 'pv_good_bars', *cached.PRICE_COLUMNS]
    windows = pd.concat([pd.read_parquet(path, columns=names) for path in r['parts_sha256']], ignore_index=True)
    return keys.merge(windows, on=['date', 'code'], how='left', validate='one_to_one').sort_values(['date', 'code']).reset_index(drop=True)


def project(log_prices):
    return (np.asarray(log_prices, float) @ WEIGHTS.T) * (MAXIMUM / SQUARES)


def measure(prices, atr_pct):
    prices, atr_pct = np.asarray(prices, float), np.asarray(atr_pct, float)
    assert prices.ndim == 2 and prices.shape[1] == 30 and atr_pct.shape == (len(prices),)
    valid = np.isfinite(prices).all(axis=1) & (prices > 0).all(axis=1)
    valid &= (np.abs(prices * 100 - np.rint(prices * 100)) <= 1e-6).all(axis=1)
    valid &= np.isfinite(atr_pct) & (atr_pct > 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        y = 100 * np.log(prices / prices[:, :1])
        values = project(y) / atr_pct[:, None]
    valid &= np.isfinite(values).all(axis=1)
    values[~valid] = np.nan
    return valid, values


def features():
    p = checked_sources()
    assert not (ROOT / 'feature_report.json').exists()
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    d = cached_windows(old[['date', 'code']])
    valid, values = measure(d[cached.PRICE_COLUMNS].to_numpy(float), old.V01.to_numpy(float))
    valid &= d[['pv_bars', 'pv_clocks', 'pv_good_bars']].eq(30).all(axis=1).to_numpy()
    values[~valid] = np.nan
    f = old.copy()
    for i, name in enumerate(NEW_EXPRESSIONS):
        f[name] = values[:, i]
    f['price_curve_valid'] = valid
    f['prior_formula_input_valid'] = old.formula_input_valid
    f['formula_input_valid'] &= valid
    assert f.date.between(p['signal_first'], p['signal_last']).all()
    ROOT.mkdir(parents=True, exist_ok=True)
    f.to_parquet(ROOT / 'features.parquet', index=False, compression='zstd')
    encoding = {}
    for name in NEW_EXPRESSIONS:
        vals = f.loc[f.formula_input_valid, name]
        x = 100*vals+10000+.000001
        encoding[name] = dict(minimum=float(vals.min()), maximum=float(vals.max()),
            below_clip=int((x < 0).sum()), above_clip=int((x > 999999).sum()),
            near_integer_boundary=int((np.abs(x-np.rint(x)) < 1e-8).sum()))
    r = dict(protocol_sha256=sha(PROTOCOL), source_hashes=p['source_hashes'],
        features_sha256=sha(ROOT / 'features.parquet'), rows=len(f), valid=int(f.formula_input_valid.sum()),
        newly_invalid=int((old.formula_input_valid & ~f.formula_input_valid).sum()),
        encoding_counts=encoding, expressions=EXPRESSIONS, native_header=HEADER,
        software_compilation_verified=False, native_source_parity_verified=False,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_report.json', r)
    return {k: v for k, v in r.items() if k not in ['source_hashes', 'expressions', 'native_header']}


def ordinary_power_fit(y):
    """Fit ordinary powers independently, then change the five coefficient axes."""
    x = np.linspace(-1, 1, 30)
    powers = np.column_stack([x**i for i in range(5)])
    basis = np.column_stack([np.ones(30), WEIGHTS.T / MAXIMUM])
    conversion = np.linalg.lstsq(powers, basis, rcond=None)[0]
    np.testing.assert_allclose(powers @ conversion, basis, rtol=0, atol=2e-13)
    beta = np.linalg.lstsq(powers, np.asarray(y, float).T, rcond=None)[0]
    return np.linalg.solve(conversion, beta)[1:].T


def verify_features():
    checked_sources()
    r = json.loads((ROOT / 'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['features_sha256'] == sha(ROOT / 'features.parquet')
    old, f = [pd.read_parquet(folder / 'features.parquet') for folder in [previous.ROOT, ROOT]]
    d = cached_windows(old[['date', 'code']])
    c = base.conn()
    c.register('cached', d)
    c.register('old', old[['date', 'code', 'V01']])
    good = ' AND '.join(f'isfinite({p}) AND {p}>0 AND abs(100*{p}-round(100*{p}))<=.000001' for p in cached.PRICE_COLUMNS)
    terms = []
    for i, (name, weights) in enumerate(zip(NEW_EXPRESSIONS, WEIGHTS)):
        total = '+'.join(f'({int(w)})*100*ln(pv_c{n+20:02d}/pv_c20)' for n, w in enumerate(weights) if w)
        terms.append(f'CASE WHEN valid THEN ({total})*{int(MAXIMUM[i])}/{int(SQUARES[i])}/V01 END AS {name}')
    expected = c.sql('''SELECT date,code,coalesce(pv_bars=30 AND pv_clocks=30 AND pv_good_bars=30
        AND isfinite(V01) AND V01>0 AND '''+good+''',false) AS valid,'''+','.join(terms)+'''
        FROM cached JOIN old USING(date,code) ORDER BY date,code''').df()
    c.close()
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')], old.drop(columns='formula_input_valid'), check_exact=True)
    pd.testing.assert_series_equal(f.prior_formula_input_valid, old.formula_input_valid, check_names=False, check_exact=True)
    pd.testing.assert_frame_equal(f[['date', 'code']], expected[['date', 'code']], check_exact=True)
    np.testing.assert_array_equal(f.price_curve_valid, expected.valid)
    final = old.formula_input_valid & expected.valid
    np.testing.assert_array_equal(f.formula_input_valid, final)
    for name in NEW_EXPRESSIONS:
        np.testing.assert_allclose(f[name], expected[name], rtol=0, atol=2e-10, equal_nan=True)
        encode = lambda x: np.floor(np.clip(100*x+10000+.000001, 0, 999999)).astype('int32')
        np.testing.assert_array_equal(encode(f.loc[final, name]), encode(expected.loc[final, name]))
    ids = np.flatnonzero(final.to_numpy())
    max_error = 0.
    prices = d[cached.PRICE_COLUMNS].to_numpy(float)
    for start in range(0, len(ids), 32768):
        subset = ids[start:start+32768]
        p = prices[subset]
        fit = ordinary_power_fit(100*np.log(p/p[:, :1])) / old.V01.to_numpy()[subset, None]
        recorded = f.iloc[subset][list(NEW_EXPRESSIONS)].to_numpy(float)
        np.testing.assert_allclose(fit, recorded, rtol=0, atol=2e-10)
        max_error = max(max_error, float(np.max(np.abs(fit-recorded))))
    names = re.findall(r'(?m)^([A-Z][A-Z0-9]*):=', HEADER) + list(EXPRESSIONS)
    assert len(names) == len({n.casefold() for n in names})
    assert r['expressions'] == EXPRESSIONS and r['native_header'] == HEADER
    assert r['rows'] == len(f) and r['valid'] == int(final.sum())
    assert r['newly_invalid'] == int((old.formula_input_valid & ~final).sum())
    proof = dict(passed=True, feature_report_sha256=sha(ROOT / 'feature_report.json'), rows=len(f), valid=int(final.sum()),
        all_original_48_values_and_keys_unchanged=True, all_projection_values_validity_and_encodings_sql_rebuilt=True,
        ordinary_power_lstsq_rows=len(ids), maximum_ordinary_power_coefficient_error=max_error,
        fixed_integer_basis_exactly_orthogonal=True, effective_input_intersection_unchanged=bool(final.equals(old.formula_input_valid)),
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_verification.json', proof)
    return proof


def native_value(prices, atr_pct, outside=1000000.):
    assert len(prices) == 30
    env = dict(C=np.r_[outside, prices, outside, outside], VP20=float(atr_pct),
        TIME=np.r_[1419, np.arange(1420, 1450), 1450, 1451], INTPART=np.floor, LN=np.log,
        REF=lambda x, n: pd.Series(x).shift(int(n)).to_numpy(),
        VALUEWHEN=lambda mask, values: np.asarray(values)[np.flatnonzero(mask)[-1]])
    for line in EXTRA_HEADER.splitlines():
        name, expr = line.rstrip(';').split(':=')
        expr = re.sub(r'(?<![<>=!])=(?!=)', '==', expr)
        env[name] = eval(expr, {'__builtins__': {}}, env)
    return {name: float(eval(expr, {'__builtins__': {}}, env)) for name, expr in NEW_EXPRESSIONS.items()}


def native():
    checked_sources()
    v = json.loads((ROOT / 'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(ROOT / 'feature_report.json')
    prior = json.loads((probes.ROOT / 'native_input_verification.json').read_text())
    assert prior['passed'] and prior['feature_report_sha256'] == sha(probes.ROOT / 'feature_report.json')
    f = pd.read_parquet(ROOT / 'features.parquet').set_index(['date', 'code'])
    samples = []
    for item in prior['samples']:
        date, code = item['date'], item['code']
        assert '2024-01-01' <= date <= '2025-12-30'
        path = MINUTES / code[:2].upper() / (code[3:] + '.parquet')
        assert sha(path) == item['source_sha256']
        raw = pd.read_parquet(path, columns=['timestamp', 'close'],
            filters=[('timestamp', '>=', pd.Timestamp(date+' 14:20')), ('timestamp', '<=', pd.Timestamp(date+' 14:49'))]).sort_values('timestamp')
        assert raw.timestamp.dt.strftime('%H:%M').tolist() == [f'14:{i}' for i in range(20, 50)]
        prices = raw.close.to_numpy(float)
        assert (np.abs(prices-prices.round(2)) <= .0001).all()
        row = f.loc[(date, code)]
        assert row.formula_input_valid
        result = native_value(prices, row.V01)
        changed = native_value(prices, row.V01, .01)
        for name in NEW_EXPRESSIONS:
            np.testing.assert_allclose(result[name], row[name], rtol=0, atol=2e-10)
            assert result[name] == changed[name]
            assert np.floor(100*result[name]+10000+.000001) == np.floor(100*row[name]+10000+.000001)
        samples.append(dict(date=date, code=code, source_sha256=item['source_sha256'], minutes=30, **result))
    assert len(samples) == 32
    proof = dict(passed=True, protocol_sha256=sha(PROTOCOL), feature_report_sha256=sha(ROOT / 'feature_report.json'),
        feature_verification_sha256=sha(ROOT / 'feature_verification.json'), samples=samples, raw_minutes=960,
        actual_native_anchored_ref_rounding_log_and_polynomial_expression_rebuilt=True,
        original_atr_numerical_proof_reused=True, outside_and_future_quotes_do_not_change_features=True,
        software_compilation_verified=False, native_source_parity_verified=False,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'native_input_verification.json', proof)
    return {k: v for k, v in proof.items() if k != 'samples'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['features', 'verify_features', 'native'])
    print(json.dumps(globals()[parser.parse_args().stage](), ensure_ascii=False, indent=2))
