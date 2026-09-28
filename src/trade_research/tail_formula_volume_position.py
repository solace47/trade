"""Late volume grouped by its minute close relative to the visible final quote."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as previous
from . import tail_formula_volume_response as cached
from .corporate_cash import MINUTES, save_json, sha

STEM = 'tail_formula_volume_position'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
NEW_EXPRESSIONS = {'VC01': '100*VCBE/MAX(VCT,0.01)', 'VC02': '100*VCEQ/MAX(VCT,0.01)'}
EXPRESSIONS = {**previous.EXPRESSIONS, **NEW_EXPRESSIONS}
EXTRA_HEADER = 'VCQ:=INTPART(Q*100+0.5);\n'
for i in range(29):
    EXTRA_HEADER += f'VCP{i}:=INTPART(VALUEWHEN(TIME=1449,REF(C,{i}))*100+0.5);\n'
    EXTRA_HEADER += f'VCV{i}:=VALUEWHEN(TIME=1449,REF(V,{i}));\n'
EXTRA_HEADER += 'VCT:=' + '+'.join(f'VCV{i}' for i in range(29)) + ';\n'
EXTRA_HEADER += 'VCBE:=' + '+'.join(f'IF(VCQ>VCP{i},VCV{i},0)' for i in range(29)) + ';\n'
EXTRA_HEADER += 'VCEQ:=' + '+'.join(f'IF(VCQ=VCP{i},VCV{i},0)' for i in range(29)) + ';\n'
HEADER = previous.HEADER + EXTRA_HEADER
native_core = base.native_core


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    for path, digest in p['source_hashes'].items():
        assert sha(Path(path)) == digest
    cached.checked_sources()
    assert p['expressions'] == NEW_EXPRESSIONS and p['new_fields'] == list(NEW_EXPRESSIONS)
    assert p['price_bars'] == p['return_bars'] == 29 and not p['new_2026_prices_allowed']
    return p


def measure(prices, volume):
    prices, volume = np.asarray(prices, float), np.asarray(volume, float)
    assert prices.ndim == volume.ndim == 2 and prices.shape == volume.shape and prices.shape[1] == 29
    cents = np.rint(prices * 100)
    valid = np.isfinite(prices).all(axis=1) & (prices > 0).all(axis=1)
    valid &= (np.abs(prices * 100 - cents) <= 1e-6).all(axis=1)
    valid &= np.isfinite(volume).all(axis=1) & (volume >= 0).all(axis=1) & (volume == np.floor(volume)).all(axis=1)
    total = volume.sum(axis=1)
    below = np.where(cents < cents[:, -1:], volume, 0).sum(axis=1)
    equal = np.where(cents == cents[:, -1:], volume, 0).sum(axis=1)
    valid &= total > 0
    values = {}
    for name, amount in [('VC01', below), ('VC02', equal)]:
        values[name] = np.divide(100 * amount, total, out=np.full(len(prices), np.nan), where=valid)
    return dict(valid=valid, vc_volume=total, vc_below=below, vc_equal=equal, **values)


def features():
    p = checked_sources()
    assert not (ROOT / 'feature_report.json').exists()
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    d = cached.cached_windows(old[['date', 'code']])
    prices = d[cached.price_source.PRICE_COLUMNS[1:]].to_numpy(float)
    np.testing.assert_allclose(prices, d[cached.volume_source.COLUMNS['c']], rtol=0, atol=0, equal_nan=True)
    result = measure(prices, d[cached.volume_source.COLUMNS['v']].to_numpy(float))
    valid = result.pop('valid')
    for field in ['bars', 'clocks', 'good_bars']:
        valid &= d['mp_' + field].eq(29).to_numpy()
    f = old.copy()
    for name, value in result.items():
        f[name] = value
    for name in NEW_EXPRESSIONS:
        f[name] = f[name].where(valid)
    f['volume_position_valid'] = valid
    f['prior_formula_input_valid'] = old.formula_input_valid
    f['formula_input_valid'] &= valid
    active = f.formula_input_valid
    np.testing.assert_array_equal(f.loc[active, 'vc_volume'], old.loc[active, 'v29'])
    assert (f.loc[active, ['VC01', 'VC02']].sum(axis=1) <= 100 + 1e-12).all()
    assert f.date.between(p['signal_first'], p['signal_last']).all()
    ROOT.mkdir(parents=True, exist_ok=True)
    f.to_parquet(ROOT / 'features.parquet', index=False, compression='zstd')
    report = dict(protocol_sha256=sha(PROTOCOL), source_hashes=p['source_hashes'],
        features_sha256=sha(ROOT / 'features.parquet'), rows=len(f), valid=int(active.sum()),
        newly_invalid=int((old.formula_input_valid & ~active).sum()),
        valid_all_volume_at_final_price=int((active & f.VC02.eq(100)).sum()),
        valid_no_volume_below_final_price=int((active & f.VC01.eq(0)).sum()),
        expressions=EXPRESSIONS, native_header=HEADER, software_compilation_verified=False,
        native_source_parity_verified=False, new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_report.json', report)
    return {key: val for key, val in report.items() if key not in ['source_hashes', 'expressions', 'native_header']}


def verify_features():
    checked_sources()
    r = json.loads((ROOT / 'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['features_sha256'] == sha(ROOT / 'features.parquet')
    old, f = [pd.read_parquet(folder / 'features.parquet') for folder in [previous.ROOT, ROOT]]
    d = cached.cached_windows(old[['date', 'code']])
    c = base.conn()
    c.register('cached', d)
    atoms = [f'struct_pack(p:=pv_c{i:02d},other:=mp_c{i:02d},v:=mp_v{i:02d})' for i in range(21, 50)]
    c.sql('SELECT date,code,pv_c49 AS final,unnest([' + ','.join(atoms) + ']) AS z FROM cached').create_view('atoms')
    stats = c.sql('''SELECT date,code,sum(z.v) AS vc_volume,
        sum(CASE WHEN floor(z.p*100+.5)<floor(final*100+.5) THEN z.v ELSE 0 END) AS vc_below,
        sum(CASE WHEN floor(z.p*100+.5)=floor(final*100+.5) THEN z.v ELSE 0 END) AS vc_equal,
        bool_and(coalesce(isfinite(z.p) AND z.p>0 AND abs(z.p*100-floor(z.p*100+.5))<=.000001
        AND z.p=z.other AND isfinite(z.v) AND z.v>=0 AND z.v=floor(z.v),false)) AS good
        FROM atoms GROUP BY date,code ORDER BY date,code''').df()
    c.register('stats', stats)
    expected = c.sql('''SELECT date,code,coalesce(mp_bars=29 AND mp_clocks=29 AND mp_good_bars=29
        AND good AND vc_volume>0,false) AS valid,
        CASE WHEN valid THEN 100*vc_below/vc_volume END AS VC01,
        CASE WHEN valid THEN 100*vc_equal/vc_volume END AS VC02
        FROM stats JOIN cached USING(date,code) ORDER BY date,code''').df()
    c.close()
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')], old.drop(columns='formula_input_valid'), check_exact=True)
    pd.testing.assert_series_equal(f.prior_formula_input_valid, old.formula_input_valid, check_exact=True, check_names=False)
    pd.testing.assert_frame_equal(f[['date', 'code']], expected[['date', 'code']], check_exact=True)
    np.testing.assert_array_equal(f.volume_position_valid, expected.valid)
    for name in ['vc_volume', 'vc_below', 'vc_equal']:
        np.testing.assert_array_equal(f.loc[expected.valid, name], stats.loc[expected.valid, name])
    final = old.formula_input_valid & expected.valid
    np.testing.assert_array_equal(f.formula_input_valid, final)
    for name in NEW_EXPRESSIONS:
        np.testing.assert_allclose(f[name], expected[name], rtol=0, atol=2e-12, equal_nan=True)
        np.testing.assert_array_equal(np.floor(100*f.loc[final, name]+10000+.000001),
            np.floor(100*expected.loc[final, name]+10000+.000001))
    names = re.findall(r'(?m)^([A-Z][A-Z0-9]*):=', HEADER) + list(EXPRESSIONS)
    assert len(names) == len({name.casefold() for name in names})
    assert r['expressions'] == EXPRESSIONS and r['native_header'] == HEADER
    assert r['rows'] == len(f) and r['valid'] == int(final.sum())
    assert r['newly_invalid'] == int((old.formula_input_valid & ~final).sum())
    assert r['valid_all_volume_at_final_price'] == int((final & expected.VC02.eq(100)).sum())
    assert r['valid_no_volume_below_final_price'] == int((final & expected.VC01.eq(0)).sum())
    proof = dict(passed=True, feature_report_sha256=sha(ROOT / 'feature_report.json'), rows=len(f), valid=int(final.sum()),
        all_29_volume_position_comparisons_totals_validities_and_encodings_sql_rebuilt=True,
        all_original_48_values_and_keys_unchanged=True, effective_input_intersection_unchanged=bool(final.equals(old.formula_input_valid)),
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_verification.json', proof)
    return proof


def native_value(prices, lots, outside=1000000.):
    assert len(prices) == len(lots) == 29
    env = dict(C=np.r_[outside, prices, outside, outside], V=np.r_[9999., lots, 9999., 9999.],
        TIME=np.r_[1420, np.arange(1421, 1450), 1450, 1451],
        REF=lambda x, n: pd.Series(x).shift(int(n)).to_numpy(),
        INTPART=np.floor, IF=np.where, MAX=np.maximum,
        VALUEWHEN=lambda mask, values: np.asarray(values)[np.flatnonzero(mask)[-1]])
    env['Q'] = env['VALUEWHEN'](env['TIME'] == 1449, env['C'])
    for line in EXTRA_HEADER.splitlines():
        name, expr = line.rstrip(';').split(':=')
        expr = re.sub(r'(?<![<>=!])=(?!=)', '==', expr)
        env[name] = eval(expr, {'__builtins__': {}}, env)
    return {name: float(eval(expr, {'__builtins__': {}}, env)) for name, expr in NEW_EXPRESSIONS.items()}


def native():
    checked_sources()
    v = json.loads((ROOT / 'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(ROOT / 'feature_report.json')
    prior = json.loads((cached.ROOT / 'native_input_verification.json').read_text())
    assert prior['passed'] and prior['feature_report_sha256'] == sha(cached.ROOT / 'feature_report.json')
    f = pd.read_parquet(ROOT / 'features.parquet').set_index(['date', 'code'])
    samples = []
    for item in prior['samples']:
        date, code = item['date'], item['code']
        assert '2024-01-01' <= date <= '2025-12-30'
        path = MINUTES / code[:2].upper() / (code[3:] + '.parquet')
        assert sha(path) == item['source_sha256']
        raw = pd.read_parquet(path, columns=['timestamp', 'close', 'volume'],
            filters=[('timestamp', '>=', pd.Timestamp(date+' 14:21')), ('timestamp', '<=', pd.Timestamp(date+' 14:49'))]).sort_values('timestamp')
        assert raw.timestamp.dt.strftime('%H:%M').tolist() == [f'14:{i}' for i in range(21, 50)]
        prices = raw.close.to_numpy(float)
        assert (np.abs(prices-prices.round(2)) <= .0001).all()
        prices = prices.round(2)
        lots = raw.volume.to_numpy(float)/100
        result, changed = native_value(prices, lots), native_value(prices, lots*100, .01)
        row = f.loc[(date, code)]
        assert row.formula_input_valid
        for name in NEW_EXPRESSIONS:
            np.testing.assert_allclose(result[name], row[name], rtol=0, atol=2e-12)
            np.testing.assert_allclose(result[name], changed[name], rtol=0, atol=2e-12)
            assert np.floor(100*result[name]+10000+.000001) == np.floor(100*row[name]+10000+.000001)
        samples.append(dict(date=date, code=code, source_sha256=item['source_sha256'], minutes=29, **result))
    assert len(samples) == 32
    result = dict(passed=True, protocol_sha256=sha(PROTOCOL), feature_report_sha256=sha(ROOT / 'feature_report.json'),
        feature_verification_sha256=sha(ROOT / 'feature_verification.json'), samples=samples, raw_minutes=928,
        actual_anchored_ref_if_integer_prices_and_volume_units_rebuilt=True,
        outside_and_future_quotes_do_not_change_features=True,
        software_compilation_verified=False, native_source_parity_verified=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'native_input_verification.json', result)
    return {key: val for key, val in result.items() if key != 'samples'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['features', 'verify_features', 'native'])
    print(json.dumps(globals()[parser.parse_args().stage](), ensure_ascii=False, indent=2))
