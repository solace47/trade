"""Volume-weighted signed and absolute minute-return deviations."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as previous
from . import tail_formula_minute_pressure as volume_source
from . import tail_formula_path_variance as price_source
from .corporate_cash import MINUTES, save_json, sha

STEM = 'tail_formula_volume_response'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
COMBINED_PROTOCOL = Path('config') / (STEM + '_combined_protocol.json')
NEW_EXPRESSIONS = {
    'IP01': 'VALUEWHEN(TIME=1449,(29*SUM(LVRT*V,29)/SUM(V,29)-SUM(LVRT,29)))/V01',
    'IP02': 'VALUEWHEN(TIME=1449,(29*SUM(ABS(LVRT)*V,29)/SUM(V,29)-SUM(ABS(LVRT),29)))/V01'}
EXPRESSIONS = {**previous.EXPRESSIONS, **NEW_EXPRESSIONS}
HEADER = previous.HEADER + 'LVRT:=100*LN(C/REF(C,1));\n'
native_core = base.native_core


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    for folder in [previous.ROOT, price_source.ROOT, volume_source.ROOT]:
        r = json.loads((folder / 'feature_report.json').read_text())
        v = json.loads((folder / 'feature_verification.json').read_text())
        assert v['passed'] and v['feature_report_sha256'] == sha(folder / 'feature_report.json')
        assert r['features_sha256'] == sha(folder / 'features.parquet')
        if folder != previous.ROOT:
            assert r['window_report_sha256'] == sha(folder / 'window_report.json')
    assert p['return_bars'] == 29 and not p['new_2026_prices_allowed']
    return p


def cached_windows(keys):
    frames = []
    for module, prefix, columns in [
        (price_source, 'pv', price_source.PRICE_COLUMNS),
        (volume_source, 'mp', volume_source.COLUMNS['c'] + volume_source.COLUMNS['v'])]:
        report = json.loads((module.ROOT / 'window_report.json').read_text())
        for file, digest in report['parts_sha256'].items():
            assert sha(Path(file)) == digest
        names = ['date', 'code', prefix + '_bars', prefix + '_clocks', prefix + '_good_bars', *columns]
        frames.append(pd.concat([pd.read_parquet(path, columns=names) for path in report['parts_sha256']], ignore_index=True))
    result = keys.copy()
    for frame in frames:
        result = result.merge(frame, on=['date', 'code'], how='left', validate='one_to_one')
    return result.sort_values(['date', 'code']).reset_index(drop=True)


def features():
    assert not (ROOT / 'feature_report.json').exists(), 'Do not replace frozen inputs'
    p = checked_sources(); old = pd.read_parquet(previous.ROOT / 'features.parquet')
    d = cached_windows(old[['date', 'code']]); prices = d[price_source.PRICE_COLUMNS].to_numpy(float)
    other_prices = d[volume_source.COLUMNS['c']].to_numpy(float); volume = d[volume_source.COLUMNS['v']].to_numpy(float)
    np.testing.assert_allclose(prices[:, 1:], other_prices, rtol=0, atol=0, equal_nan=True)
    valid = (d.pv_bars.eq(30) & d.pv_clocks.eq(30) & d.pv_good_bars.eq(30)
        & d.mp_bars.eq(29) & d.mp_clocks.eq(29) & d.mp_good_bars.eq(29)
        & np.isfinite(prices).all(axis=1) & (prices > 0).all(axis=1)
        & np.isfinite(volume).all(axis=1) & (volume >= 0).all(axis=1)
        & (volume == np.floor(volume)).all(axis=1) & old.V01.gt(0))
    with np.errstate(divide='ignore', invalid='ignore'):
        changes = 100 * np.log(prices[:, 1:] / prices[:, :-1])
    f = old.copy(); f['vr_volume'] = volume.sum(axis=1)
    f['vr_return_sum'] = changes.sum(axis=1); f['vr_absolute_sum'] = np.abs(changes).sum(axis=1)
    f['vr_weighted_return_sum'] = (changes * volume).sum(axis=1)
    f['vr_weighted_absolute_sum'] = (np.abs(changes) * volume).sum(axis=1)
    valid &= f.vr_volume.gt(0); f['volume_response_valid'] = valid
    f['IP01'] = ((29*f.vr_weighted_return_sum/f.vr_volume-f.vr_return_sum)/f.V01).where(valid)
    f['IP02'] = ((29*f.vr_weighted_absolute_sum/f.vr_volume-f.vr_absolute_sum)/f.V01).where(valid)
    f['prior_formula_input_valid'] = f.formula_input_valid
    f['formula_input_valid'] &= valid & np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
    assert f.date.between(p['signal_first'], p['signal_last']).all()
    ROOT.mkdir(parents=True, exist_ok=True); f.to_parquet(ROOT / 'features.parquet', index=False, compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL), source_hashes=p['source_hashes'],
        features_sha256=sha(ROOT / 'features.parquet'), rows=len(f), valid=int(f.formula_input_valid.sum()),
        previous_valid=int(old.formula_input_valid.sum()), newly_invalid=int((old.formula_input_valid & ~f.formula_input_valid).sum()),
        flat_price_valid=int((f.formula_input_valid & f.vr_absolute_sum.eq(0)).sum()),
        expressions=EXPRESSIONS, native_header=HEADER, all_cached_close_values_equal=True,
        software_compilation_verified=False, native_source_parity_verified=False,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_report.json', r); return {k: v for k, v in r.items() if k not in ['expressions', 'native_header']}


def verify_features():
    p = checked_sources(); r = json.loads((ROOT / 'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['source_hashes'] == p['source_hashes']
    assert r['features_sha256'] == sha(ROOT / 'features.parquet')
    old = pd.read_parquet(previous.ROOT / 'features.parquet'); f = pd.read_parquet(ROOT / 'features.parquet')
    d = cached_windows(old[['date', 'code']]); c = base.conn(); c.register('cached', d); c.register('old', old)
    atoms = []
    for i in range(21, 50):
        change = f'CASE WHEN pv_c{i:02d}>0 AND pv_c{i-1:02d}>0 THEN 100*ln(pv_c{i:02d}/pv_c{i-1:02d}) END'
        atoms.append(f'struct_pack(r:={change},v:=mp_v{i:02d},same:=pv_c{i:02d}=mp_c{i:02d})')
    c.sql('SELECT date,code,unnest(['+','.join(atoms)+']) AS z FROM cached').create_view('long_rows')
    stats = c.sql('''SELECT date,code,sum(z.v) AS vr_volume,sum(z.r) AS vr_return_sum,
        sum(abs(z.r)) AS vr_absolute_sum,sum(z.r*z.v) AS vr_weighted_return_sum,
        sum(abs(z.r)*z.v) AS vr_weighted_absolute_sum,
        bool_and(coalesce(isfinite(z.r) AND isfinite(z.v) AND z.v>=0 AND z.v=floor(z.v) AND z.same,false)) AS good
        FROM long_rows GROUP BY date,code ORDER BY date,code''').df()
    c.register('stats', stats)
    expected = c.sql('''SELECT date,code,coalesce(pv_bars=30 AND pv_clocks=30 AND pv_good_bars=30
        AND mp_bars=29 AND mp_clocks=29 AND mp_good_bars=29 AND stats.good AND vr_volume>0 AND V01>0,false) AS valid,
        CASE WHEN valid THEN (29*vr_weighted_return_sum/vr_volume-vr_return_sum)/V01 END AS IP01,
        CASE WHEN valid THEN (29*vr_weighted_absolute_sum/vr_volume-vr_absolute_sum)/V01 END AS IP02
        FROM stats JOIN cached USING(date,code) JOIN old USING(date,code) ORDER BY date,code''').df(); c.close()
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')], old.drop(columns='formula_input_valid'), check_exact=True)
    pd.testing.assert_series_equal(f.prior_formula_input_valid, old.formula_input_valid, check_names=False, check_exact=True)
    np.testing.assert_array_equal(f.volume_response_valid, expected.valid)
    pd.testing.assert_frame_equal(f[['date', 'code', 'IP01', 'IP02']], expected[['date', 'code', 'IP01', 'IP02']],
                                  check_dtype=False, rtol=0, atol=2e-10)
    columns = ['vr_volume', 'vr_return_sum', 'vr_absolute_sum', 'vr_weighted_return_sum', 'vr_weighted_absolute_sum']
    valid = f.volume_response_valid
    pd.testing.assert_frame_equal(f.loc[valid, columns].reset_index(drop=True), stats.loc[valid, columns].reset_index(drop=True),
                                  check_dtype=False, rtol=2e-13, atol=2e-8)
    target_valid = old.formula_input_valid & expected.valid & np.isfinite(expected[['IP01', 'IP02']]).all(axis=1)
    np.testing.assert_array_equal(f.formula_input_valid, target_valid)
    for name in NEW_EXPRESSIONS:
        actual = np.floor(np.clip(100*f.loc[target_valid, name]+10000+.000001, 0, 999999)).astype('int32')
        wanted = np.floor(np.clip(100*expected.loc[target_valid, name]+10000+.000001, 0, 999999)).astype('int32')
        np.testing.assert_array_equal(actual, wanted)
    assert not f.code.str[3:].str.startswith(('92', '688', '300', '301')).any()
    assert f.isST.eq(0).all() and f.tradestatus.eq(1).all()
    names = re.findall(r'(?m)^([A-Z][A-Z0-9]*):=', HEADER) + list(EXPRESSIONS)
    assert len(names) == len(set(names))
    assert len(f) == r['rows'] and int(target_valid.sum()) == r['valid']
    proof = dict(passed=True, feature_report_sha256=sha(ROOT / 'feature_report.json'), rows=len(f),
        all_48_previous_fields_and_keys_unchanged=True, all_29_minute_aggregates_rebuilt_by_sql=True,
        cached_price_links_and_volume_units_checked=True, all_validity_and_integer_encodings_rebuilt=True,
        all_native_variable_names_unique=True, new_group_outcomes_read=False,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_verification.json', proof); return proof


def native():
    checked_sources(); proof = json.loads((ROOT / 'feature_verification.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256'] == sha(ROOT / 'feature_report.json')
    f = pd.read_parquet(ROOT / 'features.parquet'); selected = f.loc[f.formula_input_valid].copy()
    selected['sample_hash'] = [hashlib.sha256((d+'|'+s+'|vr-native-v1').encode()).hexdigest() for d, s in zip(selected.date, selected.code)]
    sample = selected.sort_values('sample_hash').groupby('half', sort=True).head(8).sort_values(['date', 'code'])
    hashes = json.loads((price_source.ROOT / 'window_report.json').read_text())['source_sha256']
    receipts = []
    for row in sample.itertuples():
        path = MINUTES / row.code[:2].upper() / (row.code[3:]+'.parquet'); assert sha(path) == hashes[str(path)]
        minute = pd.read_parquet(path, columns=['timestamp', 'close', 'volume'], filters=[
            ('timestamp', '>=', pd.Timestamp(row.date+' 14:20')), ('timestamp', '<=', pd.Timestamp(row.date+' 14:49'))]).sort_values('timestamp')
        assert minute.timestamp.dt.strftime('%H:%M').tolist() == [f'14:{i:02d}' for i in range(20, 50)]
        prices = [round(float(x), 2) for x in minute.close]; lots = [float(x)/100 for x in minute.volume.iloc[1:]]
        returns = [100*math.log(prices[i]/prices[i-1]) for i in range(1, 30)]
        values = [(29*sum(v*x for v, x in zip(lots, z))/sum(lots)-sum(z))/row.V01
                  for z in [returns, [abs(x) for x in returns]]]
        np.testing.assert_allclose(values, [row.IP01, row.IP02], rtol=0, atol=2e-10)
        np.testing.assert_array_equal(np.floor(np.clip(100*np.asarray(values)+10000+.000001, 0, 999999)),
                                     np.floor(np.clip(100*np.asarray([row.IP01, row.IP02])+10000+.000001, 0, 999999)))
        receipts.append(dict(date=row.date, code=row.code, source_sha256=hashes[str(path)], minutes=len(minute)))
    assert len(receipts) == 32
    r = dict(passed=True, feature_report_sha256=sha(ROOT / 'feature_report.json'),
        feature_verification_sha256=sha(ROOT / 'feature_verification.json'), samples=receipts,
        raw_minutes=960, scalar_native_formulas_and_lot_conversion_rebuilt=True,
        software_compilation_verified=False, native_source_parity_verified=False,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'native_input_verification.json', r); return r


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['features', 'verify_features', 'native']); a = p.parse_args()
    print(json.dumps(globals()[a.stage](), ensure_ascii=False, indent=2))
