"""Scale-invariant concentration of volume within two fixed late windows."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as previous
from . import tail_formula_minute_pressure as source
from .corporate_cash import MINUTES, save_json, sha

STEM = 'tail_formula_volume_concentration'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
COMBINED_PROTOCOL = Path('config') / (STEM + '_combined_protocol.json')
NEW_EXPRESSIONS = {f'VC{n:02d}': f'VALUEWHEN(TIME=1449,100*({n}*SUM(V*V,{n})/(SUM(V,{n})*SUM(V,{n}))-1)/{n-1})'
                   for n in [29, 4]}
EXPRESSIONS = {**previous.EXPRESSIONS, **NEW_EXPRESSIONS}
HEADER = previous.HEADER
native_core = base.native_core


def concentration(volume):
    """Zero total or invalid volumes stay unknown, never a zero-concentration signal."""
    a = np.asarray(volume, dtype=float)
    assert a.ndim == 2 and a.shape[1] > 1
    total = a.sum(axis=1)
    valid = np.isfinite(a).all(axis=1) & (a >= 0).all(axis=1) & (total > 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        value = 100*(a.shape[1]*np.square(a/total[:, None]).sum(axis=1)-1)/(a.shape[1]-1)
    return np.where(valid, value, np.nan)


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    for folder in [previous.ROOT, source.ROOT]:
        r = json.loads((folder / 'feature_report.json').read_text())
        v = json.loads((folder / 'feature_verification.json').read_text())
        assert v['passed'] and v['feature_report_sha256'] == sha(folder / 'feature_report.json')
        assert r['features_sha256'] == sha(folder / 'features.parquet')
    assert p['windows'] == [29, 4] and not p['new_2026_prices_allowed']
    return p


def cached_windows(keys):
    r = json.loads((source.ROOT / 'window_report.json').read_text())
    for file, digest in r['parts_sha256'].items():
        assert sha(Path(file)) == digest
    names = ['date', 'code', 'mp_bars', 'mp_clocks', 'mp_good_bars', *source.COLUMNS['v']]
    w = pd.concat([pd.read_parquet(p, columns=names) for p in r['parts_sha256']], ignore_index=True)
    return keys.merge(w, on=['date', 'code'], how='left', validate='one_to_one').sort_values(['date', 'code']).reset_index(drop=True)


def features():
    assert not (ROOT / 'feature_report.json').exists(), 'Do not replace frozen inputs'
    p = checked_sources(); old = pd.read_parquet(previous.ROOT / 'features.parquet')
    d = cached_windows(old[['date', 'code']]); a = d[source.COLUMNS['v']].to_numpy(float)
    valid = (d.mp_bars.eq(29) & d.mp_clocks.eq(29) & d.mp_good_bars.eq(29)
        & np.isfinite(a).all(axis=1) & (a >= 0).all(axis=1) & (a == np.floor(a)).all(axis=1)
        & (a.sum(axis=1) > 0) & (a[:, -4:].sum(axis=1) > 0))
    f = old.copy(); f['volume_concentration_valid'] = valid
    for n in [29, 4]:
        f[f'vc_sum{n}'] = a[:, -n:].sum(axis=1)
        f[f'vc_square_sum{n}'] = np.square(a[:, -n:]).sum(axis=1)
        f[f'VC{n:02d}'] = pd.Series(concentration(a[:, -n:])).where(valid)
        np.testing.assert_allclose(f.loc[valid, f'vc_sum{n}'], f.loc[valid, f'v{n}'], rtol=0, atol=0)
    f['prior_formula_input_valid'] = f.formula_input_valid
    f['formula_input_valid'] &= valid & np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
    assert f.date.between(p['signal_first'], p['signal_last']).all()
    ROOT.mkdir(parents=True, exist_ok=True); f.to_parquet(ROOT / 'features.parquet', index=False, compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL), source_hashes=p['source_hashes'],
        features_sha256=sha(ROOT / 'features.parquet'), rows=len(f), valid=int(f.formula_input_valid.sum()),
        previous_valid=int(old.formula_input_valid.sum()), newly_invalid=int((old.formula_input_valid & ~f.formula_input_valid).sum()),
        expressions=EXPRESSIONS, native_header=HEADER, source_volume_units='shares; ratios invariant to lots',
        software_compilation_verified=False, native_source_parity_verified=False,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_report.json', r); return {k:v for k,v in r.items() if k not in ['expressions', 'native_header']}


def verify_features():
    p = checked_sources(); r = json.loads((ROOT / 'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['source_hashes'] == p['source_hashes']
    assert r['features_sha256'] == sha(ROOT / 'features.parquet')
    old = pd.read_parquet(previous.ROOT / 'features.parquet'); f = pd.read_parquet(ROOT / 'features.parquet')
    d = cached_windows(old[['date', 'code']]); c = base.conn(); c.register('cached', d)
    atoms = ','.join(f'struct_pack(i:={i},v:=mp_v{i:02d})' for i in range(21, 50))
    c.sql('SELECT date,code,unnest(['+atoms+']) AS z FROM cached').create_view('long_rows')
    stats = c.sql('''SELECT date,code,sum(z.v) AS vc_sum29,sum(z.v*z.v) AS vc_square_sum29,
        sum(CASE WHEN z.i>=46 THEN z.v ELSE 0 END) AS vc_sum4,
        sum(CASE WHEN z.i>=46 THEN z.v*z.v ELSE 0 END) AS vc_square_sum4,
        bool_and(coalesce(isfinite(z.v) AND z.v>=0 AND z.v=floor(z.v),false)) AS good
        FROM long_rows GROUP BY date,code ORDER BY date,code''').df()
    c.register('stats', stats)
    expected = c.sql('''SELECT date,code,coalesce(mp_bars=29 AND mp_clocks=29 AND mp_good_bars=29
        AND good AND vc_sum29>0 AND vc_sum4>0,false) AS valid,
        CASE WHEN valid THEN 100*(29*vc_square_sum29/(vc_sum29*vc_sum29)-1)/28 END AS VC29,
        CASE WHEN valid THEN 100*(4*vc_square_sum4/(vc_sum4*vc_sum4)-1)/3 END AS VC04
        FROM stats JOIN cached USING(date,code) ORDER BY date,code''').df(); c.close()
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')], old.drop(columns='formula_input_valid'), check_exact=True)
    pd.testing.assert_series_equal(f.prior_formula_input_valid, old.formula_input_valid, check_exact=True, check_names=False)
    pd.testing.assert_frame_equal(f[['date', 'code']], expected[['date', 'code']], check_exact=True)
    np.testing.assert_array_equal(f.volume_concentration_valid, expected.valid)
    valid = expected.valid
    for name in ['vc_sum29', 'vc_square_sum29', 'vc_sum4', 'vc_square_sum4']:
        np.testing.assert_allclose(f.loc[valid, name], stats.loc[valid, name], rtol=3e-15, atol=0)
    for name in NEW_EXPRESSIONS:
        np.testing.assert_allclose(f[name], expected[name], rtol=0, atol=2e-10, equal_nan=True)
        assert expected.loc[valid, name].between(-2e-10, 100+2e-10).all()
        np.testing.assert_array_equal(np.floor(100*f.loc[valid, name]+10000+.000001),
                                      np.floor(100*expected.loc[valid, name]+10000+.000001))
    np.testing.assert_array_equal(f.formula_input_valid, old.formula_input_valid & expected.valid)
    assert not f.code.str[3:].str.startswith(('92', '688', '300', '301')).any()
    assert f.isST.eq(0).all() and f.tradestatus.eq(1).all()
    names = re.findall(r'(?m)^([A-Z][A-Z0-9]*):=', HEADER) + list(EXPRESSIONS)
    assert len(names) == len(set(names))
    assert not ({x.casefold() for x in NEW_EXPRESSIONS} & {x.casefold() for x in old.columns})
    assert len(f) == r['rows'] and int(f.formula_input_valid.sum()) == r['valid']
    proof = dict(passed=True, feature_report_sha256=sha(ROOT / 'feature_report.json'), rows=len(f),
        all_48_previous_fields_and_keys_unchanged=True, all_volume_sums_and_squares_rebuilt_by_sql=True,
        all_validity_and_integer_encodings_rebuilt=True, new_group_outcomes_read=False,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_verification.json', proof); return proof


def native():
    checked_sources(); proof = json.loads((ROOT / 'feature_verification.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256'] == sha(ROOT / 'feature_report.json')
    f = pd.read_parquet(ROOT / 'features.parquet'); selected = f.loc[f.formula_input_valid].copy()
    selected['sample_hash'] = [hashlib.sha256((d+'|'+s+'|vc-native-v1').encode()).hexdigest() for d,s in zip(selected.date, selected.code)]
    sample = selected.sort_values('sample_hash').groupby('half', sort=True).head(8).sort_values(['date', 'code'])
    hashes = json.loads((source.ROOT / 'window_report.json').read_text())['source_sha256']; receipts = []
    for row in sample.itertuples():
        path = MINUTES / row.code[:2].upper() / (row.code[3:]+'.parquet'); assert sha(path) == hashes[str(path)]
        minute = pd.read_parquet(path, columns=['timestamp', 'volume'], filters=[
            ('timestamp', '>=', pd.Timestamp(row.date+' 14:21')), ('timestamp', '<=', pd.Timestamp(row.date+' 14:49'))]).sort_values('timestamp')
        assert minute.timestamp.dt.strftime('%H:%M').tolist() == [f'14:{i:02d}' for i in range(21, 50)]
        shares = [int(v) for v in minute.volume]
        for n in [29, 4]:
            volumes = shares[-n:]; total = sum(volumes)
            # Integer numerator provides an independent scale-free expression.
            numerator = n*sum(v*v for v in volumes)-total*total
            value = 100*numerator/((n-1)*total*total)
            lots = [v/100 for v in volumes]
            native_value = 100*(n*sum(v*v for v in lots)/sum(lots)**2-1)/(n-1)
            expected = getattr(row, f'VC{n:02d}')
            np.testing.assert_allclose([value, native_value], expected, rtol=0, atol=2e-10)
            np.testing.assert_array_equal(np.floor(100*np.array([value, native_value])+10000+.000001),
                                          np.repeat(np.floor(100*expected+10000+.000001), 2))
        receipts.append(dict(date=row.date, code=row.code, source_sha256=hashes[str(path)], minutes=len(minute)))
    assert len(receipts) == 32
    r = dict(passed=True, feature_report_sha256=sha(ROOT / 'feature_report.json'),
        feature_verification_sha256=sha(ROOT / 'feature_verification.json'), samples=receipts,
        raw_minutes=928, integer_numerator_and_lot_native_formulas_rebuilt=True,
        software_compilation_verified=False, native_source_parity_verified=False,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'native_input_verification.json', r); return r


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['features', 'verify_features', 'native']); a = p.parse_args()
    print(json.dumps(globals()[a.stage](), ensure_ascii=False, indent=2))
