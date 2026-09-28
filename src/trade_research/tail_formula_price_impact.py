"""Absolute minute returns per amount, with inactive minutes excluded."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as previous
from . import tail_formula_volume_response as cache
from . import tail_formula_price_impact_amount as amounts
from .corporate_cash import MINUTES, save_json, sha

STEM = 'tail_formula_price_impact'
ROOT = amounts.ROOT
PROTOCOL = amounts.PROTOCOL
NEW_EXPRESSIONS = {'IL01': 'VALUEWHEN(TIME=1449,10000000000*SUM(IF(V>0,ABS(C/REF(C,1)-1)/MAX(AMOUNT,0.01),0),29)/MAX(COUNT(V>0,29),1))'}
EXPRESSIONS = {**previous.EXPRESSIONS, **NEW_EXPRESSIONS}
HEADER = previous.HEADER + 'ILN:=VALUEWHEN(TIME=1449,COUNT(V>0,29));\n'
ORIGINAL_NATIVE_CORE = base.native_core


def native_core(*args, **kwargs):
    text = ORIGINAL_NATIVE_CORE(*args, **kwargs)
    assert text.count('CORE:SC>') == 1
    return text.replace('CORE:SC>', 'CORE:ILN>0 AND SC>')


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    for module in [previous, cache.price_source, cache.volume_source, cache]:
        r = json.loads((module.ROOT / 'feature_report.json').read_text())
        v = json.loads((module.ROOT / 'feature_verification.json').read_text())
        assert v['passed'] and v['feature_report_sha256'] == sha(module.ROOT / 'feature_report.json')
        assert r['features_sha256'] == sha(module.ROOT / 'features.parquet')
    assert p['native_expression'] == NEW_EXPRESSIONS['IL01']
    assert p['return_bars'] == 29 and not p['new_2026_prices_allowed']
    return p


def cached_windows(keys):
    r = json.loads((ROOT / 'window_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL)
    assert r['extractor_sha256'] == sha(Path(amounts.__file__))
    for path, digest in r['parts_sha256'].items():
        assert sha(Path(path)) == digest
    wide = pd.concat([pd.read_parquet(path) for path in r['parts_sha256']], ignore_index=True)
    return cache.cached_windows(keys).merge(wide, on=['date', 'code'], how='left', validate='one_to_one')


def measure(prices, volume, amount):
    """Prices have 30 bars, while volume and amount have the last 29 bars."""
    assert prices.ndim == volume.ndim == amount.ndim == 2
    assert prices.shape == (len(volume), 30) and volume.shape == amount.shape == (len(prices), 29)
    valid = np.isfinite(prices).all(axis=1) & (prices > 0).all(axis=1)
    valid &= np.isfinite(volume).all(axis=1) & (volume >= 0).all(axis=1) & (volume == np.floor(volume)).all(axis=1)
    valid &= np.isfinite(amount).all(axis=1) & (amount >= 0).all(axis=1)
    valid &= ((volume == 0) == (amount == 0)).all(axis=1) & ((volume <= 0) | (amount >= .01)).all(axis=1)
    active = volume > 0; n = active.sum(axis=1); valid &= n > 0
    with np.errstate(divide='ignore', invalid='ignore'):
        changes = np.abs(prices[:, 1:] / prices[:, :-1] - 1)
        ratios = np.divide(changes, amount, out=np.zeros_like(changes), where=active)
        value = 1e10 * ratios.sum(axis=1) / n
    return dict(valid=valid, il_active=n, il_amount=amount.sum(axis=1),
                il_ratio_sum=ratios.sum(axis=1), IL01=np.where(valid, value, np.nan))


def features():
    assert not (ROOT / 'feature_report.json').exists(), 'Do not replace frozen inputs'
    p = checked_sources(); old = pd.read_parquet(previous.ROOT / 'features.parquet')
    d = cached_windows(old[['date', 'code']])
    pd.testing.assert_frame_equal(d[['date', 'code']], old[['date', 'code']], check_exact=True)
    prices = d[cache.price_source.PRICE_COLUMNS].to_numpy(float)
    np.testing.assert_allclose(prices[:, 1:], d[cache.volume_source.COLUMNS['c']], rtol=0, atol=0, equal_nan=True)
    values = measure(prices, d[cache.volume_source.COLUMNS['v']].to_numpy(float), d[amounts.COLUMNS].to_numpy(float))
    valid = values.pop('valid')
    for prefix, count in [('pv', 30), ('mp', 29), ('il', 29)]:
        for field in ['bars', 'clocks', 'good_bars']:
            valid &= d[prefix + '_' + field].eq(count).to_numpy()
    f = old.copy()
    for name, value in values.items():
        f[name] = value
    f['IL01'] = f.IL01.where(valid)
    f['price_impact_valid'] = valid; f['prior_formula_input_valid'] = f.formula_input_valid
    f['formula_input_valid'] &= valid & np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
    active = f.formula_input_valid
    np.testing.assert_allclose(f.loc[active, 'il_amount'], old.loc[active, 'a29'], rtol=2e-13, atol=2e-6)
    assert f.date.between(p['signal_first'], p['signal_last']).all()
    f.to_parquet(ROOT / 'features.parquet', index=False, compression='zstd')
    report = dict(protocol_sha256=sha(PROTOCOL), source_hashes=p['source_hashes'],
        feature_code_sha256=sha(Path(__file__)), features_sha256=sha(ROOT / 'features.parquet'),
        window_report_sha256=sha(ROOT / 'window_report.json'), rows=len(f), valid=int(active.sum()),
        previous_valid=int(old.formula_input_valid.sum()), newly_invalid=int((old.formula_input_valid & ~active).sum()),
        minimum_active_minutes=int(f.loc[active, 'il_active'].min()),
        inactive_minutes_in_valid_windows=int((29-f.loc[active, 'il_active']).sum()),
        max_amount_sum_difference=float((f.loc[active, 'il_amount']-old.loc[active, 'a29']).abs().max()),
        expressions=EXPRESSIONS, native_header=HEADER, new_group_outcomes_read=False,
        new_2026_prices_read=False, software_compilation_verified=False, native_source_parity_verified=False, no_exit_rules=True)
    save_json(ROOT / 'feature_report.json', report)
    return {k: v for k, v in report.items() if k not in ['expressions', 'native_header', 'source_hashes']}


def verify_features():
    p = checked_sources(); r = json.loads((ROOT / 'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['source_hashes'] == p['source_hashes']
    assert r['features_sha256'] == sha(ROOT / 'features.parquet')
    assert r['window_report_sha256'] == sha(ROOT / 'window_report.json')
    old = pd.read_parquet(previous.ROOT / 'features.parquet'); f = pd.read_parquet(ROOT / 'features.parquet')
    d = cached_windows(old[['date', 'code']]); c = base.conn(); c.register('cached', d)
    atoms = [f'struct_pack(prior:=pv_c{n-1:02d},current:=pv_c{n:02d},other:=mp_c{n:02d},v:=mp_v{n:02d},a:=il_a{n:02d})' for n in range(21, 50)]
    c.sql('SELECT date,code,unnest([' + ','.join(atoms) + ']) AS z FROM cached').create_view('long_rows')
    stats = c.sql('''SELECT date,code,count(*) FILTER(WHERE z.v>0) AS il_active,
        sum(z.a) AS il_amount, sum(CASE WHEN z.v>0 THEN abs(z.current/z.prior-1)/z.a ELSE 0 END) AS il_ratio_sum,
        bool_and(coalesce(isfinite(z.current) AND z.current>0 AND isfinite(z.prior) AND z.prior>0
        AND z.current=z.other AND isfinite(z.v) AND z.v>=0 AND z.v=floor(z.v)
        AND isfinite(z.a) AND z.a>=0 AND ((z.v=0)=(z.a=0)) AND (z.v<=0 OR z.a>=.01),false)) AS good
        FROM long_rows GROUP BY date,code ORDER BY date,code''').df()
    c.register('stats', stats)
    expected = c.sql('''SELECT date,code,coalesce(pv_bars=30 AND pv_clocks=30 AND pv_good_bars=30
        AND mp_bars=29 AND mp_clocks=29 AND mp_good_bars=29 AND il_bars=29 AND il_clocks=29 AND il_good_bars=29
        AND good AND il_active>0,false) AS valid,
        CASE WHEN valid THEN 10000000000*il_ratio_sum/il_active END AS IL01
        FROM stats JOIN cached USING(date,code) ORDER BY date,code''').df(); c.close()
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')], old.drop(columns='formula_input_valid'), check_exact=True)
    pd.testing.assert_series_equal(f.prior_formula_input_valid, old.formula_input_valid, check_names=False, check_exact=True)
    np.testing.assert_array_equal(f.price_impact_valid, expected.valid)
    pd.testing.assert_frame_equal(f[['date', 'code', 'IL01']], expected[['date', 'code', 'IL01']], check_dtype=False, rtol=2e-13, atol=2e-10)
    good = expected.valid
    np.testing.assert_array_equal(f.il_active, stats.il_active)
    np.testing.assert_allclose(f.loc[good, ['il_amount', 'il_ratio_sum']], stats.loc[good, ['il_amount', 'il_ratio_sum']], rtol=2e-13, atol=2e-6)
    wanted = old.formula_input_valid & good & np.isfinite(expected.IL01)
    np.testing.assert_array_equal(f.formula_input_valid, wanted)
    np.testing.assert_allclose(f.loc[wanted, 'il_amount'], old.loc[wanted, 'a29'], rtol=2e-13, atol=2e-6)
    actual_encoded = np.floor(np.clip(100*f.loc[wanted, 'IL01']+10000+.000001, 0, 999999)).astype('int32')
    expected_encoded = np.floor(np.clip(100*expected.loc[wanted, 'IL01']+10000+.000001, 0, 999999)).astype('int32')
    np.testing.assert_array_equal(actual_encoded, expected_encoded)
    assert not f.code.str[3:].str.startswith(('92', '688', '300', '301')).any()
    assert f.isST.eq(0).all() and f.tradestatus.eq(1).all()
    names = re.findall(r'(?m)^([A-Z][A-Z0-9]*):=', HEADER) + list(EXPRESSIONS)
    assert len(names) == len(set(names))
    assert len(f) == r['rows'] and int(wanted.sum()) == r['valid']
    proof = dict(passed=True, feature_report_sha256=sha(ROOT / 'feature_report.json'), rows=len(f),
        all_48_previous_fields_and_keys_unchanged=True, all_29_minute_ratios_and_counts_rebuilt_by_sql=True,
        raw_amount_sum_matches_existing_a29=True, cached_price_links_and_units_checked=True,
        all_validity_and_integer_encodings_rebuilt=True, all_native_variable_names_unique=True,
        effective_input_intersection_unchanged=bool(wanted.equals(old.formula_input_valid)),
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_verification.json', proof); return proof


def native():
    checked_sources(); proof = json.loads((ROOT / 'feature_verification.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256'] == sha(ROOT / 'feature_report.json')
    f = pd.read_parquet(ROOT / 'features.parquet'); selected = f.loc[f.formula_input_valid].copy()
    selected['sample_hash'] = [hashlib.sha256((d+'|'+s+'|price-impact-native-v1').encode()).hexdigest() for d, s in zip(selected.date, selected.code)]
    sample = selected.sort_values('sample_hash').groupby('half', sort=True).head(8).sort_values(['date', 'code'])
    hashes = json.loads((ROOT / 'window_report.json').read_text())['source_sha256']; receipts = []
    for row in sample.itertuples():
        path = MINUTES / row.code[:2].upper() / (row.code[3:]+'.parquet'); assert sha(path) == hashes[str(path)]
        minute = pd.read_parquet(path, columns=['timestamp', 'close', 'volume', 'turnover'], filters=[
            ('timestamp', '>=', pd.Timestamp(row.date+' 14:20')), ('timestamp', '<=', pd.Timestamp(row.date+' 14:49'))]).sort_values('timestamp')
        assert minute.timestamp.dt.strftime('%H:%M').tolist() == [f'14:{i:02d}' for i in range(20, 50)]
        prices = [round(float(x), 2) for x in minute.close]
        lots = [float(x)/100 for x in minute.volume.iloc[1:]]; amounts_ = list(map(float, minute.turnover.iloc[1:]))
        count = sum(v > 0 for v in lots)
        value = 1e10*sum(abs(prices[i+1]/prices[i]-1)/max(amounts_[i], .01) if lots[i] > 0 else 0 for i in range(29))/max(count, 1)
        assert count == row.il_active and count > 0
        np.testing.assert_allclose([value, sum(amounts_)], [row.IL01, row.il_amount], rtol=2e-13, atol=2e-10)
        np.testing.assert_array_equal(np.floor(np.clip(100*np.array([value])+10000+.000001, 0, 999999)),
                                     np.floor(np.clip(100*np.array([row.IL01])+10000+.000001, 0, 999999)))
        receipts.append(dict(date=row.date, code=row.code, source_sha256=hashes[str(path)], minutes=len(minute), active_minutes=count))
    assert len(receipts) == 32 and sample.groupby('half').size().eq(8).all()
    r = dict(passed=True, protocol_sha256=sha(PROTOCOL), feature_report_sha256=sha(ROOT / 'feature_report.json'),
        feature_verification_sha256=sha(ROOT / 'feature_verification.json'), samples=receipts, raw_minutes=960,
        scalar_native_formulas_and_lot_conversion_rebuilt=True, software_compilation_verified=False,
        native_source_parity_verified=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'native_input_verification.json', r); return r


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['features', 'verify_features', 'native']); args = parser.parse_args()
    print(json.dumps(globals()[args.stage](), ensure_ascii=False, indent=2))
