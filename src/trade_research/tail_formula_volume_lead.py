"""Lag-one volume/return responses inside the already observed late window."""
import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as previous
from . import tail_formula_volume_response as cached
from .corporate_cash import MINUTES, save_json, sha

STEM = 'tail_formula_volume_lead'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
NEW_EXPRESSIONS = {
    'VL01': 'VALUEWHEN(TIME=1449,IF(SUM(REF(V,1),28)>0,28*SUM(VLR*REF(V,1),28)/MAX(SUM(REF(V,1),28),0.000000000001)-SUM(VLR,28),0))/V01',
    'VL02': 'VALUEWHEN(TIME=1449,IF(SUM(V,28)>0,28*SUM(REF(VLR,1)*V,28)/MAX(SUM(V,28),0.000000000001)-SUM(REF(VLR,1),28),0))/V01'}
EXPRESSIONS = {**previous.EXPRESSIONS, **NEW_EXPRESSIONS}
EXTRA_HEADER = 'VLR:=100*LN(C/REF(C,1));\n'
HEADER = previous.HEADER + EXTRA_HEADER


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    assert p['pairs'] == 28 and p['return_bars'] == 29 and not p['new_2026_prices_allowed']
    assert p['expressions'] == NEW_EXPRESSIONS and p['native_header'] == EXTRA_HEADER
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    cached.checked_sources()
    return p


def moments(prices, volume):
    assert prices.ndim == volume.ndim == 2 and prices.shape[1] == 30 and volume.shape[1] == 29
    with np.errstate(divide='ignore', invalid='ignore'):
        ret = 100 * np.log(prices[:, 1:] / prices[:, :-1])
    result = {}
    for name, r, v in [('vl_lead', ret[:, 1:], volume[:, :-1]), ('vl_follow', ret[:, :-1], volume[:, 1:])]:
        result[name + '_volume'] = np.sum(v, axis=1)
        result[name + '_return'] = np.sum(r, axis=1)
        result[name + '_product'] = np.sum(v*r, axis=1)
    return result


def response(total, ret, product):
    return np.where(total > 0, 28*product/np.maximum(total, 1e-12)-ret, 0)


def features():
    p = checked_sources()
    assert not (ROOT / 'feature_report.json').exists()
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    d = cached.cached_windows(old[['date', 'code']])
    prices = d[cached.price_source.PRICE_COLUMNS].to_numpy(float)
    volume = d[cached.volume_source.COLUMNS['v']].to_numpy(float)
    np.testing.assert_allclose(prices[:, 1:], d[cached.volume_source.COLUMNS['c']], rtol=0, atol=0, equal_nan=True)
    valid = (d.pv_bars.eq(30) & d.pv_clocks.eq(30) & d.pv_good_bars.eq(30) &
        d.mp_bars.eq(29) & d.mp_clocks.eq(29) & d.mp_good_bars.eq(29) &
        np.isfinite(prices).all(axis=1) & (prices > 0).all(axis=1) &
        np.isfinite(volume).all(axis=1) & (volume >= 0).all(axis=1) &
        (volume == np.floor(volume)).all(axis=1) & old.V01.gt(0))
    f = old.copy()
    for name, value in moments(prices, volume).items():
        f[name] = value
    for field, prefix in [('VL01', 'vl_lead'), ('VL02', 'vl_follow')]:
        f[field] = pd.Series(response(f[prefix+'_volume'], f[prefix+'_return'], f[prefix+'_product'])/f.V01).where(valid)
    f['volume_lead_valid'] = valid & np.isfinite(f[list(NEW_EXPRESSIONS)]).all(axis=1)
    f['prior_formula_input_valid'] = f.formula_input_valid
    f['formula_input_valid'] &= f.volume_lead_valid
    assert f.date.between(p['signal_first'], p['signal_last']).all()
    ROOT.mkdir(parents=True, exist_ok=True)
    f.to_parquet(ROOT / 'features.parquet', index=False, compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL), features_sha256=sha(ROOT / 'features.parquet'),
        source_hashes=p['source_hashes'], rows=len(f), valid=int(f.formula_input_valid.sum()),
        newly_invalid=int((old.formula_input_valid & ~f.formula_input_valid).sum()),
        zero_lead_volume_valid=int((f.formula_input_valid & f.vl_lead_volume.eq(0)).sum()),
        zero_follow_volume_valid=int((f.formula_input_valid & f.vl_follow_volume.eq(0)).sum()),
        expressions=EXPRESSIONS, native_header=HEADER, software_compilation_verified=False,
        native_source_parity_verified=False, new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_report.json', r)
    return {k:v for k,v in r.items() if k not in ['expressions', 'native_header', 'source_hashes']}


def verify_features():
    p = checked_sources()
    r = json.loads((ROOT / 'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['features_sha256'] == sha(ROOT / 'features.parquet')
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    got = pd.read_parquet(ROOT / 'features.parquet')
    d = cached.cached_windows(old[['date', 'code']])
    c = base.conn()
    c.register('cached', d)
    c.register('old', old)
    def ret(i):
        return f'(CASE WHEN pv_c{i:02d}>0 AND pv_c{i-1:02d}>0 THEN 100*ln(pv_c{i:02d}/pv_c{i-1:02d}) END)'
    atoms = [f'struct_pack(lr:={ret(i)},fr:={ret(i-1)},lv:=mp_v{i-1:02d},fv:=mp_v{i:02d})' for i in range(22,50)]
    c.sql('SELECT date,code,unnest(['+','.join(atoms)+']) AS z FROM cached').create_view('pairs')
    stats = c.sql('''SELECT date,code,sum(z.lv) AS vl_lead_volume,sum(z.lr) AS vl_lead_return,
        sum(z.lv*z.lr) AS vl_lead_product,sum(z.fv) AS vl_follow_volume,sum(z.fr) AS vl_follow_return,
        sum(z.fv*z.fr) AS vl_follow_product FROM pairs GROUP BY date,code ORDER BY date,code''').df()
    c.register('stats', stats)
    fields = []
    for i in range(21,50):
        fields.extend([f'isfinite(pv_c{i:02d}) AND pv_c{i:02d}>0 AND pv_c{i:02d}=mp_c{i:02d}',
                       f'isfinite(mp_v{i:02d}) AND mp_v{i:02d}>=0 AND mp_v{i:02d}=floor(mp_v{i:02d})'])
    fields.append('isfinite(pv_c20) AND pv_c20>0')
    valid = 'coalesce(pv_bars=30 AND pv_clocks=30 AND pv_good_bars=30 AND mp_bars=29 AND mp_clocks=29 AND mp_good_bars=29 AND V01>0 AND ' + ' AND '.join(fields) + ',false)'
    ex = c.sql(f'''SELECT old.date,old.code,{valid} AS valid,
        CASE WHEN valid THEN CASE WHEN vl_lead_volume=0 THEN 0 ELSE
        (28*vl_lead_product/vl_lead_volume-vl_lead_return)/V01 END END AS VL01,
        CASE WHEN valid THEN CASE WHEN vl_follow_volume=0 THEN 0 ELSE
        (28*vl_follow_product/vl_follow_volume-vl_follow_return)/V01 END END AS VL02
        FROM old JOIN cached USING(date,code) JOIN stats USING(date,code) ORDER BY date,code''').df()
    c.close()
    pd.testing.assert_frame_equal(got[old.columns.drop('formula_input_valid')], old.drop(columns='formula_input_valid'), check_exact=True)
    pd.testing.assert_series_equal(got.prior_formula_input_valid, old.formula_input_valid, check_names=False, check_exact=True)
    np.testing.assert_array_equal(got.volume_lead_valid, ex.valid)
    pd.testing.assert_frame_equal(got[['date', 'code', 'VL01', 'VL02']], ex[['date', 'code', 'VL01', 'VL02']],
                                  check_dtype=False, rtol=0, atol=2e-10)
    stat_columns = [name for name in stats.columns if name.startswith('vl_')]
    pd.testing.assert_frame_equal(got.loc[ex.valid, stat_columns].reset_index(drop=True), stats.loc[ex.valid, stat_columns].reset_index(drop=True),
                                  check_dtype=False, rtol=2e-13, atol=2e-8)
    valid = old.formula_input_valid & ex.valid
    np.testing.assert_array_equal(got.formula_input_valid, valid)
    encode = lambda x: np.floor(np.clip(100*x+10000+.000001, 0, 999999)).astype('int32')
    np.testing.assert_array_equal(encode(got.loc[valid, list(NEW_EXPRESSIONS)]), encode(ex.loc[valid, list(NEW_EXPRESSIONS)]))
    names = re.findall(r'(?m)^([A-Z][A-Z0-9]*):=', HEADER) + list(EXPRESSIONS)
    assert len(names) == len(set(names))
    assert got.isST.eq(0).all() and got.tradestatus.eq(1).all()
    assert not got.code.str[3:].str.startswith(('92','688','300','301')).any()
    assert len(got) == r['rows'] == p['expected_keys'] and int(valid.sum()) == r['valid']
    assert r['expressions'] == EXPRESSIONS and r['native_header'] == HEADER
    assert r['newly_invalid'] == int((old.formula_input_valid & ~valid).sum())
    for direction in ['lead', 'follow']:
        assert r['zero_'+direction+'_volume_valid'] == int((valid & got['vl_'+direction+'_volume'].eq(0)).sum())
    proof = dict(passed=True, feature_report_sha256=sha(ROOT / 'feature_report.json'), rows=len(got),
        effective_input_intersection_unchanged=bool(valid.equals(old.formula_input_valid)),
        all_48_previous_fields_and_keys_unchanged=True, all_28_lead_and_follow_pairs_aggregates_and_encodings_rebuilt=True,
        zero_volume_cases_and_all_input_validity_independently_verified=True, all_native_variable_names_unique=True,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_verification.json', proof)
    return proof


def native_values(prices, lots, atr, outside_price=1000000.):
    """Evaluate the actual REF/SUM expressions, including the frozen TIME boundary."""
    assert len(prices) == len(lots) == 30
    env = dict(C=np.r_[outside_price, outside_price, prices, outside_price, outside_price],
               V=np.r_[99999.,99999.,lots,99999.,99999.],
               TIME=np.r_[1418,1419,np.arange(1420,1450),1450,1451], V01=float(atr))
    def ref(x,n):
        return pd.Series(x).shift(int(n)).to_numpy()
    def sum_(x,n):
        # Each literal window is summed independently. Rolling accumulators can
        # retain round-off from earlier, deliberately extreme boundary probes.
        return pd.Series(x).rolling(int(n), min_periods=int(n)).apply(math.fsum, raw=True).to_numpy()
    def valuewhen(condition, values):
        return np.asarray(values)[np.flatnonzero(condition)[-1]]
    env.update(REF=ref, SUM=sum_, VALUEWHEN=valuewhen, IF=np.where, MAX=np.maximum, LN=np.log)
    for statement in EXTRA_HEADER.strip().split(';'):
        if statement:
            key, expr = statement.split(':=')
            env[key] = eval(expr, {'__builtins__': {}}, env)
    return np.array([eval(expr.replace('TIME=1449','TIME==1449'), {'__builtins__': {}}, env)
                     for expr in NEW_EXPRESSIONS.values()])


def native():
    checked_sources()
    proof = json.loads((ROOT / 'feature_verification.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256'] == sha(ROOT / 'feature_report.json')
    old = json.loads((cached.ROOT / 'native_input_verification.json').read_text())
    assert old['passed'] and old['feature_report_sha256'] == sha(cached.ROOT / 'feature_report.json')
    f = pd.read_parquet(ROOT / 'features.parquet').set_index(['date','code'])
    receipts = []
    for sample in old['samples']:
        date, code = sample['date'], sample['code']
        assert '2024-01-01' <= date <= '2025-12-30'
        path = MINUTES / code[:2].upper() / (code[3:]+'.parquet')
        assert sha(path) == sample['source_sha256']
        minute = pd.read_parquet(path, columns=['timestamp','close','volume'], filters=[
            ('timestamp','>=',pd.Timestamp(date+' 14:20')),('timestamp','<=',pd.Timestamp(date+' 14:49'))]).sort_values('timestamp')
        assert minute.timestamp.dt.strftime('%H:%M').tolist() == [f'14:{i}' for i in range(20,50)]
        prices = minute.close.to_numpy(float).round(2)
        lots = minute.volume.to_numpy(float)/100
        row = f.loc[(date,code)]
        assert row.formula_input_valid
        value = native_values(prices, lots, row.V01)
        np.testing.assert_allclose(value, [row.VL01,row.VL02], rtol=0, atol=2e-10)
        encode = lambda x: np.floor(np.clip(100*np.array(x)+10000+.000001,0,999999))
        np.testing.assert_array_equal(encode(value), encode([row.VL01,row.VL02]))
        np.testing.assert_array_equal(value, native_values(prices,lots,row.V01,.01))
        receipts.append(dict(date=date,code=code,source_sha256=sample['source_sha256'],minutes=30))
    assert len(receipts) == 32
    result = dict(passed=True, protocol_sha256=sha(PROTOCOL), feature_report_sha256=sha(ROOT / 'feature_report.json'),
        feature_verification_sha256=sha(ROOT / 'feature_verification.json'), samples=receipts, raw_minutes=960,
        actual_generated_expressions_ref_sum_time_boundary_and_lot_conversion_rebuilt=True,
        outside_window_extreme_prices_no_effect=True, software_compilation_verified=False,
        native_source_parity_verified=False, new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'native_input_verification.json', result)
    return {k:v for k,v in result.items() if k != 'samples'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['features','verify_features','native'])
    print(json.dumps(globals()[parser.parse_args().stage](), ensure_ascii=False, indent=2))
