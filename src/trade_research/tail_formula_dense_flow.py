"""Retain the fixed ordered price path and all 29 minute volume fractions.

The first screen evaluates only 2025. The volumes are an observable trading
pattern, not proof of an investor's identity or a claim of future profit.
"""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_dense_path as price
from .corporate_cash import MINUTES, save_json, sha

STEM = 'tail_formula_dense_flow'
ROOT = Path('data/research') / STEM
INPUTS = ROOT / 'inputs'
PROTOCOL = Path('config') / (STEM + '_protocol.json')
INTENT = Path('config/tail_formula_dense_flow_intent.json')
VOLUME = Path('data/research/tail_formula_minute_pressure')
META = price.META
VOLUME_COLUMNS = [f'mp_v{i:02d}' for i in range(21, 50)]
NEW_EXPRESSIONS = {f'DVC{i:02d}': f'IF(VRREADY,100*VALUEWHEN(TIME=1449,REF(V,{i}))/VRVOL,DRAWNULL)'
                   for i in range(29)}
ARMS = {'control': price.ARMS['control'], 'price': price.ARMS['path'],
        'flow': {**price.ARMS['path'], **NEW_EXPRESSIONS}}
EXTRA_HEADER = ('VRVOL:=VALUEWHEN(TIME=1449,SUM(V,29));\n'
                'VRGOOD:=VALUEWHEN(TIME=1449,COUNT(V>=0,29));\n'
                'VRREADY:=RTREADY AND VRGOOD=29 AND VRVOL>0;\n')
HEADER = price.HEADER + EXTRA_HEADER


def checked():
    p = json.loads(PROTOCOL.read_text()); price.checked()
    assert p['arms'] == ARMS and p['native_header'] == HEADER and p['expected_keys'] == 1258085
    assert p['intent_sha256'] == sha(INTENT) and list(p['folds']) == ['2025h1', '2025h2']
    assert p['threshold'] == .995 and not p['new_2026_prices_allowed']
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    for folder in [price.INPUTS, VOLUME]:
        fr = json.loads((folder / 'feature_report.json').read_text())
        fv = json.loads((folder / 'feature_verification.json').read_text())
        assert fv['passed'] and fv['feature_report_sha256'] == sha(folder / 'feature_report.json')
        assert fr['features_sha256'] == sha(folder / 'features.parquet')
    return p


def fractions(volumes):
    """Newest minute first; zero minutes stay zero, invalid windows stay NaN."""
    v = np.asarray(volumes, float)
    assert v.ndim == 2 and v.shape[1] == 29
    good = np.isfinite(v).all(axis=1) & (v >= 0).all(axis=1)
    good &= v.sum(axis=1) > 0
    with np.errstate(all='ignore'):
        result = 100 * v[:, ::-1] / v.sum(axis=1)[:, None]
    result[~good] = np.nan
    return result


def volume_inputs():
    return pd.read_parquet(VOLUME / 'features.parquet',
                           columns=['date', 'code', 'mp_bars', 'mp_clocks', *VOLUME_COLUMNS])


def prepare():
    checked(); assert not (INPUTS / 'feature_report.json').exists()
    INPUTS.mkdir(parents=True, exist_ok=True)
    f = pd.read_parquet(price.INPUTS / 'features.parquet',
                        columns=[*META, *ARMS['price'], 'dense_path_valid'], filters=[('date', '>=', '2024-01-01')])
    v = volume_inputs(); pd.testing.assert_frame_equal(f[['date', 'code']], v[['date', 'code']], check_exact=True)
    vv = v[VOLUME_COLUMNS].to_numpy(float)
    values = fractions(vv)
    good = v.mp_bars.eq(29).to_numpy() & v.mp_clocks.eq(29).to_numpy()
    good &= np.isfinite(vv).all(axis=1) & (vv >= 0).all(axis=1) & (vv == np.floor(vv)).all(axis=1)
    good &= np.isfinite(values).all(axis=1)
    good &= f.dense_path_valid.to_numpy()
    values[~good] = np.nan
    f['prior_formula_input_valid'] = f.formula_input_valid
    for i, name in enumerate(NEW_EXPRESSIONS):
        f[name] = values[:, i]
    f['dense_flow_valid'] = good; f['formula_input_valid'] &= good
    f.to_parquet(INPUTS / 'features.parquet', index=False, compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL), features_sha256=sha(INPUTS / 'features.parquet'),
             rows=len(f), valid=int(f.formula_input_valid.sum()),
             newly_invalid=int((f.prior_formula_input_valid & ~f.formula_input_valid).sum()),
             expressions=ARMS['flow'], native_header=HEADER, all_29_ordered_volume_fractions_retained=True,
             no_new_raw_window_extraction=True, no_new_2023_volume_extraction=True,
             zero_volume_minutes_retained=True, source_volume_unit='shares',
             native_lot_units_cancel_in_ratios=True, native_fractional_lots_not_rejected=True,
             new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True,
             software_compilation_verified=False, native_source_parity_verified=False)
    save_json(INPUTS / 'feature_report.json', r)
    for file in ['full_labels.parquet', 'full_label_report.json', 'full_label_verification.json']:
        (INPUTS / file).symlink_to((price.INPUTS / file).resolve())
    return {k: r[k] for k in ['rows', 'valid', 'newly_invalid']}


def verify():
    checked(); r = json.loads((INPUTS / 'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['features_sha256'] == sha(INPUTS / 'features.parquet')
    f = pd.read_parquet(INPUTS / 'features.parquet')
    old = pd.read_parquet(price.INPUTS / 'features.parquet', columns=[*META, *ARMS['price']],
                          filters=[('date', '>=', '2024-01-01')])
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')],
                                  old.drop(columns='formula_input_valid'), check_exact=True)
    np.testing.assert_array_equal(f.prior_formula_input_valid, old.formula_input_valid)
    c = base.conn(); c.register('volumes', volume_inputs())
    c.read_parquet(str(price.INPUTS / 'features.parquet')).create_view('prices')
    total = '+'.join(VOLUME_COLUMNS)
    good = ' AND '.join(f'isfinite({n}) AND {n}>=0 AND {n}=floor({n})' for n in VOLUME_COLUMNS)
    fields = ','.join(f'CASE WHEN valid THEN 100*mp_v{49-i:02d}/total END AS DVC{i:02d}' for i in range(29))
    expected = c.sql(f'''WITH a AS(SELECT v.*,p.dense_path_valid,({total}) AS total FROM volumes v
        JOIN prices p USING(date,code)), b AS(SELECT *,coalesce(dense_path_valid AND mp_bars=29 AND mp_clocks=29
        AND {good} AND total>0,false) AS valid FROM a)
        SELECT date,code,valid,{fields} FROM b ORDER BY date,code''').df(); c.close()
    pd.testing.assert_frame_equal(f[['date', 'code']], expected[['date', 'code']], check_exact=True)
    np.testing.assert_array_equal(f.dense_flow_valid, expected.valid)
    encode = lambda x: np.floor(np.clip(100*x+10000+.000001, 0, 999999))
    for name in NEW_EXPRESSIONS:
        np.testing.assert_allclose(f[name], expected[name], rtol=0, atol=2e-11, equal_nan=True)
        ok = np.isfinite(expected[name]); np.testing.assert_array_equal(encode(f.loc[ok, name]), encode(expected.loc[ok, name]))
    final = old.formula_input_valid & expected.valid
    np.testing.assert_array_equal(f.formula_input_valid, final)
    names = re.findall(r'(?m)^([A-Z][A-Z0-9]*):=', HEADER) + list(ARMS['flow'])
    assert len(names) == len({n.casefold() for n in names})
    proof = dict(passed=True, feature_report_sha256=sha(INPUTS / 'feature_report.json'), rows=len(f), valid=int(final.sum()),
                 all_77_parent_values_and_metadata_unchanged=True, all_29_sql_fractions_and_encodings_equal=True,
                 effective_input_intersection_unchanged=bool(final.equals(old.formula_input_valid)),
                 no_unknown_window_zero_imputation=True, source_shares_nonnegative_integers_checked=True,
                 new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(INPUTS / 'feature_verification.json', proof); return proof


def native_value(volumes, outside=1000000.):
    """Literal native volume header/expression replay, including fractional lots."""
    env = dict(V=np.r_[outside, volumes, outside, outside], RTREADY=True, DRAWNULL=np.nan,
               TIME=np.r_[1420, np.arange(1421, 1450), 1450, 1451], IF=np.where,
               REF=lambda x,n: pd.Series(x).shift(int(n)).to_numpy(),
               SUM=lambda x,n: pd.Series(x).rolling(int(n),min_periods=int(n)).sum().to_numpy(),
               COUNT=lambda x,n: pd.Series(np.asarray(x,float)).rolling(int(n),min_periods=int(n)).sum().to_numpy(),
               VALUEWHEN=lambda mask,values: np.asarray(values)[np.flatnonzero(mask)[-1]])
    for line in EXTRA_HEADER.splitlines():
        name, expr = line.rstrip(';').split(':=')
        expr = re.sub(r'(?<![<>=!])=(?!=)', '==', expr).replace(' AND ', ' and ')
        env[name] = eval(expr, {'__builtins__': {}}, env)
    return np.asarray([float(eval(re.sub(r'(?<![<>=!])=(?!=)', '==', expr), {'__builtins__': {}}, env))
                       for expr in NEW_EXPRESSIONS.values()])


def native():
    checked(); fv = json.loads((INPUTS / 'feature_verification.json').read_text())
    assert fv['passed'] and fv['feature_report_sha256'] == sha(INPUTS / 'feature_report.json')
    f = pd.read_parquet(INPUTS / 'features.parquet'); v = volume_inputs(); checks = 0
    pd.testing.assert_frame_equal(f[['date', 'code']], v[['date', 'code']], check_exact=True)
    encode = lambda x: np.floor(np.clip(100*x+10000+.000001, 0, 999999))
    for start in range(0, len(f), 50000):
        ff = f.iloc[start:start+50000]; vv = v.iloc[start:start+50000][VOLUME_COLUMNS].to_numpy(float)
        # VALUEWHEN/REF/SUM below are vector projections of the literal header;
        # raw probes separately replay the full sequence and clock operations.
        env = dict(VRREADY=ff.dense_flow_valid.to_numpy(), VRVOL=vv.sum(axis=1), IF=np.where, DRAWNULL=np.nan)
        for i, (name, expr) in enumerate(NEW_EXPRESSIONS.items()):
            literal = expr.replace(f'VALUEWHEN(TIME=1449,REF(V,{i}))', 'LAGV'); env['LAGV'] = vv[:, 28-i]
            with np.errstate(all='ignore'):
                got = eval(literal, {'__builtins__': {}}, env)
            expected = ff[name].to_numpy(float); np.testing.assert_allclose(got, expected, rtol=0, atol=2e-11, equal_nan=True)
            ok = np.isfinite(expected); np.testing.assert_array_equal(encode(got[ok]), encode(expected[ok])); checks += len(ff)
    sample = json.loads((price.prior.INPUTS / 'native_input_verification.json').read_text())['samples']
    sample = [s for s in sample if s['date'] >= '2024-01-01']; assert len(sample) == 32
    indexed = f.set_index(['date', 'code']); receipts = []
    for item in sample:
        date, code = item['date'], item['code']; row = indexed.loc[(date, code)]
        file = MINUTES / code[:2].upper() / (code[3:] + '.parquet'); assert sha(file) == item['source_sha256']
        raw = pd.read_parquet(file, columns=['timestamp', 'close', 'volume'],
                              filters=[('timestamp', '>=', pd.Timestamp(date+' 14:20')),
                                       ('timestamp', '<=', pd.Timestamp(date+' 14:49'))]).sort_values('timestamp')
        assert raw.timestamp.dt.strftime('%H%M').tolist() == [f'14{i:02d}' for i in range(20,50)]
        volumes = raw.volume.to_numpy(float)[1:]
        got = native_value(volumes); expected = row[list(NEW_EXPRESSIONS)].to_numpy(float)
        np.testing.assert_allclose(got, expected, rtol=0, atol=2e-11)
        np.testing.assert_array_equal(got, native_value(volumes, .01))
        lots = native_value(volumes / 100.)
        np.testing.assert_allclose(got, lots, rtol=0, atol=2e-11)
        np.testing.assert_array_equal(encode(got), encode(lots))
        coords = price.native_value(raw.close.to_numpy(float), row.V01)
        np.testing.assert_allclose(coords, row[list(price.NEW_EXPRESSIONS)].to_numpy(float), rtol=0, atol=2e-11)
        receipts.append(item)
    helper = INPUTS / 'dense_flow_inputs.tdx'
    helper.write_text(price.EXTRA_HEADER + EXTRA_HEADER + '\n'.join(f'{k}:={v};' for k,v in {**price.NEW_EXPRESSIONS, **NEW_EXPRESSIONS}.items())+'\n')
    proof = dict(passed=True, protocol_sha256=sha(PROTOCOL), feature_report_sha256=sha(INPUTS / 'feature_report.json'),
                 feature_verification_sha256=sha(INPUTS / 'feature_verification.json'), helper_sha256=sha(helper),
                 scalar_checks=checks, samples=receipts, all_29_volume_literal_arithmetic_and_encodings_replayed=True,
                 all_32_raw_price_volume_windows_replayed=True, fractional_lot_ratios_and_probe_encodings_equal=True,
                 future_and_outside_window_volume_excluded=True, no_new_2023_volume_extraction=True,
                 new_2026_prices_read=False, no_exit_rules=True, software_compilation_verified=False, native_source_parity_verified=False)
    save_json(INPUTS / 'native_input_verification.json', proof); return {k:v for k,v in proof.items() if k != 'samples'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('stage', choices=['prepare','verify','native'])
    print(json.dumps(globals()[parser.parse_args().stage](), ensure_ascii=False, indent=2))
