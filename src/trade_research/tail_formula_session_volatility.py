"""Matched 29-change opening and late-session volatility, using cached prices."""
import argparse
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as previous
from . import tail_formula_path_variance as late
from .corporate_cash import MINUTES, save_json, sha

STEM = 'tail_formula_session_volatility'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
MORNING = Path('data/research/tail_formula_morning_history')
PROBES = Path('data/research/tail_formula_volume_response/native_input_verification.json')
CLOCKS = [f'09{i:02d}' for i in range(31, 60)] + ['1000']
OPEN_COLUMNS = [f'sv_p{i:02d}' for i in range(30)]
NEW_EXPRESSIONS = {'SV01': 'SQRT(SVL/29)/V01', 'SV02': 'SQRT(SVO/29)/V01',
                   'SV03': '(SVL-SVO)/MAX(SVL+SVO,0.000000000001)'}
EXPRESSIONS = {**previous.EXPRESSIONS, **NEW_EXPRESSIONS}
EXTRA_HEADER = '''SVR:=100*LN(C/REF(C,1));
SVO:=VALUEWHEN(TIME=1000,SUM(SVR*SVR,29));
SVL:=VALUEWHEN(TIME=1449,SUM(SVR*SVR,29));
'''
HEADER = previous.HEADER + EXTRA_HEADER
native_core = base.native_core


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    for path, digest in p['source_hashes'].items():
        assert sha(Path(path)) == digest
    for root in [previous.ROOT, late.ROOT]:
        r = json.loads((root / 'feature_report.json').read_text())
        v = json.loads((root / 'feature_verification.json').read_text())
        assert v['passed'] and v['feature_report_sha256'] == sha(root / 'feature_report.json')
        assert r['features_sha256'] == sha(root / 'features.parquet')
    r = json.loads((MORNING / 'window_report.json').read_text())
    v = json.loads((MORNING / 'window_verification.json').read_text())
    assert v['passed'] and v['window_report_sha256'] == sha(MORNING / 'window_report.json')
    assert r['raw_report_sha256'] == sha(MORNING / 'raw_report.json')
    assert p['new_fields'] == list(NEW_EXPRESSIONS) and not p['new_2026_prices_allowed']
    return p


def parts(root, report):
    r = json.loads((root / report).read_text())
    for path, digest in r['parts_sha256'].items():
        assert sha(Path(path)) == digest
    return list(r['parts_sha256'])


def late_windows(keys):
    paths = parts(late.ROOT, 'window_report.json')
    columns = ['date', 'code', 'pv_bars', 'pv_clocks', 'pv_good_bars', *late.PRICE_COLUMNS]
    d = pd.concat([pd.read_parquet(path, columns=columns) for path in paths], ignore_index=True)
    return keys.merge(d, on=['date', 'code'], how='left', validate='one_to_one')


def opening_source(c, keys):
    c.read_parquet(parts(MORNING, 'raw_report.json')).create_view('raw')
    c.register('keys', keys)
    c.sql('''SELECT r.*,coalesce(timestamp=date_trunc('minute',timestamp)
        AND strftime(timestamp,'%Y-%m-%d')=r.date AND strftime(timestamp,'%H%M')=clock
        AND isfinite(close) AND close>0 AND abs(close-round(close,2))<=.0001
        AND isfinite(volume) AND volume>=0 AND volume=floor(volume),false) AS good
        FROM raw r JOIN keys USING(date,code)
        WHERE r.date BETWEEN '2024-01-01' AND '2025-12-30'
        AND clock BETWEEN '0931' AND '1000' ''').create_view('morning')


def measure(opening, closing, atr):
    opening, closing, atr = np.asarray(opening, float), np.asarray(closing, float), np.asarray(atr, float)
    assert opening.shape == closing.shape and opening.ndim == 2 and opening.shape[1] == 30
    assert atr.shape == (len(opening),)
    valid = np.isfinite(opening).all(axis=1) & (opening > 0).all(axis=1)
    valid &= np.isfinite(closing).all(axis=1) & (closing > 0).all(axis=1) & np.isfinite(atr) & (atr > 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        a = 100*np.log(opening[:, 1:]/opening[:, :-1])
        b = 100*np.log(closing[:, 1:]/closing[:, :-1])
        e, l = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
        values = [np.sqrt(l/29)/atr, np.sqrt(e/29)/atr, (l-e)/np.maximum(l+e, 1e-12)]
    valid &= np.isfinite(values).all(axis=0)
    return dict(valid=valid, sv_early_square=e, sv_late_square=l,
                **{name: np.where(valid, value, np.nan) for name, value in zip(NEW_EXPRESSIONS, values)})


def features():
    p = checked_sources(); assert not (ROOT / 'feature_report.json').exists()
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    c = base.conn(); opening_source(c, old[['date', 'code']])
    pivots = ','.join(f"max(round(close,2)) FILTER(WHERE clock='{clock}') AS {name}" for clock, name in zip(CLOCKS, OPEN_COLUMNS))
    early = c.sql('''SELECT date,code,count(*) AS sv_bars,count(DISTINCT clock) AS sv_clocks,
        count(*) FILTER(WHERE good) AS sv_good,''' + pivots + ''' FROM morning GROUP BY date,code ORDER BY date,code''').df()
    c.close()
    early = old[['date', 'code']].merge(early, on=['date', 'code'], how='left', validate='one_to_one')
    ending = late_windows(old[['date', 'code']])
    for d in [early, ending]: pd.testing.assert_frame_equal(d[['date', 'code']], old[['date', 'code']], check_exact=True)
    pieces = []
    for start in range(0, len(old), 50000):
        end = start + 50000
        pieces.append(pd.DataFrame(measure(early.iloc[start:end][OPEN_COLUMNS], ending.iloc[start:end][late.PRICE_COLUMNS], old.iloc[start:end].V01)))
    values = pd.concat(pieces, ignore_index=True)
    good = values.valid & early.sv_bars.eq(30) & early.sv_clocks.eq(30) & early.sv_good.eq(30)
    good &= ending.pv_bars.eq(30) & ending.pv_clocks.eq(30) & ending.pv_good_bars.eq(30)
    f = old.copy()
    for name in values.columns.drop('valid'): f[name] = values[name]
    for name in NEW_EXPRESSIONS: f[name] = f[name].where(good)
    f['session_volatility_valid'] = good
    f['prior_formula_input_valid'] = old.formula_input_valid
    f['formula_input_valid'] &= good
    ROOT.mkdir(parents=True, exist_ok=True)
    early.to_parquet(ROOT / 'opening_windows.parquet', index=False, compression='zstd')
    f.to_parquet(ROOT / 'features.parquet', index=False, compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL), source_hashes=p['source_hashes'],
        opening_windows_sha256=sha(ROOT / 'opening_windows.parquet'), features_sha256=sha(ROOT / 'features.parquet'),
        rows=len(f), valid=int(f.formula_input_valid.sum()), newly_invalid=int((old.formula_input_valid & ~good).sum()),
        valid_flat_early=int((f.formula_input_valid & f.sv_early_square.eq(0)).sum()),
        valid_flat_late=int((f.formula_input_valid & f.sv_late_square.eq(0)).sum()),
        expressions={**previous.EXPRESSIONS, **NEW_EXPRESSIONS}, native_header=HEADER,
        software_compilation_verified=False, native_source_parity_verified=False,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_report.json', r)
    return {k: v for k, v in r.items() if k not in ['source_hashes', 'expressions', 'native_header']}


def verify_features():
    checked_sources(); r = json.loads((ROOT / 'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['features_sha256'] == sha(ROOT / 'features.parquet')
    assert r['opening_windows_sha256'] == sha(ROOT / 'opening_windows.parquet')
    old = pd.read_parquet(previous.ROOT / 'features.parquet'); actual = pd.read_parquet(ROOT / 'features.parquet')
    pd.testing.assert_frame_equal(actual[old.columns.drop('formula_input_valid')], old.drop(columns='formula_input_valid'), check_exact=True)
    assert actual.prior_formula_input_valid.equals(old.formula_input_valid)
    c = base.conn(); opening_source(c, old[['date', 'code']])
    # Sum adjacent changes directly from original long rows, independent of the pivot.
    c.sql('''SELECT *,lag(round(close,2)) OVER(PARTITION BY date,code ORDER BY timestamp) AS pc
        FROM morning''').create_view('adjacent')
    early = c.sql('''SELECT date,code,count(*)=30 AND count(DISTINCT clock)=30
        AND min(clock)='0931' AND max(clock)='1000' AND bool_and(good) AS early_good,
        sum(CASE WHEN pc>0 AND close>0 THEN pow(100*ln(round(close,2)/pc),2) ELSE 0 END) AS early_square
        FROM adjacent GROUP BY date,code ORDER BY date,code''').df()
    ending = late_windows(old[['date', 'code']]); c.register('ending', ending)
    atoms = ','.join(f'struct_pack(p:=pv_c{i-1:02d},q:=pv_c{i:02d})' for i in range(21, 50))
    c.sql('SELECT date,code,unnest([' + atoms + ']) AS v FROM ending').create_view('tail_atoms')
    tail = c.sql('''SELECT date,code,bool_and(coalesce(isfinite(v.p) AND isfinite(v.q) AND v.p>0 AND v.q>0,false)) AS tail_good,
        sum(CASE WHEN v.p>0 AND v.q>0 THEN pow(100*ln(v.q/v.p),2) END) AS late_square
        FROM tail_atoms GROUP BY date,code ORDER BY date,code''').df()
    c.register('early', early); c.register('tail', tail); c.register('old', old[['date', 'code', 'V01']])
    expected = c.sql('''WITH h AS(SELECT *,coalesce(early_good AND tail_good AND pv_bars=30 AND pv_clocks=30
        AND pv_good_bars=30 AND isfinite(V01) AND V01>0,false) AS valid
        FROM old LEFT JOIN early USING(date,code) LEFT JOIN tail USING(date,code)
        LEFT JOIN (SELECT date,code,pv_bars,pv_clocks,pv_good_bars FROM ending) USING(date,code))
        SELECT date,code,valid,early_square,late_square,
        CASE WHEN valid THEN sqrt(late_square/29)/V01 END AS SV01,
        CASE WHEN valid THEN sqrt(early_square/29)/V01 END AS SV02,
        CASE WHEN valid THEN (late_square-early_square)/greatest(late_square+early_square,1e-12) END AS SV03
        FROM h ORDER BY date,code''').df(); c.close()
    pd.testing.assert_frame_equal(actual[['date', 'code']], expected[['date', 'code']], check_exact=True)
    np.testing.assert_array_equal(actual.session_volatility_valid, expected.valid)
    final = old.formula_input_valid & expected.valid
    np.testing.assert_array_equal(actual.formula_input_valid, final)
    for new, key in [('sv_early_square', 'early_square'), ('sv_late_square', 'late_square')]:
        np.testing.assert_allclose(actual.loc[expected.valid, new], expected.loc[expected.valid, key], rtol=1e-12, atol=2e-10)
    for name in NEW_EXPRESSIONS:
        np.testing.assert_allclose(actual[name], expected[name], rtol=0, atol=2e-10, equal_nan=True)
        np.testing.assert_array_equal(np.floor(100*actual.loc[final, name]+10000+1e-6).clip(0, 999999),
                                     np.floor(100*expected.loc[final, name]+10000+1e-6).clip(0, 999999))
    prior = pd.read_parquet(late.ROOT / 'features.parquet', columns=['date', 'code', 'Z01'])
    pd.testing.assert_frame_equal(prior[['date', 'code']], old[['date', 'code']], check_exact=True)
    np.testing.assert_allclose(actual.loc[final, 'SV01'], prior.loc[final, 'Z01'], rtol=0, atol=0)
    names = re.findall(r'(?m)^([A-Z][A-Z0-9]*):=', HEADER) + list(previous.EXPRESSIONS) + list(NEW_EXPRESSIONS)
    assert len(names) == len({n.casefold() for n in names})
    assert r['rows'] == len(actual) and r['valid'] == int(final.sum())
    assert r['newly_invalid'] == int((old.formula_input_valid & ~final).sum())
    assert r['expressions'] == {**previous.EXPRESSIONS, **NEW_EXPRESSIONS} and r['native_header'] == HEADER
    proof = dict(passed=True, feature_report_sha256=sha(ROOT / 'feature_report.json'), rows=len(actual), valid=int(final.sum()),
        effective_input_intersection_unchanged=bool(final.equals(old.formula_input_valid)),
        original_morning_long_rows_and_tail_pairs_sql_rebuilt=True, old_Z01_exactly_reused_for_late_control=True,
        all_original_48_values_validity_and_integer_encodings_verified=True,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_verification.json', proof); return proof


def native_value(prices, clocks, atr):
    env = dict(C=np.asarray(prices, float), TIME=np.asarray(clocks), V01=float(atr), LN=np.log,
        MAX=np.maximum, SQRT=np.sqrt, REF=lambda x, n: pd.Series(x).shift(int(n)).to_numpy(),
        SUM=lambda x, n: pd.Series(x).rolling(int(n), min_periods=int(n)).apply(math.fsum, raw=True).to_numpy(),
        VALUEWHEN=lambda mask, value: np.asarray(value)[np.flatnonzero(mask)[-1]])
    with np.errstate(divide='ignore', invalid='ignore'):
        for line in EXTRA_HEADER.splitlines():
            name, expr = line.rstrip(';').split(':=')
            env[name] = eval(re.sub(r'(?<![<>=!])=(?!=)', '==', expr), {'__builtins__': {}}, env)
        return {name: float(eval(expr, {'__builtins__': {}}, env)) for name, expr in NEW_EXPRESSIONS.items()}


def native():
    checked_sources(); proof = json.loads((ROOT / 'feature_verification.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256'] == sha(ROOT / 'feature_report.json')
    samples = json.loads(PROBES.read_text()); assert samples['passed']
    actual = pd.read_parquet(ROOT / 'features.parquet').set_index(['date', 'code']); details = []
    for sample in samples['samples']:
        day, code = sample['date'], sample['code']; assert '2024-01-01' <= day <= '2025-12-30'
        path = MINUTES / code[:2].upper() / (code[3:] + '.parquet'); assert sha(path) == sample['source_sha256']
        raw = pd.read_parquet(path, columns=['timestamp', 'close'], filters=[('timestamp', '>=', pd.Timestamp(day+' 09:30')),
            ('timestamp', '<=', pd.Timestamp(day+' 14:49'))]).sort_values('timestamp')
        clocks = (100*raw.timestamp.dt.hour+raw.timestamp.dt.minute).to_numpy()
        assert len(raw) == 230 and len(set(clocks)) == 230
        values = raw.close.to_numpy(float); assert np.isfinite(values).all() and (values > 0).all()
        assert (np.abs(values-values.round(2)) <= .0001).all(); values = values.round(2)
        row = actual.loc[day, code]; assert row.formula_input_valid
        result = native_value(values, clocks, row.V01)
        changed = native_value(np.r_[values, .01, 1000000.], np.r_[clocks, 1450, 1451], row.V01)
        for name in NEW_EXPRESSIONS:
            np.testing.assert_allclose(result[name], row[name], rtol=0, atol=2e-10)
            assert changed[name] == result[name]
            assert np.floor(100*result[name]+10000+1e-6).clip(0, 999999) == np.floor(100*row[name]+10000+1e-6).clip(0, 999999)
        perturbed = values.copy(); perturbed[clocks > 1000] = 1000000.
        assert native_value(perturbed, clocks, row.V01)['SV02'] == result['SV02']
        details.append(dict(date=day, code=code, source_sha256=sample['source_sha256'], **result))
    assert len(details) == 32
    r = dict(passed=True, protocol_sha256=sha(PROTOCOL), feature_report_sha256=sha(ROOT / 'feature_report.json'),
        feature_verification_sha256=sha(ROOT / 'feature_verification.json'), samples=details, raw_prefix_rows=7360,
        native_ref_log_sum_sqrt_and_two_anchors_rebuilt=True, future_perturbations=32, opening_anchor_perturbations=32,
        software_compilation_verified=False, native_source_parity_verified=False,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'native_input_verification.json', r); return {k: v for k, v in r.items() if k != 'samples'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('stage', choices=['features', 'verify_features', 'native'])
    print(json.dumps(globals()[p.parse_args().stage](), ensure_ascii=False, indent=2))
