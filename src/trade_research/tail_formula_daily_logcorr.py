"""Prior 20 complete-day price/log-volume co-movement, with explicit units."""
import argparse
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_daily_response as source
from . import tail_formula_float as previous
from .corporate_cash import save_json, sha

STEM = 'tail_formula_daily_logcorr'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
PRICES = source.prices.PRICE_COLUMNS[:20]
VOLUMES = source.VOLUMES
EXTRA_HEADER = ''.join(f'LCV{i:02d}:=LN(100*HDV{i:02d}+1);\n' for i in range(1, 21))
EXTRA_HEADER += 'LCPM:=(' + '+'.join(f'DCP{i}' for i in range(1, 21)) + ')/20;\n'
EXTRA_HEADER += 'LCVM:=(' + '+'.join(f'LCV{i:02d}' for i in range(1, 21)) + ')/20;\n'
EXTRA_HEADER += 'LCXX:=' + '+'.join(f'(DCP{i}-LCPM)*(DCP{i}-LCPM)' for i in range(1, 21)) + ';\n'
EXTRA_HEADER += 'LCYY:=' + '+'.join(f'(LCV{i:02d}-LCVM)*(LCV{i:02d}-LCVM)' for i in range(1, 21)) + ';\n'
EXTRA_HEADER += 'LCXY:=' + '+'.join(f'(DCP{i}-LCPM)*(LCV{i:02d}-LCVM)' for i in range(1, 21)) + ';\n'
HEADER = source.volumes.HEADER + EXTRA_HEADER
NEW_EXPRESSIONS = {'LC01': 'MIN(MAX(100*LCXY/MAX(SQRT(LCXX*LCYY),0.000000000001),-100),100)'}
EXPRESSIONS = {**previous.EXPRESSIONS, **NEW_EXPRESSIONS}


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    for path, digest in p['source_hashes'].items():
        assert sha(Path(path)) == digest
    for root in [previous.ROOT, source.ROOT, source.prices.ROOT, source.volumes.ROOT]:
        r = json.loads((root / 'feature_report.json').read_text())
        v = json.loads((root / 'feature_verification.json').read_text())
        assert v['passed'] and v['feature_report_sha256'] == sha(root / 'feature_report.json')
        assert r['features_sha256'] == sha(root / 'features.parquet')
    assert p['history_days'] == 20 and p['new_fields'] == list(NEW_EXPRESSIONS)
    control = json.loads(Path('data/research/tail_formula_exchange_context/constant_control_precheck.json').read_text())
    assert control['passed'] and all(x['full_frame_equal'] for x in control['checks'])
    for x in control['checks']:
        assert x['left_sha256'] == sha(Path(x['left']) / 'selection.parquet')
        assert x['right_sha256'] == sha(Path(x['right']) / 'selection.parquet')
    return p


def correlation(closes, volumes):
    closes, volumes = np.asarray(closes, float), np.asarray(volumes, float)
    assert closes.ndim == 2 and closes.shape == volumes.shape and closes.shape[1] == 20
    valid = np.isfinite(closes).all(axis=1) & (closes > 0).all(axis=1)
    valid &= (np.abs(closes*100-np.rint(closes*100)) <= 1e-6).all(axis=1)
    valid &= np.isfinite(volumes).all(axis=1) & (volumes >= 0).all(axis=1)
    valid &= (volumes == np.floor(volumes)).all(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_volume = np.log1p(volumes)
        x = closes-closes.mean(axis=1, keepdims=True)
        y = log_volume-log_volume.mean(axis=1, keepdims=True)
        denominator = np.sqrt(np.square(x).sum(axis=1)*np.square(y).sum(axis=1))
        value = np.clip(100*(x*y).sum(axis=1)/np.maximum(denominator, 1e-12), -100, 100)
    valid &= np.isfinite(value)
    return np.where(valid, value, np.nan)


def features():
    p = checked_sources(); assert not (ROOT / 'feature_report.json').exists()
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    cols = ['date', 'code', *PRICES, *VOLUMES, *source.HISTORY, 'daily_response_valid']
    history = pd.read_parquet(source.ROOT / 'features.parquet', columns=cols)
    pd.testing.assert_frame_equal(old[['date', 'code']], history[['date', 'code']], check_exact=True)
    f = old.copy()
    for field in cols[2:]:
        f[field] = history[field]
    parts = []
    for start in range(0, len(f), 50000):
        batch = f.iloc[start:start+50000]
        parts.append(correlation(batch[PRICES], batch[VOLUMES]))
    values = np.concatenate(parts)
    good = f.daily_response_valid & np.isfinite(values)
    f['LC01'] = pd.Series(values).where(good)
    f['daily_logcorr_valid'] = good
    f['formula_input_valid'] &= good
    ROOT.mkdir(exist_ok=True)
    f.to_parquet(ROOT / 'features.parquet', index=False, compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL), source_hashes=p['source_hashes'],
        features_sha256=sha(ROOT / 'features.parquet'), rows=len(f), valid=int(f.formula_input_valid.sum()),
        newly_invalid=int((old.formula_input_valid & ~f.formula_input_valid).sum()),
        valid_with_reference_breaks=int((f.formula_input_valid & f.hd_reference_breaks.gt(0)).sum()),
        expressions=EXPRESSIONS, native_header=HEADER, no_new_raw_extraction=True,
        software_compilation_verified=False, new_group_outcomes_read=False,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_report.json', r)
    return {k: v for k, v in r.items() if k not in ['source_hashes', 'expressions', 'native_header']}


def verify_features():
    checked_sources(); r = json.loads((ROOT / 'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['features_sha256'] == sha(ROOT / 'features.parquet')
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    f = pd.read_parquet(ROOT / 'features.parquet')
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')], old.drop(columns='formula_input_valid'), check_exact=True)
    src = pd.read_parquet(source.ROOT / 'features.parquet', columns=['date', 'code', *PRICES, *VOLUMES, *source.HISTORY, 'daily_response_valid'])
    pd.testing.assert_frame_equal(f[src.columns], src, check_exact=True)
    results = []
    flat_price = flat_volume = 0
    for start in range(0, len(src), 50000):
        c = base.conn(); c.register('batch', src.iloc[start:start+50000])
        pairs = ','.join(f'({cl},{vol})' for cl, vol in zip(PRICES, VOLUMES))
        d = c.sql(f'''WITH points AS (SELECT date,code,p,v,
            CASE WHEN isfinite(v) AND v>=0 THEN ln(v+1) END AS l FROM batch,
            LATERAL (VALUES {pairs}) AS pair(p,v)), moments AS (
            SELECT date,code,count(*) AS n,count(*) FILTER(WHERE isfinite(p) AND p>0
                AND abs(p*100-round(p*100))<=1e-6 AND isfinite(v) AND v>=0 AND v=floor(v)) AS good,
                corr(p,l) AS rho,20*var_pop(p) AS xx,20*var_pop(l) AS yy
            FROM points GROUP BY date,code)
            SELECT *,CASE WHEN NOT isfinite(rho) OR xx=0 OR yy=0 THEN 0e0 ELSE
                least(greatest(100*rho*least(sqrt(xx*yy)/1e-12,1),-100),100) END AS val
            FROM moments ORDER BY date,code''').df(); c.close()
        flat_price += int(d.xx.eq(0).sum()); flat_volume += int(d.yy.eq(0).sum())
        results.append(d)
    rebuilt = pd.concat(results, ignore_index=True)
    pd.testing.assert_frame_equal(rebuilt[['date', 'code']], f[['date', 'code']], check_exact=True)
    good = src.daily_response_valid & rebuilt.n.eq(20) & rebuilt.good.eq(20) & np.isfinite(rebuilt.val)
    expected = rebuilt.val.where(good)
    np.testing.assert_array_equal(f.daily_logcorr_valid, good)
    np.testing.assert_allclose(f.LC01, expected, rtol=0, atol=2e-8, equal_nan=True)
    final = old.formula_input_valid & good
    np.testing.assert_array_equal(f.formula_input_valid, final)
    for values in [f.LC01, expected]:
        assert np.isfinite(values[final]).all() and values[final].between(-100, 100).all()
    encode = lambda x: np.floor(np.clip(100*x+10000+1e-6, 0, 999999)).astype(np.int32)
    np.testing.assert_array_equal(encode(f.loc[final, 'LC01']), encode(expected[final]))
    assert (f.loc[final, 'hd_last_date'] < f.loc[final, 'date']).all()
    names = re.findall(r'(?m)^([A-Z][A-Z0-9]*):=', HEADER) + list(EXPRESSIONS)
    assert len(names) == len({x.casefold() for x in names})
    v = dict(passed=True, feature_report_sha256=sha(ROOT / 'feature_report.json'), rows=len(f), valid=int(final.sum()),
        effective_input_intersection_unchanged=bool(final.equals(old.formula_input_valid)),
        all_original_48_values_keys_and_new_encodings_checked=True, strict_prior_source_dates_reused_and_verified=True,
        all_price_log_volume_correlations_independently_sql_rebuilt=True, flat_price_rows=flat_price,
        flat_volume_rows=flat_volume, max_difference=float((f.LC01-expected).abs().max()),
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_verification.json', v)
    return v


def native_value(closes, lots):
    env = {'LN': math.log, 'SQRT': math.sqrt, 'MAX': max, 'MIN': min}
    for i, value in enumerate(closes, 1): env[f'DCP{i}'] = float(value)
    for i, value in enumerate(lots, 1): env[f'HDV{i:02d}'] = float(value)
    for line in EXTRA_HEADER.strip().split(';'):
        if line.strip():
            name, expr = line.strip().split(':=')
            env[name] = eval(expr, {'__builtins__': {}}, env)
    return float(eval(NEW_EXPRESSIONS['LC01'], {'__builtins__': {}}, env))


def verify_native():
    checked_sources(); v = json.loads((ROOT / 'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(ROOT / 'feature_report.json')
    prior = source.volumes.ROOT / 'native_input_verification.json'
    proof = json.loads(prior.read_text())
    assert proof['passed'] and proof['samples'] == 32 and proof['maximum_daily_volume_difference_shares'] == 0
    assert proof['raw_close_and_volume_checked_against_daily_source']
    f = pd.read_parquet(ROOT / 'features.parquet').set_index(['date', 'code']); cases = []
    for case in proof['cases']:
        row = f.loc[(case['date'], case['code'])]; valid = bool(row.daily_logcorr_valid)
        if valid:
            prices = row[PRICES].to_numpy(float); volumes = row[VOLUMES].to_numpy(float)
            m = math.fsum(prices)/20; logs = [math.log1p(x) for x in volumes]; n = math.fsum(logs)/20
            x = [p-m for p in prices]; y = [z-n for z in logs]
            scalar = min(max(100*math.fsum(a*b for a,b in zip(x,y))/max(math.sqrt(math.fsum(a*a for a in x)*math.fsum(b*b for b in y)),1e-12),-100),100)
            native = native_value(prices, volumes/100)
            np.testing.assert_allclose([scalar, native], row.LC01, rtol=0, atol=2e-8)
            np.testing.assert_array_equal(np.floor(100*np.array([scalar, native])+10000+1e-6), np.repeat(np.floor(100*row.LC01+10000+1e-6),2))
        else:
            assert pd.isna(row.LC01)
        cases.append(dict(date=case['date'], code=case['code'], valid=valid))
    r = dict(passed=True, protocol_sha256=sha(PROTOCOL), feature_report_sha256=sha(ROOT / 'feature_report.json'),
        feature_verification_sha256=sha(ROOT / 'feature_verification.json'), prior_day_native_proof_sha256=sha(prior),
        source_cases_reused=32, cases=cases, actual_new_expressions_and_integer_encodings_rebuilt=True,
        shares_and_native_lots_explicitly_converted=True, no_new_minute_extraction=True,
        software_compilation_verified=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'native_input_verification.json', r)
    return {k: v for k, v in r.items() if k != 'cases'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['features', 'verify_features', 'verify_native'])
    print(json.dumps(globals()[p.parse_args().stage](), ensure_ascii=False, indent=2))
