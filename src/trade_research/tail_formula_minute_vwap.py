"""Minute close versus that minute's actual amount/volume, using proven caches."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_morning_range as prior
from . import tail_formula_path_variance as quotes
from . import tail_formula_minute_pressure as bars
from . import tail_formula_price_impact as amounts
from .corporate_cash import MINUTES, save_json, sha

STEM = 'tail_formula_minute_vwap'
ROOT = Path('data/research') / STEM
INPUTS = ROOT / 'inputs'
PROTOCOL = Path('config') / (STEM + '_input_protocol.json')
INTENT = Path('config') / (STEM + '_intent.json')
META = prior.META
EXTRA_HEADER = '''MVDC:=ROUND(C*100)*ROUND(V*100)-ROUND(AMOUNT*100+0.000001);
MVSG:=IF(MVDC>0,1,IF(MVDC<0,-1,0));
MVV29:=VALUEWHEN(TIME=1449,SUM(V,29));
MVV4:=VALUEWHEN(TIME=1449,SUM(V,4));
MVS29:=VALUEWHEN(TIME=1449,SUM(V*MVSG,29));
MVS4:=VALUEWHEN(TIME=1449,SUM(V*MVSG,4));
MVREADY:=AMREADY AND VALUEWHEN(TIME=1449,DATE)=DATE AND MVV29>0 AND MVV4>0 AND MVV29-MVV4>0;
'''
HEADER = prior.HEADER + EXTRA_HEADER
CORE_GATE = 'MVREADY'
NEW_EXPRESSIONS = {
    'MV01': 'IF(MVREADY,100*MVS29/MAX(MVV29,0.000001),DRAWNULL)',
    'MV02': 'IF(MVREADY,100*(MVS4/MAX(MVV4,0.000001)-(MVS29-MVS4)/MAX(MVV29-MVV4,0.000001)),DRAWNULL)'}
ARMS = {'control': prior.EXPRESSIONS, 'vwap': {**prior.EXPRESSIONS, **NEW_EXPRESSIONS}}
SOURCES = [quotes.ROOT, bars.ROOT, amounts.ROOT]


def checked():
    prior.checked(); p = json.loads(PROTOCOL.read_text())
    assert p['intent_sha256'] == sha(INTENT) and p['arms'] == ARMS and p['native_header'] == HEADER
    assert p['expected_keys'] == 1258085 and p['no_new_production_raw_extraction']
    assert not p['new_2026_prices_allowed']
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest, file
    complete = json.loads(Path(p['conditional_completion']).read_text())
    gate = json.loads(Path(p['conditional_gate']).read_text())
    assert complete['passed'] and gate['passed'] and not gate['supports_2024_extension']
    for root in SOURCES:
        r = json.loads((root / 'window_report.json').read_text())
        for file, digest in r['parts_sha256'].items():
            assert sha(Path(file)) == digest, file
    return p


def cache_parts():
    plans = [json.loads((s / 'window_report.json').read_text()) for s in SOURCES]
    assert all(len(p['parts_sha256']) == 50 for p in plans)
    for files in zip(*(p['parts_sha256'] for p in plans)):
        frames = [pd.read_parquet(f) for f in files]
        for f in frames[1:]:
            pd.testing.assert_frame_equal(frames[0][['date', 'code']], f[['date', 'code']], check_exact=True)
        for f in frames:
            assert not f.duplicated(['date', 'code']).any()
        yield frames


def measure(close, volume, amount, high, low):
    assert close.shape == volume.shape == amount.shape == high.shape == low.shape and close.shape[1] == 29
    good = np.isfinite(np.stack([close, volume, amount, high, low])).all(axis=(0, 2))
    good &= (close > 0).all(axis=1) & (high >= close).all(axis=1) & (close >= low).all(axis=1) & (low > 0).all(axis=1)
    good &= (np.abs(close * 100 - np.floor(close * 100 + .5)) <= .01).all(axis=1)
    good &= (volume >= 0).all(axis=1) & (volume == np.floor(volume)).all(axis=1) & (amount >= 0).all(axis=1)
    good &= ((volume == 0) == (amount == 0)).all(axis=1)
    with np.errstate(all='ignore'):
        average = np.divide(amount, volume, out=np.full_like(amount, np.nan), where=volume > 0)
        good &= ((volume == 0) | ((average >= low - .0101) & (average <= high + .0101))).all(axis=1)
        # Same cent epsilon as the native expression, for double half-cent ties.
        pc = np.floor(close * 100 + .5); ac = np.floor(amount * 100 + .5 + .000001)
        good &= (np.abs(pc * volume) < 2**53).all(axis=1) & (ac < 2**53).all(axis=1)
        direction = np.sign(pc * volume - ac)
        v29, v4 = volume.sum(axis=1), volume[:, -4:].sum(axis=1)
        s29, s4 = (volume * direction).sum(axis=1), (volume[:, -4:] * direction[:, -4:]).sum(axis=1)
        good &= (v29 > 0) & (v4 > 0) & (v29 - v4 > 0)
        first = 100 * s29 / v29
        second = 100 * (s4 / v4 - (s29 - s4) / (v29 - v4))
    return dict(mv_input_valid=good, mv_v29=v29, mv_v4=v4, mv_s29=s29, mv_s4=s4,
                MV01=np.where(good, first, np.nan), MV02=np.where(good, second, np.nan))


def features():
    checked(); INPUTS.mkdir(parents=True, exist_ok=True)
    assert not (INPUTS / 'feature_report.json').exists()
    old = pd.read_parquet(prior.INPUTS / 'features.parquet'); pieces = []
    for i, (p, b, a) in enumerate(cache_parts(), 1):
        close, volume, high, low = [b[bars.COLUMNS[k]].to_numpy(float) for k in ['c', 'v', 'h', 'l']]
        np.testing.assert_array_equal(p[quotes.PRICE_COLUMNS[1:]].to_numpy(float), close)
        values = measure(close, volume, a[amounts.amounts.COLUMNS].to_numpy(float), high, low)
        good = values['mv_input_valid']
        for f, prefix, count in [(p, 'pv', 30), (b, 'mp', 29), (a, 'il', 29)]:
            for field in ['bars', 'clocks', 'good_bars']:
                good &= f[prefix + '_' + field].eq(count).to_numpy()
        d = b[['date', 'code']].copy()
        for name, value in values.items():
            d[name] = value
        d.loc[~good, list(NEW_EXPRESSIONS)] = np.nan
        pieces.append(d); print(json.dumps(dict(computed_parts=i, total_parts=50)), flush=True)
    a = pd.concat(pieces, ignore_index=True)
    f = old.merge(a, on=['date', 'code'], how='left', validate='one_to_one', sort=False)
    pd.testing.assert_frame_equal(f[old.columns], old, check_exact=True)
    f['mv_prior_formula_input_valid'] = old.formula_input_valid
    f['formula_input_valid'] &= f.mv_input_valid.eq(True)
    assert len(f) == 1258085
    f.to_parquet(INPUTS / 'features.parquet', index=False, compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL), features_sha256=sha(INPUTS / 'features.parquet'),
        original_feature_report_sha256=sha(prior.INPUTS / 'feature_report.json'), rows=len(f), valid=int(f.formula_input_valid.sum()),
        newly_invalid=int((f.mv_prior_formula_input_valid & ~f.formula_input_valid).sum()),
        all_existing_complete_minute_caches_reused=True, no_new_production_raw_extraction=True,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True,
        expressions=ARMS['vwap'], native_header=HEADER, software_compilation_verified=False, native_source_parity_verified=False)
    save_json(INPUTS / 'feature_report.json', r)
    for name in ['full_labels.parquet', 'full_label_report.json', 'full_label_verification.json']:
        (INPUTS / name).symlink_to((prior.INPUTS / name).resolve())
    return {k: r[k] for k in ['rows', 'valid', 'newly_invalid']}


def verify():
    checked(); r = json.loads((INPUTS / 'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['features_sha256'] == sha(INPUTS / 'features.parquet')
    old = pd.read_parquet(prior.INPUTS / 'features.parquet'); f = pd.read_parquet(INPUTS / 'features.parquet')
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')], old.drop(columns='formula_input_valid'), check_exact=True)
    c = base.conn(); c.execute('SET threads=1'); c.execute("SET memory_limit='3GB'"); pieces = []
    for i, (p, b, a) in enumerate(cache_parts(), 1):
        wide = b.merge(a, on=['date', 'code'], validate='one_to_one').merge(p[['date', 'code', 'pv_bars', 'pv_clocks', 'pv_good_bars']], on=['date', 'code'], validate='one_to_one')
        c.register('wide', wide)
        queries = [f'''SELECT date,code,{j} AS pos,mp_c{j:02d} AS pc,mp_v{j:02d} AS v,mp_h{j:02d} AS h,
            mp_l{j:02d} AS l,il_a{j:02d} AS a FROM wide''' for j in range(21, 50)]
        c.sql(' UNION ALL '.join(queries)).create_view('long', replace=True)
        c.sql('''SELECT *,round(pc*100)*v-round(a*100+.000001) AS dc,
            coalesce(isfinite(pc) AND isfinite(v) AND isfinite(h) AND isfinite(l) AND isfinite(a)
            AND pc>0 AND h>=pc AND pc>=l AND l>0 AND abs(pc*100-round(pc*100))<=.01
            AND v>=0 AND v=floor(v) AND a>=0 AND (v=0)=(a=0)
            AND (v=0 OR a/v BETWEEN l-.0101 AND h+.0101)
            AND abs(round(pc*100)*v)<9007199254740992 AND round(a*100+.000001)<9007199254740992,false) AS good
            FROM long''').create_view('atoms', replace=True)
        d = c.sql('''WITH z AS(SELECT date,code,bool_and(good) AS good,
            sum(v) AS v29,sum(v) FILTER(WHERE pos>=46) AS v4,
            sum(v*CASE WHEN dc>0 THEN 1 WHEN dc<0 THEN -1 ELSE 0 END) AS s29,
            sum(v*CASE WHEN dc>0 THEN 1 WHEN dc<0 THEN -1 ELSE 0 END) FILTER(WHERE pos>=46) AS s4
            FROM atoms GROUP BY date,code), k AS(SELECT *,good AND v29>0 AND v4>0 AND v29-v4>0
            AND pv_bars=30 AND pv_clocks=30 AND pv_good_bars=30
            AND mp_bars=29 AND mp_clocks=29 AND mp_good_bars=29
            AND il_bars=29 AND il_clocks=29 AND il_good_bars=29 AS ok FROM z JOIN wide USING(date,code))
            SELECT date,code,ok,CASE WHEN ok THEN 100.*s29/v29 END AS MV01,
            CASE WHEN ok THEN 100.*(s4/v4-(s29-s4)/(v29-v4)) END AS MV02 FROM k ORDER BY date,code''').df()
        pieces.append(d); print(json.dumps(dict(verified_parts=i, total_parts=50)), flush=True)
    c.close(); expected = f[['date', 'code']].merge(pd.concat(pieces, ignore_index=True), on=['date', 'code'], validate='one_to_one', how='left')
    expected['ok'] = expected.ok.eq(True)
    np.testing.assert_array_equal(f.mv_input_valid.eq(True), expected.ok)
    maximum = 0.; encode = lambda x: np.floor(np.clip(100 * x + 10000 + .000001, 0, 999999))
    for name in NEW_EXPRESSIONS:
        np.testing.assert_allclose(f[name], expected[name], rtol=0, atol=2e-12, equal_nan=True)
        np.testing.assert_array_equal(encode(f.loc[expected.ok, name]), encode(expected.loc[expected.ok, name]))
        maximum = max(maximum, float((f[name] - expected[name]).abs().max()))
    good = old.formula_input_valid & expected.ok
    np.testing.assert_array_equal(f.formula_input_valid, good)
    out = dict(passed=True, feature_report_sha256=sha(INPUTS / 'feature_report.json'), rows=len(f), valid=int(good.sum()),
        max_difference=maximum, all_original_50_values_and_metadata_unchanged=True,
        all_minute_amount_rounding_volume_directions_quality_and_encodings_sql_rebuilt=True,
        effective_input_intersection_unchanged=bool(good.equals(old.formula_input_valid)),
        same_quality_controls_required_before_fitting=not good.equals(old.formula_input_valid),
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(INPUTS / 'feature_verification.json', out); return out


def native():
    checked(); proof = json.loads((INPUTS / 'feature_verification.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256'] == sha(INPUTS / 'feature_report.json')
    f = pd.read_parquet(INPUTS / 'features.parquet').set_index(['date', 'code'])
    old = json.loads((amounts.ROOT / 'native_input_verification.json').read_text()); cases = []
    assert old['passed'] and len(old['samples']) == 32
    for sample in old['samples']:
        day, code = sample['date'], sample['code']; file = MINUTES / code[:2].upper() / (code[3:] + '.parquet')
        assert sha(file) == sample['source_sha256']
        d = pd.read_parquet(file, columns=['timestamp', 'close', 'volume', 'turnover'], filters=[
            ('timestamp', '>=', pd.Timestamp(day + ' 14:21')), ('timestamp', '<=', pd.Timestamp(day + ' 14:49'))]).sort_values('timestamp')
        assert d.timestamp.dt.strftime('%H%M').tolist() == [f'14{j:02d}' for j in range(21, 50)]
        value, lots = [], []
        for row in d.itertuples():
            v = float(row.volume) / 100
            dc = np.floor(float(row.close) * 100 + .5) * np.floor(v * 100 + .5) - np.floor(float(row.turnover) * 100 + .5 + .000001)
            value.append(v * (1 if dc > 0 else -1 if dc < 0 else 0)); lots.append(v)
        v29, v4, s29, s4 = sum(lots), sum(lots[-4:]), sum(value), sum(value[-4:])
        got = [100 * s29 / v29, 100 * (s4 / v4 - (s29 - s4) / (v29 - v4))]
        target = f.loc[(day, code), list(NEW_EXPRESSIONS)].to_numpy(float)
        np.testing.assert_allclose(got, target, atol=2e-12, rtol=0)
        enc = lambda x: np.floor(np.clip(100 * np.array(x) + 10000 + .000001, 0, 999999))
        np.testing.assert_array_equal(enc(got), enc(target))
        cases.append(dict(**sample, reused_original_source_identity=True, native_lots_and_amount_cents_replayed=True))
    out = dict(passed=True, feature_report_sha256=sha(INPUTS / 'feature_report.json'),
        feature_verification_sha256=sha(INPUTS / 'feature_verification.json'), samples=cases,
        existing_32_source_cases_replayed_for_new_kernel_only=True, no_new_production_raw_extraction=True,
        all_full_values_and_codes_sql_rebuilt=True, software_compilation_verified=False,
        native_source_parity_verified=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(INPUTS / 'native_input_verification.json', out); return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['features', 'verify', 'native']); args = parser.parse_args()
    print(json.dumps(globals()[args.stage](), ensure_ascii=False, indent=2))
