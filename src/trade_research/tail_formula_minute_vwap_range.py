"""Distances to active minute amount/volume extrema, with the proven input domain."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_morning_range as prior
from . import tail_formula_minute_vwap as source
from .corporate_cash import MINUTES, save_json, sha

STEM = 'tail_formula_minute_vwap_range'
ROOT = Path('data/research') / STEM
INPUTS = ROOT / 'inputs'
PROTOCOL = Path('config') / (STEM + '_input_protocol.json')
INTENT = Path('config') / (STEM + '_intent.json')
META = prior.META
EXTRA_HEADER = '''MVAV:=ROUND(AMOUNT*100+0.000001)/MAX(ROUND(V*100),1);
MVHC:=VALUEWHEN(TIME=1449,HHV(IF(V>0,MVAV,0),29));
MVLC:=VALUEWHEN(TIME=1449,LLV(IF(V>0,MVAV,999999999),29));
MVRV29:=VALUEWHEN(TIME=1449,SUM(V,29));
MVRV4:=VALUEWHEN(TIME=1449,SUM(V,4));
MVRREADY:=AMREADY AND RTREADY AND VALUEWHEN(TIME=1449,DATE)=DATE AND MVRV29>0 AND MVRV4>0 AND MVRV29-MVRV4>0 AND MVHC>0 AND MVLC>0 AND MVLC<999999999;
'''
HEADER = prior.HEADER + EXTRA_HEADER
CORE_GATE = 'MVRREADY'
NEW_EXPRESSIONS = {
    'MVH': 'IF(MVRREADY,100*(ROUND(100*Q)/MVHC-1)/VP20,DRAWNULL)',
    'MVL': 'IF(MVRREADY,100*(ROUND(100*Q)/MVLC-1)/VP20,DRAWNULL)'}
ARMS = {'control': prior.EXPRESSIONS, 'range': {**prior.EXPRESSIONS, **NEW_EXPRESSIONS}}


def checked():
    source.checked(); p = json.loads(PROTOCOL.read_text())
    assert p['intent_sha256'] == sha(INTENT) and p['arms'] == ARMS and p['native_header'] == HEADER
    assert p['expected_keys'] == 1258085 and p['expected_valid'] == 1117396
    assert p['no_new_production_raw_extraction'] and not p['new_2026_prices_allowed']
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest, file
    complete = json.loads(Path(p['conditional_completion']).read_text())
    gate = json.loads(Path(p['conditional_gate']).read_text())
    assert complete['passed'] and gate['passed'] and not gate['supports_2024_extension']
    fr = json.loads((source.INPUTS / 'feature_report.json').read_text())
    fv = json.loads((source.INPUTS / 'feature_verification.json').read_text())
    assert fv['passed'] and fv['feature_report_sha256'] == sha(source.INPUTS / 'feature_report.json')
    assert fr['features_sha256'] == sha(source.INPUTS / 'features.parquet') and fv['valid'] == 1117396
    return p


def measure(volume, amount, quote, atr, parent_valid):
    assert volume.shape == amount.shape and volume.shape[1] == 29
    assert quote.shape == atr.shape == parent_valid.shape == (len(volume),)
    good = parent_valid.copy()
    good &= np.isfinite(volume).all(axis=1) & np.isfinite(amount).all(axis=1)
    good &= (volume >= 0).all(axis=1) & (volume == np.floor(volume)).all(axis=1)
    good &= (amount >= 0).all(axis=1) & ((volume == 0) == (amount == 0)).all(axis=1)
    good &= np.isfinite(quote) & (quote > 0) & np.isfinite(atr) & (atr > 0)
    with np.errstate(all='ignore'):
        ac = np.floor(amount * 100 + .5 + .000001)
        av = np.divide(ac, volume, out=np.full_like(amount, np.nan), where=volume > 0)
        highest = np.max(np.where(volume > 0, av, -np.inf), axis=1)
        lowest = np.min(np.where(volume > 0, av, np.inf), axis=1)
        good &= np.isfinite(highest) & np.isfinite(lowest) & (lowest > 0) & (highest < 999999999)
        q = np.floor(100 * quote + .5)
        good &= (np.abs(100 * quote - q) <= .01) & (ac < 2**53).all(axis=1)
        high_distance = 100 * (q / highest - 1) / atr
        low_distance = 100 * (q / lowest - 1) / atr
    return dict(mvr_input_valid=good, mvr_high_cent=highest, mvr_low_cent=lowest,
                MVH=np.where(good, high_distance, np.nan), MVL=np.where(good, low_distance, np.nan))


def features():
    checked(); INPUTS.mkdir(parents=True, exist_ok=True)
    assert not (INPUTS / 'feature_report.json').exists()
    old = pd.read_parquet(source.INPUTS / 'features.parquet')
    lookup = old.set_index(['date', 'code']); pieces = []
    for i, (p, b, a) in enumerate(source.cache_parts(), 1):
        keys = pd.MultiIndex.from_frame(b[['date', 'code']]); matched = lookup.reindex(keys)
        assert len(matched) == len(b)
        close, volume, high, low = [b[source.bars.COLUMNS[k]].to_numpy(float) for k in ['c', 'v', 'h', 'l']]
        np.testing.assert_array_equal(p[source.quotes.PRICE_COLUMNS[1:]].to_numpy(float), close)
        amount = a[source.amounts.amounts.COLUMNS].to_numpy(float)
        good = source.measure(close, volume, amount, high, low)['mv_input_valid']
        for frame, prefix, count in [(p, 'pv', 30), (b, 'mp', 29), (a, 'il', 29)]:
            for field in ['bars', 'clocks', 'good_bars']:
                good &= frame[prefix + '_' + field].eq(count).to_numpy()
        np.testing.assert_array_equal(good, matched.mv_input_valid.eq(True))
        values = measure(volume, amount, matched.A04.to_numpy(float), matched.V01.to_numpy(float), good)
        d = b[['date', 'code']].copy()
        for name, value in values.items():
            d[name] = value
        pieces.append(d); print(json.dumps(dict(computed_parts=i, total_parts=50)), flush=True)
    f = old.merge(pd.concat(pieces, ignore_index=True), on=['date', 'code'], validate='one_to_one', how='left', sort=False)
    pd.testing.assert_frame_equal(f[old.columns], old, check_exact=True)
    f['formula_input_valid'] &= f.mvr_input_valid.eq(True)
    np.testing.assert_array_equal(f.formula_input_valid, old.formula_input_valid)
    assert len(f) == 1258085 and f.formula_input_valid.sum() == 1117396
    f.to_parquet(INPUTS / 'features.parquet', index=False, compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL), features_sha256=sha(INPUTS / 'features.parquet'), rows=len(f),
        valid=int(f.formula_input_valid.sum()), domain_reference=str(source.INPUTS),
        parent_feature_report_sha256=sha(source.INPUTS / 'feature_report.json'),
        effective_domain_exactly_parent=True, all_parent_values_metadata_and_keys_unchanged=True,
        unused_parent_direction_columns_retained_as_provenance_only=True, expressions=ARMS['range'], native_header=HEADER,
        no_new_production_raw_extraction=True, new_group_outcomes_read=False, new_2026_prices_read=False,
        no_exit_rules=True, software_compilation_verified=False, native_source_parity_verified=False)
    save_json(INPUTS / 'feature_report.json', r)
    for name in ['full_labels.parquet', 'full_label_report.json', 'full_label_verification.json']:
        (INPUTS / name).symlink_to((source.INPUTS / name).resolve())
    return {k: r[k] for k in ['rows', 'valid', 'effective_domain_exactly_parent']}


def verify():
    checked(); r = json.loads((INPUTS / 'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['features_sha256'] == sha(INPUTS / 'features.parquet')
    old = pd.read_parquet(source.INPUTS / 'features.parquet'); f = pd.read_parquet(INPUTS / 'features.parquet')
    pd.testing.assert_frame_equal(f[old.columns], old, check_exact=True)
    c = base.conn(); c.execute('SET threads=1'); c.execute("SET memory_limit='3GB'"); pieces = []
    for i, (_, b, a) in enumerate(source.cache_parts(), 1):
        wide = a.merge(b[['date', 'code']], on=['date', 'code'], validate='one_to_one').merge(
            old[['date', 'code', 'A04', 'V01', 'mv_input_valid']], on=['date', 'code'], validate='one_to_one')
        c.register('wide', wide)
        # Rebuild the new arithmetic independently. The identical parent's full atomic quality proof is pinned.
        c.register('vwide', wide.merge(b[['date', 'code', *source.bars.COLUMNS['v']]], on=['date', 'code'], validate='one_to_one'))
        c.sql(' UNION ALL '.join(f'SELECT date,code,mp_v{j:02d} AS v,il_a{j:02d} AS a FROM vwide' for j in range(21, 50))).create_view('long', replace=True)
        d = c.sql('''WITH z AS(SELECT date,code,
            max(CASE WHEN v>0 THEN round(a*100+.000001)/v END) AS hc,
            min(CASE WHEN v>0 THEN round(a*100+.000001)/v END) AS lc,
            bool_and(coalesce(isfinite(v) AND isfinite(a) AND v>=0 AND v=floor(v) AND a>=0
                AND (v=0)=(a=0) AND round(a*100+.000001)<9007199254740992,false)) AS atoms_good
            FROM long GROUP BY date,code), k AS(SELECT *,coalesce(mv_input_valid AND atoms_good
                AND isfinite(A04) AND A04>0 AND abs(A04*100-round(A04*100))<=.01
                AND isfinite(V01) AND V01>0 AND isfinite(hc) AND isfinite(lc) AND lc>0 AND hc<999999999,false) AS ok
            FROM z JOIN wide USING(date,code)) SELECT date,code,ok,
            CASE WHEN ok THEN 100.*(round(A04*100)/hc-1)/V01 END AS MVH,
            CASE WHEN ok THEN 100.*(round(A04*100)/lc-1)/V01 END AS MVL FROM k ORDER BY date,code''').df()
        pieces.append(d); print(json.dumps(dict(verified_parts=i, total_parts=50)), flush=True)
    c.close()
    expected = f[['date', 'code']].merge(pd.concat(pieces, ignore_index=True), on=['date', 'code'], how='left', validate='one_to_one')
    expected['ok'] = expected.ok.eq(True)
    np.testing.assert_array_equal(f.mvr_input_valid.eq(True), expected.ok)
    maximum = 0.; encode = lambda x: np.floor(np.clip(100 * x + 10000 + .000001, 0, 999999))
    for name in NEW_EXPRESSIONS:
        np.testing.assert_allclose(f[name], expected[name], rtol=0, atol=2e-10, equal_nan=True)
        np.testing.assert_array_equal(encode(f.loc[expected.ok, name]), encode(expected.loc[expected.ok, name]))
        maximum = max(maximum, float((f[name] - expected[name]).abs().max()))
    np.testing.assert_array_equal(f.formula_input_valid, old.formula_input_valid & expected.ok)
    out = dict(passed=True, feature_report_sha256=sha(INPUTS / 'feature_report.json'), rows=len(f),
        valid=int(f.formula_input_valid.sum()), max_difference=maximum,
        all_active_minute_amount_extrema_distances_and_codes_sql_rebuilt=True,
        unchanged_atomic_quality_proof_reused_after_full_parent_frame_equality=True,
        parent_feature_verification_sha256=sha(source.INPUTS / 'feature_verification.json'),
        effective_input_intersection_unchanged=True, domain_reference=str(source.INPUTS),
        original_50_domain_has_one_previously_verified_evaluation_only_delta=True,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(INPUTS / 'feature_verification.json', out); return out


def native():
    checked(); proof = json.loads((INPUTS / 'feature_verification.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256'] == sha(INPUTS / 'feature_report.json')
    f = pd.read_parquet(INPUTS / 'features.parquet').set_index(['date', 'code'])
    old = json.loads((source.amounts.ROOT / 'native_input_verification.json').read_text()); cases = []
    assert old['passed'] and len(old['samples']) == 32
    for sample in old['samples']:
        day, code = sample['date'], sample['code']; file = MINUTES / code[:2].upper() / (code[3:] + '.parquet')
        assert sha(file) == sample['source_sha256']
        d = pd.read_parquet(file, columns=['timestamp', 'close', 'volume', 'turnover'], filters=[
            ('timestamp', '>=', pd.Timestamp(day + ' 14:21')), ('timestamp', '<=', pd.Timestamp(day + ' 14:49'))]).sort_values('timestamp')
        assert d.timestamp.dt.strftime('%H%M').tolist() == [f'14{j:02d}' for j in range(21, 50)]
        averages = []
        for row in d.itertuples():
            lots = float(row.volume) / 100
            if lots > 0:
                averages.append(np.floor(float(row.turnover) * 100 + .5 + .000001) / np.floor(lots * 100 + .5))
        target_row = f.loc[(day, code)]; q = np.floor(float(d.iloc[-1].close) * 100 + .5)
        got = 100 * (q / np.array([max(averages), min(averages)]) - 1) / target_row.V01
        target = target_row[list(NEW_EXPRESSIONS)].to_numpy(float)
        np.testing.assert_allclose(got, target, atol=2e-10, rtol=0)
        enc = lambda x: np.floor(np.clip(100 * np.array(x) + 10000 + .000001, 0, 999999))
        np.testing.assert_array_equal(enc(got), enc(target))
        cases.append(dict(**sample, reused_original_source_identity=True, native_lots_and_active_extrema_replayed=True))
    out = dict(passed=True, feature_report_sha256=sha(INPUTS / 'feature_report.json'),
        feature_verification_sha256=sha(INPUTS / 'feature_verification.json'), samples=cases,
        existing_32_source_cases_replayed_for_new_kernel_only=True, no_new_production_raw_extraction=True,
        all_new_full_values_and_codes_sql_rebuilt=True, software_compilation_verified=False,
        native_source_parity_verified=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(INPUTS / 'native_input_verification.json', out); return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['features', 'verify', 'native']); args = parser.parse_args()
    print(json.dumps(globals()[args.stage](), ensure_ascii=False, indent=2))
