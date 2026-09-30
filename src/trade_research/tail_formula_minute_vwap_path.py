"""Complete ordered contributions of actual minute amounts relative to the quote."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_morning_range as prior
from . import tail_formula_minute_vwap as source
from .corporate_cash import MINUTES, save_json, sha

STEM = 'tail_formula_minute_vwap_path'
ROOT = Path('data/research') / STEM
INPUTS = ROOT / 'inputs'
PROTOCOL = Path('config') / (STEM + '_input_protocol.json')
INTENT = Path('config') / (STEM + '_intent.json')
META = prior.META
EXTRA_HEADER = '''MUQC:=ROUND(100*Q);
MUV29:=VALUEWHEN(TIME=1449,ROUND(100*SUM(V,29)));
MUV4:=VALUEWHEN(TIME=1449,ROUND(100*SUM(V,4)));
MUREADY:=AMREADY AND RTREADY AND VALUEWHEN(TIME=1449,DATE)=DATE AND MUQC>0 AND MUV29>0 AND MUV4>0 AND MUV29-MUV4>0;
'''
HEADER = prior.HEADER + EXTRA_HEADER
CORE_GATE = 'MUREADY'
NEW_EXPRESSIONS = {
    f'MU{i:02d}': f'IF(MUREADY,2900*(MUQC*VALUEWHEN(TIME=1449,ROUND(100*REF(V,{i})))-VALUEWHEN(TIME=1449,ROUND(100*REF(AMOUNT,{i})+0.000001)))/(MUQC*MUV29)/VP20,DRAWNULL)'
    for i in range(29)}
ARMS = {'control': prior.EXPRESSIONS, 'path': {**prior.EXPRESSIONS, **NEW_EXPRESSIONS}}


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
        ac = np.floor(amount * 100 + .5 + .000001); q = np.floor(quote * 100 + .5)
        total = volume.sum(axis=1); product = q[:, None] * volume; denominator = q * total
        good &= (total > 0) & (denominator < 2**53) & (np.abs(quote * 100 - q) <= .01)
        good &= (product < 2**53).all(axis=1) & (ac < 2**53).all(axis=1)
        contribution = 2900 * (product - ac) / denominator[:, None] / atr[:, None]
    return dict(mu_input_valid=good, **{name: np.where(good, contribution[:, 28-i], np.nan)
                for i, name in enumerate(NEW_EXPRESSIONS)})


def features():
    checked(); INPUTS.mkdir(parents=True, exist_ok=True)
    assert not (INPUTS / 'feature_report.json').exists()
    old = pd.read_parquet(source.INPUTS / 'features.parquet'); lookup = old.set_index(['date', 'code']); pieces = []
    for i, (p, b, a) in enumerate(source.cache_parts(), 1):
        matched = lookup.reindex(pd.MultiIndex.from_frame(b[['date', 'code']]))
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
        for name, value in values.items(): d[name] = value
        pieces.append(d); print(json.dumps(dict(computed_parts=i, total_parts=50)), flush=True)
    f = old.merge(pd.concat(pieces, ignore_index=True), on=['date', 'code'], validate='one_to_one', how='left', sort=False)
    pd.testing.assert_frame_equal(f[old.columns], old, check_exact=True)
    f['formula_input_valid'] &= f.mu_input_valid.eq(True)
    np.testing.assert_array_equal(f.formula_input_valid, old.formula_input_valid)
    assert len(f) == 1258085 and f.formula_input_valid.sum() == 1117396
    f.to_parquet(INPUTS / 'features.parquet', index=False, compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL), features_sha256=sha(INPUTS / 'features.parquet'), rows=len(f),
        valid=int(f.formula_input_valid.sum()), domain_reference=str(source.INPUTS),
        parent_feature_report_sha256=sha(source.INPUTS / 'feature_report.json'),
        effective_domain_exactly_parent=True, all_parent_values_metadata_and_keys_unchanged=True,
        unused_parent_direction_columns_retained_as_provenance_only=True, expressions=ARMS['path'], native_header=HEADER,
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
        wide = a.merge(b[['date', 'code', *source.bars.COLUMNS['v']]], on=['date', 'code'], validate='one_to_one').merge(
            old[['date', 'code', 'A04', 'V01', 'mv_input_valid']], on=['date', 'code'], validate='one_to_one')
        c.register('wide', wide)
        c.sql(' UNION ALL '.join(f'SELECT date,code,{49-j} AS lag,mp_v{j:02d} AS v,il_a{j:02d} AS a FROM wide' for j in range(21, 50))).create_view('long', replace=True)
        c.sql('''SELECT *,sum(v) OVER(PARTITION BY date,code) AS vt,
            bool_and(coalesce(isfinite(v) AND isfinite(a) AND v>=0 AND v=floor(v) AND a>=0
                AND (v=0)=(a=0) AND round(a*100+.000001)<9007199254740992,false))
                OVER(PARTITION BY date,code) AS atoms_good FROM long''').create_view('totals', replace=True)
        c.sql('''SELECT *,coalesce(mv_input_valid AND atoms_good AND vt>0
                AND isfinite(A04) AND A04>0 AND abs(A04*100-round(A04*100))<=.01
                AND isfinite(V01) AND V01>0 AND round(A04*100)*vt<9007199254740992,false) AS ok
            FROM totals JOIN wide USING(date,code)''').create_view('atoms', replace=True)
        c.sql('''SELECT date,code,lag,ok,CASE WHEN ok THEN
            2900.*(round(A04*100)*v-round(a*100+.000001))/(round(A04*100)*vt)/V01 END AS contribution
            FROM atoms''').create_view('values_long', replace=True)
        fields = ','.join(f'max(contribution) FILTER(WHERE lag={j}) AS {name}' for j, name in enumerate(NEW_EXPRESSIONS))
        d = c.sql(f'SELECT date,code,bool_and(ok) AS ok,{fields} FROM values_long GROUP BY date,code ORDER BY date,code').df()
        pieces.append(d); print(json.dumps(dict(verified_parts=i, total_parts=50)), flush=True)
    c.close()
    expected = f[['date', 'code']].merge(pd.concat(pieces, ignore_index=True), on=['date', 'code'], how='left', validate='one_to_one')
    expected['ok'] = expected.ok.eq(True)
    np.testing.assert_array_equal(f.mu_input_valid.eq(True), expected.ok)
    maximum = 0.; encode = lambda x: np.floor(np.clip(100 * x + 10000 + .000001, 0, 999999))
    for name in NEW_EXPRESSIONS:
        np.testing.assert_allclose(f[name], expected[name], rtol=0, atol=2e-10, equal_nan=True)
        np.testing.assert_array_equal(encode(f.loc[expected.ok, name]), encode(expected.loc[expected.ok, name]))
        maximum = max(maximum, float((f[name] - expected[name]).abs().max()))
    np.testing.assert_array_equal(f.formula_input_valid, old.formula_input_valid & expected.ok)
    out = dict(passed=True, feature_report_sha256=sha(INPUTS / 'feature_report.json'), rows=len(f),
        valid=int(f.formula_input_valid.sum()), max_difference=maximum,
        all_29_amount_contributions_order_zero_semantics_and_codes_sql_rebuilt=True,
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
        target_row = f.loc[(day, code)]; q = np.floor(float(d.iloc[-1].close) * 100 + .5)
        lots = d.volume.to_numpy(float) / 100; total = np.floor(sum(lots) * 100 + .5); got = []
        for i in range(29):
            row = d.iloc[28-i]; shares = np.floor(lots[28-i] * 100 + .5)
            ac = np.floor(float(row.turnover) * 100 + .5 + .000001)
            got.append(2900 * (q * shares - ac) / (q * total) / target_row.V01)
        target = target_row[list(NEW_EXPRESSIONS)].to_numpy(float)
        np.testing.assert_allclose(got, target, atol=2e-10, rtol=0)
        enc = lambda x: np.floor(np.clip(100 * np.array(x) + 10000 + .000001, 0, 999999))
        np.testing.assert_array_equal(enc(got), enc(target))
        cases.append(dict(**sample, reused_original_source_identity=True, native_ref29_lots_and_cents_replayed=True))
    out = dict(passed=True, feature_report_sha256=sha(INPUTS / 'feature_report.json'),
        feature_verification_sha256=sha(INPUTS / 'feature_verification.json'), samples=cases,
        existing_32_source_cases_replayed_for_new_kernel_only=True, no_new_production_raw_extraction=True,
        all_29_full_values_and_codes_sql_rebuilt=True, software_compilation_verified=False,
        native_source_parity_verified=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(INPUTS / 'native_input_verification.json', out); return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['features', 'verify', 'native']); args = parser.parse_args()
    print(json.dumps(globals()[args.stage](), ensure_ascii=False, indent=2))
