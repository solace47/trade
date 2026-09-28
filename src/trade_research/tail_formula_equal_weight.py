"""Stock performance relative to the covered equal-weight main-board pool."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as previous
from . import tail_formula_cross_breadth as members_source
from .corporate_cash import DAILY, MINUTES, save_json, sha

STEM = 'tail_formula_equal_weight'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
COMBINED_PROTOCOL = Path('config') / (STEM + '_combined_protocol.json')
NEW_EXPRESSIONS = {'EW01': '(A01-EQDAY)/V01', 'EW02': '(A05-EQTAIL)/V01'}
EXPRESSIONS = {**previous.EXPRESSIONS, **NEW_EXPRESSIONS}
HEADER = previous.HEADER + """EQN:=INSUM('沪深Ａ股','YJEW20',1,0);
EQDAY:=INSUM('沪深Ａ股','YJEW20',2,0)/MAX(EQN,1);
EQTAIL:=INSUM('沪深Ａ股','YJEW20',3,0)/MAX(EQN,1);
"""
HELPER = members_source.HELPER.split('NB:')[0] + """NB:IF(OK,1,0);
DR:IF(OK,100*(P49/MAX(PC,1)-1),0);
TR:IF(OK,100*(P49/MAX(P20,1)-1),0);
"""
GATE = 'EQN>=2000'
ORIGINAL_NATIVE_CORE = base.native_core


def native_core(*args, **kwargs):
    text = ORIGINAL_NATIVE_CORE(*args, **kwargs)
    assert text.count('CORE:SC>') == 1
    return text.replace('CORE:SC>', 'CORE:'+GATE+' AND SC>')


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    for path, digest in p['source_hashes'].items():
        assert sha(Path(path)) == digest
    members_source.checked_sources()
    r = json.loads((members_source.ROOT/'feature_report.json').read_text())
    for name in ['feature_verification', 'coverage_verification', 'native_input_verification']:
        proof = json.loads((members_source.ROOT/(name+'.json')).read_text())
        assert proof['passed'] and proof['feature_report_sha256'] == sha(members_source.ROOT/'feature_report.json')
    assert r['members_sha256'] == sha(members_source.ROOT/'members.parquet')
    assert r['breadth_sha256'] == sha(members_source.ROOT/'breadth.parquet')
    assert p['minimum_members'] == 2000 and not p['new_2026_prices_allowed']
    return p


def features():
    assert not (ROOT/'feature_report.json').exists(), 'Do not replace frozen equal-weight inputs'
    p = checked_sources(); m = pd.read_parquet(members_source.ROOT/'members.parquet')
    m['ew_day_atom'] = 100*(m.p49-m.pc)/m.pc
    m['ew_tail_atom'] = 100*(m.p49-m.p20)/m.p20
    assert np.isfinite(m[['ew_day_atom', 'ew_tail_atom']]).all().all()
    daily = m.groupby('date', sort=True).agg(ew_members=('code', 'size'),
        ew_day_mean=('ew_day_atom', 'mean'), ew_tail_mean=('ew_tail_atom', 'mean')).reset_index()
    coverage = pd.read_parquet(members_source.ROOT/'breadth.parquet', columns=['date', 'expected_local_members', 'missing_local_members'])
    daily = daily.merge(coverage, on='date', validate='one_to_one')
    daily['equal_weight_valid'] = daily.ew_members.ge(p['minimum_members'])
    old = pd.read_parquet(previous.ROOT/'features.parquet')
    f = old.merge(daily, on='date', how='left', validate='many_to_one').sort_values(['date', 'code']).reset_index(drop=True)
    valid = f.equal_weight_valid.fillna(False) & f.V01.gt(0)
    f['EW01'] = ((f.A01-f.ew_day_mean)/f.V01).where(valid)
    f['EW02'] = ((f.A05-f.ew_tail_mean)/f.V01).where(valid)
    f['prior_formula_input_valid'] = f.formula_input_valid
    f['formula_input_valid'] &= valid & np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
    ROOT.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(ROOT/'market_reference.parquet', index=False, compression='zstd')
    f.to_parquet(ROOT/'features.parquet', index=False, compression='zstd'); (ROOT/'YJEW20.tdx').write_text(HELPER)
    r = dict(protocol_sha256=sha(PROTOCOL), source_hashes=p['source_hashes'],
        features_sha256=sha(ROOT/'features.parquet'), market_reference_sha256=sha(ROOT/'market_reference.parquet'),
        helper_sha256=sha(ROOT/'YJEW20.tdx'), members_sha256=sha(members_source.ROOT/'members.parquet'),
        rows=len(f), valid=int(f.formula_input_valid.sum()), previous_valid=int(old.formula_input_valid.sum()),
        newly_invalid=int((old.formula_input_valid & ~f.formula_input_valid).sum()),
        member_rows=len(m), member_codes=m.code.nunique(), days=len(daily),
        minimum_members=int(daily.ew_members.min()), maximum_members=int(daily.ew_members.max()),
        invalid_days=int((~daily.equal_weight_valid).sum()), missing_local_member_days=int(daily.missing_local_members.sum()),
        expressions=EXPRESSIONS, native_header=HEADER, native_core_gate=GATE,
        common_reference_includes_current_stock=True, native_auxiliary_indicator_required=True,
        software_compilation_verified=False, native_source_parity_verified=False,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT/'feature_report.json', r); return {k:v for k,v in r.items() if k not in ['expressions', 'native_header']}


def verify_features():
    checked_sources(); r = json.loads((ROOT/'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL)
    for key, file in [('features', 'features.parquet'), ('market_reference', 'market_reference.parquet'), ('helper', 'YJEW20.tdx')]:
        assert r[key+'_sha256'] == sha(ROOT/file)
    c = base.conn()
    c.read_parquet(str(members_source.SOURCE/'visible_base.parquet')).create_view('source')
    expected_members = c.sql('''SELECT date,code,round(price_1420*100)::BIGINT AS p20,
        round(price_1449*100)::BIGINT AS p49,round(preclose*100)::BIGINT AS pc
        FROM source WHERE code LIKE 'sh.60%' OR code LIKE 'sz.00%' ORDER BY date,code''').df()
    m = pd.read_parquet(members_source.ROOT/'members.parquet')
    pd.testing.assert_frame_equal(m[expected_members.columns], expected_members, check_exact=True)
    assert m.isST.eq(0).all() and m.tradestatus.eq(1).all() and m.listing_age_sessions.ge(20).all()
    assert m.date.between('2024-01-01', '2025-12-30').all()
    c.register('members', expected_members)
    stats = c.sql('''SELECT date,count(*) AS ew_members,
        avg(100*(p49::DOUBLE/pc-1)) AS ew_day_mean,
        avg(100*(p49::DOUBLE/p20-1)) AS ew_tail_mean FROM members GROUP BY date ORDER BY date''').df()
    daily = pd.read_parquet(ROOT/'market_reference.parquet')
    pd.testing.assert_frame_equal(daily[stats.columns], stats, check_dtype=False, rtol=0, atol=2e-12)
    coverage = pd.read_parquet(members_source.ROOT/'breadth.parquet', columns=['date', 'expected_local_members', 'missing_local_members'])
    pd.testing.assert_frame_equal(daily[coverage.columns], coverage, check_exact=True)
    assert daily.equal_weight_valid.equals(daily.ew_members.ge(2000))
    old = pd.read_parquet(previous.ROOT/'features.parquet'); f = pd.read_parquet(ROOT/'features.parquet')
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')], old.drop(columns='formula_input_valid'), check_exact=True)
    pd.testing.assert_series_equal(f.prior_formula_input_valid, old.formula_input_valid, check_exact=True, check_names=False)
    # Independently rebuild the stock component from the integer quotes too.
    c.register('old', old[['date', 'code', 'V01']]); c.register('stats', stats)
    expected = c.sql('''SELECT old.date,old.code,
        CASE WHEN V01>0 AND ew_members>=2000 THEN (100*(p49::DOUBLE/pc-1)-ew_day_mean)/V01 END AS EW01,
        CASE WHEN V01>0 AND ew_members>=2000 THEN (100*(p49::DOUBLE/p20-1)-ew_tail_mean)/V01 END AS EW02
        FROM old LEFT JOIN members USING(date,code) LEFT JOIN stats USING(date) ORDER BY date,code''').df(); c.close()
    pd.testing.assert_frame_equal(f[['date', 'code']], expected[['date', 'code']], check_exact=True)
    for name in NEW_EXPRESSIONS:
        np.testing.assert_allclose(f[name], expected[name], rtol=0, atol=2e-10, equal_nan=True)
    wanted_valid = old.formula_input_valid & np.isfinite(expected[list(NEW_EXPRESSIONS)]).all(axis=1)
    np.testing.assert_array_equal(f.formula_input_valid, wanted_valid)
    encoder = lambda x: np.floor(np.clip(100*x+10000+.000001, 0, 999999))
    np.testing.assert_array_equal(encoder(f.loc[wanted_valid, list(NEW_EXPRESSIONS)]),
                                  encoder(expected.loc[wanted_valid, list(NEW_EXPRESSIONS)]))
    assert (ROOT/'YJEW20.tdx').read_text() == HELPER
    names = re.findall(r'(?m)^([A-Z][A-Z0-9]*):=', HEADER)+list(EXPRESSIONS)
    assert len(names) == len(set(names))
    assert r['expressions'] == EXPRESSIONS and r['native_header'] == HEADER and r['native_core_gate'] == GATE
    proof = dict(passed=True, feature_report_sha256=sha(ROOT/'feature_report.json'), rows=len(f),
        members=len(m), days=len(daily), valid=int(wanted_valid.sum()),
        all_members_integer_prices_group_means_and_stock_components_independently_rebuilt=True,
        all_previous_48_inputs_and_keys_unchanged=True, all_new_encodings_and_validity_rebuilt=True,
        source_coverage_proof_reused=True, new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT/'feature_verification.json', proof); return proof


def native():
    checked_sources(); v = json.loads((ROOT/'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(ROOT/'feature_report.json')
    hashes = members_source.checked_sources()
    raw_hashes = json.loads(Path('data/research/economic_winner/input_manifest.json').read_text())['source_sha256']
    f = pd.read_parquet(ROOT/'features.parquet'); selected = f.loc[f.formula_input_valid].copy()
    selected['sample_hash'] = [hashlib.sha256((d+'|'+c+'|ew-native-v1').encode()).hexdigest() for d,c in zip(selected.date, selected.code)]
    sample = selected.sort_values('sample_hash').groupby('half', sort=True).head(8).sort_values(['date', 'code'])
    receipts = []
    for row in sample.itertuples():
        path = MINUTES/row.code[:2].upper()/(row.code[3:]+'.parquet'); assert sha(path) == raw_hashes[str(path)]
        minute = pd.read_parquet(path, columns=['timestamp', 'close', 'volume', 'turnover'], filters=[
            ('timestamp', '>=', pd.Timestamp(row.date)), ('timestamp', '<=', pd.Timestamp(row.date+' 14:49'))])
        assert len(minute) == 230 and not minute.timestamp.duplicated().any()
        clocks = minute.timestamp.dt.strftime('%H%M')
        p20, p49 = [int(round(float(minute.loc[clocks.eq(t), 'close'].iloc[0])*100)) for t in ['1420', '1449']]
        daily_path = DAILY/(row.code.replace('.', '_')+'.parquet'); assert sha(daily_path) == hashes[str(daily_path)]
        raw = pd.read_parquet(daily_path, columns=['date', 'preclose', 'isST', 'tradestatus'], filters=[('date', '==', row.date)]).iloc[0]
        pc = int(round(float(raw.preclose)*100))
        assert raw.isST == 0 and raw.tradestatus == 1 and minute.volume.sum() > 0 and minute.turnover.sum() > 0
        day, tail = 100*(p49/pc-1), 100*(p49/p20-1)
        values = np.array([(day-row.ew_day_mean)/row.V01, (tail-row.ew_tail_mean)/row.V01])
        expected = np.array([row.EW01, row.EW02])
        np.testing.assert_allclose(values, expected, rtol=0, atol=2e-10)
        encode = lambda x: np.floor(np.clip(100*x+10000+.000001, 0, 999999))
        np.testing.assert_array_equal(encode(values), encode(expected))
        receipts.append(dict(date=row.date, code=row.code, minutes=len(minute), source_sha256=raw_hashes[str(path)],
            helper_outputs=[1, day, tail]))
    assert len(receipts) == 32
    r = dict(passed=True, feature_report_sha256=sha(ROOT/'feature_report.json'),
        feature_verification_sha256=sha(ROOT/'feature_verification.json'), samples=receipts, raw_minutes=7360,
        helper_and_stock_expressions_rebuilt_from_raw=True, entire_cached_pool_aggregation_independently_verified=True,
        software_compilation_verified=False, native_source_parity_verified=False,
        full_client_member_and_history_guards_verified=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT/'native_input_verification.json', r); return r


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['features', 'verify_features', 'native']); args = parser.parse_args()
    print(json.dumps(globals()[args.stage](), ensure_ascii=False, indent=2))
