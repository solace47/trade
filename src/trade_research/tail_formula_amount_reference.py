"""Stock strength relative to the same covered pool with amount weights."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as previous
from . import tail_formula_equal_weight as equal
from . import tail_formula_cross_breadth as members_source
from .corporate_cash import MINUTES, save_json, sha

STEM = 'tail_formula_amount_reference'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
NEW_EXPRESSIONS = {'AW01': '(A01-AMDAY)/V01', 'AW02': '(A05-AMTAIL)/V01'}
EXPRESSIONS = {**previous.EXPRESSIONS, **NEW_EXPRESSIONS}
HEADER = previous.HEADER + """AMN:=INSUM('沪深Ａ股','YJAM20',1,0);
AMD:=INSUM('沪深Ａ股','YJAM20',2,0);
AMT:=INSUM('沪深Ａ股','YJAM20',4,0);
AMDAY:=INSUM('沪深Ａ股','YJAM20',3,0)/MAX(AMD,0.0000000001);
AMTAIL:=INSUM('沪深Ａ股','YJAM20',5,0)/MAX(AMT,0.0000000001);
"""
HELPER = equal.HELPER.split('NB:')[0] + """TAM:=VALUEWHEN(TIME=1449,SUM(AMOUNT,29));
NB:IF(OK,1,0);
DW:IF(OK,DA/100000000,0);
DR:IF(OK,DA/100000000*100*(P49/MAX(PC,1)-1),0);
TW:IF(OK,TAM/100000000,0);
TR:IF(OK,TAM/100000000*100*(P49/MAX(P20,1)-1),0);
"""
GATE = 'AMN>=2000 AND AMD>=0.0000000001 AND AMT>=0.0000000001'
ORIGINAL_NATIVE_CORE = base.native_core


def native_core(*args, **kwargs):
    text = ORIGINAL_NATIVE_CORE(*args, **kwargs)
    assert text.count('CORE:SC>') == 1
    return text.replace('CORE:SC>', 'CORE:' + GATE + ' AND SC>')


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    equal.checked_sources()
    for module in [previous, equal]:
        r = json.loads((module.ROOT / 'feature_report.json').read_text())
        v = json.loads((module.ROOT / 'feature_verification.json').read_text())
        assert v['passed'] and v['feature_report_sha256'] == sha(module.ROOT / 'feature_report.json')
        assert r['features_sha256'] == sha(module.ROOT / 'features.parquet')
    assert p['minimum_members'] == 2000 and not p['new_2026_prices_allowed']
    return p


def member_inputs():
    m = pd.read_parquet(members_source.ROOT / 'members.parquet')
    tail = pd.read_parquet(members_source.SOURCE / 'visible_base.parquet', columns=['date', 'code', 'amount_last29'])
    return m.merge(tail, on=['date', 'code'], how='left', validate='one_to_one').sort_values(['date', 'code']).reset_index(drop=True)


def reference(m, minimum_members=2000):
    """Keep every input member; any invalid amount invalidates its whole date."""
    assert not m.duplicated(['date', 'code']).any()
    d = m[['date', 'code']].copy()
    day, tail = m.amount_1449.to_numpy(float), m.amount_last29.to_numpy(float)
    good = np.isfinite(day) & np.isfinite(tail) & (day >= .01) & ((tail == 0) | (tail >= .01)) & (tail <= day)
    d['bad'] = ~good; d['day_amount'] = day; d['tail_amount'] = tail
    d['day_numerator'] = day * (100*(m.p49-m.pc)/m.pc)
    d['tail_numerator'] = tail * (100*(m.p49-m.p20)/m.p20)
    out = d.groupby('date', sort=True).agg(aw_members=('code', 'size'), aw_bad_members=('bad', 'sum'),
        aw_day_amount=('day_amount', 'sum'), aw_tail_amount=('tail_amount', 'sum'),
        aw_day_numerator=('day_numerator', 'sum'), aw_tail_numerator=('tail_numerator', 'sum')).reset_index()
    out['amount_reference_valid'] = out.aw_bad_members.eq(0) & out.aw_members.ge(minimum_members)
    out['amount_reference_valid'] &= out.aw_day_amount.ge(.01) & out.aw_tail_amount.ge(.01)
    out['aw_day_mean'] = (out.aw_day_numerator / out.aw_day_amount).where(out.amount_reference_valid)
    out['aw_tail_mean'] = (out.aw_tail_numerator / out.aw_tail_amount).where(out.amount_reference_valid)
    return out


def features():
    p = checked_sources(); assert not (ROOT / 'feature_report.json').exists()
    m = member_inputs(); stats = reference(m)
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    f = old.merge(stats, on='date', how='left', validate='many_to_one').sort_values(['date', 'code']).reset_index(drop=True)
    valid = f.amount_reference_valid.fillna(False) & f.V01.gt(0)
    f['AW01'] = ((f.A01-f.aw_day_mean)/f.V01).where(valid)
    f['AW02'] = ((f.A05-f.aw_tail_mean)/f.V01).where(valid)
    f['prior_formula_input_valid'] = f.formula_input_valid
    f['formula_input_valid'] &= valid & np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
    ROOT.mkdir(parents=True, exist_ok=True)
    f.to_parquet(ROOT / 'features.parquet', index=False, compression='zstd')
    stats.to_parquet(ROOT / 'market_reference.parquet', index=False, compression='zstd')
    (ROOT / 'YJAM20.tdx').write_text(HELPER)
    r = dict(protocol_sha256=sha(PROTOCOL), source_hashes=p['source_hashes'], features_sha256=sha(ROOT / 'features.parquet'),
        market_reference_sha256=sha(ROOT / 'market_reference.parquet'), helper_sha256=sha(ROOT / 'YJAM20.tdx'),
        rows=len(f), valid=int(f.formula_input_valid.sum()), newly_invalid=int((old.formula_input_valid & ~f.formula_input_valid).sum()),
        member_rows=len(m), days=len(stats), bad_amount_members=int(stats.aw_bad_members.sum()),
        zero_tail_amount_members=int(m.amount_last29.eq(0).sum()), invalid_reference_days=int((~stats.amount_reference_valid).sum()),
        minimum_day_amount=float(stats.aw_day_amount.min()), minimum_tail_amount=float(stats.aw_tail_amount.min()),
        expressions=EXPRESSIONS, native_header=HEADER, native_core_gate=GATE,
        original_equal_weight_members_preserved=True, software_compilation_verified=False, native_source_parity_verified=False,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_report.json', r)
    return {k: v for k, v in r.items() if k not in ['source_hashes', 'expressions', 'native_header']}


def verify_features():
    p = checked_sources(); r = json.loads((ROOT / 'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['source_hashes'] == p['source_hashes']
    for key, file in [('features', 'features.parquet'), ('market_reference', 'market_reference.parquet'), ('helper', 'YJAM20.tdx')]:
        assert r[key+'_sha256'] == sha(ROOT / file)
    c = base.conn(); c.read_parquet(str(members_source.SOURCE / 'visible_base.parquet')).create_view('source')
    m = c.sql('''SELECT date,code,round(price_1420*100)::BIGINT AS p20,round(price_1449*100)::BIGINT AS p49,
        round(preclose*100)::BIGINT AS pc,amount_1449,amount_last29 FROM source
        WHERE code LIKE 'sh.60%' OR code LIKE 'sz.00%' ORDER BY date,code''').df()
    actual_members = member_inputs(); pd.testing.assert_frame_equal(actual_members[m.columns], m, check_exact=True)
    assert actual_members.isST.eq(0).all() and actual_members.tradestatus.eq(1).all()
    assert actual_members.date.between(p['signal_first'], p['signal_last']).all()
    c.register('members', m)
    stats = c.sql('''SELECT date,count(*) AS aw_members,
        count(*) FILTER(WHERE NOT coalesce(isfinite(amount_1449) AND amount_1449>=.01
        AND isfinite(amount_last29) AND (amount_last29=0 OR amount_last29>=.01) AND amount_last29<=amount_1449,false)) AS aw_bad_members,
        sum(amount_1449) AS aw_day_amount,sum(amount_last29) AS aw_tail_amount,
        sum(amount_1449*100*(p49::DOUBLE/pc-1)) AS aw_day_numerator,
        sum(amount_last29*100*(p49::DOUBLE/p20-1)) AS aw_tail_numerator,
        aw_members>=2000 AND aw_bad_members=0 AND aw_day_amount>=.01 AND aw_tail_amount>=.01 AS amount_reference_valid,
        CASE WHEN amount_reference_valid THEN aw_day_numerator/aw_day_amount END AS aw_day_mean,
        CASE WHEN amount_reference_valid THEN aw_tail_numerator/aw_tail_amount END AS aw_tail_mean
        FROM members GROUP BY date ORDER BY date''').df()
    actual = pd.read_parquet(ROOT / 'market_reference.parquet')
    pd.testing.assert_frame_equal(actual[stats.columns], stats, check_dtype=False, rtol=2e-12, atol=2e-8)
    ew_stats = pd.read_parquet(equal.ROOT / 'market_reference.parquet')
    pd.testing.assert_frame_equal(actual[['date', 'aw_members']].rename(columns={'aw_members': 'ew_members'}), ew_stats[['date', 'ew_members']], check_dtype=False, check_exact=True)
    old = pd.read_parquet(previous.ROOT / 'features.parquet'); f = pd.read_parquet(ROOT / 'features.parquet')
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')], old.drop(columns='formula_input_valid'), check_exact=True)
    pd.testing.assert_series_equal(f.prior_formula_input_valid, old.formula_input_valid, check_names=False, check_exact=True)
    c.register('old', old[['date', 'code', 'V01']]); c.register('stats', stats)
    expected = c.sql('''SELECT old.date,old.code,
        CASE WHEN V01>0 AND amount_reference_valid THEN (100*(p49::DOUBLE/pc-1)-aw_day_mean)/V01 END AS AW01,
        CASE WHEN V01>0 AND amount_reference_valid THEN (100*(p49::DOUBLE/p20-1)-aw_tail_mean)/V01 END AS AW02
        FROM old LEFT JOIN members USING(date,code) LEFT JOIN stats USING(date) ORDER BY date,code''').df(); c.close()
    pd.testing.assert_frame_equal(f[['date', 'code']], expected[['date', 'code']], check_exact=True)
    np.testing.assert_allclose(f[list(NEW_EXPRESSIONS)], expected[list(NEW_EXPRESSIONS)], rtol=0, atol=2e-10, equal_nan=True)
    valid = old.formula_input_valid & np.isfinite(expected[list(NEW_EXPRESSIONS)]).all(axis=1)
    np.testing.assert_array_equal(f.formula_input_valid, valid)
    encode = lambda x: np.floor(np.clip(100*x+10000+.000001, 0, 999999))
    np.testing.assert_array_equal(encode(f.loc[valid, list(NEW_EXPRESSIONS)]), encode(expected.loc[valid, list(NEW_EXPRESSIONS)]))
    names = re.findall(r'(?m)^([A-Z][A-Z0-9]*):=', HEADER)+list(EXPRESSIONS)
    assert len(names) == len(set(names)) and (ROOT / 'YJAM20.tdx').read_text() == HELPER
    proof = dict(passed=True, feature_report_sha256=sha(ROOT / 'feature_report.json'), rows=len(f),
        all_members_prices_amount_weights_and_means_sql_rebuilt=True, all_original_48_fields_and_keys_unchanged=True,
        equal_weight_reference_members_identical=True, all_new_encodings_and_validity_rebuilt=True,
        effective_input_intersection_unchanged=bool(valid.equals(old.formula_input_valid)),
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_verification.json', proof); return proof


def native():
    checked_sources(); v = json.loads((ROOT / 'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(ROOT / 'feature_report.json')
    original = json.loads((equal.ROOT / 'native_input_verification.json').read_text())
    assert original['passed']
    f = pd.read_parquet(ROOT / 'features.parquet'); m = member_inputs()
    receipts = []
    for sample in original['samples']:
        date, code = sample['date'], sample['code']; row = f.loc[f.date.eq(date) & f.code.eq(code)].iloc[0]
        raw_path = MINUTES / code[:2].upper() / (code[3:] + '.parquet')
        assert sha(raw_path) == sample['source_sha256']
        raw = pd.read_parquet(raw_path, columns=['timestamp', 'turnover'], filters=[
            ('timestamp', '>=', pd.Timestamp(date)), ('timestamp', '<=', pd.Timestamp(date+' 14:49'))])
        assert len(raw) == 230 and not raw.timestamp.duplicated().any()
        tail = raw.loc[raw.timestamp.ge(pd.Timestamp(date+' 14:21'))]; assert len(tail) == 29
        source = m.loc[m.date.eq(date) & m.code.eq(code)].iloc[0]
        np.testing.assert_allclose([raw.turnover.sum(), tail.turnover.sum()], [source.amount_1449, source.amount_last29], rtol=2e-13, atol=2e-6)
        pool = m.loc[m.date.eq(date)]
        dw = [float(x)/1e8 for x in pool.amount_1449]; tw = [float(x)/1e8 for x in pool.amount_last29]
        d = [100*(q/r-1) for q, r in zip(pool.p49, pool.pc)]; t = [100*(q/r-1) for q, r in zip(pool.p49, pool.p20)]
        outputs = [len(pool), sum(dw), sum(w*x for w, x in zip(dw, d)), sum(tw), sum(w*x for w, x in zip(tw, t))]
        assert outputs[0] >= 2000 and outputs[1] >= 1e-10 and outputs[3] >= 1e-10
        day, tail_mean = outputs[2]/max(outputs[1], 1e-10), outputs[4]/max(outputs[3], 1e-10)
        values = [(100*(source.p49/source.pc-1)-day)/row.V01, (100*(source.p49/source.p20-1)-tail_mean)/row.V01]
        wanted = [row.AW01, row.AW02]
        np.testing.assert_allclose(values, wanted, rtol=0, atol=2e-10)
        np.testing.assert_array_equal(np.floor(np.clip(100*np.array(values)+10000+.000001, 0, 999999)), np.floor(np.clip(100*np.array(wanted)+10000+.000001, 0, 999999)))
        receipts.append(dict(date=date, code=code, source_sha256=sample['source_sha256'], amount_bars=230, tail_amount_bars=29))
    assert len(receipts) == 32
    proof = dict(passed=True, protocol_sha256=sha(PROTOCOL), feature_report_sha256=sha(ROOT / 'feature_report.json'),
        feature_verification_sha256=sha(ROOT / 'feature_verification.json'), samples=receipts,
        original_equal_weight_sample_and_price_proofs_reused=True, raw_amounts_and_five_helper_outputs_rebuilt=True,
        software_compilation_verified=False, native_source_parity_verified=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'native_input_verification.json', proof); return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['features', 'verify_features', 'native']); args = parser.parse_args()
    print(json.dumps(globals()[args.stage](), ensure_ascii=False, indent=2))
