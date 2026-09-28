"""Cross-stock amount concentration with unchanged visible pool membership."""
import argparse
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_amount_reference as parent
from . import tail_formula_float as previous
from .corporate_cash import save_json, sha

STEM = 'tail_formula_amount_concentration'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
NEW_EXPRESSIONS = {
    'AC01': 'MAX(ACN*ACD2/(MAX(ACD,0.0000000001)*MAX(ACD,0.0000000001))-1,0)',
    'AC02': 'MAX(ACN*ACT2/(MAX(ACT,0.0000000001)*MAX(ACT,0.0000000001))-1,0)'}
EXPRESSIONS = {**previous.EXPRESSIONS, **NEW_EXPRESSIONS}
HEADER = previous.HEADER + ''.join(
    f"{name}:=INSUM('沪深Ａ股','YJAC20',{i},0);\n"
    for i, name in enumerate(['ACN', 'ACD', 'ACD2', 'ACT', 'ACT2'], 1))
HELPER = parent.equal.HELPER.split('NB:')[0] + '''TAM:=VALUEWHEN(TIME=1449,SUM(AMOUNT,29));
NB:IF(OK,1,0);
DW:IF(OK,DA/100000000,0);
DS:IF(OK,(DA/100000000)*(DA/100000000),0);
TW:IF(OK,TAM/100000000,0);
TS:IF(OK,(TAM/100000000)*(TAM/100000000),0);
'''
GATE = 'ACN>=2000 AND ACD>=0.0000000001 AND ACT>=0.0000000001'
ORIGINAL_NATIVE_CORE = base.native_core


def native_core(*args, **kwargs):
    text = ORIGINAL_NATIVE_CORE(*args, **kwargs)
    assert text.count('CORE:SC>') == 1
    return text.replace('CORE:SC>', 'CORE:' + GATE + ' AND SC>')


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    parent.checked_sources()
    r = json.loads((parent.ROOT / 'feature_report.json').read_text())
    v = json.loads((parent.ROOT / 'feature_verification.json').read_text())
    n = json.loads((parent.ROOT / 'native_input_verification.json').read_text())
    assert v['passed'] and n['passed']
    assert v['feature_report_sha256'] == n['feature_report_sha256'] == sha(parent.ROOT / 'feature_report.json')
    assert n['feature_verification_sha256'] == sha(parent.ROOT / 'feature_verification.json')
    assert r['features_sha256'] == sha(parent.ROOT / 'features.parquet')
    assert p['minimum_members'] == 2000 and p['new_fields'] == list(NEW_EXPRESSIONS)
    assert not p['new_2026_prices_allowed']
    return p


def reference(m, minimum_members=2000):
    assert not m.duplicated(['date', 'code']).any()
    day, tail = m.amount_1449.to_numpy(float), m.amount_last29.to_numpy(float)
    good = np.isfinite(day) & np.isfinite(tail) & (day >= .01) & ((tail == 0) | (tail >= .01)) & (tail <= day)
    f = m[['date', 'code']].copy()
    f['bad'] = ~good
    f['day'] = day / 1e8
    f['tail'] = tail / 1e8
    f['day2'] = f.day ** 2
    f['tail2'] = f['tail'] ** 2
    stats = f.groupby('date', sort=True).agg(ac_members=('code', 'size'), ac_bad_members=('bad', 'sum'),
        ac_day_sum=('day', 'sum'), ac_day_square_sum=('day2', 'sum'),
        ac_tail_sum=('tail', 'sum'), ac_tail_square_sum=('tail2', 'sum')).reset_index()
    valid = (stats.ac_members.ge(minimum_members) & stats.ac_bad_members.eq(0)
        & stats.ac_day_sum.ge(1e-10) & stats.ac_tail_sum.ge(1e-10))
    stats['amount_concentration_valid'] = valid
    for side in ['day', 'tail']:
        total = stats['ac_' + side + '_sum']
        ratio = (stats.ac_members * stats['ac_' + side + '_square_sum'] / total.replace(0, np.nan) ** 2 - 1).clip(lower=0)
        stats['ac_' + side + '_concentration'] = ratio.where(valid)
    return stats


def features():
    p = checked_sources()
    assert not (ROOT / 'feature_report.json').exists()
    members = parent.member_inputs()
    stats = reference(members)
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    f = old.merge(stats, on='date', how='left', validate='many_to_one').sort_values(['date', 'code']).reset_index(drop=True)
    f['AC01'] = f.ac_day_concentration
    f['AC02'] = f.ac_tail_concentration
    f['prior_formula_input_valid'] = f.formula_input_valid
    f['formula_input_valid'] &= f.amount_concentration_valid.eq(True) & np.isfinite(f[list(NEW_EXPRESSIONS)]).all(axis=1)
    ROOT.mkdir(parents=True, exist_ok=True)
    f.to_parquet(ROOT / 'features.parquet', index=False, compression='zstd')
    stats.to_parquet(ROOT / 'market_reference.parquet', index=False, compression='zstd')
    (ROOT / 'YJAC20.tdx').write_text(HELPER)
    r = dict(protocol_sha256=sha(PROTOCOL), source_hashes=p['source_hashes'],
        features_sha256=sha(ROOT / 'features.parquet'), market_reference_sha256=sha(ROOT / 'market_reference.parquet'),
        helper_sha256=sha(ROOT / 'YJAC20.tdx'), rows=len(f), valid=int(f.formula_input_valid.sum()),
        newly_invalid=int((old.formula_input_valid & ~f.formula_input_valid).sum()),
        member_rows=len(members), days=len(stats), bad_amount_members=int(stats.ac_bad_members.sum()),
        invalid_reference_days=int((~stats.amount_concentration_valid).sum()),
        zero_tail_amount_members=int(members.amount_last29.eq(0).sum()),
        minimum_members=int(stats.ac_members.min()), expressions=EXPRESSIONS, native_header=HEADER,
        native_core_gate=GATE, original_amount_reference_members_preserved=True,
        native_auxiliary_indicator_required=True, native_source_parity_verified=False, software_compilation_verified=False,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_report.json', r)
    return {k: v for k, v in r.items() if k not in ['source_hashes', 'expressions', 'native_header']}


def verify_features():
    p = checked_sources()
    r = json.loads((ROOT / 'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL)
    for key, file in [('features', 'features.parquet'), ('market_reference', 'market_reference.parquet'), ('helper', 'YJAC20.tdx')]:
        assert r[key + '_sha256'] == sha(ROOT / file)
    c = base.conn()
    c.read_parquet(str(parent.members_source.SOURCE / 'visible_base.parquet')).create_view('source')
    m = c.sql("""SELECT date,code,amount_1449,amount_last29 FROM source
        WHERE code LIKE 'sh.60%' OR code LIKE 'sz.00%' ORDER BY date,code""").df()
    members = parent.member_inputs()
    pd.testing.assert_frame_equal(members[m.columns], m, check_exact=True)
    c.register('members', m)
    stats = c.sql('''WITH a AS(SELECT date,code,amount_1449,amount_last29,
        amount_1449/1e8 AS d,amount_last29/1e8 AS t FROM members),
        s AS(SELECT date,count(*) AS ac_members,
            count(*) FILTER(WHERE NOT coalesce(isfinite(amount_1449) AND amount_1449>=.01
                AND isfinite(amount_last29) AND (amount_last29=0 OR amount_last29>=.01)
                AND amount_last29<=amount_1449,false)) AS ac_bad_members,
            sum(d) AS ac_day_sum,sum(d*d) AS ac_day_square_sum,
            sum(t) AS ac_tail_sum,sum(t*t) AS ac_tail_square_sum FROM a GROUP BY date)
        SELECT *,ac_members>=2000 AND ac_bad_members=0 AND ac_day_sum>=1e-10 AND ac_tail_sum>=1e-10 AS amount_concentration_valid,
            CASE WHEN amount_concentration_valid THEN greatest(ac_members*ac_day_square_sum/power(ac_day_sum,2)-1,0) END AS ac_day_concentration,
            CASE WHEN amount_concentration_valid THEN greatest(ac_members*ac_tail_square_sum/power(ac_tail_sum,2)-1,0) END AS ac_tail_concentration
        FROM s ORDER BY date''').df()
    c.close()
    pd.testing.assert_frame_equal(pd.read_parquet(ROOT / 'market_reference.parquet'), stats, check_dtype=False, rtol=3e-13, atol=2e-10)
    # Independent share-normalized squares avoid using the aggregate formula.
    for day, group in members.groupby('date', sort=True):
        row = stats.loc[stats.date.eq(day)].iloc[0]
        assert row.amount_concentration_valid
        for side, column in [('day', 'amount_1449'), ('tail', 'amount_last29')]:
            amounts = [float(x) for x in group[column]]
            total = math.fsum(amounts)
            value = max(len(group) * math.fsum((x / total) ** 2 for x in amounts) - 1, 0)
            np.testing.assert_allclose(value, row['ac_' + side + '_concentration'], rtol=0, atol=2e-10)
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    f = pd.read_parquet(ROOT / 'features.parquet')
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')], old.drop(columns='formula_input_valid'), check_exact=True)
    pd.testing.assert_series_equal(f.prior_formula_input_valid, old.formula_input_valid, check_exact=True, check_names=False)
    ex = old[['date', 'code']].merge(stats, on='date', how='left', validate='many_to_one')
    expected = ex[['ac_day_concentration', 'ac_tail_concentration']].to_numpy(float)
    np.testing.assert_allclose(f[list(NEW_EXPRESSIONS)], expected, rtol=0, atol=2e-10, equal_nan=True)
    valid = old.formula_input_valid & ex.amount_concentration_valid.eq(True) & np.isfinite(expected).all(axis=1)
    np.testing.assert_array_equal(f.formula_input_valid, valid)
    encode = lambda v: np.floor(np.clip(100 * v + 10000 + .000001, 0, 999999))
    np.testing.assert_array_equal(encode(f.loc[valid, list(NEW_EXPRESSIONS)]), encode(expected[valid]))
    assert r['rows'] == len(f) and r['valid'] == int(valid.sum()) and r['newly_invalid'] == int((old.formula_input_valid & ~valid).sum())
    assert r['member_rows'] == len(m) == 1483109 and r['days'] == len(stats) == 484
    assert r['bad_amount_members'] == int(stats.ac_bad_members.sum()) == 0
    assert r['invalid_reference_days'] == int((~stats.amount_concentration_valid).sum()) == 0
    assert r['zero_tail_amount_members'] == int(m.amount_last29.eq(0).sum())
    assert r['minimum_members'] == int(stats.ac_members.min())
    names = re.findall(r'(?m)^([A-Z][A-Z0-9]*):=', HEADER) + list(EXPRESSIONS)
    assert len(names) == len({n.casefold() for n in names})
    assert (ROOT / 'YJAC20.tdx').read_text() == HELPER and r['expressions'] == EXPRESSIONS and r['native_header'] == HEADER
    v = dict(passed=True, feature_report_sha256=sha(ROOT / 'feature_report.json'),
        all_484_dates_amount_concentration_and_integer_encodings_sql_rebuilt=True,
        all_date_values_independently_checked_using_normalized_shares=True,
        all_original_members_keys_and_48_values_preserved=True,
        effective_input_intersection_unchanged=bool(valid.equals(old.formula_input_valid)),
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_verification.json', v)
    return v


def native_values(day_amounts, tail_amounts):
    # Evaluate the real helper's five output expressions and the real core
    # expressions, rather than an independently copied version of the formula.
    outputs = HELPER.split('NB:')[1]
    outputs = 'NB:' + outputs
    sums = [0.] * 5
    for day, tail in zip(day_amounts, tail_amounts):
        env = {'OK': True, 'DA': float(day), 'TAM': float(tail), 'IF': lambda cond, a, b: a if cond else b}
        for i, line in enumerate(outputs.strip().splitlines()):
            expression = line.rstrip(';').split(':', 1)[1]
            sums[i] += eval(expression, {'__builtins__': {}}, env)
    assert sums[0] >= 2 and sums[1] >= 1e-10 and sums[3] >= 1e-10
    env = dict(zip(['ACN', 'ACD', 'ACD2', 'ACT', 'ACT2'], sums))
    env['MAX'] = max
    return [eval(expr, {'__builtins__': {}}, env) for expr in NEW_EXPRESSIONS.values()]


def native():
    checked_sources()
    v = json.loads((ROOT / 'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(ROOT / 'feature_report.json')
    original = json.loads((parent.ROOT / 'native_input_verification.json').read_text())
    assert original['passed'] and len(original['samples']) == 32
    m = parent.member_inputs()
    f = pd.read_parquet(ROOT / 'features.parquet', columns=['date', 'code', 'AC01', 'AC02']).set_index(['date', 'code'])
    receipts = []
    for sample in original['samples']:
        day, code = sample['date'], sample['code']
        pool = m.loc[m.date.eq(day)]
        assert len(pool) >= 2000
        rebuilt = native_values(pool.amount_1449, pool.amount_last29)
        wanted = f.loc[(day, code)].to_numpy(float)
        np.testing.assert_allclose(rebuilt, wanted, rtol=0, atol=2e-10)
        np.testing.assert_array_equal(np.floor(100*np.array(rebuilt)+10000+.000001), np.floor(100*wanted+10000+.000001))
        receipts.append(dict(date=day, code=code, members=len(pool), AC01=rebuilt[0], AC02=rebuilt[1],
            original_amount_source_sha256=sample['source_sha256']))
    proof = dict(passed=True, protocol_sha256=sha(PROTOCOL), feature_report_sha256=sha(ROOT / 'feature_report.json'),
        feature_verification_sha256=sha(ROOT / 'feature_verification.json'),
        original_amount_native_proof_sha256=sha(parent.ROOT / 'native_input_verification.json'), samples=receipts,
        all_new_actual_helper_outputs_and_native_expressions_rebuilt=True, raw_amount_source_proofs_reused_without_reextraction=True,
        native_source_parity_verified=False, software_compilation_verified=False,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'native_input_verification.json', proof)
    return {k: v for k, v in proof.items() if k != 'samples'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['features', 'verify_features', 'native'])
    print(json.dumps(globals()[parser.parse_args().stage](), ensure_ascii=False, indent=2))
