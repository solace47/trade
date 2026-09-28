"""Predeclared market context from price-limit quotes and retreat after touches."""
import argparse
import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_before1000 as labels
from . import tail_formula_cross_breadth as peers
from . import tail_formula_float as previous
from .corporate_cash import DAILY, MINUTES, save_json, sha

STEM = 'tail_formula_limit_breadth'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
COMBINED_PROTOCOL = Path('config') / (STEM + '_combined_protocol.json')
EXPRESSIONS = {**previous.EXPRESSIONS,
    'LB01': '100*(MLU-MLD)/MAX(MLN,1)',
    'LB02': '100*(MLT-MLU)/MAX(MLT,1)'}
HEADER = previous.HEADER + ''.join(f"{name}:=INSUM('沪深Ａ股','YJLB20',{i},0);\n"
    for i, name in enumerate(['MLA', 'MLN', 'MLU', 'MLD', 'MLT'], 1))
GATE = 'MLA>=2000 AND MLN>=0.99*MLA'
HELPER = peers.HELPER.split('NB:')[0] + '''HIX:=INTPART(VALUEWHEN(TIME=1449,HHV(IF(V>0,H,0),B0))*100+0.5);
LOX:=INTPART(VALUEWHEN(TIME=1449,LLV(IF(V>0,L,1000000),B0))*100+0.5);
UPX:=INTPART((11*PC+5)/10);
DNX:=INTPART((9*PC+5)/10);
COK:=OK AND DNX<=LOX AND LOX<=P49 AND P49<=HIX AND HIX<=UPX;
N0:IF(OK,1,0);
NC:IF(COK,1,0);
LU:IF(COK AND P49=UPX,1,0);
LD:IF(COK AND P49=DNX,1,0);
TU:IF(COK AND HIX=UPX,1,0);
'''
ORIGINAL_NATIVE_CORE = base.native_core


def native_core(*args, **kwargs):
    text = ORIGINAL_NATIVE_CORE(*args, **kwargs)
    assert text.count('CORE:SC>') == 1
    return text.replace('CORE:SC>', 'CORE:' + GATE + ' AND SC>')


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    for path, digest in p['source_hashes'].items():
        assert sha(Path(path)) == digest, path
    for proof in ['feature_verification', 'coverage_verification']:
        r = json.loads((peers.ROOT / (proof + '.json')).read_text())
        assert r['passed'] and r['feature_report_sha256'] == sha(peers.ROOT / 'feature_report.json')
    assert p['minimum_members'] == 2000 and p['minimum_classifiable_fraction'] == .99
    return p


def features():
    checked_sources()
    if (ROOT / 'feature_report.json').exists():
        raise ValueError('Do not replace limit-price context inputs')
    m = pd.read_parquet(peers.ROOT / 'members.parquet')
    raw = pd.read_parquet(peers.SOURCE / 'visible_base.parquet', columns=['date', 'code', 'high_1449', 'low_1449'])
    m = m.merge(raw, on=['date', 'code'], validate='one_to_one')
    assert m.date.between('2024-01-01', '2025-12-30').all() and len(m) == 1483109
    for col, target in [('high_1449', 'hi'), ('low_1449', 'lo')]:
        assert np.isfinite(m[col]).all() and m[col].gt(0).all()
        m[target] = np.floor(m[col] * 100 + .5).astype('int64')
    m['upper'] = (11 * m.pc + 5) // 10; m['lower'] = (9 * m.pc + 5) // 10
    m['classifiable'] = m.lo.ge(m.lower) & m.lo.le(m.p49) & m.p49.le(m.hi) & m.hi.le(m.upper)
    m['at_upper'] = m.classifiable & m.p49.eq(m.upper)
    m['at_lower'] = m.classifiable & m.p49.eq(m.lower)
    m['touched_upper'] = m.classifiable & m.hi.eq(m.upper)
    assert (~m.at_upper | m.touched_upper).all()
    d = m.groupby('date', sort=True).agg(all_members=('code', 'size'), members=('classifiable', 'sum'),
        at_upper=('at_upper', 'sum'), at_lower=('at_lower', 'sum'), touched_upper=('touched_upper', 'sum')).reset_index()
    d['unclassified'] = d.all_members - d.members
    d['classification_coverage'] = d.members / d.all_members
    d['limit_context_valid'] = d.all_members.ge(2000) & d.classification_coverage.ge(.99)
    d['LB01'] = (100 * (d.at_upper - d.at_lower) / d.members.clip(lower=1)).where(d.limit_context_valid)
    d['LB02'] = (100 * (d.touched_upper - d.at_upper) / d.touched_upper.clip(lower=1)).where(d.limit_context_valid)
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    f = old.merge(d, on='date', how='left', validate='many_to_one').sort_values(['date', 'code']).reset_index(drop=True)
    f['prior_formula_input_valid'] = f.formula_input_valid
    f['formula_input_valid'] &= f.limit_context_valid.fillna(False) & np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
    ROOT.mkdir(parents=True, exist_ok=True)
    for frame, name in [(m, 'members'), (d, 'limit_context'), (f, 'features')]:
        frame.to_parquet(ROOT / (name + '.parquet'), index=False, compression='zstd')
    (ROOT / 'YJLB20.tdx').write_text(HELPER)
    r = dict(protocol_sha256=sha(PROTOCOL), features_sha256=sha(ROOT / 'features.parquet'),
        members_sha256=sha(ROOT / 'members.parquet'), context_sha256=sha(ROOT / 'limit_context.parquet'),
        helper_sha256=sha(ROOT / 'YJLB20.tdx'), rows=len(f), valid=int(f.formula_input_valid.sum()),
        previous_valid=int(old.formula_input_valid.sum()), newly_invalid=int((old.formula_input_valid & ~f.formula_input_valid).sum()),
        member_rows=len(m), days=len(d), unclassified_rows=int((~m.classifiable).sum()),
        invalid_dates=int((~d.limit_context_valid).sum()), minimum_classification_coverage=float(d.classification_coverage.min()),
        days_with_no_upper_touch=int(d.touched_upper.eq(0).sum()), source_peer_feature_report_sha256=sha(peers.ROOT / 'feature_report.json'),
        expressions=EXPRESSIONS, native_header=HEADER, native_core_gate=GATE,
        both_features_same_for_every_stock_on_date=True, full_price_limit_order_books_not_available=True,
        native_auxiliary_indicator_required=True, software_compilation_verified=False,
        native_client_membership_and_source_parity_verified=False,
        new_selection_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_report.json', r)
    return {k: v for k, v in r.items() if k not in ['expressions', 'native_header']}


def verify_features():
    checked_sources(); r = json.loads((ROOT / 'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL)
    for key, file in [('features', 'features.parquet'), ('members', 'members.parquet'),
                      ('context', 'limit_context.parquet'), ('helper', 'YJLB20.tdx')]:
        assert r[key + '_sha256'] == sha(ROOT / file)
    m = pd.read_parquet(ROOT / 'members.parquet'); oldm = pd.read_parquet(peers.ROOT / 'members.parquet')
    pd.testing.assert_frame_equal(m[oldm.columns], oldm, check_exact=True)
    pc = m.pc.to_numpy(); q = m.p49.to_numpy()
    # Independent decimal half-up for every distinct reference price.
    limits = {int(x): tuple(int((Decimal(int(x)) * Decimal(v)).quantize(Decimal(1), rounding=ROUND_HALF_UP))
                           for v in ['1.1', '.9']) for x in np.unique(pc)}
    up = np.array([limits[int(x)][0] for x in pc]); dn = np.array([limits[int(x)][1] for x in pc])
    raw = pd.read_parquet(peers.SOURCE / 'visible_base.parquet', columns=['date', 'code', 'high_1449', 'low_1449'])
    raw = m[['date', 'code']].merge(raw, on=['date', 'code'], validate='one_to_one')
    hi = np.rint(raw.high_1449.to_numpy() * 100).astype('int64')
    lo = np.rint(raw.low_1449.to_numpy() * 100).astype('int64')
    np.testing.assert_array_equal(m[['upper', 'lower', 'hi', 'lo']], np.column_stack([up, dn, hi, lo]))
    valid = (dn <= lo) & (lo <= q) & (q <= hi) & (hi <= up)
    expected = m[['date', 'code']].copy()
    expected['classifiable'] = valid; expected['at_upper'] = valid & (q == up)
    expected['at_lower'] = valid & (q == dn); expected['touched_upper'] = valid & (hi == up)
    pd.testing.assert_frame_equal(m[expected.columns], expected, check_exact=True)
    c = base.conn(); c.register('members', expected)
    counts = c.sql('''SELECT date,count(*) AS all_members,sum(classifiable::INT) AS members,
        sum(at_upper::INT) AS at_upper,sum(at_lower::INT) AS at_lower,sum(touched_upper::INT) AS touched_upper
        FROM members GROUP BY date ORDER BY date''').df(); c.close()
    d = pd.read_parquet(ROOT / 'limit_context.parquet')
    pd.testing.assert_frame_equal(d[counts.columns], counts, check_dtype=False, check_exact=True)
    good = (counts.all_members >= 2000) & (100 * counts.members >= 99 * counts.all_members)
    values = np.column_stack([100 * (counts.at_upper - counts.at_lower) / np.maximum(counts.members, 1),
                              100 * (counts.touched_upper - counts.at_upper) / np.maximum(counts.touched_upper, 1)])
    values[~good] = np.nan
    np.testing.assert_array_equal(d.limit_context_valid, good)
    np.testing.assert_allclose(d[['LB01', 'LB02']], values, atol=2e-14, rtol=0, equal_nan=True)
    assert (d.unclassified == d.all_members - d.members).all()
    np.testing.assert_allclose(d.classification_coverage, counts.members / counts.all_members, atol=0, rtol=0)
    f = pd.read_parquet(ROOT / 'features.parquet'); old = pd.read_parquet(previous.ROOT / 'features.parquet')
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')], old.drop(columns='formula_input_valid'), check_exact=True)
    ex = old[['date', 'code']].merge(d, on='date', how='left', validate='many_to_one')
    pd.testing.assert_frame_equal(f[ex.columns], ex, check_exact=True)
    np.testing.assert_array_equal(f.formula_input_valid, old.formula_input_valid & ex.limit_context_valid.fillna(False))
    encode = lambda x: np.floor(np.clip(x * 100 + 10000 + .000001, 0, 999999))
    np.testing.assert_array_equal(encode(d.loc[good, ['LB01', 'LB02']]), encode(values[good]))
    assert (ROOT / 'YJLB20.tdx').read_text() == HELPER
    assert r['expressions'] == EXPRESSIONS and r['native_header'] == HEADER and r['native_core_gate'] == GATE
    proof = dict(passed=True, feature_report_sha256=sha(ROOT / 'feature_report.json'), rows=len(f), members=len(m), days=len(d),
        all_limits_integer_classifications_counts_ratios_encodings_and_original_48_values_rebuilt=True,
        no_leave_one_out=True, source_local_peer_coverage_reused=True, outcomes_read=False,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_verification.json', proof); return proof


def native():
    checked_sources()
    proof = json.loads((ROOT / 'feature_verification.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256'] == sha(ROOT / 'feature_report.json')
    m = pd.read_parquet(ROOT / 'members.parquet')
    m['kind'] = np.select([~m.classifiable, m.at_upper, m.at_lower, m.touched_upper],
                          ['unknown', 'upper', 'lower', 'retreated'], default='ordinary')
    m['half'] = m.date.str[:4] + np.where(m.date.str[5:7].le('06'), 'H1', 'H2')
    m['identity'] = [hashlib.sha256(('limit-context-native-v1' + d + c).encode()).hexdigest() for d, c in zip(m.date, m.code)]
    samples = m.sort_values('identity').groupby(['half', 'kind'], sort=True).head(2)
    manifest = json.loads(Path('data/research/economic_winner/input_manifest.json').read_text())['source_sha256']
    daily_hashes = json.loads((peers.SOURCE / 'source_manifest.json').read_text())['sha256']
    checked = {}; cases = []
    for row in samples.itertuples(index=False):
        path = MINUTES / row.code[:2].upper() / (row.code[3:] + '.parquet')
        if str(path) not in checked:
            assert sha(path) == manifest[str(path)]; checked[str(path)] = manifest[str(path)]
        q = pd.read_parquet(path, filters=[('timestamp', '>=', pd.Timestamp(row.date)),
            ('timestamp', '<=', pd.Timestamp(row.date + ' 14:49'))])
        assert len(q) == 230 and not q.timestamp.duplicated().any()
        clock = q.timestamp.dt.strftime('%H%M')
        p49 = int(np.floor(float(q.loc[clock.eq('1449'), 'close'].iloc[0]) * 100 + .5))
        hi = int(np.floor(float(q.loc[q.volume.gt(0), 'high'].max()) * 100 + .5))
        lo = int(np.floor(float(q.loc[q.volume.gt(0), 'low'].min()) * 100 + .5))
        path = DAILY / (row.code.replace('.', '_') + '.parquet')
        assert sha(path) == daily_hashes[str(path)]
        daily = pd.read_parquet(path, filters=[('date', '==', row.date)]).iloc[0]
        pc = int(np.floor(float(daily.preclose) * 100 + .5))
        up = int((11 * pc + 5) / 10); dn = int((9 * pc + 5) / 10)
        valid = dn <= lo <= p49 <= hi <= up
        assert (pc, p49, hi, lo, up, dn) == (row.pc, row.p49, row.hi, row.lo, row.upper, row.lower)
        outputs = [1, int(valid), int(valid and p49 == up), int(valid and p49 == dn), int(valid and hi == up)]
        assert outputs[1:] == [int(row.classifiable), int(row.at_upper), int(row.at_lower), int(row.touched_upper)]
        cases.append(dict(date=row.date, code=row.code, kind=row.kind, half=row.half, raw_minutes=len(q), outputs=outputs))
    r = dict(passed=True, feature_report_sha256=sha(ROOT / 'feature_report.json'),
        feature_verification_sha256=sha(ROOT / 'feature_verification.json'), samples=len(cases), cases=cases,
        raw_minutes=sum(x['raw_minutes'] for x in cases), source_sha256=checked,
        integer_native_arithmetic_verified=True, full_client_member_history_and_quote_parity_verified=False,
        software_compilation_verified=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'native_input_verification.json', r)
    return {k: v for k, v in r.items() if k not in ['cases', 'source_sha256']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['features', 'verify_features', 'native'])
    print(json.dumps(globals()[p.parse_args().stage](), ensure_ascii=False, indent=2))
