"""Two point-in-time breadth inputs from the covered, historical main-board pool."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as previous
from . import tail_formula_prior_day as adapter
from .corporate_cash import DAILY, MINUTES, save_json, sha

STEM = 'tail_formula_cross_breadth'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
COMBINED_PROTOCOL = Path('config') / (STEM + '_combined_protocol.json')
SOURCE = Path('data/research/next_day_winner')
COLUMNS = ['date', 'code', 'board', 'price_1420', 'price_1449', 'preclose',
           'isST', 'tradestatus', 'listing_age_sessions', 'volume_1449', 'amount_1449']
EXPRESSIONS = {**previous.EXPRESSIONS,
    'CB01': '100*MCU49/MAX(MCU49+MCD49,1)',
    'CB02': '100*(MCU49/MAX(MCU49+MCD49,1)-MCU20/MAX(MCU20+MCD20,1))'}
HEADER = previous.HEADER + ''.join(
    f"{name}:=INSUM('沪深Ａ股','YJKB20',{i},0);\n"
    for i, name in enumerate(['MCN', 'MCU49', 'MCD49', 'MCU20', 'MCD20'], 1))
HELPER = 'Q49:=VALUEWHEN(TIME=1449,C);\nQ20:=VALUEWHEN(TIME=1420,C);\nDD:=DATE<>REF(DATE,1);\nB0:=BARSLAST(DD)+1;\n'
HELPER += ''.join(f'B{i}:=B{i-1}+REF(BARSLAST(DD)+1,B{i-1});\n' for i in range(1, 20))
HELPER += '''PC:=INTPART(DYNAINFO(3)*100+0.5);
P49:=INTPART(Q49*100+0.5);
P20:=INTPART(Q20*100+0.5);
DV:=VALUEWHEN(TIME=1449,SUM(V,B0));
DA:=VALUEWHEN(TIME=1449,SUM(AMOUNT,B0));
OK49:=VALUEWHEN(TIME=1449,DATE)=DATE;
OK20:=VALUEWHEN(TIME=1420,DATE)=DATE;
MB:=FINANCE(3)=1 AND NOT(NAMELIKE('ST')) AND NOT(NAMELIKE('*ST'));
OK:=MB AND OK49 AND OK20 AND REF(DATE,B19)>0 AND DV>0 AND DA>0 AND PC>0 AND P49>0 AND P20>0;
NB:IF(OK,1,0);
U49:IF(OK AND P49>PC,1,0);
D49:IF(OK AND P49<PC,1,0);
U20:IF(OK AND P20>PC,1,0);
D20:IF(OK AND P20<PC,1,0);
'''
GATE = 'MCN>=2000 AND MCU49+MCD49>=1000 AND MCU20+MCD20>=1000'
ORIGINAL_NATIVE_CORE = base.native_core


def native_core(*args, **kwargs):
    s = ORIGINAL_NATIVE_CORE(*args, **kwargs)
    assert s.count('CORE:SC>') == 1
    return s.replace('CORE:SC>', 'CORE:' + GATE + ' AND SC>')


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    for key, path in [('previous_feature_report_sha256', previous.ROOT / 'feature_report.json'),
                      ('source_base_report_sha256', SOURCE / 'base_report.json'),
                      ('source_base_sha256', SOURCE / 'visible_base.parquet'),
                      ('source_base_checks_sha256', SOURCE / 'independent_base_checks.json'),
                      ('source_prefix_audit_sha256', Path('data/research/minute_prefix_1449/input_audit.json'))]:
        assert p[key] == sha(path)
    r = json.loads((previous.ROOT / 'feature_report.json').read_text())
    v = json.loads((previous.ROOT / 'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(previous.ROOT / 'feature_report.json')
    assert r['features_sha256'] == sha(previous.ROOT / 'features.parquet')
    b = json.loads((SOURCE / 'base_report.json').read_text())
    assert b['base_sha256'] == p['source_base_sha256']
    assert b['source_manifest_sha256'] == sha(SOURCE / 'source_manifest.json')
    assert p['minimum_members'] == 2000 and p['minimum_nonflat'] == 1000
    return json.loads((SOURCE / 'source_manifest.json').read_text())['sha256']


def read_members():
    f = pd.read_parquet(SOURCE / 'visible_base.parquet', columns=COLUMNS)
    f = f.loc[f.board.eq('main')].sort_values(['date', 'code']).reset_index(drop=True)
    assert f.date.between('2024-01-01', '2025-12-30').all()
    assert f.isST.eq(0).all() and f.tradestatus.eq(1).all() and f.listing_age_sessions.ge(20).all()
    assert not f.duplicated(['date', 'code']).any()
    for source, name in [('price_1420', 'p20'), ('price_1449', 'p49'), ('preclose', 'pc')]:
        assert np.isfinite(f[source]).all() and f[source].gt(0).all()
        f[name] = np.floor(f[source] * 100 + .5).astype('int64')
    return f


def coverage(source_hashes):
    files = sorted(path for path in source_hashes if path.startswith(str(DAILY) + '/') and path.endswith('.parquet'))
    assert files
    for path in files:
        assert sha(Path(path)) == source_hashes[path]
    c = base.conn(); c.read_parquet(files).create_view('daily')
    out = c.sql('''WITH active AS (
        SELECT date,code,isST,preclose,row_number() OVER(PARTITION BY code ORDER BY date)-1 AS age
        FROM daily WHERE date<='2025-12-30' AND tradestatus=1
        AND (code LIKE 'sh.60%' OR code LIKE 'sz.00%'))
        SELECT date,count(*) AS expected_local_members FROM active
        WHERE date BETWEEN '2024-01-01' AND '2025-12-30' AND isST=0 AND age>=20 AND preclose>0
        GROUP BY date ORDER BY date''').df()
    c.close(); return out


def features():
    hashes = checked_sources(); assert not (ROOT / 'feature_report.json').exists()
    m = read_members(); c = base.conn(); c.register('members', m)
    daily = c.sql('''SELECT date,count(*) AS members,
        sum((p49>pc)::INT) AS up49,sum((p49<pc)::INT) AS down49,
        sum((p20>pc)::INT) AS up20,sum((p20<pc)::INT) AS down20
        FROM members GROUP BY date ORDER BY date''').df(); c.close()
    daily = daily.merge(coverage(hashes), on='date', validate='one_to_one')
    daily['missing_local_members'] = daily.expected_local_members - daily.members
    assert daily.missing_local_members.ge(0).all()
    daily['coverage'] = daily.members / daily.expected_local_members
    daily['cross_breadth_valid'] = daily.members.ge(2000) & (daily.up49 + daily.down49).ge(1000) & (daily.up20 + daily.down20).ge(1000)
    daily['CB01'] = (100 * daily.up49 / (daily.up49 + daily.down49)).where(daily.cross_breadth_valid)
    daily['CB02'] = (100 * (daily.up49 / (daily.up49 + daily.down49) - daily.up20 / (daily.up20 + daily.down20))).where(daily.cross_breadth_valid)
    for suffix in ['20', '49']:
        denominator = daily['up' + suffix] + daily['down' + suffix] + daily.missing_local_members
        daily['coverage_lower' + suffix] = 100 * daily['up' + suffix] / denominator
        daily['coverage_upper' + suffix] = 100 * (daily['up' + suffix] + daily.missing_local_members) / denominator
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    f = old.merge(daily, on='date', how='left', validate='many_to_one').sort_values(['date', 'code']).reset_index(drop=True)
    f['prior_formula_input_valid'] = f.formula_input_valid
    f['formula_input_valid'] &= f.cross_breadth_valid.fillna(False) & np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
    ROOT.mkdir(parents=True, exist_ok=True)
    m.to_parquet(ROOT / 'members.parquet', index=False, compression='zstd')
    daily.to_parquet(ROOT / 'breadth.parquet', index=False, compression='zstd')
    f.to_parquet(ROOT / 'features.parquet', index=False, compression='zstd')
    (ROOT / 'YJKB20.tdx').write_text(HELPER)
    r = dict(protocol_sha256=sha(PROTOCOL), source_base_report_sha256=sha(SOURCE / 'base_report.json'),
        previous_feature_report_sha256=sha(previous.ROOT / 'feature_report.json'),
        features_sha256=sha(ROOT / 'features.parquet'), members_sha256=sha(ROOT / 'members.parquet'),
        breadth_sha256=sha(ROOT / 'breadth.parquet'), helper_sha256=sha(ROOT / 'YJKB20.tdx'),
        rows=len(f), valid=int(f.formula_input_valid.sum()), previous_valid=int(old.formula_input_valid.sum()),
        newly_invalid=int((old.formula_input_valid & ~f.formula_input_valid).sum()),
        member_rows=len(m), member_codes=m.code.nunique(), days=len(daily), invalid_days=int((~daily.cross_breadth_valid).sum()),
        minimum_daily_members=int(daily.members.min()), maximum_daily_members=int(daily.members.max()),
        minimum_local_coverage=float(daily.coverage.min()), missing_local_stock_days=int(daily.missing_local_members.sum()),
        maximum_coverage_interval_width_pp=float((daily.coverage_upper49 - daily.coverage_lower49).max()),
        expressions=EXPRESSIONS, native_header=HEADER, native_core_gate=GATE,
        coverage_diagnostic_not_used_to_filter_dates=True, includes_current_stock_without_leave_one_out=True,
        native_auxiliary_indicator_required=True, native_client_membership_and_source_parity_verified=False,
        software_compilation_verified=False, new_selection_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_report.json', r)
    return {k: v for k, v in r.items() if k not in ['expressions', 'native_header']}


def verify_features():
    checked_sources(); r = json.loads((ROOT / 'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL)
    for key, file in [('features', 'features.parquet'), ('members', 'members.parquet'), ('breadth', 'breadth.parquet'), ('helper', 'YJKB20.tdx')]:
        assert r[key + '_sha256'] == sha(ROOT / file)
    m = pd.read_parquet(ROOT / 'members.parquet')
    source = pd.read_parquet(SOURCE / 'visible_base.parquet', columns=COLUMNS)
    source = source.loc[source.code.str.match(r'(sh\.60|sz\.00)')].sort_values(['date', 'code']).reset_index(drop=True)
    pd.testing.assert_frame_equal(m[COLUMNS], source, check_exact=True)
    for col, name in [('price_1420', 'p20'), ('price_1449', 'p49'), ('preclose', 'pc')]:
        np.testing.assert_array_equal(m[name], np.rint(source[col] * 100).astype('int64'))
    records = []
    for date, rows in m.groupby('date', sort=True):
        record = dict(date=date, members=len(rows))
        for suffix in ['20', '49']:
            signs = np.sign(rows['p' + suffix].to_numpy() - rows.pc.to_numpy())
            record['up' + suffix] = int(np.count_nonzero(signs > 0))
            record['down' + suffix] = int(np.count_nonzero(signs < 0))
        records.append(record)
    counts = pd.DataFrame(records)
    d = pd.read_parquet(ROOT / 'breadth.parquet')
    pd.testing.assert_frame_equal(d[counts.columns], counts, check_dtype=False, check_exact=True)
    valid = counts.members.ge(2000) & (counts.up49 + counts.down49).ge(1000) & (counts.up20 + counts.down20).ge(1000)
    assert valid.equals(d.cross_breadth_valid)
    h1, h2 = counts.up49 + counts.down49, counts.up20 + counts.down20
    x1 = (counts.up49 / h1 * 100).where(valid)
    x2 = (100 * (counts.up49 * h2 - counts.up20 * h1) / (h1 * h2)).where(valid)
    np.testing.assert_allclose(d[['CB01', 'CB02']], np.column_stack([x1, x2]), rtol=0, atol=4e-14, equal_nan=True)
    f = pd.read_parquet(ROOT / 'features.parquet'); old = pd.read_parquet(previous.ROOT / 'features.parquet')
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')], old.drop(columns='formula_input_valid'), check_exact=True)
    expected = old[['date', 'code']].merge(d, on='date', how='left', validate='many_to_one')
    pd.testing.assert_frame_equal(f[expected.columns], expected, check_exact=True)
    assert f.formula_input_valid.equals(old.formula_input_valid & expected.cross_breadth_valid)
    encode = lambda x: np.floor(np.clip(x * 100 + 10000 + .000001, 0, 999999))
    np.testing.assert_array_equal(encode(d.loc[valid, ['CB01', 'CB02']]), encode(np.column_stack([x1[valid], x2[valid]])))
    assert (ROOT / 'YJKB20.tdx').read_text() == HELPER
    assert r['expressions'] == EXPRESSIONS and r['native_header'] == HEADER and r['native_core_gate'] == GATE
    proof = dict(passed=True, feature_report_sha256=sha(ROOT / 'feature_report.json'),
        rows=len(f), members=len(m), days=len(d), valid=int(f.formula_input_valid.sum()),
        all_member_keys_signs_counts_daily_ratios_and_integer_encodings_independently_rebuilt=True,
        original_48_inputs_unchanged=True, no_leave_one_out_mechanical_signal=True,
        coverage_counts_are_diagnostics_not_input_gates=True,
        new_selection_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_verification.json', proof); return proof


def native():
    hashes = checked_sources()
    v = json.loads((ROOT / 'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(ROOT / 'feature_report.json')
    m = pd.read_parquet(ROOT / 'members.parquet')
    m['half'] = m.date.str[:4] + np.where(m.date.str[5:7].le('06'), 'H1', 'H2')
    m['identity'] = [hashlib.sha256(f'cross_breadth_native|{d}|{c}'.encode()).hexdigest() for d, c in zip(m.date, m.code)]
    samples = m.sort_values('identity').groupby('half', sort=True).head(8)
    manifest = Path('data/research/economic_winner/input_manifest.json')
    raw_hashes = json.loads(manifest.read_text())['source_sha256']
    checked = {}; cases = []; total = 0
    for row in samples.itertuples(index=False):
        path = MINUTES / row.code[:2].upper() / (row.code[3:] + '.parquet')
        if str(path) not in checked:
            assert sha(path) == raw_hashes[str(path)]; checked[str(path)] = raw_hashes[str(path)]
        q = pd.read_parquet(path, columns=['timestamp', 'close', 'volume', 'turnover'], filters=[
            ('timestamp', '>=', pd.Timestamp(row.date)), ('timestamp', '<=', pd.Timestamp(row.date + ' 14:49:00'))])
        labels = q.timestamp.dt.strftime('%H%M')
        assert len(q) == 230 and not q.timestamp.duplicated().any()
        p20 = int(np.floor(float(q.loc[labels.eq('1420'), 'close'].iloc[0]) * 100 + .5))
        p49 = int(np.floor(float(q.loc[labels.eq('1449'), 'close'].iloc[0]) * 100 + .5))
        dp = DAILY / (row.code.replace('.', '_') + '.parquet')
        assert sha(dp) == hashes[str(dp)]
        raw = pd.read_parquet(dp, columns=['date', 'preclose', 'tradestatus', 'isST'], filters=[('date', '==', row.date)]).iloc[0]
        pc = int(np.floor(float(raw.preclose) * 100 + .5))
        assert (p20, p49, pc) == (row.p20, row.p49, row.pc)
        assert raw.tradestatus == 1 and raw.isST == 0 and q.volume.sum() > 0 and q.turnover.sum() > 0
        total += len(q); cases.append(dict(date=row.date, code=row.code, raw_minutes=len(q),
            outputs=[1, int(p49 > pc), int(p49 < pc), int(p20 > pc), int(p20 < pc)]))
    assert len(cases) == 32
    r = dict(passed=True, feature_report_sha256=sha(ROOT / 'feature_report.json'),
        feature_verification_sha256=sha(ROOT / 'feature_verification.json'),
        samples=len(cases), raw_minutes=total, cases=cases, source_sha256=checked,
        current_day_helper_arithmetic_rebuilt_from_raw_quotes=True,
        all_cross_section_sums_verified_by_feature_proof=True,
        full_native_history_member_guards_and_client_coverage_verified=False,
        software_compilation_verified=False, new_selection_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'native_input_verification.json', r)
    return {k: v for k, v in r.items() if k not in ['cases', 'source_sha256']}


def verify_coverage():
    hashes = checked_sources(); frames = []
    for name in sorted(hashes):
        path = Path(name)
        if path.parent != DAILY or not path.stem.startswith(('sh_60', 'sz_00')):
            continue
        assert sha(path) == hashes[name]
        d = pd.read_parquet(path, columns=['date', 'code', 'isST', 'tradestatus', 'preclose'],
                            filters=[('date', '<=', '2025-12-30')])
        d = d.loc[d.tradestatus.eq(1)].sort_values('date').reset_index(drop=True)
        assert not d.date.duplicated().any()
        mask = (d.index >= 20) & d.date.ge('2024-01-01') & d.isST.eq(0) & d.preclose.gt(0)
        frames.append(d.loc[mask, ['date', 'code']])
    expected = pd.concat(frames, ignore_index=True).sort_values(['date', 'code']).reset_index(drop=True)
    m = pd.read_parquet(ROOT / 'members.parquet', columns=['date', 'code'])
    joined = expected.merge(m, on=['date', 'code'], how='outer', validate='one_to_one', indicator=True)
    assert not joined['_merge'].eq('right_only').any()
    count = expected.groupby('date').size()
    missing = joined.loc[joined['_merge'].eq('left_only')].groupby('date').size()
    d = pd.read_parquet(ROOT / 'breadth.parquet')
    np.testing.assert_array_equal(d.expected_local_members, d.date.map(count))
    np.testing.assert_array_equal(d.missing_local_members, d.date.map(missing).fillna(0))
    np.testing.assert_allclose(d.coverage, 1 - d.missing_local_members / d.expected_local_members, rtol=0, atol=2e-16)
    for suffix in ['20', '49']:
        up = d['up' + suffix]; down = d['down' + suffix]; absent = d.missing_local_members
        np.testing.assert_allclose(d['coverage_lower' + suffix], up / (up + down + absent) * 100, rtol=0, atol=2e-14)
        np.testing.assert_allclose(d['coverage_upper' + suffix], (up + absent) / (up + down + absent) * 100, rtol=0, atol=2e-14)
    r = dict(passed=True, feature_report_sha256=sha(ROOT / 'feature_report.json'),
        source_manifest_sha256=sha(SOURCE / 'source_manifest.json'), expected_local_stock_days=len(expected),
        missing_stock_days=int(d.missing_local_members.sum()),
        all_local_daily_stock_ages_eligibility_identities_and_coverage_bounds_independently_rebuilt=True,
        local_daily_universe_is_not_verified_exchange_master=True,
        no_new_selection_outcomes_read=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'coverage_verification.json', r); return r


def configure():
    checked_sources(); v = json.loads((ROOT / 'native_input_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(ROOT / 'feature_report.json')
    assert v['feature_verification_sha256'] == sha(ROOT / 'feature_verification.json')
    coverage_proof = json.loads((ROOT / 'coverage_verification.json').read_text())
    assert coverage_proof['passed'] and coverage_proof['feature_report_sha256'] == sha(ROOT / 'feature_report.json')
    adapter.STEM = STEM; adapter.ROOT = ROOT; adapter.PROTOCOL = PROTOCOL
    adapter.COMBINED_PROTOCOL = COMBINED_PROTOCOL; adapter.CONTROL = Path('data/research') / (STEM + '_control')
    adapter.EXPRESSIONS = EXPRESSIONS; adapter.HEADER = HEADER; base.native_core = native_core
    for fold in ['2024', 'recent', 'combined']:
        assert json.loads((Path('config') / (STEM + '_' + fold + '_protocol.json')).read_text())['inputs_protocol_sha256'] == sha(PROTOCOL)


if __name__ == '__main__':
    from . import tail_formula_context_2024 as linkage
    from . import tail_formula_recent as study
    from . import tail_formula_relative as relative
    from .tail_formula_offset_logit48 import verify_scores
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['features', 'verify_features', 'verify_coverage', 'native', 'model', 'verify_model', 'scores', 'verify_scores', 'freeze', 'verify', 'analyze'])
    p.add_argument('--fold', choices=['2024', 'recent', 'combined', 'control'], default='2024'); a = p.parse_args()
    if a.stage in ['features', 'verify_features', 'verify_coverage', 'native']:
        result = globals()[a.stage]()
    else:
        configure()
        if a.stage == 'analyze':
            assert all((Path('data/research') / (STEM + '_' + f) / 'selection_verification.json').exists() for f in ['2024', 'recent', '2025', 'control'])
        if a.fold == 'control':
            assert a.stage in ['freeze', 'verify', 'analyze']
            result = (linkage.common_analysis(adapter.CONTROL, COMBINED_PROTOCOL) if a.stage == 'analyze' else adapter.control(a.stage + '_control'))
        else:
            adapter.setup(a.fold)
            if a.fold == 'combined':
                assert a.stage in ['freeze', 'verify', 'analyze']
                result = (linkage.common_analysis(linkage.COMBINED, linkage.PROTOCOL) if a.stage == 'analyze' else getattr(linkage, 'combine' if a.stage == 'freeze' else 'verify_combined')())
            elif a.stage in ['model', 'verify_model']:
                result = getattr(relative, a.stage)('relative')
            elif a.stage == 'verify_scores':
                result = verify_scores()
            elif a.stage in ['freeze', 'verify']:
                result = getattr(study, a.stage)()
            else:
                result = getattr(base, a.stage)()
    print(json.dumps(result, ensure_ascii=False, indent=2))
