"""Same-day cross-sectional rotation and incremental tail strength."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_equal_weight as parent
from . import tail_formula_float as previous
from .corporate_cash import save_json, sha

STEM = 'tail_formula_cross_rotation'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
NEW_EXPRESSIONS = {
    'XR01': 'MAX(MIN(100*XRXY/MAX(SQRT(XRXX*XRYY),0.0001),100),-100)',
    'XR02': '(XRSELFY-XRMY-XRB*(XRSELFX-XRMX))/MAX(XRSD,0.01)'}
EXPRESSIONS = {**previous.EXPRESSIONS, **NEW_EXPRESSIONS}
EXTRA_HEADER = """XRN:=INSUM('沪深Ａ股','YJXR20',1,0);
XRMX:=INSUM('沪深Ａ股','YJXR20',2,0)/MAX(XRN,1);
XRMY:=INSUM('沪深Ａ股','YJXR20',3,0)/MAX(XRN,1);
XRXX:=MAX(INSUM('沪深Ａ股','YJXR20',4,0)/MAX(XRN,1)-XRMX*XRMX,0);
XRYY:=MAX(INSUM('沪深Ａ股','YJXR20',5,0)/MAX(XRN,1)-XRMY*XRMY,0);
XRXY:=INSUM('沪深Ａ股','YJXR20',6,0)/MAX(XRN,1)-XRMX*XRMY;
XRB:=XRXY/MAX(XRXX,0.0001);
XRSD:=SQRT(MAX(XRYY-2*XRB*XRXY+XRB*XRB*XRXX,0));
XRPC:=INTPART(DYNAINFO(3)*100+0.5);
XRP20:=INTPART(P20*100+0.5);
XRP49:=INTPART(Q*100+0.5);
XRSELFX:=100*(XRP20/MAX(XRPC,1)-1);
XRSELFY:=100*(XRP49/MAX(XRP20,1)-1);
"""
HEADER = previous.HEADER + EXTRA_HEADER
HELPER_SUFFIX = """XE:=100*(P20/MAX(PC,1)-1);
XT:=100*(P49/MAX(P20,1)-1);
NB:IF(OK,1,0);
EX:IF(OK,XE,0);
TY:IF(OK,XT,0);
EX2:IF(OK,XE*XE,0);
TY2:IF(OK,XT*XT,0);
EXT:IF(OK,XE*XT,0);
"""
HELPER = parent.members_source.HELPER.split('NB:')[0] + HELPER_SUFFIX
GATE = 'XRN>=2000'
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
    assert p['minimum_members'] == 2000 and p['new_fields'] == list(NEW_EXPRESSIONS)
    assert not p['new_2026_prices_allowed']
    proof = json.loads((parent.ROOT / 'native_input_verification.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256'] == sha(parent.ROOT / 'feature_report.json')
    return p


def reference(m, minimum_members=2000):
    """Keep bad members; they invalidate their entire date."""
    assert not m.duplicated(['date', 'code']).any()
    prices = m[['pc', 'p20', 'p49']].to_numpy(float)
    good = np.isfinite(prices).all(axis=1) & (prices > 0).all(axis=1)
    atoms = m[['date', 'code']].copy()
    with np.errstate(divide='ignore', invalid='ignore'):
        atoms['xr_x'] = 100 * (prices[:, 1] - prices[:, 0]) / prices[:, 0]
        atoms['xr_y'] = 100 * (prices[:, 2] - prices[:, 1]) / prices[:, 1]
    atoms['xr_bad'] = ~good
    rows = []
    for day, group in atoms.groupby('date', sort=True):
        valid = len(group) >= minimum_members and not group.xr_bad.any()
        row = dict(date=day, xr_members=len(group), xr_bad_members=int(group.xr_bad.sum()), xr_valid=bool(valid))
        names = ['xr_mx', 'xr_my', 'xr_vx', 'xr_vy', 'xr_cov', 'xr_beta', 'xr_std', 'xr_correlation']
        if valid:
            x, y = group[['xr_x', 'xr_y']].to_numpy(float).T
            mx, my = x.mean(), y.mean()
            dx, dy = x - mx, y - my
            vx, vy, cov = np.dot(dx, dx) / len(x), np.dot(dy, dy) / len(x), np.dot(dx, dy) / len(x)
            beta = cov / max(vx, .0001)
            std = np.sqrt(max(vy - 2 * beta * cov + beta * beta * vx, 0))
            corr = np.clip(100 * cov / max(np.sqrt(vx * vy), .0001), -100, 100)
            row.update(zip(names, [mx, my, vx, vy, cov, beta, std, corr]))
        else:
            row.update({name: np.nan for name in names})
        rows.append(row)
    return atoms, pd.DataFrame(rows)


def mapped_inputs(keys, atoms, daily):
    f = keys.merge(atoms[['date', 'code', 'xr_x', 'xr_y']], on=['date', 'code'], how='left', validate='one_to_one')
    f = f.merge(daily, on='date', how='left', validate='many_to_one')
    f['XR01'] = f.xr_correlation.where(f.xr_valid.eq(True))
    f['XR02'] = ((f.xr_y - f.xr_my - f.xr_beta * (f.xr_x - f.xr_mx)) / f.xr_std.clip(lower=.01)).where(f.xr_valid.eq(True))
    return f


def features():
    p = checked_sources()
    assert not (ROOT / 'feature_report.json').exists()
    members = pd.read_parquet(parent.members_source.ROOT / 'members.parquet')
    atoms, daily = reference(members)
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    f = mapped_inputs(old, atoms, daily).sort_values(['date', 'code']).reset_index(drop=True)
    f['prior_formula_input_valid'] = f.formula_input_valid
    f['formula_input_valid'] &= f.xr_valid.eq(True) & np.isfinite(f[list(NEW_EXPRESSIONS)]).all(axis=1)
    ROOT.mkdir(parents=True, exist_ok=True)
    f.to_parquet(ROOT / 'features.parquet', index=False, compression='zstd')
    daily.to_parquet(ROOT / 'market_reference.parquet', index=False, compression='zstd')
    (ROOT / 'YJXR20.tdx').write_text(HELPER)
    r = dict(protocol_sha256=sha(PROTOCOL), source_hashes=p['source_hashes'],
        features_sha256=sha(ROOT / 'features.parquet'), market_reference_sha256=sha(ROOT / 'market_reference.parquet'),
        helper_sha256=sha(ROOT / 'YJXR20.tdx'), rows=len(f), valid=int(f.formula_input_valid.sum()),
        newly_invalid=int((old.formula_input_valid & ~f.formula_input_valid).sum()),
        member_rows=len(members), days=len(daily), minimum_members=int(daily.xr_members.min()),
        bad_members=int(daily.xr_bad_members.sum()), invalid_reference_days=int((~daily.xr_valid).sum()),
        early_variance_floor_days=int(daily.xr_vx.lt(.0001).sum()), residual_scale_floor_days=int(daily.xr_std.lt(.01).sum()),
        expressions=EXPRESSIONS, native_header=HEADER, native_core_gate=GATE,
        original_members_preserved=True, native_auxiliary_indicator_required=True,
        native_source_parity_verified=False, software_compilation_verified=False,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_report.json', r)
    return {k: v for k, v in r.items() if k not in ['source_hashes', 'expressions', 'native_header']}


def verify_features():
    checked_sources()
    r = json.loads((ROOT / 'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL)
    for key, file in [('features', 'features.parquet'), ('market_reference', 'market_reference.parquet'), ('helper', 'YJXR20.tdx')]:
        assert r[key + '_sha256'] == sha(ROOT / file)
    c = base.conn()
    c.read_parquet(str(parent.members_source.SOURCE / 'visible_base.parquet')).create_view('source')
    m = c.sql("""SELECT date,code,round(preclose*100)::BIGINT AS pc,
        round(price_1420*100)::BIGINT AS p20,round(price_1449*100)::BIGINT AS p49 FROM source
        WHERE code LIKE 'sh.60%' OR code LIKE 'sz.00%' ORDER BY date,code""").df()
    members = pd.read_parquet(parent.members_source.ROOT / 'members.parquet')
    pd.testing.assert_frame_equal(members[m.columns], m, check_exact=True)
    c.register('members', m)
    c.sql('''SELECT date,code,100*(p20::DOUBLE/pc-1) AS x,100*(p49::DOUBLE/p20-1) AS y
        FROM members''').create_view('atoms')
    stats = c.sql('''WITH m AS(SELECT date,count(*) AS xr_members,avg(x) AS xr_mx,avg(y) AS xr_my,
        greatest(avg(x*x)-avg(x)*avg(x),0) AS xr_vx,greatest(avg(y*y)-avg(y)*avg(y),0) AS xr_vy,
        avg(x*y)-avg(x)*avg(y) AS xr_cov FROM atoms GROUP BY date),
        b AS(SELECT *,xr_cov/greatest(xr_vx,.0001) AS xr_beta FROM m)
        SELECT *,sqrt(greatest(xr_vy-2*xr_beta*xr_cov+xr_beta*xr_beta*xr_vx,0)) AS xr_std,
        greatest(least(100*xr_cov/greatest(sqrt(xr_vx*xr_vy),.0001),100),-100) AS xr_correlation
        FROM b ORDER BY date''').df()
    assert np.isfinite(m[['pc', 'p20', 'p49']]).all().all() and (m[['pc', 'p20', 'p49']] > 0).all().all()
    daily = pd.read_parquet(ROOT / 'market_reference.parquet')
    pd.testing.assert_frame_equal(daily[stats.columns], stats, check_dtype=False, rtol=0, atol=2e-10)
    assert daily.xr_valid.all() and daily.xr_bad_members.eq(0).all()
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    f = pd.read_parquet(ROOT / 'features.parquet')
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')], old.drop(columns='formula_input_valid'), check_exact=True)
    pd.testing.assert_series_equal(f.prior_formula_input_valid, old.formula_input_valid, check_names=False, check_exact=True)
    c.register('stats', stats)
    c.register('keys', old[['date', 'code']])
    expected = c.sql('''SELECT k.date,k.code,s.xr_members,s.xr_correlation AS XR01,
        (a.y-s.xr_my-s.xr_beta*(a.x-s.xr_mx))/greatest(s.xr_std,.01) AS XR02
        FROM keys k LEFT JOIN atoms a USING(date,code) LEFT JOIN stats s USING(date) ORDER BY date,code''').df()
    np.testing.assert_allclose(f[list(NEW_EXPRESSIONS)], expected[list(NEW_EXPRESSIONS)], rtol=0, atol=2e-10, equal_nan=True)
    valid = old.formula_input_valid & expected.xr_members.ge(2000) & np.isfinite(expected[list(NEW_EXPRESSIONS)]).all(axis=1)
    np.testing.assert_array_equal(f.formula_input_valid, valid)
    encode = lambda x: np.floor(np.clip(100 * np.asarray(x) + 10000 + .000001, 0, 999999))
    np.testing.assert_array_equal(encode(f.loc[valid, list(NEW_EXPRESSIONS)]), encode(expected.loc[valid, list(NEW_EXPRESSIONS)]))
    atoms = c.sql('SELECT * FROM atoms ORDER BY date,code').df()
    c.close()
    checked = 0
    for day, group in atoms.groupby('date', sort=True):
        row = daily.loc[daily.date.eq(day)].iloc[0]
        if row.xr_vx < .0001:
            continue
        x, y = group[['x', 'y']].to_numpy().T
        design = np.column_stack([np.ones(len(x)), x])
        coeff = np.linalg.lstsq(design, y, rcond=None)[0]
        np.testing.assert_allclose(coeff, [row.xr_my-row.xr_beta*row.xr_mx, row.xr_beta], rtol=0, atol=2e-10)
        residual = y - design @ coeff
        np.testing.assert_allclose(np.sqrt(np.mean(residual**2)), row.xr_std, rtol=0, atol=2e-10)
        np.testing.assert_allclose(design.T @ residual / len(x), [0, 0], rtol=0, atol=2e-10)
        checked += 1
    assert r['rows'] == len(f) and r['valid'] == int(valid.sum()) and r['newly_invalid'] == int((old.formula_input_valid & ~valid).sum())
    assert r['member_rows'] == len(m) == 1483109 and r['days'] == len(stats) == 484
    assert r['bad_members'] == r['invalid_reference_days'] == 0
    assert r['minimum_members'] == int(daily.xr_members.min())
    assert r['early_variance_floor_days'] == int(daily.xr_vx.lt(.0001).sum())
    assert r['residual_scale_floor_days'] == int(daily.xr_std.lt(.01).sum())
    names = re.findall(r'(?m)^([A-Z][A-Z0-9]*):=', HEADER) + list(EXPRESSIONS)
    assert len(names) == len({name.casefold() for name in names})
    assert (ROOT / 'YJXR20.tdx').read_text() == HELPER and r['expressions'] == EXPRESSIONS and r['native_header'] == HEADER
    proof = dict(passed=True, feature_report_sha256=sha(ROOT / 'feature_report.json'),
        all_market_moments_stock_residuals_validity_and_encodings_sql_rebuilt=True,
        least_squares_and_residual_orthogonality_days=checked,
        original_members_keys_and_48_values_preserved=True,
        effective_input_intersection_unchanged=bool(valid.equals(old.formula_input_valid)),
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_verification.json', proof)
    return proof


def native_values(pool, stock):
    env = {'P20': pool.p20.to_numpy(float), 'P49': pool.p49.to_numpy(float), 'PC': pool.pc.to_numpy(float),
        'OK': np.ones(len(pool), dtype=bool), 'MAX': np.maximum, 'IF': np.where}
    sums = []
    for line in HELPER_SUFFIX.strip().splitlines():
        if ':=' in line:
            name, expression = line.rstrip(';').split(':=')
            env[name] = eval(expression, {'__builtins__': {}}, env)
        else:
            _, expression = line.rstrip(';').split(':')
            sums.append(float(np.sum(eval(expression, {'__builtins__': {}}, env))))
    assert len(sums) == 6 and sums[0] == len(pool)
    env = {'INSUM': lambda block, helper, index, mode: sums[index-1], 'MAX': max, 'MIN': min,
        'SQRT': np.sqrt, 'INTPART': np.trunc, 'DYNAINFO': lambda index: stock.pc/100,
        'P20': stock.p20/100, 'Q': stock.p49/100}
    for line in EXTRA_HEADER.strip().splitlines():
        name, expression = line.rstrip(';').split(':=')
        env[name] = eval(expression, {'__builtins__': {}}, env)
    return [eval(expression, {'__builtins__': {}}, env) for expression in NEW_EXPRESSIONS.values()]


def native():
    checked_sources()
    v = json.loads((ROOT / 'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(ROOT / 'feature_report.json')
    original = json.loads((parent.ROOT / 'native_input_verification.json').read_text())
    assert len(original['samples']) == 32
    members = pd.read_parquet(parent.members_source.ROOT / 'members.parquet')
    f = pd.read_parquet(ROOT / 'features.parquet', columns=['date', 'code', 'XR01', 'XR02']).set_index(['date', 'code'])
    receipts = []
    for sample in original['samples']:
        day, code = sample['date'], sample['code']
        pool = members.loc[members.date.eq(day)]
        stock = pool.loc[pool.code.eq(code)].iloc[0]
        np.testing.assert_allclose(sample['helper_outputs'], [1, 100*(stock.p49/stock.pc-1), 100*(stock.p49/stock.p20-1)], rtol=0, atol=2e-10)
        rebuilt = native_values(pool, stock)
        wanted = f.loc[(day, code)].to_numpy(float)
        np.testing.assert_allclose(rebuilt, wanted, rtol=0, atol=2e-10)
        np.testing.assert_array_equal(np.floor(100*np.array(rebuilt)+10000+.000001), np.floor(100*wanted+10000+.000001))
        receipts.append(dict(date=day, code=code, members=len(pool), XR01=rebuilt[0], XR02=rebuilt[1], source_sha256=sample['source_sha256']))
    proof = dict(passed=True, protocol_sha256=sha(PROTOCOL), feature_report_sha256=sha(ROOT / 'feature_report.json'),
        feature_verification_sha256=sha(ROOT / 'feature_verification.json'), samples=receipts,
        reused_raw_sample_report_sha256=sha(parent.ROOT / 'native_input_verification.json'),
        actual_six_helper_outputs_and_main_formula_expressions_rebuilt=True, original_raw_proofs_reused_without_reextraction=True,
        native_source_parity_verified=False, software_compilation_verified=False,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'native_input_verification.json', proof)
    return {k: v for k, v in proof.items() if k != 'samples'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['features', 'verify_features', 'native'])
    print(json.dumps(globals()[p.parse_args().stage](), ensure_ascii=False, indent=2))
