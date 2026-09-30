"""Fixed, integer-price cross-sectional tail-return bins; no future inputs."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_cross_breadth as members_source
from . import tail_formula_morning_range as prior
from .corporate_cash import save_json, sha

STEM = 'tail_formula_market_bins'
ROOT = Path('data/research') / STEM
INPUTS = ROOT / 'inputs'
PROTOCOL = Path('config') / (STEM + '_input_protocol.json')
INTENT = Path('config') / (STEM + '_intent.json')
META = prior.META
BOUNDARIES = [-200, -100, -50, 0, 50, 100, 200]
COUNT_NAMES = ['mb_members', *[f'mb_lt{i}' for i in range(1, 8)]]
HELPER = members_source.HELPER.split('NB:')[0] + 'NB:IF(OK,1,0);\n'
HELPER += ''.join(f'LT{i}:IF(OK AND 10000*(P49-P20)<({b})*P20,1,0);\n'
                  for i, b in enumerate(BOUNDARIES, 1))
EXTRA_HEADER = ''.join(f"{n}:=INSUM('沪深Ａ股','YJMB20',{i},0);\n"
                       for i, n in enumerate(['MBN', *[f'MBL{j}' for j in range(1, 8)]], 1))
EXTRA_HEADER += 'MBP20:=ROUND(VALUEWHEN(TIME=1420,C)*100);\nMBP49:=ROUND(Q*100);\n'
EXTRA_HEADER += 'MBBIN:=' + '+'.join(f'IF(10000*(MBP49-MBP20)>=({b})*MBP20,1,0)' for b in BOUNDARIES) + ';\n'


def choose(values):
    out = values[-1]
    for i in range(6, -1, -1):
        out = f'IF(MBBIN={i},{values[i]},{out})'
    return out


EXTRA_HEADER += 'MBLOW:=' + choose(['0', *[f'MBL{i}' for i in range(1, 8)]]) + ';\n'
EXTRA_HEADER += 'MBHIGH:=' + choose([*[f'MBL{i}' for i in range(1, 8)], 'MBN']) + ';\n'
EXTRA_HEADER += ('MBREADY:=MBN>=2000 AND MBP20>0 AND MBP49>0 '
                 'AND VALUEWHEN(TIME=1420,DATE)=DATE AND VALUEWHEN(TIME=1449,DATE)=DATE '
                 'AND MBHIGH>=MBLOW AND MBHIGH<=MBN;\n')
HEADER = prior.HEADER + EXTRA_HEADER
CORE_GATE = 'MBREADY'
NEW_EXPRESSIONS = {
    'MB01': 'IF(MBREADY,100*(MBLOW+0.5*(MBHIGH-MBLOW))/MAX(MBN,1),DRAWNULL)',
    'MB02': 'IF(MBREADY,100*(MBHIGH-MBLOW)/MAX(MBN,1),DRAWNULL)'}
ARMS = {'control': prior.EXPRESSIONS, 'bins': {**prior.EXPRESSIONS, **NEW_EXPRESSIONS}}
EXPRESSIONS = ARMS['bins']


def checked():
    p = json.loads(PROTOCOL.read_text())
    prior.checked()
    assert p['intent_sha256'] == sha(INTENT) and p['arms'] == ARMS
    assert p['boundaries_bps'] == BOUNDARIES and p['minimum_members'] == 2000
    assert p['native_header'] == HEADER and p['native_helper'] == HELPER
    assert not p['new_2026_prices_allowed'] and p['no_new_raw_extraction']
    for f, d in p['source_hashes'].items():
        assert sha(Path(f)) == d, f
    gate = json.loads(Path(p['conditional_gate']).read_text())
    complete = json.loads(Path(p['conditional_completion']).read_text())
    assert gate['passed'] and not gate['supports_2024_extension'] and complete['passed']
    for f, d in complete['source_hashes'].items():
        assert sha(Path(f)) == d, f
    for name in ['feature_verification', 'coverage_verification', 'native_input_verification']:
        v = json.loads((members_source.ROOT / (name + '.json')).read_text())
        assert v['passed'] and v['feature_report_sha256'] == sha(members_source.ROOT / 'feature_report.json')
    r = json.loads((members_source.ROOT / 'feature_report.json').read_text())
    assert r['members_sha256'] == sha(members_source.ROOT / 'members.parquet')
    assert r['breadth_sha256'] == sha(members_source.ROOT / 'breadth.parquet')
    return p


def bin_index(p20, p49):
    a, b = np.asarray(p20), np.asarray(p49)
    assert a.shape == b.shape and a.dtype.kind in 'iu' and b.dtype.kind in 'iu'
    assert ((a > 0) & (a < 10**9) & (b > 0) & (b < 10**9)).all()
    a, b = a.astype('int64'), b.astype('int64')
    return sum((10000 * (b - a) >= boundary * a).astype('int64') for boundary in BOUNDARIES)


def helper_atoms(p20, p49, eligible=True):
    if not eligible:
        return [0] * 8
    bin_index(np.array([p20], dtype='int64'), np.array([p49], dtype='int64'))
    return [1, *[int(10000 * (int(p49) - int(p20)) < b * int(p20)) for b in BOUNDARIES]]


def native_values(p20, p49, counts):
    n, *cum = map(int, counts)
    assert len(cum) == 7 and all(0 <= x <= n for x in cum) and cum == sorted(cum)
    if n < 2000 or p20 <= 0 or p49 <= 0:
        return np.array([np.nan, np.nan])
    # Literal IF-bin selection, independent of vectorized production grouping.
    i = sum(int(10000 * (int(p49) - int(p20)) >= b * int(p20)) for b in BOUNDARIES)
    low, high = [0, *cum][i], [*cum, n][i]
    return np.array([100 * (low + .5 * (high - low)) / n, 100 * (high - low) / n])


def features():
    checked()
    assert not (INPUTS / 'feature_report.json').exists(), 'Do not replace frozen bin inputs'
    original = pd.read_parquet(prior.INPUTS / 'features.parquet', columns=[*META, *prior.EXPRESSIONS])
    m = pd.read_parquet(members_source.ROOT / 'members.parquet', columns=['date', 'code', 'p20', 'p49'])
    assert len(original) == 1258085 and original.date.between('2024-01-01', '2025-12-30').all()
    m['mb_bin'] = bin_index(m.p20.to_numpy(), m.p49.to_numpy())
    daily = m.groupby(['date', 'mb_bin']).size().unstack(fill_value=0).reindex(columns=range(8), fill_value=0)
    counts = pd.DataFrame({'mb_members': daily.sum(axis=1)})
    for i in range(1, 8):
        counts[f'mb_lt{i}'] = daily.iloc[:, :i].sum(axis=1)
    counts = counts.reset_index()
    a = m.merge(counts, on='date', validate='many_to_one')
    cumulative = np.column_stack([np.zeros(len(a), dtype='int64'), a[COUNT_NAMES].to_numpy()[:, 1:], a.mb_members])
    bins = a.mb_bin.to_numpy(); rows = np.arange(len(a))
    a['mb_low'], a['mb_high'] = cumulative[rows, bins], cumulative[rows, bins + 1]
    a['MB01'] = 100 * (a.mb_low + .5 * (a.mb_high - a.mb_low)) / a.mb_members
    a['MB02'] = 100 * (a.mb_high - a.mb_low) / a.mb_members
    f = original.merge(a.rename(columns={'p20': 'mb_p20', 'p49': 'mb_p49'}), on=['date', 'code'], how='left', validate='one_to_one')
    valid = f.mb_members.ge(2000) & f.mb_p20.gt(0) & f.mb_p49.gt(0) & np.isfinite(f[list(NEW_EXPRESSIONS)]).all(axis=1)
    f['prior_formula_input_valid'] = original.formula_input_valid
    f['formula_input_valid'] = original.formula_input_valid & valid
    f.loc[~valid, list(NEW_EXPRESSIONS)] = np.nan
    INPUTS.mkdir(parents=True, exist_ok=True)
    f.to_parquet(INPUTS / 'features.parquet', index=False, compression='zstd')
    counts.to_parquet(INPUTS / 'bin_counts.parquet', index=False, compression='zstd')
    (INPUTS / 'YJMB20.tdx').write_text(HELPER)
    for file in ['full_labels.parquet', 'full_label_report.json', 'full_label_verification.json']:
        (INPUTS / file).symlink_to((prior.INPUTS / file).resolve())
    r = dict(protocol_sha256=sha(PROTOCOL), source_hashes=json.loads(PROTOCOL.read_text())['source_hashes'],
             features_sha256=sha(INPUTS / 'features.parquet'), counts_sha256=sha(INPUTS / 'bin_counts.parquet'),
             helper_sha256=sha(INPUTS / 'YJMB20.tdx'), rows=len(f), valid=int(f.formula_input_valid.sum()),
             previous_valid=int(original.formula_input_valid.sum()), newly_invalid=int((original.formula_input_valid & ~f.formula_input_valid).sum()),
             member_rows=len(m), days=len(counts), minimum_members=int(counts.mb_members.min()),
             maximum_members=int(counts.mb_members.max()), expressions=EXPRESSIONS, native_header=HEADER,
             native_core_gate=CORE_GATE, includes_own_stock=True, coarse_bin_cdf_not_exact_tie_rank=True,
             software_compilation_verified=False, native_source_parity_verified=False,
             no_new_raw_extraction=True, new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(INPUTS / 'feature_report.json', r)
    return {k: v for k, v in r.items() if k not in ['source_hashes', 'expressions', 'native_header']}


def verify():
    checked(); r = json.loads((INPUTS / 'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL)
    for key, file in [('features', 'features.parquet'), ('counts', 'bin_counts.parquet'), ('helper', 'YJMB20.tdx')]:
        assert r[key + '_sha256'] == sha(INPUTS / file)
    # Rebuild cents directly from the independently proven visible source, not the new cache.
    c = base.conn()
    c.sql(f"""SELECT date,code,round(price_1420*100)::BIGINT AS p20,round(price_1449*100)::BIGINT AS p49
        FROM read_parquet('{members_source.SOURCE}/visible_base.parquet')
        WHERE code LIKE 'sh.60%' OR code LIKE 'sz.00%'""").create_view('m')
    member = c.sql('SELECT * FROM m ORDER BY date,code').df()
    old_member = pd.read_parquet(members_source.ROOT / 'members.parquet', columns=member.columns.tolist())
    pd.testing.assert_frame_equal(member, old_member, check_exact=True)
    fields = ['count(*)::BIGINT AS mb_members']
    fields += [f'sum((10000*(p49-p20)<({b})*p20)::BIGINT)::BIGINT AS mb_lt{i}' for i, b in enumerate(BOUNDARIES, 1)]
    c.sql('SELECT date,' + ','.join(fields) + ' FROM m GROUP BY date').create_view('daily')
    expected_counts = c.sql('SELECT * FROM daily ORDER BY date').df()
    counts = pd.read_parquet(INPUTS / 'bin_counts.parquet')
    pd.testing.assert_frame_equal(counts, expected_counts, check_exact=True)
    when = ' '.join(f'WHEN 10000*(p49-p20)<({b})*p20 THEN {i}' for i, b in enumerate(BOUNDARIES))
    c.sql(f'SELECT *,CASE {when} ELSE 7 END::BIGINT AS mb_bin FROM m').create_view('bins')
    lows = ' '.join(f'WHEN {i} THEN mb_lt{i}' for i in range(1, 8))
    highs = ' '.join(f'WHEN {i} THEN mb_lt{i+1}' for i in range(7))
    c.sql(f'''SELECT b.date,code,p20 AS mb_p20,p49 AS mb_p49,mb_bin,{','.join(COUNT_NAMES)},
        CASE mb_bin {lows} ELSE 0 END::BIGINT AS mb_low,
        CASE mb_bin {highs} ELSE mb_members END::BIGINT AS mb_high
        FROM bins b JOIN daily d ON b.date=d.date''').create_view('atoms')
    fields = ','.join('a.'+n for n in ['mb_p20', 'mb_p49', 'mb_bin', *COUNT_NAMES, 'mb_low', 'mb_high'])
    c.sql(f"SELECT date,code FROM read_parquet('{prior.INPUTS}/features.parquet')").create_view('keys')
    expected = c.sql(f'''SELECT k.*, {fields},
        50.*(mb_low+mb_high)/mb_members AS MB01,100.*(mb_high-mb_low)/mb_members AS MB02
        FROM keys k LEFT JOIN atoms a USING(date,code) ORDER BY date,code''').df(); c.close()
    f = pd.read_parquet(INPUTS / 'features.parquet')
    original = pd.read_parquet(prior.INPUTS / 'features.parquet', columns=[*META, *prior.EXPRESSIONS])
    pd.testing.assert_frame_equal(f[[*META, *prior.EXPRESSIONS]], original, check_exact=True)
    pd.testing.assert_frame_equal(f[expected.columns.drop(list(NEW_EXPRESSIONS))], expected.drop(columns=list(NEW_EXPRESSIONS)), check_exact=True)
    a, b = f[list(NEW_EXPRESSIONS)].to_numpy(), expected[list(NEW_EXPRESSIONS)].to_numpy()
    np.testing.assert_allclose(a, b, rtol=0, atol=2e-12, equal_nan=True)
    encode = lambda x: np.floor(np.clip(100*x + 10000 + .000001, 0, 999999)).astype('int32')
    np.testing.assert_array_equal(encode(a), encode(b))
    assert np.isfinite(a).all() and f.formula_input_valid.equals(original.formula_input_valid)
    assert f.mb_high.ge(f.mb_low).all() and f.mb_high.le(f.mb_members).all()
    assert (INPUTS / 'YJMB20.tdx').read_text() == HELPER
    proof = dict(passed=True, feature_report_sha256=sha(INPUTS / 'feature_report.json'), rows=len(f),
                 member_rows=len(member), days=len(counts), max_feature_difference=float(np.abs(a-b).max()),
                 all_members_cents_strict_boundaries_counts_bins_ratios_and_encodings_independently_rebuilt=True,
                 effective_input_intersection_unchanged=True, all_original_50_values_and_metadata_exact=True,
                 original_full_local_coverage_proof_reused_after_member_equality=True,
                 no_new_raw_extraction=True, new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(INPUTS / 'feature_verification.json', proof); return proof


def native():
    checked(); f = pd.read_parquet(INPUTS / 'features.parquet')
    v = json.loads((INPUTS / 'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(INPUTS / 'feature_report.json')
    counts = f[COUNT_NAMES].to_numpy(dtype='int64'); bins = f.mb_bin.to_numpy(dtype='int64')
    low = np.select([bins == i for i in range(8)], [np.zeros(len(f)), *[counts[:, i] for i in range(1, 8)]])
    high = np.select([bins == i for i in range(8)], [*[counts[:, i] for i in range(1, 8)], counts[:, 0]])
    literal = np.column_stack([100*(low+.5*(high-low))/counts[:, 0], 100*(high-low)/counts[:, 0]])
    np.testing.assert_allclose(literal, f[list(NEW_EXPRESSIONS)], rtol=0, atol=2e-12)
    encode = lambda x: np.floor(np.clip(100*x+10000+.000001,0,999999)).astype('int32')
    np.testing.assert_array_equal(encode(literal), encode(f[list(NEW_EXPRESSIONS)].to_numpy()))
    m = pd.read_parquet(members_source.ROOT / 'members.parquet'); sources = json.loads((members_source.ROOT / 'native_input_verification.json').read_text())
    for path, digest in sources['source_sha256'].items():
        assert sha(Path(path)) == digest, path
    samples = []
    for sample in sources['cases']:
        row = m.loc[m.date.eq(sample['date']) & m.code.eq(sample['code'])].iloc[0]
        x = f.loc[f.date.eq(row.date) & f.code.eq(row.code)]
        # Some original raw-proof members need not be in the signal pool.
        day = pd.read_parquet(INPUTS / 'bin_counts.parquet').set_index('date').loc[row.date]
        atoms = helper_atoms(int(row.p20), int(row.p49))
        if len(x):
            np.testing.assert_allclose(native_values(int(row.p20), int(row.p49), day[COUNT_NAMES]), x[list(NEW_EXPRESSIONS)].iloc[0], rtol=0, atol=2e-12)
        samples.append(dict(date=row.date,code=row.code,helper_outputs=atoms,raw_minutes_proof_reused=sample['raw_minutes']))
    assert len(samples) == 32
    # Exact boundary inclusivity, nonuniform counts, empty bins and no member are tested separately.
    from fractions import Fraction
    for p20 in [100, 10000, 100000000]:
        for boundary in BOUNDARIES:
            threshold = Fraction(p20) * (1 + Fraction(boundary, 10000))
            for p49 in [int(threshold)-1, int(threshold), int(threshold)+1]:
                if p49 > 0:
                    expected = sum(Fraction(p49-p20,p20)*10000 >= b for b in BOUNDARIES)
                    assert bin_index(np.array([p20]), np.array([p49]))[0] == expected
    assert np.isnan(native_values(10000,10000,[1999,0,0,0,1000,1000,1500,1999])).all()
    assert helper_atoms(0,0,False) == [0]*8
    proof = dict(passed=True,feature_report_sha256=sha(INPUTS / 'feature_report.json'),
                 feature_verification_sha256=sha(INPUTS / 'feature_verification.json'), samples=samples,
                 original_50_native_proof_sha256=sha(prior.INPUTS / 'native_input_verification.json'),
                 member_raw_proof_sha256=sha(members_source.ROOT / 'native_input_verification.json'),
                 new_literal_values_and_integer_encodings_checked=2*len(f),
                 reused_fixed_raw_source_cases=32,no_new_raw_extraction=True,
                 strict_boundary_cross_products_IF_selection_and_count_sums_verified=True,
                 software_compilation_verified=False,native_source_parity_verified=False,
                 new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(INPUTS / 'native_input_verification.json',proof)
    return {k:v for k,v in proof.items() if k!='samples'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['features', 'verify', 'native'])
    print(json.dumps(globals()[parser.parse_args().stage](), ensure_ascii=False, indent=2))
