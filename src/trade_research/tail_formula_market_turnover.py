"""Stock turnover relative to the contemporaneous valid-input stock pool."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as previous
from .corporate_cash import save_json, sha

STEM = 'tail_formula_market_turnover'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
EXPRESSIONS = {**previous.EXPRESSIONS, 'MT01': 'S02/MTD', 'MT02': 'S03/MTT'}
HEADER = previous.HEADER + """{YJMTO48完整成员实现及客户端历史股本一致性尚未核准，仅研究数值结构。}
MTN:=INSUM('沪深Ａ股','YJMTO48',1,0);
MTD:=INSUM('沪深Ａ股','YJMTO48',2,0)/MAX(MTN,1);
MTT:=INSUM('沪深Ａ股','YJMTO48',3,0)/MAX(MTN,1);
"""
ORIGINAL_NATIVE_CORE = base.native_core


def native_core(*args, **kwargs):
    text = ORIGINAL_NATIVE_CORE(*args, **kwargs)
    assert text.count('CORE:SC>') == 1
    return text.replace('CORE:SC>', 'CORE:MTN>0 AND MTD>0 AND MTT>0 AND SC>')


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    r = json.loads((previous.ROOT / 'feature_report.json').read_text())
    v = json.loads((previous.ROOT / 'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(previous.ROOT / 'feature_report.json')
    assert r['features_sha256'] == sha(previous.ROOT / 'features.parquet')
    return p


def relative_turnover(old):
    """Only input validity and same-date S02/S03 define the reference pool."""
    assert not old.duplicated(['date', 'code']).any()
    valid = old.formula_input_valid
    assert np.isfinite(old.loc[valid, ['S02', 'S03']]).all().all()
    stats = old.loc[valid].groupby('date', sort=True).agg(
        mt_members=('code', 'size'), mt_day_mean=('S02', 'mean'), mt_tail_mean=('S03', 'mean')).reset_index()
    f = old.merge(stats, on='date', how='left', validate='many_to_one').sort_values(['date', 'code']).reset_index(drop=True)
    good = f.mt_members.gt(0) & np.isfinite(f[['mt_day_mean', 'mt_tail_mean']]).all(axis=1)
    good &= f.mt_day_mean.gt(0) & f.mt_tail_mean.gt(0)
    f['prior_formula_input_valid'] = f.formula_input_valid
    f['formula_input_valid'] &= good
    f['MT01'] = (f.S02 / f.mt_day_mean).where(f.formula_input_valid)
    f['MT02'] = (f.S03 / f.mt_tail_mean).where(f.formula_input_valid)
    f['formula_input_valid'] &= np.isfinite(f[['MT01', 'MT02']]).all(axis=1)
    return f, stats


def features():
    p = checked_sources()
    assert not (ROOT / 'feature_report.json').exists()
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    f, stats = relative_turnover(old)
    assert len(f) == p['all_keys_expected'] and int(old.formula_input_valid.sum()) == p['all_members_count_expected']
    ROOT.mkdir(parents=True, exist_ok=True)
    f.to_parquet(ROOT / 'features.parquet', index=False, compression='zstd')
    stats.to_parquet(ROOT / 'market_reference.parquet', index=False, compression='zstd')
    interface = dict(name='YJMTO48', outputs=[
        '完整原硬过滤且48项有效者输出1，否则0',
        '同一标记下S02，否则0', '同一标记下S03，否则0'],
        full_client_membership_implementation_verified=False,
        historical_FINANCE7_proxy_parity_verified=False,
        mathematical_interface_only=True)
    save_json(ROOT / 'helper_interface.json', interface)
    report = dict(protocol_sha256=sha(PROTOCOL), previous_feature_report_sha256=sha(previous.ROOT / 'feature_report.json'),
        features_sha256=sha(ROOT / 'features.parquet'), market_reference_sha256=sha(ROOT / 'market_reference.parquet'),
        helper_interface_sha256=sha(ROOT / 'helper_interface.json'), rows=len(f), valid=int(f.formula_input_valid.sum()),
        newly_invalid=int((f.prior_formula_input_valid & ~f.formula_input_valid).sum()),
        member_rows=int(stats.mt_members.sum()), days=len(stats), minimum_members=int(stats.mt_members.min()),
        maximum_members=int(stats.mt_members.max()), minimum_day_mean=float(stats.mt_day_mean.min()),
        minimum_tail_mean=float(stats.mt_tail_mean.min()), expressions=EXPRESSIONS, native_header=HEADER,
        software_compilation_verified=False, full_client_membership_implementation_verified=False,
        native_FINANCE7_parity_verified=False, new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_report.json', report)
    return {k: v for k, v in report.items() if k not in ['expressions', 'native_header']}


def verify_features():
    checked_sources()
    report = json.loads((ROOT / 'feature_report.json').read_text())
    assert report['protocol_sha256'] == sha(PROTOCOL)
    for key, file in [('features', 'features.parquet'), ('market_reference', 'market_reference.parquet'), ('helper_interface', 'helper_interface.json')]:
        assert report[key + '_sha256'] == sha(ROOT / file)
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    actual = pd.read_parquet(ROOT / 'features.parquet')
    pd.testing.assert_frame_equal(actual[old.columns.drop('formula_input_valid')], old.drop(columns='formula_input_valid'), check_exact=True)
    c = base.conn()
    c.register('original', old[['date', 'code', 'formula_input_valid', 'volume_1449', 'v29', 'float_prior_volume', 'float_prior_turn']])
    c.execute('''CREATE VIEW members AS SELECT date,code,
        100*volume_1449/(100*float_prior_volume/float_prior_turn) AS day_turn,
        100*v29/(100*float_prior_volume/float_prior_turn) AS tail_turn
        FROM original WHERE formula_input_valid''')
    rebuilt_members = c.sql('SELECT * FROM members ORDER BY date,code').df()
    members = old.loc[old.formula_input_valid]
    pd.testing.assert_frame_equal(rebuilt_members[['date', 'code']], members[['date', 'code']].reset_index(drop=True), check_exact=True)
    np.testing.assert_allclose(rebuilt_members[['day_turn', 'tail_turn']], members[['S02', 'S03']], rtol=0, atol=2e-12)
    c.execute('''CREATE VIEW stats AS SELECT date,count(*) AS mt_members,
        sum(day_turn)/count(*) AS mt_day_mean,sum(tail_turn)/count(*) AS mt_tail_mean FROM members GROUP BY date''')
    stats = c.sql('SELECT * FROM stats ORDER BY date').df()
    pd.testing.assert_frame_equal(pd.read_parquet(ROOT / 'market_reference.parquet'), stats, check_dtype=False, rtol=0, atol=2e-12)
    expected = c.sql('''SELECT o.date,o.code,coalesce(o.formula_input_valid AND s.mt_members>0
        AND isfinite(s.mt_day_mean) AND s.mt_day_mean>0 AND isfinite(s.mt_tail_mean) AND s.mt_tail_mean>0,false) AS valid,
        CASE WHEN valid THEN m.day_turn/s.mt_day_mean END AS MT01,
        CASE WHEN valid THEN m.tail_turn/s.mt_tail_mean END AS MT02
        FROM original o LEFT JOIN members m USING(date,code) LEFT JOIN stats s USING(date) ORDER BY date,code''').df()
    c.close()
    np.testing.assert_array_equal(actual.formula_input_valid, expected.valid)
    np.testing.assert_allclose(actual[['MT01', 'MT02']], expected[['MT01', 'MT02']], rtol=0, atol=2e-11, equal_nan=True)
    encode = lambda x: np.floor(np.clip(100 * np.asarray(x) + 10000 + .000001, 0, 999999))
    np.testing.assert_array_equal(encode(actual.loc[expected.valid, ['MT01', 'MT02']]), encode(expected.loc[expected.valid, ['MT01', 'MT02']]))
    assert len(actual) == report['rows'] and int(expected.valid.sum()) == report['valid']
    assert stats.mt_members.sum() == report['member_rows'] and len(stats) == report['days']
    names = re.findall(r'(?m)^([A-Z][A-Z0-9]*):=', HEADER) + list(EXPRESSIONS)
    assert len(names) == len(set(names))
    assert report['expressions'] == EXPRESSIONS and report['native_header'] == HEADER
    unchanged = bool(np.array_equal(actual.formula_input_valid, old.formula_input_valid))
    proof = dict(passed=True, feature_report_sha256=sha(ROOT / 'feature_report.json'),
        rows=len(actual), all_source_volume_float_proxy_members_means_ratios_and_encodings_rebuilt=True,
        effective_input_intersection_unchanged=unchanged, all_original_48_fields_and_keys_unchanged=True,
        independent_sql_member_rows=len(rebuilt_members), independent_sql_days=len(stats),
        full_client_membership_implementation_verified=False, new_group_outcomes_read=False,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_verification.json', proof)
    return proof


def native():
    checked_sources()
    proof = json.loads((ROOT / 'feature_verification.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256'] == sha(ROOT / 'feature_report.json')
    f = pd.read_parquet(ROOT / 'features.parquet')
    visible = f.loc[f.prior_formula_input_valid].copy()
    visible['sample_order'] = [hashlib.sha256((d + '|' + c + '|market-turnover-native-v1').encode()).hexdigest()
        for d, c in zip(visible.date, visible.code)]
    samples = visible.sort_values('sample_order').groupby('half', sort=True).head(8)
    assert len(samples) == 32 and samples.groupby('half').size().eq(8).all()
    receipts = []
    for row in samples.itertuples():
        peers = visible.loc[visible.date.eq(row.date)]
        # Native V is in lots, while the validated research source is in shares.
        shares = 100 * peers.float_prior_volume.to_numpy() / peers.float_prior_turn.to_numpy()
        d = 10000 * (peers.volume_1449.to_numpy() / 100) / shares
        t = 10000 * (peers.v29.to_numpy() / 100) / shares
        n, ds, ts = len(peers), float(d.sum()), float(t.sum())
        np.testing.assert_allclose([ds/n, ts/n], [row.mt_day_mean, row.mt_tail_mean], rtol=0, atol=2e-12)
        values = [row.S02/(ds/n), row.S03/(ts/n)]
        np.testing.assert_allclose(values, [row.MT01, row.MT02], rtol=0, atol=2e-11)
        receipts.append(dict(date=row.date, code=row.code, helper_aggregate_outputs=[n, ds, ts], values=values))
    result = dict(passed=True, protocol_sha256=sha(PROTOCOL), feature_report_sha256=sha(ROOT / 'feature_report.json'),
        feature_verification_sha256=sha(ROOT / 'feature_verification.json'),
        reused_source_feature_verification_sha256=sha(previous.ROOT / 'feature_verification.json'),
        helper_interface_sha256=sha(ROOT / 'helper_interface.json'), samples=receipts,
        new_raw_prices_extracted=0, original_48_verification_reused=True,
        lot_share_units_and_INSUM_arithmetic_verified=True, mathematical_interface_only=True,
        full_client_membership_implementation_verified=False, software_compilation_verified=False,
        native_FINANCE7_parity_verified=False, new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'native_input_verification.json', result)
    return {k: v for k, v in result.items() if k != 'samples'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['features', 'verify_features', 'native'])
    print(json.dumps(globals()[p.parse_args().stage](), ensure_ascii=False, indent=2))
