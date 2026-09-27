"""Bounded 2026 Q1 inputs for the already-frozen next-morning model."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as original
from . import tail_formula_replay48 as replay
from .corporate_cash import DAILY, MINUTES, save_json, sha
from .sector_resilience_1449 import DELIST_NOTICES
from .tail_formula_forward import ROOT, PROTOCOL, checked_model, policy
from .turnover_reference import CALENDAR

OUT = ROOT / 'inputs'
LEGACY = Path('data/research/alpha158_pool_2026')


def checked_sources():
    checked_model(); p = policy()
    manifest = json.loads((OUT / 'source_manifest.json').read_text())
    assert manifest['protocol_sha256'] == sha(PROTOCOL)
    assert manifest['model_freeze_report_sha256'] == sha(ROOT / 'model_freeze_report.json')
    assert manifest['legacy_replay_verification_sha256'] == sha(ROOT / 'legacy_replay_verification.json')
    proof = json.loads((ROOT / 'legacy_replay_verification.json').read_text())
    assert proof['passed'] and proof['adapter_sha256'] == sha(Path(replay.__file__))
    return p, manifest


def prepare():
    checked_model(); p = policy()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'base_report.json').exists():
        raise ValueError('Do not replace forward source and base inputs')
    proof = json.loads((ROOT / 'legacy_replay_verification.json').read_text())
    assert proof['passed'] and proof['adapter_sha256'] == sha(Path(replay.__file__))
    old = json.loads((LEGACY / 'base_report.json').read_text())
    assert old['manifest_sha256'] == sha(LEGACY / 'prefix_manifest.json')
    source = json.loads((LEGACY / 'prefix_manifest.json').read_text())
    assert source['calendar_sha256'] == sha(CALENDAR)
    assert source['code_sha256']['minute_prefix_1449.py'] == sha(Path(__file__).with_name('minute_prefix_1449.py'))
    for name, digest in old['prefix_sha256'].items():
        assert sha(Path(name)) == digest
    daily_hashes = {}
    for i, (name, digest) in enumerate(sorted(source['minute_sha256'].items()), 1):
        path = Path(name)
        assert sha(path) == digest
        code = path.parent.name.lower() + '.' + path.stem
        daily = DAILY / (code.replace('.', '_') + '.parquet')
        if daily.exists():
            daily_hashes[str(daily)] = sha(daily)
        if i % 400 == 0:
            print(json.dumps(dict(source_files_checked=i, total=len(source['minute_sha256']))), flush=True)
    for name, digest in source['snapshot_sha256'].items():
        assert sha(Path(name)) == digest
    manifest = dict(protocol_sha256=sha(PROTOCOL), model_freeze_report_sha256=sha(ROOT / 'model_freeze_report.json'),
        legacy_replay_verification_sha256=sha(ROOT / 'legacy_replay_verification.json'),
        legacy_base_report_sha256=sha(LEGACY / 'base_report.json'), legacy_prefix_manifest_sha256=sha(LEGACY / 'prefix_manifest.json'),
        prefix_sha256=old['prefix_sha256'], minute_sha256=source['minute_sha256'],
        snapshot_sha256=source['snapshot_sha256'], daily_sha256=daily_hashes,
        calendar_sha256=sha(CALENDAR), delist_notices_sha256=sha(DELIST_NOTICES),
        source_file_hashing_reads_bytes_not_out_of_scope_price_values=True)
    if (OUT / 'source_manifest.json').exists():
        assert json.loads((OUT / 'source_manifest.json').read_text()) == manifest
    save_json(OUT / 'source_manifest.json', manifest)
    c = base.conn()
    c.read_parquet(list(old['prefix_sha256'])).create_view('prefix')
    c.read_parquet(list(source['snapshot_sha256'])).create_view('snapshots')
    c.read_parquet(str(DELIST_NOTICES)).create_view('notices')
    prefix = c.execute('SELECT * FROM prefix WHERE date BETWEEN ? AND ? ORDER BY date,code',
                       [p['signal_first'], p['signal_last']]).df()
    assert not prefix.duplicated(['date', 'code']).any()
    prefix.to_parquet(OUT / 'prefix.parquet', index=False, compression='zstd')
    c.register('bounded_prefix', prefix)
    missing = c.sql('''SELECT p.date,p.code,p.amount_1449 FROM bounded_prefix p
        ANTI JOIN snapshots s USING(date,code) WHERE p.amount_1449>=30000000''').df()
    missing.to_parquet(OUT / 'missing_states.parquet', index=False, compression='zstd')
    assert len(missing) == 0, 'State coverage incomplete; preserve missing-state audit before proceeding'
    f = c.sql('''SELECT p.*,s.preclose,s.open_1450 AS daily_open,s.isST,s.tradestatus,
        s.listing_age_sessions,s.reference_gap,
        EXISTS(SELECT 1 FROM notices n WHERE n.code=p.code AND n.notice_date<p.date
            AND regexp_matches(n.title,'进入退市整理|退市整理期交易')) AS known_delisting
        FROM bounded_prefix p JOIN snapshots s USING(date,code)
        WHERE s.isST=0 AND s.tradestatus=1 AND s.listing_age_sessions>=60
          AND (p.code LIKE 'sh.60%' OR p.code LIKE 'sz.00%')
          AND NOT p.quote_outside_traded_range AND p.amount_1449>0 AND p.volume_1449>0
          AND s.preclose>0 AND s.open_1450>0 AND p.price_1449>0
          AND abs(p.price_1449-round(p.price_1449,2))<=.0001
        ORDER BY p.date,p.code''').df()
    c.close()
    for name in ['price_1449','price_1420','price_1435','high_1449','low_1449']:
        f[name] = f[name].round(2)
    cents = np.rint(f.price_1449*100).astype('int64')
    f['decision_shares'] = (2000000//cents)//100*100
    prev_cents = np.rint(f.preclose*100).astype('int64')
    f['upper_limit'] = ((prev_cents*110+50)//100)/100
    f['necessary_tradeable'] = (f.price_1449+np.maximum(f.price_1449*.0015,.005)<f.upper_limit-.005) \
        & f.decision_shares.gt(0) & f.amount_1449.ge(3e7) & ~f.reference_gap & ~f.known_delisting
    f = f.loc[f.necessary_tradeable & f.price_1449.le(200)].copy()
    f['board'] = 'main'; f['half'] = '2026H1'
    f = f.sort_values(['date','code']).reset_index(drop=True)
    assert not f.duplicated(['date','code']).any()
    assert all(str(DAILY/(code.replace('.','_')+'.parquet')) in daily_hashes for code in f.code.unique())
    f.to_parquet(OUT / 'universe.parquet', index=False, compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL), source_manifest_sha256=sha(OUT / 'source_manifest.json'),
        prefix_sha256=sha(OUT / 'prefix.parquet'), universe_sha256=sha(OUT / 'universe.parquet'),
        missing_states_sha256=sha(OUT / 'missing_states.parquet'), missing_amount_eligible_states=len(missing),
        rows=len(f), codes=f.code.nunique(), dates=f.date.nunique(), first=f.date.min(), last=f.date.max(),
        new_2026_prices_read=True, no_2026_after_q1_signal_prices_read=True, next_morning_stock_outcomes_read=False)
    save_json(OUT / 'base_report.json', r)
    return r


def index_inputs():
    p, _ = checked_sources()
    folder = ROOT / 'indices'
    dr = json.loads((folder / 'daily_report.json').read_text())
    mr = json.loads((folder / 'minute_report.json').read_text())
    assert dr['protocol_sha256'] == mr['protocol_sha256'] == sha(PROTOCOL)
    assert dr['daily_sha256'] == sha(folder / 'daily.parquet')
    for name, digest in dr['raw_files_sha256'].items():
        assert sha(Path(name)) == digest
    indices = pd.read_parquet(folder / 'daily.parquet')
    assert indices.date.between(p['warmup_first'], p['signal_last']).all()
    cal = pd.read_parquet(CALENDAR)
    days = set(cal.loc[cal.is_trading_day.eq('1') & cal.calendar_date.between(p['signal_first'],p['signal_last']), 'calendar_date'])
    assert {(s['symbol'],s['date']) for s in mr['sessions']} == {(code,date) for code in ['sh.000001','sz.399001'] for date in days}
    points = []; audits = []
    for item in mr['sessions']:
        rows = []
        if 'path' in item:
            path = Path(item['path']); assert sha(path) == item['sha256']
            rows = json.loads(path.read_text())
            for w in item['wire']:
                assert sha(Path(w['response'])) == w['response_sha256'] and sha(Path(w['request'])) == w['request_sha256']
        points.append(dict(date=item['date'], index_code=item['symbol'], **original.context.points(rows)))
        day = indices.loc[indices.code.eq(item['symbol']) & indices.date.eq(item['date'])].iloc[0]
        values = np.array([x['price_raw']/100 for x in rows])
        valid = bool(len(values)==240 and (values>0).all() and (values>=day.low-.0051).all()
            and (values<=day.high+.0051).all() and abs(values[-1]-day.close)<=.0051)
        audits.append(dict(date=item['date'], index_code=item['symbol'], full_day_source_valid=valid))
    return indices, pd.DataFrame(points).sort_values(['date','index_code']).reset_index(drop=True), pd.DataFrame(audits)


def features():
    p, sources = checked_sources()
    if (OUT / 'feature_report.json').exists():
        raise ValueError('Do not replace forward feature inputs')
    r = json.loads((OUT / 'base_report.json').read_text())
    assert r['source_manifest_sha256'] == sha(OUT / 'source_manifest.json')
    assert r['universe_sha256'] == sha(OUT / 'universe.parquet')
    universe = pd.read_parquet(OUT / 'universe.parquet')
    indices, points, audit = index_inputs()
    points.to_parquet(OUT / 'index_points.parquet', index=False, compression='zstd')
    audit.to_parquet(OUT / 'index_audit.parquet', index=False, compression='zstd')
    codes = sorted(universe.code.unique()); parts = {}; folder = OUT / 'parts'; folder.mkdir(exist_ok=True)
    adapter = sha(Path(replay.__file__))
    for start in range(0,len(codes),64):
        subset = codes[start:start+64]
        path = folder / f'part_{start//64:03d}.parquet'; meta_path = path.with_suffix('.json')
        identity = dict(codes=subset, source_manifest_sha256=sha(OUT/'source_manifest.json'),
            adapter_sha256=adapter, base_report_sha256=sha(OUT/'base_report.json'),
            index_daily_report_sha256=sha(ROOT/'indices/daily_report.json'),
            index_minute_report_sha256=sha(ROOT/'indices/minute_report.json'))
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            assert all(meta[k]==v for k,v in identity.items()) and meta['sha256']==sha(path)
        else:
            keys = universe.loc[universe.code.isin(subset)]
            files = [MINUTES/code[:2].upper()/(code[3:]+'.parquet') for code in subset]
            daily_files = [DAILY/(code.replace('.','_')+'.parquet') for code in subset]
            for f in files:
                assert sha(f)==sources['minute_sha256'][str(f)]
            for f in daily_files:
                assert sha(f)==sources['daily_sha256'][str(f)]
            aggregates = replay.afternoon_aggregates(keys, files, p['signal_first'], p['signal_last'])
            daily = replay.read_daily(daily_files, p['warmup_first'], p['signal_last'])
            f = replay.combine(keys, aggregates, daily, indices, points)
            f.to_parquet(path, index=False, compression='zstd')
            meta = dict(**identity, sha256=sha(path), rows=len(f)); save_json(meta_path, meta)
        parts[str(path)] = meta['sha256']
        print(json.dumps(dict(codes=min(start+64,len(codes)), total=len(codes))), flush=True)
    f = pd.concat([pd.read_parquet(x) for x in parts], ignore_index=True).sort_values(['date','code']).reset_index(drop=True)
    pd.testing.assert_frame_equal(f[['date','code']], universe[['date','code']], check_exact=True)
    f.to_parquet(OUT/'features.parquet', index=False, compression='zstd')
    report = dict(protocol_sha256=sha(PROTOCOL), model_freeze_report_sha256=sha(ROOT/'model_freeze_report.json'),
        source_manifest_sha256=sha(OUT/'source_manifest.json'), base_report_sha256=sha(OUT/'base_report.json'),
        adapter_sha256=adapter, legacy_replay_verification_sha256=sha(ROOT/'legacy_replay_verification.json'),
        parts_sha256=parts, features_sha256=sha(OUT/'features.parquet'), rows=len(f), valid=int(f.formula_input_valid.sum()),
        expressions=original.EXPRESSIONS, native_header=original.HEADER,
        index_points_sha256=sha(OUT/'index_points.parquet'), index_audit_sha256=sha(OUT/'index_audit.parquet'),
        index_full_day_conflicts=int((~audit.full_day_source_valid).sum()),
        index_daily_report_sha256=sha(ROOT/'indices/daily_report.json'), index_minute_report_sha256=sha(ROOT/'indices/minute_report.json'),
        new_2026_prices_read=True, no_2026_after_q1_signal_prices_read=True,
        next_morning_stock_outcomes_read=False, native_source_parity_verified=False, no_exit_rules=True)
    save_json(OUT/'feature_report.json', report)
    return {k:v for k,v in report.items() if k not in ['parts_sha256','expressions','native_header']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare','features'])
    print(json.dumps(globals()[parser.parse_args().stage](), ensure_ascii=False, indent=2))
