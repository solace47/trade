"""Earlier inputs for a fixed three-year version of the original 48-field model."""
import argparse
from datetime import date
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as original
from . import tail_formula_replay48 as replay
from .corporate_cash import API, DAILY, MINUTES, curl, save_json, sha
from .turnover_reference import CALENDAR

ROOT = Path('data/research/tail_formula_long48')
OUT = ROOT/'inputs'
PROTOCOL = Path('config/tail_formula_long48_inputs_protocol.json')
PROBE = Path('data/research/tail_formula_long48_probe')
LEGACY = Path('data/research/long_history_ridge_1449')
NOTICES = OUT/'notices.parquet'
LEGACY_PROOF = Path('data/research/tail_formula_forward_2026q1/legacy_replay_verification.json')


def policy():
    p = json.loads(PROTOCOL.read_text())
    assert p['signal_first']=='2022-01-01' and p['signal_last']=='2023-12-29'
    assert p['warmup_first']=='2021-06-01' and p['observation_last']=='2024-01-02'
    assert not p['new_2026_prices_allowed'] and p['no_exit_rule_research']
    for path,digest in p['source_sha256'].items():
        assert sha(Path(path))==digest, path
    OUT.mkdir(parents=True,exist_ok=True)
    return p


def notices():
    from . import cash_dividend_notices as provider
    p = policy()
    if (OUT/'notice_report.json').exists():
        raise ValueError('Do not replace the frozen earlier announcement index')
    def page(first,last,keyword,number,cache):
        assert first.year in p['notices']['years'] and first.year==last.year
        assert keyword in p['notices']['keywords']
        path = cache/keyword/f'{first}_{last}_{number:03d}.json'
        if path.exists():
            payload = json.loads(path.read_text())
        else:
            payload = json.loads(curl(API,dict(stock='',tabName='fulltext',pageSize='30',pageNum=str(number),
                column='sse',seDate=f'{first}~{last}',searchkey=keyword,isHLtitle='true')))
            path.parent.mkdir(parents=True,exist_ok=True); save_json(path,payload); time.sleep(.1)
        total,rows = payload.get('totalAnnouncement'),payload.get('announcements')
        assert isinstance(total,int) and total>=0 and (isinstance(rows,list) or (total==0 and rows is None))
        return payload
    # Reuse the independently tested full pagination logic, not its 2024/25 bounds.
    saved = provider.fetch_page; provider.fetch_page = page
    raw = []; coverage = []
    try:
        for year in p['notices']['years']:
            for keyword in p['notices']['keywords']:
                part = provider.range_rows(date(year,1,1),date(year,12,31),keyword,OUT/'notice_pages')
                raw.extend(part); coverage.append(dict(year=year,keyword=keyword,rows=len(part)))
                print(json.dumps(coverage[-1]),flush=True)
    finally:
        provider.fetch_page = saved
    result = []
    for row in raw:
        code = row['secCode']; title = re.sub('<[^>]*>','',row['announcementTitle']).strip()
        if not re.fullmatch(r'(?:60|00)\d{4}',code):
            continue
        stamp = pd.Timestamp(row['announcementTime'],unit='ms',tz='UTC').tz_convert('Asia/Shanghai')
        result.append(dict(code=('sh.' if code.startswith('60') else 'sz.')+code,
            notice_date=stamp.strftime('%Y-%m-%d'),title=title,announcement_id=str(row['announcementId']),
            adjunct_url=row.get('adjunctUrl','')))
    f = pd.DataFrame(result).drop_duplicates().sort_values(['notice_date','code','announcement_id'])
    assert not f.duplicated(['code','announcement_id']).any()
    f.to_parquet(NOTICES,index=False,compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL),notices_sha256=sha(NOTICES),queries=coverage,rows=len(f),
        matching_rows=int(f.title.str.contains(p['notices']['title_filter'],regex=True).sum()),
        page_sha256={str(x):sha(x) for x in sorted((OUT/'notice_pages').rglob('*.json'))},
        complete_pagination_checked=True,all_market_completeness_proven=False,pdfs_downloaded=0,
        new_2026_prices_read=False,stock_outcomes_read=False)
    save_json(OUT/'notice_report.json',r)
    return {k:v for k,v in r.items() if k!='page_sha256'}


def checked_sources():
    p = policy(); m = json.loads((OUT/'source_manifest.json').read_text())
    assert m['protocol_sha256']==sha(PROTOCOL)
    proof = json.loads(LEGACY_PROOF.read_text())
    assert proof['passed'] and proof['adapter_sha256']==sha(Path(replay.__file__))
    return p,m


def prepare():
    p = policy()
    if (OUT/'base_report.json').exists():
        raise ValueError('Do not replace earlier base inputs')
    old = json.loads((LEGACY/'feature_report.json').read_text())
    source = json.loads((LEGACY/'prefix_manifest.json').read_text())
    assert old['prefix_manifest_sha256']==sha(LEGACY/'prefix_manifest.json')
    for path,digest in old['prefix_output_sha256'].items():
        assert sha(Path(path))==digest
    nr = json.loads((OUT/'notice_report.json').read_text())
    assert nr['protocol_sha256']==sha(PROTOCOL) and nr['notices_sha256']==sha(NOTICES)
    for path,digest in nr['page_sha256'].items():
        assert sha(Path(path))==digest
    daily_hashes = {}
    for i,(name,digest) in enumerate(sorted(source['minute_sha256'].items()),1):
        path = Path(name); assert sha(path)==digest
        daily = DAILY/(path.parent.name.lower()+'_'+path.stem+'.parquet')
        if daily.exists():
            daily_hashes[str(daily)] = sha(daily)
        if i%400==0:
            print(json.dumps(dict(source_files_checked=i,total=len(source['minute_sha256']))),flush=True)
    snapshots = {str(x):sha(x) for x in sorted(Path('data/research/market_snapshots_ci').glob('*.parquet'))}
    assert len(snapshots)==120
    manifest = dict(protocol_sha256=sha(PROTOCOL),legacy_feature_report_sha256=sha(LEGACY/'feature_report.json'),
        legacy_prefix_manifest_sha256=sha(LEGACY/'prefix_manifest.json'),legacy_replay_verification_sha256=sha(LEGACY_PROOF),
        prefix_sha256=old['prefix_output_sha256'],minute_sha256=source['minute_sha256'],
        daily_sha256=daily_hashes,snapshot_sha256=snapshots,calendar_sha256=sha(CALENDAR),
        notice_report_sha256=sha(OUT/'notice_report.json'),source_file_hashing_reads_bytes_not_out_of_scope_price_values=True)
    if (OUT/'source_manifest.json').exists():
        assert json.loads((OUT/'source_manifest.json').read_text())==manifest
    save_json(OUT/'source_manifest.json',manifest)
    c = base.conn(); c.read_parquet(list(manifest['prefix_sha256'])).create_view('prefix')
    c.read_parquet(list(snapshots)).create_view('snapshots'); c.read_parquet(str(NOTICES)).create_view('notices')
    prefix = c.execute('SELECT * FROM prefix WHERE date BETWEEN ? AND ? ORDER BY date,code',
        [p['signal_first'],p['signal_last']]).df()
    assert not prefix.duplicated(['date','code']).any()
    prefix.to_parquet(OUT/'prefix.parquet',index=False,compression='zstd'); c.register('bounded_prefix',prefix)
    missing = c.sql('''SELECT p.date,p.code,p.amount_1449 FROM bounded_prefix p
        ANTI JOIN snapshots s USING(date,code) WHERE p.amount_1449>=30000000''').df()
    missing.to_parquet(OUT/'missing_states.parquet',index=False,compression='zstd')
    assert len(missing)==0, 'Incomplete earlier state coverage'
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
        ORDER BY p.date,p.code''').df(); c.close()
    for name in ['price_1449','price_1420','price_1435','high_1449','low_1449']:
        f[name] = f[name].round(2)
    cents = np.rint(f.price_1449*100).astype('int64')
    f['decision_shares'] = (2000000//cents)//100*100
    prev_cents = np.rint(f.preclose*100).astype('int64')
    f['upper_limit'] = ((prev_cents*110+50)//100)/100
    f['necessary_tradeable'] = (f.price_1449+np.maximum(f.price_1449*.0015,.005)<f.upper_limit-.005) \
        & f.decision_shares.gt(0) & f.amount_1449.ge(3e7) & ~f.reference_gap & ~f.known_delisting
    f = f.loc[f.necessary_tradeable & f.price_1449.le(200)].copy()
    f['board'] = 'main'; f['half'] = f.date.str[:4]+np.where(f.date.str[5:7].le('06'),'H1','H2')
    f = f.sort_values(['date','code']).reset_index(drop=True)
    assert not f.duplicated(['date','code']).any()
    assert all(str(DAILY/(code.replace('.','_')+'.parquet')) in daily_hashes for code in f.code.unique())
    f.to_parquet(OUT/'universe.parquet',index=False,compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL),source_manifest_sha256=sha(OUT/'source_manifest.json'),
        prefix_sha256=sha(OUT/'prefix.parquet'),universe_sha256=sha(OUT/'universe.parquet'),
        missing_states_sha256=sha(OUT/'missing_states.parquet'),missing_amount_eligible_states=len(missing),
        rows=len(f),codes=f.code.nunique(),dates=f.date.nunique(),first=f.date.min(),last=f.date.max(),
        by_half=f.groupby('half').agg(rows=('code','size'),days=('date','nunique')).reset_index().to_dict('records'),
        new_2026_prices_read=False,next_morning_stock_outcomes_read=False)
    save_json(OUT/'base_report.json',r); return r


def index_inputs():
    policy(); folder = ROOT/'indices'
    proof = json.loads((folder/'verification.json').read_text())
    assert proof['passed'] and proof['source_report_sha256']==sha(folder/'source_report.json')
    assert proof['index_daily_sha256']==sha(PROBE/'index_daily.parquet')
    for name,digest in proof['output_sha256'].items():
        assert sha(folder/name)==digest
    return (pd.read_parquet(PROBE/'index_daily.parquet'),pd.read_parquet(folder/'points.parquet'),
            pd.read_parquet(folder/'audit.parquet'))


def features():
    p,sources = checked_sources()
    if (OUT/'feature_report.json').exists():
        raise ValueError('Do not replace earlier feature inputs')
    r = json.loads((OUT/'base_report.json').read_text())
    assert r['source_manifest_sha256']==sha(OUT/'source_manifest.json') and r['universe_sha256']==sha(OUT/'universe.parquet')
    universe = pd.read_parquet(OUT/'universe.parquet'); indices,points,audit = index_inputs()
    points.to_parquet(OUT/'index_points.parquet',index=False,compression='zstd')
    audit.to_parquet(OUT/'index_audit.parquet',index=False,compression='zstd')
    codes = sorted(universe.code.unique()); parts = {}; folder = OUT/'parts'; folder.mkdir(exist_ok=True)
    for start in range(0,len(codes),64):
        subset = codes[start:start+64]; path = folder/f'part_{start//64:03d}.parquet'; meta_path = path.with_suffix('.json')
        identity = dict(codes=subset,source_manifest_sha256=sha(OUT/'source_manifest.json'),
            adapter_sha256=sha(Path(replay.__file__)),base_report_sha256=sha(OUT/'base_report.json'),
            index_verification_sha256=sha(ROOT/'indices/verification.json'),producer_sha256=sha(Path(__file__)))
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            assert all(meta[k]==v for k,v in identity.items()) and meta['sha256']==sha(path)
        else:
            keys = universe.loc[universe.code.isin(subset)]
            files = [MINUTES/code[:2].upper()/(code[3:]+'.parquet') for code in subset]
            daily_files = [DAILY/(code.replace('.','_')+'.parquet') for code in subset]
            for file in files:
                assert sha(file)==sources['minute_sha256'][str(file)]
            for file in daily_files:
                assert sha(file)==sources['daily_sha256'][str(file)]
            aggregates = replay.afternoon_aggregates(keys,files,p['signal_first'],p['signal_last'])
            daily = replay.read_daily(daily_files,p['warmup_first'],p['signal_last'])
            f = replay.combine(keys,aggregates,daily,indices,points)
            f.to_parquet(path,index=False,compression='zstd')
            meta = dict(**identity,sha256=sha(path),rows=len(f)); save_json(meta_path,meta)
        parts[str(path)] = meta['sha256']
        print(json.dumps(dict(codes=min(start+64,len(codes)),total=len(codes))),flush=True)
    # DuckDB writes parts without keeping all wide frames twice in memory.
    c = base.conn(); c.read_parquet(list(parts)).create_view('parts')
    c.execute(f"COPY (SELECT * FROM parts ORDER BY date,code) TO '{OUT}/features.parquet' (FORMAT PARQUET,COMPRESSION ZSTD)")
    stats = c.sql('SELECT count(*) AS rows,sum(formula_input_valid)::BIGINT AS valid FROM parts').df().iloc[0].to_dict()
    assert stats['rows']==len(universe); c.close()
    report = dict(protocol_sha256=sha(PROTOCOL),source_manifest_sha256=sha(OUT/'source_manifest.json'),
        base_report_sha256=sha(OUT/'base_report.json'),adapter_sha256=sha(Path(replay.__file__)),
        legacy_replay_verification_sha256=sha(LEGACY_PROOF),parts_sha256=parts,
        features_sha256=sha(OUT/'features.parquet'),**stats,expressions=original.EXPRESSIONS,native_header=original.HEADER,
        index_points_sha256=sha(OUT/'index_points.parquet'),index_audit_sha256=sha(OUT/'index_audit.parquet'),
        index_full_day_conflicts=int((~audit.full_day_source_valid).sum()),
        index_verification_sha256=sha(ROOT/'indices/verification.json'),new_2026_prices_read=False,
        next_morning_stock_outcomes_read=False,native_source_parity_verified=False,no_exit_rules=True)
    save_json(OUT/'feature_report.json',report)
    return {k:v for k,v in report.items() if k not in ['parts_sha256','expressions','native_header']}


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('stage',choices=['notices','prepare','features'])
    print(json.dumps(globals()[p.parse_args().stage](),ensure_ascii=False,indent=2))
