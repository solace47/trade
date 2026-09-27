"""Complete earlier training keys and reused company-action exposure dates."""
import argparse
import atexit
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
import json
from pathlib import Path
import socket
import time

import pandas as pd

from .corporate_cash import save_json, sha
from .tail_formula_long48_inputs import ROOT, OUT, PROTOCOL, LEGACY, checked_sources, policy
from .turnover_reference import CALENDAR

LABELS = ROOT/'labels'
CATALOG = LABELS/'catalog'


def prepare():
    p,sources = checked_sources()
    proof = json.loads((OUT/'feature_verification.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256']==sha(OUT/'feature_report.json')
    if (LABELS/'input_manifest.json').exists():
        raise ValueError('Do not replace historical observation keys')
    keys = pd.read_parquet(OUT/'universe.parquet')
    cal = pd.read_parquet(CALENDAR)
    days = sorted(cal.loc[cal.is_trading_day.eq('1') & cal.calendar_date.between(p['signal_first'],p['observation_last']),'calendar_date'])
    keys['next_date'] = keys.date.map(dict(zip(days[:-1],days[1:])))
    assert keys.next_date.gt(keys.date).all() and keys.next_date.le(p['observation_last']).all()
    LABELS.mkdir(parents=True,exist_ok=True); CATALOG.mkdir(exist_ok=True); (CATALOG/'vendor').mkdir(exist_ok=True)
    keys.to_parquet(LABELS/'keys.parquet',index=False,compression='zstd')
    jobs = pd.concat([keys[['code']].assign(year=keys.date.str[:4]),keys[['code']].assign(year=keys.next_date.str[:4])])
    jobs = jobs.drop_duplicates().sort_values(['code','year']).reset_index(drop=True)
    assert jobs.year.isin(['2022','2023','2024']).all()
    jobs.to_parquet(CATALOG/'jobs.parquet',index=False,compression='zstd')
    old_roots = [LEGACY/'historical_catalog',LEGACY/'cross_year_catalog',Path('data/research/cash_dividend_catalog')]
    cached = {}; old_reports = {}
    for folder in old_roots:
        report = json.loads((folder/'catalog_report.json').read_text()); old_reports[str(folder/'catalog_report.json')] = sha(folder/'catalog_report.json')
        for name,digest in report['sha256'].items():
            path = Path(name)
            if path.parent.name=='vendor' and path.suffix=='.json':
                assert path.name not in cached or cached[path.name][1]==digest
                cached[path.name] = (path,digest)
    reused = {}
    for job in jobs.itertuples():
        name = f'{job.code}_{job.year}.json'
        if name in cached:
            path,digest = cached[name]; assert sha(path)==digest
            (CATALOG/'vendor'/name).write_bytes(path.read_bytes()); reused[str(path)] = digest
    save_json(CATALOG/'manifest.json',dict(protocol_sha256=sha(PROTOCOL),jobs_sha256=sha(CATALOG/'jobs.parquet'),
        code_years=len(jobs),old_reports_sha256=old_reports,reused_catalog_sha256=reused))
    r = dict(protocol_sha256=sha(PROTOCOL),source_manifest_sha256=sha(OUT/'source_manifest.json'),
        feature_verification_sha256=sha(OUT/'feature_verification.json'),keys_sha256=sha(LABELS/'keys.parquet'),
        catalog_manifest_sha256=sha(CATALOG/'manifest.json'),calendar_sha256=sha(CALENDAR),rows=len(keys),
        dates=keys.date.nunique(),codes=keys.code.nunique(),catalog_jobs=len(jobs),catalog_reused=len(reused),
        first_signal=keys.date.min(),last_signal=keys.date.max(),last_observation=keys.next_date.max(),
        next_morning_stock_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(LABELS/'input_manifest.json',r); return r


def checked_keys():
    p = policy(); m = json.loads((LABELS/'input_manifest.json').read_text())
    assert m['protocol_sha256']==sha(PROTOCOL) and m['source_manifest_sha256']==sha(OUT/'source_manifest.json')
    assert m['feature_verification_sha256']==sha(OUT/'feature_verification.json')
    assert m['catalog_manifest_sha256']==sha(CATALOG/'manifest.json')
    assert m['keys_sha256']==sha(LABELS/'keys.parquet') and m['calendar_sha256']==sha(CALENDAR)
    return p,m,pd.read_parquet(LABELS/'keys.parquet')


def date_records(records,code,year):
    assert isinstance(records,list)
    for row in records:
        assert row['code']==code and row['dividOperateDate'].startswith(year+'-')
        for field in ['dividOperateDate','dividRegistDate']:
            value = row.get(field,'')
            assert field!='dividOperateDate' or value
            assert not value or date.fromisoformat(value).isoformat()==value
    return records


def login():
    import baostock as bs
    from .ingest import _login
    socket.setdefaulttimeout(20); _login(); atexit.register(bs.logout)


def fetch_one(job):
    import baostock as bs
    from .ingest import _login, _rows
    code,year = job; path = CATALOG/'vendor'/f'{code}_{year}.json'; errors = []
    for attempt in range(2):
        try:
            records = _rows(bs.query_dividend_data(code,year=year,yearType='operate')).to_dict('records')
            date_records(records,code,year)
            break
        except Exception as exc:
            errors.append(dict(attempt=attempt,error_type=type(exc).__name__,error=str(exc)))
            bs.logout()
            if attempt==0:
                time.sleep(.5); _login()
    else:
        save_json(path.with_suffix('.error.json'),dict(code=code,year=year,attempts=errors))
        return dict(code=code,year=year,status='error')
    save_json(path,records)
    if errors:
        save_json(path.with_suffix('.retries.json'),dict(code=code,year=year,attempts=errors))
    time.sleep(.05)
    return dict(code=code,year=year,status='received')


def catalog():
    checked_keys()
    if (CATALOG/'date_report.json').exists():
        raise ValueError('Do not replace the historical date catalogue')
    jobs = pd.read_parquet(CATALOG/'jobs.parquet')
    manifest = json.loads((CATALOG/'manifest.json').read_text())
    assert manifest['jobs_sha256']==sha(CATALOG/'jobs.parquet')
    missing = []
    for job in jobs.itertuples():
        path = CATALOG/'vendor'/f'{job.code}_{job.year}.json'
        if path.exists():
            date_records(json.loads(path.read_text()),job.code,job.year)
        else:
            missing.append((job.code,job.year))
    errors = []
    if missing:
        with ProcessPoolExecutor(max_workers=4,initializer=login) as pool:
            futures = [pool.submit(fetch_one,x) for x in missing]
            for i,future in enumerate(as_completed(futures),1):
                result = future.result()
                if result['status']=='error': errors.append(result)
                if i%100==0 or i==len(missing):
                    print(json.dumps(dict(processed=i,total_missing=len(missing),errors=len(errors))),flush=True)
    fetch = dict(protocol_sha256=sha(PROTOCOL),jobs=len(jobs),new_jobs=len(missing),errors=errors,complete=not errors)
    save_json(CATALOG/'fetch_report.json',fetch)
    if errors:
        return fetch
    events = []; coverage = []; sources = {}; duplicate_jobs = []
    for job in jobs.itertuples():
        path = CATALOG/'vendor'/f'{job.code}_{job.year}.json'
        records = date_records(json.loads(path.read_text()),job.code,job.year); sources[str(path)] = sha(path)
        duplicate = len({x['dividOperateDate'] for x in records})<len(records)
        if duplicate: duplicate_jobs.append(dict(code=job.code,year=job.year))
        for row in records:
            events.append(dict(code=job.code,dividOperateDate=row['dividOperateDate'],dividRegistDate=row.get('dividRegistDate',''),
                source_path=str(path),cash_terms_unresolved=duplicate))
        coverage.append(dict(code=job.code,year=job.year,records=len(records),cash_terms_unresolved=duplicate))
    e = pd.DataFrame(events).sort_values(['code','dividOperateDate','dividRegistDate']).reset_index(drop=True)
    v = pd.DataFrame(coverage).sort_values(['code','year']).reset_index(drop=True)
    e.to_parquet(CATALOG/'date_events.parquet',index=False,compression='zstd'); v.to_parquet(CATALOG/'date_coverage.parquet',index=False,compression='zstd')
    r = dict(complete=True,protocol_sha256=sha(PROTOCOL),input_manifest_sha256=sha(LABELS/'input_manifest.json'),
        fetch_report_sha256=sha(CATALOG/'fetch_report.json'),jobs_sha256=sha(CATALOG/'jobs.parquet'),
        source_sha256=sources,events_sha256=sha(CATALOG/'date_events.parquet'),coverage_sha256=sha(CATALOG/'date_coverage.parquet'),
        code_years=len(jobs),records=len(e),duplicate_jobs=duplicate_jobs,all_original_event_dates_retained=True,
        cash_terms_not_used=True,not_used_for_selection=True,no_dividend_adjusted_profits_claimed=True,issuer_notice_verified=False)
    save_json(CATALOG/'date_report.json',r)
    return {k:v for k,v in r.items() if k!='source_sha256'}


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('stage',choices=['prepare','catalog'])
    print(json.dumps(globals()[p.parse_args().stage](),ensure_ascii=False,indent=2))
