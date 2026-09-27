"""Independently reconstruct cached announcement and action-date metadata."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re

import pandas as pd

from trade_research.corporate_cash import save_json, sha
from trade_research.tail_formula_long48_inputs import OUT, PROTOCOL, NOTICES, policy
from trade_research.tail_formula_long48_observations import LABELS, CATALOG, checked_keys


def notices():
    policy(); r = json.loads((OUT/'notice_report.json').read_text())
    assert r['protocol_sha256']==sha(PROTOCOL) and r['notices_sha256']==sha(NOTICES)
    for path,digest in r['page_sha256'].items():
        assert sha(Path(path))==digest
    rows = []
    for query in r['queries']:
        year = query['year']; part = []
        for number in range(1,math.ceil(query['rows']/30)+1):
            path = OUT/'notice_pages'/query['keyword']/f'{year}-01-01_{year}-12-31_{number:03d}.json'
            payload = json.loads(path.read_text()); assert payload['totalAnnouncement']==query['rows']
            part.extend(payload.get('announcements') or [])
        assert len(part)==query['rows'] and len({(x['secCode'],x['announcementId'],x['adjunctUrl']) for x in part})==len(part)
        for item in part:
            day = pd.Timestamp(item['announcementTime'],unit='ms',tz='UTC').tz_convert('Asia/Shanghai').strftime('%Y-%m-%d')
            assert day.startswith(str(year)+'-')
            code = item['secCode']
            if re.fullmatch(r'(?:60|00)\d{4}',code):
                rows.append((('sh.' if code[:2]=='60' else 'sz.')+code,day,
                    re.sub('<[^>]*>','',item['announcementTitle']).strip(),str(item['announcementId']),item.get('adjunctUrl','')))
    actual = pd.read_parquet(NOTICES)
    assert len(rows)==len(set(rows))==len(actual)
    assert Counter(rows)==Counter(actual.itertuples(index=False,name=None))
    assert int(actual.title.str.contains('进入退市整理|退市整理期交易',regex=True).sum())==r['matching_rows']
    proof = dict(passed=True,notice_report_sha256=sha(OUT/'notice_report.json'),rows=len(actual),
        all_queries_and_disclosure_dates_rebuilt=True,no_future_delist_date_filter=True,pdfs_downloaded=0)
    save_json(OUT/'notice_verification.json',proof); return proof


def actions():
    checked_keys(); r = json.loads((CATALOG/'date_report.json').read_text())
    assert r['complete'] and r['input_manifest_sha256']==sha(LABELS/'input_manifest.json')
    for path,digest in r['source_sha256'].items():
        assert sha(Path(path))==digest
    jobs = pd.read_parquet(CATALOG/'jobs.parquet'); assert r['jobs_sha256']==sha(CATALOG/'jobs.parquet')
    manifest = json.loads((CATALOG/'manifest.json').read_text())
    for path,digest in manifest['old_reports_sha256'].items():
        assert sha(Path(path))==digest
    for path,digest in manifest['reused_catalog_sha256'].items():
        assert sha(Path(path))==digest and sha(CATALOG/'vendor'/Path(path).name)==digest
    events = []; coverage = []; duplicates = []
    for job in jobs.itertuples():
        path = CATALOG/'vendor'/f'{job.code}_{job.year}.json'
        records = json.loads(path.read_text()); assert isinstance(records,list) and sha(path)==r['source_sha256'][str(path)]
        duplicate = len({x['dividOperateDate'] for x in records})<len(records)
        if duplicate: duplicates.append(dict(code=job.code,year=job.year))
        for row in records:
            assert row['code']==job.code and row['dividOperateDate'].startswith(job.year+'-')
            for field in ['dividOperateDate','dividRegistDate']:
                value = row.get(field,''); assert not value or pd.Timestamp(value).strftime('%Y-%m-%d')==value
            events.append((job.code,row['dividOperateDate'],row.get('dividRegistDate',''),str(path),duplicate))
        coverage.append((job.code,job.year,len(records),duplicate))
    e = pd.read_parquet(CATALOG/'date_events.parquet'); v = pd.read_parquet(CATALOG/'date_coverage.parquet')
    assert Counter(events)==Counter(e.itertuples(index=False,name=None)) and Counter(coverage)==Counter(v.itertuples(index=False,name=None))
    assert duplicates==r['duplicate_jobs'] and len(jobs)==r['code_years'] and len(e)==r['records']
    assert r['events_sha256']==sha(CATALOG/'date_events.parquet') and r['coverage_sha256']==sha(CATALOG/'date_coverage.parquet')
    proof = dict(passed=True,date_report_sha256=sha(CATALOG/'date_report.json'),code_years=len(jobs),records=len(e),
        all_original_dates_and_duplicate_records_preserved=True,duplicate_jobs=duplicates,
        cash_amounts_not_used_for_labels=True,not_used_for_selection=True)
    save_json(CATALOG/'date_verification.json',proof); return proof


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('stage',choices=['notices','actions'])
    print(json.dumps(globals()[p.parse_args().stage](),ensure_ascii=False,indent=2))
