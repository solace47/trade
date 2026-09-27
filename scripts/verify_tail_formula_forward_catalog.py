"""Verify the exposure-date union without resolving or using cash terms."""
from collections import Counter
import json
from pathlib import Path

import pandas as pd

from trade_research.corporate_cash import save_json, sha
from trade_research.tail_formula_forward_observations import CATALOG, checked_keys


def main():
    checked_keys();r=json.loads((CATALOG/'date_report.json').read_text())
    for path,digest in r['source_sha256'].items():
        assert sha(Path(path))==digest
    jobs=pd.read_parquet(CATALOG/'jobs.parquet');assert r['jobs_sha256']==sha(CATALOG/'jobs.parquet')
    events=[];coverage=[];duplicates=[]
    for job in jobs.itertuples():
        path=CATALOG/'vendor'/f'{job.code}_{job.year}.json';error=path.with_suffix('.error.json')
        unresolved=error.exists()
        if unresolved:
            assert json.loads(error.read_text())['error']=='Multiple actions on the same date require manual reconciliation'
            path=CATALOG/'source_conflicts'/f'{job.code}_{job.year}_raw.json';duplicates.append(job.code)
        records=json.loads(path.read_text());assert isinstance(records,list)
        assert sha(path)==r['source_sha256'][str(path)]
        for row in records:
            assert row['code']==job.code and row['dividOperateDate'].startswith(job.year+'-')
            for field in ['dividOperateDate','dividRegistDate']:
                value=row.get(field,'')
                assert not value or pd.Timestamp(value).strftime('%Y-%m-%d')==value
            events.append((job.code,row['dividOperateDate'],row.get('dividRegistDate',''),str(path),unresolved))
        coverage.append((job.code,job.year,len(records),unresolved))
    actual=pd.read_parquet(CATALOG/'date_events.parquet');v=pd.read_parquet(CATALOG/'date_coverage.parquet')
    assert Counter(events)==Counter(actual.itertuples(index=False,name=None))
    assert Counter(coverage)==Counter(v.itertuples(index=False,name=None))
    assert duplicates==r['duplicate_cash_terms_unresolved'] and len(jobs)==r['code_years'] and len(actual)==r['records']
    assert r['events_sha256']==sha(CATALOG/'date_events.parquet') and r['coverage_sha256']==sha(CATALOG/'date_coverage.parquet')
    proof=dict(passed=True,date_report_sha256=sha(CATALOG/'date_report.json'),code_years=len(jobs),records=len(actual),
        all_original_dates_and_duplicate_records_preserved=True,unresolved_cash_codes=duplicates,
        cash_amounts_not_used_for_labels=True,not_used_for_selection=True)
    save_json(CATALOG/'date_verification.json',proof)
    print(json.dumps(proof,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
