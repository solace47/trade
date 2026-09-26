"""Resume the frozen missing code-years using four isolated public-data sessions."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import socket
import time

import pandas as pd

from trade_research.cash_dividend_catalog import checked_response,reviewed_duplicates,action_type,jobs_for
from trade_research.corporate_cash import save_json,sha
from trade_research.economic_winner import ROOT,OLD_EVENTS,OLD_COVERAGE


def occurrence_records(records,code,year,review):
    """Retain conflicting cash terms: this study uses the union of event dates only."""
    if review:return checked_response(records,code,year,review)
    if not isinstance(records,list):raise ValueError('A complete response must be a list')
    for row in records:
        checked_response([row],code,year)
        for field in ['dividRegistDate','dividPayDate','dividStockMarketDate']:
            date=row.get(field,'')
            if date and pd.Timestamp(date).strftime('%Y-%m-%d')!=date:raise ValueError('Invalid event date')
    return records


def worker(args):
    import baostock as bs
    from trade_research.ingest import _login,_rows
    index,jobs,reviews=args;folder=ROOT/'catalog';cache=folder/'vendor'
    socket.setdefaulttimeout(20);downloaded=cached=0;errors=[];consecutive_errors=0
    _login()
    try:
        for number,(code,year) in enumerate(jobs,1):
            path=cache/(code+'_'+year+'.json');review=reviews.get((code,year))
            try:
                if path.exists():
                    occurrence_records(json.loads(path.read_text()),code,year,review);cached+=1
                else:
                    conflict_path=folder/'source_conflicts'/(code+'_'+year+'_raw.json')
                    records=(json.loads(conflict_path.read_text()) if conflict_path.exists() else
                        _rows(bs.query_dividend_data(code,year=year,yearType='operate')).to_dict('records'))
                    try:records=occurrence_records(records,code,year,review)
                    except ValueError:
                        conflicts=folder/'source_conflicts';conflicts.mkdir(exist_ok=True)
                        save_json(conflicts/(code+'_'+year+'_raw.json'),records);raise
                    save_json(path,records);downloaded+=1
                    time.sleep(.1)
                path.with_suffix('.error.json').unlink(missing_ok=True);consecutive_errors=0
            except Exception as error:
                detail={'code':code,'year':year,'error':str(error)};errors.append(detail)
                save_json(path.with_suffix('.error.json'),detail);consecutive_errors+=1
                if consecutive_errors>=3:break
                bs.logout();time.sleep(.5);_login()
            if number%100==0 or number==len(jobs):
                print(json.dumps({'worker':index,'processed':number,'total':len(jobs),'downloaded':downloaded,'cached':cached,'errors':len(errors)}),flush=True)
    finally:bs.logout()
    return {'worker':index,'processed':number,'jobs':len(jobs),'downloaded':downloaded,'cached':cached,'errors':errors}


def main():
    folder=ROOT/'catalog';manifest=json.loads((ROOT/'input_manifest.json').read_text())
    assert sha(OLD_EVENTS)==manifest['old_events_sha256'] and sha(OLD_COVERAGE)==manifest['old_coverage_sha256']
    jobs=jobs_for(folder);assert jobs.year.isin(['2024','2025']).all()
    reviews=reviewed_duplicates();pairs=list(jobs[['code','year']].itertuples(index=False,name=None))
    (folder/'vendor').mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(worker,[(i,pairs[i::4],reviews) for i in range(4)]))
    complete=all(r['processed']==r['jobs'] and not r['errors'] for r in results)
    save_json(folder/'parallel_fetch_report.json',{'workers':4,'complete':complete,'results':results,
        'jobs_sha256':sha(folder/'jobs.parquet'),'new_2026_prices_read':False})
    if not complete:raise RuntimeError('Some frozen catalog queries remain incomplete; do not label them as no events')
    records=[];queries=[];duplicates=[];hashes={}
    for code,year in pairs:
        path=folder/'vendor'/(code+'_'+year+'.json')
        assert not path.with_suffix('.error.json').exists()
        rows=occurrence_records(json.loads(path.read_text()),code,year,reviews.get((code,year)))
        hashes[str(path)]=sha(path);queries.append({'code':code,'year':year,'events':len(rows)})
        dates=pd.Series([r['dividOperateDate'] for r in rows]).value_counts()
        for day,n in dates.loc[dates.gt(1)].items():duplicates.append({'code':code,'year':year,'operate_date':day,'records':int(n)})
        records.extend([dict(r,implementation_year=year,action_type=action_type(r)) for r in rows])
    new_events=pd.DataFrame(records).sort_values(['code','dividOperateDate'])
    new_events.to_parquet(folder/'events.parquet',index=False)
    pd.DataFrame(queries).to_parquet(folder/'query_coverage.parquet',index=False)
    save_json(folder/'catalog_report.json',{'code_year_queries':len(pairs),'event_rows':len(records),
        'duplicate_action_dates_retained':duplicates,'sha256':hashes,'events_sha256':sha(folder/'events.parquet'),
        'method':'date_union_only_all_cash_amount_versions_retained_and_no_event_pnl_assigned',
        'new_2026_prices_read':False})
    events=pd.concat([pd.read_parquet(OLD_EVENTS),pd.read_parquet(folder/'events.parquet')],ignore_index=True).sort_values(['code','dividOperateDate'])
    coverage=pd.concat([pd.read_parquet(OLD_COVERAGE),pd.read_parquet(folder/'query_coverage.parquet')],ignore_index=True)
    assert not coverage.duplicated(['code','year']).any()
    needed=pd.read_parquet(folder/'needed.parquet')
    assert needed.merge(coverage[['code','year']],on=['code','year'],how='left',indicator=True)._merge.eq('both').all()
    events.to_parquet(folder/'combined_events.parquet',index=False);coverage.to_parquet(folder/'combined_coverage.parquet',index=False)
    report={'complete':True,'needed_code_years':len(needed),'added_code_years':len(jobs),
        'events_sha256':sha(folder/'combined_events.parquet'),'coverage_sha256':sha(folder/'combined_coverage.parquet'),
        'catalog_report_sha256':sha(folder/'catalog_report.json'),'parallel_fetch_report_sha256':sha(folder/'parallel_fetch_report.json'),
        'economic_labels_computed':False,'new_2026_prices_read':False}
    save_json(folder/'coverage_report.json',report);print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
