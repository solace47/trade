"""Post-freeze source preparation for the 2026 Q1 next-morning evaluation."""
import argparse
import json
from pathlib import Path

import pandas as pd

from .corporate_cash import save_json, sha
from .tail_formula_forward import ROOT, PROTOCOL, checked_model, policy
from .tail_formula_forward_inputs import OUT, checked_sources
from .turnover_reference import CALENDAR

LABELS = ROOT / 'labels'
CATALOG = LABELS / 'catalog'


def prepare():
    p, sources = checked_sources(); checked_model()
    if (LABELS/'input_manifest.json').exists():
        raise ValueError('Do not replace forward observation identities')
    proof = json.loads((ROOT/'selection_verification.json').read_text())
    assert proof['passed'] and proof['selection_report_sha256']==sha(ROOT/'selection_report.json')
    selected = pd.read_parquet(ROOT/'all/selection.parquet')
    report = json.loads((ROOT/'all/selection_report.json').read_text())
    assert report['selection_sha256']==sha(ROOT/'all/selection.parquet')
    dates = sorted(selected.loc[selected.selected,'date'].unique())
    universe = pd.read_parquet(OUT/'universe.parquet')
    keys = universe.loc[universe.date.isin(dates)].copy()
    cal = pd.read_parquet(CALENDAR)
    days = sorted(cal.loc[cal.is_trading_day.eq('1') & cal.calendar_date.between(p['signal_first'],p['observation_last']), 'calendar_date'])
    keys['next_date'] = keys.date.map(dict(zip(days[:-1],days[1:])))
    assert keys.next_date.gt(keys.date).all() and keys.next_date.le(p['observation_last']).all()
    LABELS.mkdir(parents=True,exist_ok=True); CATALOG.mkdir(exist_ok=True)
    keys.to_parquet(LABELS/'keys.parquet',index=False,compression='zstd')
    jobs = pd.DataFrame({'code':sorted(keys.code.unique()),'year':'2026'})
    jobs.to_parquet(CATALOG/'jobs.parquet',index=False)
    save_json(CATALOG/'manifest.json',dict(protocol_sha256=sha(PROTOCOL), jobs_sha256=sha(CATALOG/'jobs.parquet'),
        code_years=len(jobs),selection_report_sha256=sha(ROOT/'selection_report.json')))
    # Reuse completed old queries before issuing missing requests. The catalogue
    # is outcome-side data and is never joined to the already-frozen selection.
    old = Path('data/research/alpha158_pool_2026/catalog')
    old_report = json.loads((old/'catalog_report.json').read_text())
    vendor = CATALOG/'vendor'; vendor.mkdir(exist_ok=True); reused = {}
    for code in jobs.code:
        path = old/'vendor'/f'{code}_2026.json'
        if path.exists() and not path.with_suffix('.error.json').exists():
            assert old_report['sha256'][str(path)]==sha(path)
            (vendor/path.name).write_bytes(path.read_bytes()); reused[str(path)]=sha(path)
    r = dict(protocol_sha256=sha(PROTOCOL), selection_report_sha256=sha(ROOT/'selection_report.json'),
        selection_verification_sha256=sha(ROOT/'selection_verification.json'),
        source_manifest_sha256=sha(OUT/'source_manifest.json'),keys_sha256=sha(LABELS/'keys.parquet'),
        calendar_sha256=sha(CALENDAR), rows=len(keys), dates=len(dates), codes=len(jobs),
        first_signal=keys.date.min(),last_signal=keys.date.max(),last_observation=keys.next_date.max(),
        old_catalog_report_sha256=sha(old/'catalog_report.json'), reused_catalog_sha256=reused,
        period_symbol_audit_range=[p['signal_first'],p['signal_last']],
        next_day_full_day_issue_keys_through=p['observation_last'],
        first_april_new_raw_price_window='09:31-10:00 only if required by the March 31 signal',
        next_morning_stock_outcomes_read=False,no_exit_rules=True)
    save_json(LABELS/'input_manifest.json',r)
    return {k:v for k,v in r.items() if k!='reused_catalog_sha256'}


def checked_keys():
    p = policy(); checked_model()
    m = json.loads((LABELS/'input_manifest.json').read_text())
    assert m['protocol_sha256']==sha(PROTOCOL) and m['selection_report_sha256']==sha(ROOT/'selection_report.json')
    assert m['source_manifest_sha256']==sha(OUT/'source_manifest.json') and m['keys_sha256']==sha(LABELS/'keys.parquet')
    assert m['calendar_sha256']==sha(CALENDAR)
    return p,m,pd.read_parquet(LABELS/'keys.parquet')


def catalog():
    checked_keys()
    from .cash_dividend_catalog import fetch, assemble
    result = fetch(CATALOG)
    if not result['complete']:
        return result
    return assemble(CATALOG)


def catalog_dates():
    """Keep duplicate cash records unresolved; only union their exposure dates."""
    from datetime import date
    checked_keys()
    if (CATALOG/'date_report.json').exists():
        raise ValueError('Do not replace the forward exposure-date catalogue')
    jobs = pd.read_parquet(CATALOG/'jobs.parquet')
    assert json.loads((CATALOG/'manifest.json').read_text())['jobs_sha256']==sha(CATALOG/'jobs.parquet')
    fetched = json.loads((CATALOG/'fetch_report.json').read_text())
    assert fetched['jobs']==len(jobs)
    events=[]; coverage=[]; sources={}; unresolved=[]
    for job in jobs.itertuples():
        path=CATALOG/'vendor'/f'{job.code}_{job.year}.json'
        error=path.with_suffix('.error.json'); duplicate_only=False
        if error.exists():
            failure=json.loads(error.read_text())
            assert failure['error']=='Multiple actions on the same date require manual reconciliation', failure
            path=CATALOG/'source_conflicts'/f'{job.code}_{job.year}_raw.json'
            sources[str(error)]=sha(error); duplicate_only=True
        records=json.loads(path.read_text()); assert isinstance(records,list)
        assert not duplicate_only or len({r['dividOperateDate'] for r in records})<len(records)
        sources[str(path)]=sha(path)
        for row in records:
            assert row['code']==job.code and row['dividOperateDate'].startswith(job.year+'-')
            for field in ['dividOperateDate','dividRegistDate']:
                value=row.get(field,'')
                assert field!='dividOperateDate' or value
                assert not value or date.fromisoformat(value).isoformat()==value
            events.append(dict(code=job.code,dividOperateDate=row['dividOperateDate'],
                dividRegistDate=row.get('dividRegistDate',''),source_path=str(path),cash_terms_unresolved=duplicate_only))
        coverage.append(dict(code=job.code,year=job.year,records=len(records),cash_terms_unresolved=duplicate_only))
        if duplicate_only:
            unresolved.append(job.code)
    e=pd.DataFrame(events,columns=['code','dividOperateDate','dividRegistDate','source_path','cash_terms_unresolved'])
    e=e.sort_values(['code','dividOperateDate','dividRegistDate']).reset_index(drop=True)
    v=pd.DataFrame(coverage).sort_values(['code','year']).reset_index(drop=True)
    e.to_parquet(CATALOG/'date_events.parquet',index=False,compression='zstd')
    v.to_parquet(CATALOG/'date_coverage.parquet',index=False,compression='zstd')
    r=dict(complete=True,protocol_sha256=sha(PROTOCOL),input_manifest_sha256=sha(LABELS/'input_manifest.json'),
        fetch_report_sha256=sha(CATALOG/'fetch_report.json'),jobs_sha256=sha(CATALOG/'jobs.parquet'),
        source_sha256=sources,events_sha256=sha(CATALOG/'date_events.parquet'),coverage_sha256=sha(CATALOG/'date_coverage.parquet'),
        code_years=len(jobs),records=len(e),duplicate_cash_terms_unresolved=unresolved,
        all_original_event_dates_retained=True,cash_terms_not_used=True,not_used_for_selection=True,
        no_dividend_adjusted_profits_claimed=True,issuer_notice_verified=False)
    save_json(CATALOG/'date_report.json',r)
    return {k:v for k,v in r.items() if k!='source_sha256'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','catalog','catalog_dates'])
    result = globals()[parser.parse_args().stage]()
    print(json.dumps({k:v for k,v in result.items() if k!='sha256'},ensure_ascii=False,indent=2))
