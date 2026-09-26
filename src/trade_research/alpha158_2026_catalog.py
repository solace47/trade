"""Collect only the frozen 2026 selections' implementation-year catalogue."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json

import pandas as pd

from .alpha158_pool_2026 import ROOT, RULE_COMMIT, policy, unchanged
from .cash_dividend_catalog import assemble, reviewed_duplicates
from .corporate_cash import save_json, sha
from .long_history_inputs import _catalog_login, _catalog_job


def collect() -> dict:
    policy()
    selection = json.loads((ROOT/'input_report.json').read_text())
    sources, codes = {}, set()
    for name, detail in selection['models'].items():
        path = ROOT/name/'signals.parquet'
        if sha(path) != detail['signals_sha256']:
            raise ValueError('Validation selection changed before catalogue collection')
        sources[str(path)] = sha(path)
        frame = pd.read_parquet(path,columns=['date','code'])
        if not frame.date.str.startswith('2026-').all():
            raise ValueError('Unexpected implementation year')
        codes.update(frame.code)
    output = ROOT/'catalog'; output.mkdir(parents=True,exist_ok=True)
    jobs = pd.DataFrame({'code':sorted(codes),'year':'2026'})
    path = output/'jobs.parquet'
    if path.exists():
        pd.testing.assert_frame_equal(pd.read_parquet(path),jobs)
    else:
        jobs.to_parquet(path,index=False)
    unchanged(output/'manifest.json',{'rule_commit':RULE_COMMIT,'source_signals_sha256':sources,
        'jobs_sha256':sha(path),'code_years':len(jobs),'strategy_returns_read':False,'holdout_metadata_read':True})
    (output/'vendor').mkdir(exist_ok=True);(output/'source_conflicts').mkdir(exist_ok=True)
    reviews = reviewed_duplicates()
    items = [(r.code,r.year,str(output),reviews.get((r.code,r.year))) for r in jobs.itertuples()]
    errors=[];cached=downloaded=0
    with ProcessPoolExecutor(max_workers=4,initializer=_catalog_login) as pool:
        futures=[pool.submit(_catalog_job,job) for job in items]
        for count,future in enumerate(as_completed(futures),1):
            result=future.result();cached+=int(result.get('cached',False));downloaded+=int(result.get('downloaded',False))
            if 'error' in result:errors.append(result['error'])
            if count%100==0 or count==len(jobs):
                progress={'processed':count,'total':len(jobs),'cached':cached,'downloaded':downloaded,'errors':len(errors)}
                save_json(output/'fetch_progress.json',progress);print(progress,flush=True)
    report={'jobs':len(jobs),'cached':cached,'downloaded':downloaded,'errors':errors,'complete':not errors,
        'holdout_metadata_read':True,'strategy_returns_read':False}
    save_json(output/'fetch_report.json',report)
    if errors:raise ValueError(f'{len(errors)} incomplete 2026 catalogue queries; never treat them as empty')
    result=assemble(output)
    result.update(holdout_read=True,strategy_returns_read=False)
    save_json(output/'catalog_report.json',result)
    return {k:v for k,v in result.items() if k!='sha256'}


if __name__=='__main__':
    print(json.dumps(collect(),ensure_ascii=False,indent=2))
