"""Explicit 2022--2023 training inputs; default recent evaluation is unchanged."""
from __future__ import annotations

import argparse
import atexit
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import socket
import json
from pathlib import Path

import duckdb
import pandas as pd

from .absolute_ridge_1449 import feature_frame
from .corporate_cash import save_json,sha
from .downside_ridge_inputs import ROOT as RECENT
from .minute_prefix_1449 import build_year
from .shallow_tree_1449 import decision_buyable
from .reference_gain_eval import MINUTES

ROOT=Path('data/research/long_history_ridge_1449')
RULE_COMMIT='802cc1e'


def prepare(output: Path = ROOT) -> dict:
    if (output/'long/repriced.parquet').exists():
        raise ValueError('Training inputs cannot change after selected outcomes exist')
    output.mkdir(parents=True,exist_ok=True)
    paths=sorted([*list((MINUTES/'SH').glob('60*.parquet')),*list((MINUTES/'SZ').glob('00*.parquet'))])
    frozen={'rule_commit':RULE_COMMIT,'years':[2022,2023],'minute_sha256':{str(p):sha(p) for p in paths},'holdout_prices_read':False}
    manifest=output/'prefix_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=frozen:
        raise ValueError('Frozen historical minute sources changed')
    save_json(manifest,frozen)
    for year in (2022,2023):
        for start in range(0,len(paths),256):
            part=output/'prefix'/str(year)/f'part_{start//256:03d}.parquet'
            if not part.exists():
                build_year([str(p) for p in paths[start:start+256]],year,part,threads=4,historical_training=True)
            print(f'Historical prefix {year} {min(start+256,len(paths))}/{len(paths)}',flush=True)
    c=duckdb.connect();c.execute('SET threads=4')
    historical=feature_frame(c,prefix_pattern=str(output/'prefix/*/*.parquet'),date_ranges=(('2022-01-01','2023-12-31'),))
    c.close()
    historical['decision_buyable']=[decision_buyable(r.code,r.price_1449,r.preclose) for r in historical.itertuples()]
    historical=historical.loc[historical.decision_buyable & historical.price_1449.ge(5)].sort_values(['date','code']).reset_index(drop=True)
    c=duckdb.connect();c.execute('SET threads=4')
    current=feature_frame(c);c.close()
    current['decision_buyable']=[decision_buyable(r.code,r.price_1449,r.preclose) for r in current.itertuples()]
    current=current.loc[current.decision_buyable & current.price_1449.ge(5)].sort_values(['date','code']).reset_index(drop=True)
    original=pd.read_parquet(RECENT/'features.parquet').sort_values(['date','code']).reset_index(drop=True)
    pd.testing.assert_frame_equal(current,original,check_dtype=False,atol=1e-12,rtol=0)
    historical.to_parquet(output/'features_2022_2023.parquet',index=False,compression='zstd')
    features=pd.concat([historical,original],ignore_index=True).sort_values(['date','code'])
    features.to_parquet(output/'features.parquet',index=False,compression='zstd')
    jobs=historical.assign(year=historical.date.str[:4]).groupby(['code','year']).agg(active_days=('date','size'),first_active=('date','min'),last_active=('date','max')).reset_index()
    folder=output/'historical_catalog';folder.mkdir(exist_ok=True)
    jobs.to_parquet(folder/'jobs.parquet',index=False)
    save_json(folder/'manifest.json',{'rule_commit':RULE_COMMIT,'jobs_sha256':sha(folder/'jobs.parquet'),'scope':'historical_training_catalogue_only','code_years':len(jobs),'years':[2022,2023],'holdout_read':False})
    result={'rule_commit':RULE_COMMIT,'historical_rows':len(historical),'recent_rows_unchanged':len(original),'catalog_code_years':len(jobs),
        'by_year':historical.groupby(historical.date.str[:4]).agg(rows=('code','size'),days=('date','nunique'),symbols=('code','nunique')).reset_index().to_dict('records'),
        'historical_features_sha256':sha(output/'features_2022_2023.parquet'),'features_sha256':sha(output/'features.parquet'),
        'recent_feature_source_sha256':sha(RECENT/'features.parquet'),'prefix_manifest_sha256':sha(manifest),
        'prefix_output_sha256':{str(p):sha(p) for p in sorted((output/'prefix').glob('*/*.parquet'))},
        'historical_labels_read':False,'new_test_returns_read':False,'holdout_read':False}
    save_json(output/'feature_report.json',result)
    return result


def _catalog_login() -> None:
    import baostock as bs
    from .ingest import _login
    socket.setdefaulttimeout(20)
    _login()
    atexit.register(bs.logout)


def _catalog_job(job) -> dict:
    import baostock as bs
    from .cash_dividend_catalog import checked_response
    from .ingest import _login,_rows
    code,year,folder,review=job
    root=Path(folder);path=root/'vendor'/f'{code}_{year}.json';error_path=path.with_suffix('.error.json')
    try:
        if path.exists():
            checked_response(json.loads(path.read_text()),code,year,review)
            error_path.unlink(missing_ok=True)
            return {'cached':True}
        for attempt in range(3):
            try:
                raw=_rows(bs.query_dividend_data(code,year=year,yearType='operate')).to_dict('records')
                try:
                    rows=checked_response(raw,code,year,review)
                except ValueError:
                    save_json(root/'source_conflicts'/f'{code}_{year}_raw.json',raw)
                    raise
                temporary=path.with_suffix('.tmp.json');save_json(temporary,rows);temporary.replace(path)
                error_path.unlink(missing_ok=True);time.sleep(.05)
                return {'downloaded':True}
            except ValueError:
                raise
            except Exception:
                if attempt==2:
                    raise
                bs.logout();time.sleep(.5);_login()
    except Exception as error:
        record={'code':code,'year':year,'error':str(error)};save_json(error_path,record)
        return {'error':record}


def fetch_catalog_parallel(output: Path = ROOT, workers: int = 4) -> dict:
    from .cash_dividend_catalog import jobs_for,reviewed_duplicates
    root=output/'historical_catalog';jobs=jobs_for(root);reviews=reviewed_duplicates()
    (root/'vendor').mkdir(exist_ok=True);(root/'source_conflicts').mkdir(exist_ok=True)
    if not jobs.year.isin(['2022','2023']).all() or not 1<=workers<=4:
        raise ValueError('Parallel collection is bounded to the fixed historical catalogue')
    items=[(r.code,r.year,str(root),reviews.get((r.code,r.year))) for r in jobs.itertuples()]
    downloaded=cached=0;errors=[]
    with ProcessPoolExecutor(max_workers=workers,initializer=_catalog_login) as pool:
        futures=[pool.submit(_catalog_job,job) for job in items]
        for count,future in enumerate(as_completed(futures),1):
            result=future.result();downloaded+=int(result.get('downloaded',False));cached+=int(result.get('cached',False))
            if 'error' in result:
                errors.append(result['error'])
            if count%100==0 or count==len(jobs):
                progress={'processed':count,'total':len(jobs),'downloaded':downloaded,'cached':cached,'errors':len(errors)}
                save_json(root/'fetch_progress.json',progress);print(progress,flush=True)
    report={'jobs':len(jobs),'downloaded':downloaded,'cached':cached,'errors':errors,'complete':not errors,
        'holdout_read':False,'strategy_returns_read':False,'workers':workers}
    save_json(root/'fetch_report.json',report);return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['features','catalog_fetch','catalog_parallel','catalog_assemble'])
    args=parser.parse_args()
    if args.stage=='features':
        result=prepare()
    elif args.stage=='catalog_parallel':
        result=fetch_catalog_parallel()
    else:
        from .cash_dividend_catalog import fetch,assemble
        result=(fetch if args.stage=='catalog_fetch' else assemble)(ROOT/'historical_catalog')
    print(json.dumps({k:v for k,v in result.items() if 'sha256' not in k},ensure_ascii=False,indent=2))
