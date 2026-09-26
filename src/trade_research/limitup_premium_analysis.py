"""Date-clustered outcome association for frozen prior-limit-up environments."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import t

from .corporate_cash import save_json, sha
from .reference_gain_accounting import weekly_interval
from .limitup_premium_environment import ROOT, BASE

DETAILS=Path('config/limitup_premium_environment_analysis.json')
LABELS=BASE.with_name('labels.parquet')
METRICS=['up_lower','up_upper','down_lower','down_upper','reference_mean']
BINS=['strong','neutral','weak','unknown']
HALVES=['2024H1','2024H2','2025H1','2025H2']


def number(value):
    return float(value) if pd.notna(value) and np.isfinite(value) else None


def matched_daily(cells: pd.DataFrame,control: str,excluded_week: str|None=None) -> pd.DataFrame:
    p=cells if excluded_week is None else cells.loc[cells.week.ne(excluded_week)]
    p=p.loc[p.known.gt(0)]
    comparison=p.loc[p.environment_bin.eq(control)].groupby('stratum')[METRICS].mean()
    candidate=p.loc[p.environment_bin.eq('strong')]
    joined=candidate.merge(comparison,left_on='stratum',right_index=True,how='inner',suffixes=('','_control'))
    if joined.empty:return pd.DataFrame(columns=['date','covered_rows',*METRICS])
    for metric in METRICS:
        joined[metric]=(joined[metric]-joined[metric+'_control'])*joined.n
    sums=joined.groupby('date').agg(covered_rows=('n','sum'),**{m:(m,'sum') for m in METRICS})
    for metric in METRICS:sums[metric]/=sums.covered_rows
    return sums.reset_index()


def evaluate() -> dict:
    if (ROOT/'analysis_report.json').exists():raise ValueError('Do not overwrite inspected environment results')
    inputs=json.loads((ROOT/'input_report.json').read_text())
    check=json.loads((ROOT/'input_verification.json').read_text())
    assert check['passed'] and check['input_report_sha256']==sha(ROOT/'input_report.json')
    for name,digest in inputs['outputs_sha256'].items():assert sha(ROOT/(name+'.parquet'))==digest
    source_report=json.loads(BASE.with_name('base_report.json').read_text())
    assert sha(BASE)==inputs['sha256'][str(BASE)] and sha(LABELS)==source_report['labels_sha256']
    manifest={'input_report_sha256':sha(ROOT/'input_report.json'),'input_verification_sha256':sha(ROOT/'input_verification.json'),
      'details_sha256':sha(DETAILS),'labels_sha256':sha(LABELS),'previously_exposed_2024_2025':True,'new_2026_prices_read':False}
    save_json(ROOT/'analysis_manifest.json',manifest)
    c=duckdb.connect();c.execute('SET threads=4');c.execute("SET memory_limit='6GB'")
    c.read_parquet(str(ROOT/'features.parquet')).create_view('features')
    c.read_parquet(str(BASE)).create_view('base')
    c.read_parquet(str(LABELS)).create_view('labels')
    c.execute('''CREATE TABLE observations AS SELECT f.date,f.code,f.board,f.half,f.necessary_tradeable,f.primary_pool,
      f.environment_bin,f.main_market_return,f.main_market_rising,
      b.return_1449,b.return20_prior_adjusted,b.price_1449,b.amount_1449,
      l.known_label,l.winner IS TRUE AS up,
      l.known_label AND round(l.next_close*100)::BIGINT*100<=round(l.next_preclose*100)::BIGINT*95 AS down,
      CASE WHEN l.known_label THEN l.next_gain END AS reference_gain
      FROM features f JOIN base b USING(date,code) JOIN labels l USING(date,code)''')
    scopes=[(f'{scope}_{board}',f"board='{board}'"+(' AND necessary_tradeable' if scope=='necessary' else ''))
            for scope in ['all','necessary'] for board in ['main','chinext','star']]
    scopes.append(('primary','primary_pool IS TRUE'))
    frames=[]
    for scope,where in scopes:
        part=c.sql('''SELECT date,half,environment_bin,count(*) AS n,sum(known_label::INT) AS known,
            sum(up::INT) AS up_count,sum(down::INT) AS down_count,avg(reference_gain) AS reference_mean
            FROM observations WHERE '''+where+''' GROUP BY date,half,environment_bin ORDER BY date,environment_bin''').df()
        part['scope']=scope;part['up_lower']=part.up_count/part.n;part['down_lower']=part.down_count/part.n
        part['up_upper']=(part.up_count+part.n-part.known)/part.n
        part['down_upper']=(part.down_count+part.n-part.known)/part.n
        frames.append(part)
    daily=pd.concat(frames,ignore_index=True)
    daily.to_parquet(ROOT/'group_daily.parquet',index=False,compression='zstd')
    summaries=[]
    context=pd.read_parquet(ROOT/'daily_features.parquet')
    for scope,_ in scopes:
        for half in HALVES:
            for group in BINS:
                p=daily.loc[daily.scope.eq(scope)&daily.half.eq(half)&daily.environment_bin.eq(group)].sort_values('date')
                item={'scope':scope,'half':half,'environment_bin':group,'dates':len(p),'stock_days':int(p.n.sum()),
                    'known_labels':int(p.known.sum()),'unknown_labels':int((p.n-p.known).sum()),
                    'up_cases':int(p.up_count.sum()),'down_cases':int(p.down_count.sum())}
                for metric in METRICS:
                    values=p.set_index('date')[metric]
                    item[metric]=number(values.mean());item[metric+'_week_interval']=weekly_interval(values)
                descriptions=context.loc[context.date.isin(p.date)]
                item['input_descriptions']={name:number(descriptions[name].mean()) for name in
                    ['raw_premium','excess_premium','rising_fraction','resealed_fraction','opened_after_touch_fraction','market_return','market_rising']}
                summaries.append(item)
    primary=c.sql('''SELECT *,floor(main_market_return/.01)::INT AS market_bin,
      floor(main_market_rising/.1)::INT AS breadth_bin,floor(return_1449/.02)::INT AS stock_day_bin,
      floor(return20_prior_adjusted/.1)::INT AS prior20_bin,floor(log2(price_1449))::INT AS price_bin,
      floor(log2(amount_1449))::INT AS amount_bin FROM observations WHERE primary_pool IS TRUE''').df()
    adjusted,matched,jackknife,cells_saved=[],[],[],[]
    for adjustment in ['market','market_and_stock']:
        strata=['market_bin','breadth_bin']+(['stock_day_bin','prior20_bin','price_bin','amount_bin'] if adjustment=='market_and_stock' else [])
        usable=primary[strata].notna().all(axis=1)
        p=primary.loc[usable].copy()
        p['stratum']=pd.MultiIndex.from_frame(p[['half',*strata]]).factorize(sort=True)[0]
        cells=p.groupby(['date','half','environment_bin','stratum'],observed=True).agg(
            n=('code','size'),known=('known_label','sum'),up_count=('up','sum'),down_count=('down','sum'),reference_mean=('reference_gain','mean')).reset_index()
        cells['week']=pd.to_datetime(cells.date).dt.to_period('W-SUN').astype(str)
        cells['up_lower']=cells.up_count/cells.n;cells['down_lower']=cells.down_count/cells.n
        cells['up_upper']=(cells.up_count+cells.n-cells.known)/cells.n
        cells['down_upper']=(cells.down_count+cells.n-cells.known)/cells.n
        cells['adjustment']=adjustment;cells_saved.append(cells)
        for half in HALVES:
            for control in ['neutral','weak']:
                g=cells.loc[cells.half.eq(half)&cells.environment_bin.isin(['strong',control])]
                out=matched_daily(g,control)
                common={'half':half,'adjustment':adjustment,'comparison':'strong_minus_'+control}
                matched.append(out.assign(**common))
                weeks=sorted(g.week.unique());estimates=[]
                for week in weeks:
                    deleted=matched_daily(g,control,week)
                    estimates.append(deleted[METRICS].mean().to_numpy(dtype=float))
                    jackknife.append(dict(common,deleted_week=week,matched_dates=len(deleted),
                        **{m:number(deleted[m].mean()) for m in METRICS}))
                estimates=np.array(estimates)
                candidates=primary.loc[primary.half.eq(half)&primary.environment_bin.eq('strong')]
                reference=primary.loc[primary.half.eq(half)&primary.environment_bin.eq(control)]
                item=dict(common,candidate_dates=candidates.date.nunique(),candidate_rows=len(candidates),
                    control_dates=reference.date.nunique(),control_rows=len(reference),matched_dates=len(out),
                    covered_rows=int(out.covered_rows.sum()),unknown_label_cells=int(g.known.eq(0).sum()),
                    weeks=len(weeks),interval_method='delete_one_entire_week_recompute_both_sides_t_jackknife',
                    input_missing_candidate_rows=int((~usable&primary.half.eq(half)&primary.environment_bin.eq('strong')).sum()))
                for i,metric in enumerate(METRICS):
                    point=number(out[metric].mean());item[metric]=point
                    values=estimates[:,i] if len(estimates) else np.array([])
                    if point is None or len(values)<2 or not np.isfinite(values).all():interval=None
                    else:
                        variance=(len(values)-1)/len(values)*np.square(values-values.mean()).sum()
                        margin=float(t.ppf(.975,len(values)-1)*np.sqrt(variance))
                        interval=[point-margin,point+margin]
                    item[metric+'_week_interval']=interval
                adjusted.append(item)
                print(json.dumps({'adjustment':adjustment,'half':half,'control':control,'matched_dates':len(out)}),flush=True)
    pd.concat(cells_saved,ignore_index=True).to_parquet(ROOT/'adjustment_cells.parquet',index=False,compression='zstd')
    pd.concat(matched,ignore_index=True).to_parquet(ROOT/'adjusted_daily.parquet',index=False,compression='zstd')
    pd.DataFrame(jackknife).to_parquet(ROOT/'jackknife.parquet',index=False,compression='zstd')
    result={'interpretation':'exploratory_reference_price_association_not_executed_returns_or_investor_identity',
        'group_summaries':summaries,'adjusted_summaries':adjusted,
        'analysis_manifest_sha256':sha(ROOT/'analysis_manifest.json'),
        'outputs_sha256':{name:sha(ROOT/(name+'.parquet')) for name in ['group_daily','adjustment_cells','adjusted_daily','jackknife']},
        'new_2026_prices_read':False}
    save_json(ROOT/'analysis_report.json',result)
    return result


if __name__=='__main__':
    result=evaluate()
    print(json.dumps({'primary':[s for s in result['group_summaries'] if s['scope']=='primary'],
                     'adjusted':result['adjusted_summaries']},ensure_ascii=False,indent=2))
