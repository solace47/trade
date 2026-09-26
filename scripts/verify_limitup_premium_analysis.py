"""Rebuild environment associations from labels, SQL cells and deleted weeks."""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import t

from trade_research.corporate_cash import save_json, sha

root=Path('data/research/limitup_premium_environment')
base=Path('data/research/next_day_winner/visible_base.parquet')
manifest=json.loads((root/'analysis_manifest.json').read_text())
report=json.loads((root/'analysis_report.json').read_text())
inputs=json.loads((root/'input_report.json').read_text())
assert report['analysis_manifest_sha256']==sha(root/'analysis_manifest.json')
for key,path in [('input_report',root/'input_report.json'),('input_verification',root/'input_verification.json'),
                 ('details',Path('config/limitup_premium_environment_analysis.json')),('labels',base.with_name('labels.parquet'))]:
    assert manifest[key+'_sha256']==sha(path)
assert json.loads((root/'input_verification.json').read_text())['passed']
assert inputs['sha256'][str(base)]==sha(base)
for name,digest in inputs['outputs_sha256'].items():assert sha(root/(name+'.parquet'))==digest
for name,digest in report['outputs_sha256'].items():assert sha(root/(name+'.parquet'))==digest
c=duckdb.connect();c.execute('SET threads=4');c.execute("SET memory_limit='6GB'")
c.read_parquet(str(base)).create_view('base')
c.read_parquet(str(base.with_name('labels.parquet'))).create_view('labels')
c.read_parquet(str(root/'features.parquet')).create_view('features')
c.execute('''CREATE TABLE observations AS SELECT f.date,f.code,f.half,f.board,f.necessary_tradeable,
    f.primary_pool,f.environment_bin,l.known_label AS known,
    l.known_label AND round(l.next_close*100)::BIGINT*100>=round(l.next_preclose*100)::BIGINT*105 AS up,
    l.known_label AND round(l.next_close*100)::BIGINT*100<=round(l.next_preclose*100)::BIGINT*95 AS down,
    CASE WHEN l.known_label THEN l.next_close/l.next_preclose-1 END AS gain,
    floor(f.main_market_return/.01)::INT AS market_bin,floor(f.main_market_rising/.1)::INT AS breadth_bin,
    floor(b.return_1449/.02)::INT AS stock_day_bin,floor(b.return20_prior_adjusted/.1)::INT AS prior20_bin,
    floor(log2(b.price_1449))::INT AS price_bin,floor(log2(b.amount_1449))::INT AS amount_bin
    FROM features f JOIN base b USING(date,code) JOIN labels l USING(date,code)''')
assert c.sql('SELECT count(*) FROM observations').fetchone()[0]==inputs['stock_days']
assert c.sql('SELECT count(*)-count(DISTINCT(date,code)) FROM observations').fetchone()[0]==0
metrics=['up_lower','up_upper','down_lower','down_upper','reference_mean']
aggregates='''count(*) AS n,sum(known::INT) AS known,sum(up::INT) AS up_count,sum(down::INT) AS down_count,
    avg(gain) AS reference_mean,sum(up::INT)*1./count(*) AS up_lower,
    (sum(up::INT)+count(*)-sum(known::INT))*1./count(*) AS up_upper,
    sum(down::INT)*1./count(*) AS down_lower,
    (sum(down::INT)+count(*)-sum(known::INT))*1./count(*) AS down_upper'''
scopes={f'{scope}_{board}':f"board='{board}'"+(' AND necessary_tradeable' if scope=='necessary' else '')
    for scope in ['all','necessary'] for board in ['main','chinext','star']}
scopes['primary']='primary_pool IS TRUE'
daily=pd.concat([c.sql(f"SELECT date,half,environment_bin,'{scope}' AS scope,{aggregates} FROM observations "
    f"WHERE {where} GROUP BY date,half,environment_bin").df() for scope,where in scopes.items()],ignore_index=True)

def compare(frame,name,keys):
    actual=pd.read_parquet(root/(name+'.parquet')).set_index(keys).sort_index()
    expected=frame.set_index(keys).sort_index()[actual.columns]
    pd.testing.assert_frame_equal(expected,actual,check_dtype=False,atol=2e-12,rtol=0)
    return actual

compare(daily,'group_daily',['scope','half','environment_bin','date'])
cells=[]
for adjustment,columns in [('market',['market_bin','breadth_bin']),
    ('market_and_stock',['market_bin','breadth_bin','stock_day_bin','prior20_bin','price_bin','amount_bin'])]:
    where=' AND '.join(col+' IS NOT NULL' for col in columns)
    ranked=f"SELECT *,dense_rank() OVER(ORDER BY half,{','.join(columns)})-1 AS stratum FROM observations WHERE primary_pool IS TRUE AND {where}"
    p=c.sql(f"WITH ranked AS ({ranked}) SELECT date,half,environment_bin,stratum,{aggregates} FROM ranked "
            "GROUP BY date,half,environment_bin,stratum").df()
    p['week']=pd.to_datetime(p.date).dt.to_period('W-SUN').astype(str)
    p['adjustment']=adjustment;cells.append(p)
cells=pd.concat(cells,ignore_index=True)
compare(cells,'adjustment_cells',['adjustment','half','date','environment_bin','stratum'])
c.register('independent_cells',cells)

def estimate(adjustment,half,control,week=None):
    where="adjustment=? AND half=? AND environment_bin IN ('strong',?)"
    params=[adjustment,half,control]
    if week is not None:where+=' AND week<>?';params.append(week)
    controls=','.join(f'avg({m}) AS {m}' for m in metrics)
    differences=','.join(f'sum((s.{m}-r.{m})*s.n)/sum(s.n) AS {m}' for m in metrics)
    return c.execute(f'''WITH part AS (SELECT * FROM independent_cells WHERE {where} AND known>0),
        reference AS (SELECT stratum,{controls} FROM part WHERE environment_bin<>'strong' GROUP BY stratum)
        SELECT s.date,sum(s.n)::BIGINT AS covered_rows,{differences}
        FROM part s JOIN reference r USING(stratum) WHERE s.environment_bin='strong'
        GROUP BY s.date ORDER BY s.date''',params).df()

matched=[];deleted=[]
for summary in report['adjusted_summaries']:
    adjustment,half,comparison=[summary[k] for k in ['adjustment','half','comparison']]
    control=comparison.removeprefix('strong_minus_')
    common=dict(adjustment=adjustment,half=half,comparison=comparison)
    p=estimate(adjustment,half,control);matched.append(p.assign(**common))
    g=cells.loc[cells.adjustment.eq(adjustment)&cells.half.eq(half)&cells.environment_bin.isin(['strong',control])]
    for week in sorted(g.week.unique()):
        e=estimate(adjustment,half,control,week)
        deleted.append(dict(common,deleted_week=week,matched_dates=len(e),**e[metrics].mean().to_dict()))
matched=pd.concat(matched,ignore_index=True);deleted=pd.DataFrame(deleted)
compare(matched,'adjusted_daily',['adjustment','half','comparison','date'])
compare(deleted,'jackknife',['adjustment','half','comparison','deleted_week'])

def equivalent(actual,expected):
    if expected is None or (np.isscalar(expected) and pd.isna(expected)):assert actual is None
    else:np.testing.assert_allclose(actual,expected,atol=2e-12,rtol=0)

def interval(p,metric):
    if p.empty or p[metric].isna().any():return None
    weeks=pd.to_datetime(p.date).dt.to_period('W-SUN').astype(str)
    blocks=p[metric].groupby(weeks).agg(['sum','size']).sort_index()
    if len(blocks)<2:return None
    sample=np.random.default_rng(20260926).integers(0,len(blocks),(10000,len(blocks)))
    totals=blocks['sum'].to_numpy()[sample].sum(axis=1)
    counts=blocks['size'].to_numpy()[sample].sum(axis=1)
    return np.percentile(totals/counts,[2.5,97.5])

context=pd.read_parquet(root/'daily_features.parquet')
for summary in report['group_summaries']:
    p=daily.loc[daily.scope.eq(summary['scope'])&daily.half.eq(summary['half'])&
                daily.environment_bin.eq(summary['environment_bin'])].sort_values('date')
    counts={'dates':len(p),'stock_days':p.n.sum(),'known_labels':p.known.sum(),'unknown_labels':(p.n-p.known).sum(),
        'up_cases':p.up_count.sum(),'down_cases':p.down_count.sum()}
    for key,value in counts.items():assert summary[key]==value
    for metric in metrics:
        equivalent(summary[metric],p[metric].mean())
        equivalent(summary[metric+'_week_interval'],interval(p,metric))
    for key,value in summary['input_descriptions'].items():
        equivalent(value,context.loc[context.date.isin(p.date),key].mean())

for summary in report['adjusted_summaries']:
    a,h,comparison=[summary[k] for k in ['adjustment','half','comparison']]
    control=comparison.removeprefix('strong_minus_')
    p=matched.loc[matched.adjustment.eq(a)&matched.half.eq(h)&matched.comparison.eq(comparison)]
    j=deleted.loc[deleted.adjustment.eq(a)&deleted.half.eq(h)&deleted.comparison.eq(comparison)]
    g=cells.loc[cells.adjustment.eq(a)&cells.half.eq(h)&cells.environment_bin.isin(['strong',control])]
    assert summary['matched_dates']==len(p) and summary['covered_rows']==p.covered_rows.sum()
    assert summary['unknown_label_cells']==g.known.eq(0).sum() and summary['weeks']==g.week.nunique()
    columns=['market_bin','breadth_bin']+(['stock_day_bin','prior20_bin','price_bin','amount_bin'] if a=='market_and_stock' else [])
    nulls=' OR '.join(col+' IS NULL' for col in columns)
    for label,group in [('candidate','strong'),('control',control)]:
        counts=c.execute("SELECT count(DISTINCT date),count(*) FROM observations WHERE primary_pool IS TRUE AND half=? AND environment_bin=?",[h,group]).fetchone()
        assert counts==(summary[label+'_dates'],summary[label+'_rows'])
    missing=c.execute(f"SELECT count(*) FROM observations WHERE primary_pool IS TRUE AND half=? AND environment_bin='strong' AND ({nulls})",[h]).fetchone()[0]
    assert summary['input_missing_candidate_rows']==missing
    for metric in metrics:
        point=p[metric].mean();equivalent(summary[metric],point)
        values=j[metric].to_numpy()
        if not np.isfinite(point) or len(values)<2 or not np.isfinite(values).all():ci=None
        else:
            radius=np.sqrt((len(values)-1)*np.var(values,ddof=0))*t.ppf(.975,len(values)-1)
            ci=[point-radius,point+radius]
        equivalent(summary[metric+'_week_interval'],ci)

result={'passed':True,'original_stock_days':inputs['stock_days'],'group_dates':len(daily),
    'adjustment_cells':len(cells),'adjusted_dates':len(matched),'deleted_week_recomputations':len(deleted),
    'group_summary_values_and_intervals':len(report['group_summaries'])*len(metrics),
    'adjusted_summary_values_and_intervals':len(report['adjusted_summaries'])*len(metrics),
    'analysis_report_sha256':sha(root/'analysis_report.json'),'new_2026_prices_read':False}
save_json(root/'analysis_verification.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
