"""Rebuild every visible rank and economic feature group independently in SQL."""
import argparse
import json
import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from trade_research.corporate_cash import save_json,sha
from trade_research.next_day_winner_analysis import FEATURES

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,default=Path('data/research/economic_winner'))
root=parser.parse_args().root;portrait=Path('data/research/next_day_winner')
report=json.loads((root/'analysis_report.json').read_text())
assert report['label_report_sha256']==sha(root/'label_report.json')
assert report['label_verification_sha256']==sha(root/'label_verification.json')
assert json.loads((root/'label_verification.json').read_text())['passed']
assert report['source_cohort_sha256']==sha(portrait/'cohort.parquet')
for name,digest in report['outputs_sha256'].items():assert sha(root/(name+'.parquet'))==digest
c=duckdb.connect();c.execute('SET threads=4');c.execute("SET memory_limit='6GB'")
for name,path in [('out',root/'labels.parquet'),('cohort',portrait/'cohort.parquet'),
    ('ranks',root/'feature_ranks.parquet'),('feature_daily',root/'feature_daily.parquet')]:c.read_parquet(str(path)).create_view(name)
metrics=['winner_lower','winner_upper','loser_lower','loser_upper','positive_lower','positive_upper','net_mean']
agg='''count(*) AS n,count(net_return15) AS known,count(*) FILTER(WHERE label15='unknown') AS unknown,
    count(*) FILTER(WHERE label15='no_trade') AS no_trade,count(*) FILTER(WHERE label15='economic_winner') AS winner_count,
    count(*) FILTER(WHERE label15='economic_loser') AS loser_count,count(*) FILTER(WHERE net_return15>0) AS positive_count,
    avg(net_return15) AS net_mean,
    avg((label15='economic_winner')::INT) AS winner_lower,avg((label15 IN('economic_winner','unknown'))::INT) AS winner_upper,
    avg((label15='economic_loser')::INT) AS loser_lower,avg((label15 IN('economic_loser','unknown'))::INT) AS loser_upper,
    avg(CASE WHEN net_return15>0 THEN 1 ELSE 0 END) AS positive_lower,
    avg(CASE WHEN net_return15>0 OR label15='unknown' THEN 1 ELSE 0 END) AS positive_upper'''
baseline=c.sql(f'SELECT date,half,board,{agg} FROM out WHERE necessary_tradeable GROUP BY date,half,board ORDER BY date').df()
stored=pd.read_parquet(root/'baseline_daily.parquet').set_index(['date','board']).sort_index()
pd.testing.assert_frame_equal(baseline.set_index(['date','board']).sort_index()[stored.columns],stored,check_dtype=False,atol=2e-12,rtol=0)
c.register('independent_baseline',baseline)
cross=c.sql('''SELECT half,board,necessary_tradeable,
    CASE WHEN NOT original_known_label THEN 'unknown' WHEN original_winner IS TRUE THEN 'up5' ELSE 'not_up5' END AS reference_class,
    label15,count(*) AS rows FROM out GROUP BY ALL''').df()
keys=['half','board','necessary_tradeable','reference_class','label15']
pd.testing.assert_frame_equal(cross.set_index(keys).sort_index(),pd.read_parquet(root/'reference_cross.parquet').set_index(keys).sort_index(),check_dtype=False)
draws={}


def equal(a,b):
    if b is None or(np.isscalar(b) and pd.isna(b)):assert a is None
    else:np.testing.assert_allclose(a,b,atol=2e-12,rtol=0)


def interval(p,metric):
    if p.empty or p[metric].isna().any():return None
    block=p[metric].groupby(pd.to_datetime(p.date).dt.to_period('W-SUN').astype(str)).agg(['sum','size']).sort_index()
    n=len(block)
    if n<2:return None
    if n not in draws:draws[n]=np.random.default_rng(20260926).integers(0,n,(10000,n))
    sample=draws[n]
    values=block['sum'].to_numpy()[sample].sum(axis=1)/block['size'].to_numpy()[sample].sum(axis=1)
    return np.percentile(values,[2.5,97.5])


def verify_summary(item,p):
    counts={'stock_days':p.n.sum(),'dates':len(p),'known':p.known.sum(),'unknown':p.unknown.sum(),
        'no_trade':p.no_trade.sum(),'winner_cases':p.winner_count.sum(),'loser_cases':p.loser_count.sum(),
        'positive_cases':p.positive_count.sum(),'valid_net_dates':p.net_mean.notna().sum()}
    for name,value in counts.items():assert item[name]==value
    for name in metrics:
        equal(item[name],p[name].mean());equal(item[name+'_week_interval'],interval(p,name))
        if name+'_delta' in p:
            equal(item[name+'_delta'],p[name+'_delta'].mean())
            equal(item[name+'_delta_week_interval'],interval(p,name+'_delta'))


for item in report['baselines']:
    p=baseline.loc[baseline.board.eq(item['board'])&baseline.half.eq(item['half'])]
    verify_summary(item,p)
for item in report['distributions']:
    bps=item['cost_bps']
    p=c.execute(f'SELECT date,net_return{bps} AS value,label{bps} AS label FROM out WHERE necessary_tradeable AND board=? AND half=?',
        [item['board'],item['half']]).df()
    values=p.value.dropna();wins=values.loc[values.gt(0)];losses=values.loc[values.lt(0)]
    assert item['valid_rows']==len(values) and item['unknown_rows']==p.label.eq('unknown').sum() and item['no_trade']==p.label.eq('no_trade').sum()
    expected={'positive_frequency':values.gt(0).mean(),'median':values.median(),'mean_positive':wins.mean(),
        'mean_negative':losses.mean(),'payoff_ratio':wins.mean()/(-losses.mean()) if len(wins) and len(losses) else None,
        'worst_five_percent_mean':values.sort_values().iloc[:max(1,math.ceil(len(values)*.05))].mean(),'minimum':values.min()}
    for name,value in expected.items():equal(item[name],value)
    dates=p.groupby('date').value.mean().reset_index()
    equal(item['mean_signal_day'],dates.value.mean());equal(item['mean_signal_day_week_interval'],interval(dates,'value'))

rank_values=0;daily_rows=0;maximum_rank_error=0.
for feature in FEATURES:
    c.execute(f'''CREATE OR REPLACE TEMP TABLE independent_rank AS SELECT date,code,board,
      CASE WHEN {feature} IS NOT NULL THEN
        (rank() OVER(PARTITION BY date,board ORDER BY {feature} NULLS LAST)+
         (count(*) OVER(PARTITION BY date,board,{feature})-1)/2.)/
          nullif(count({feature}) OVER(PARTITION BY date,board),0) END AS score
      FROM cohort''')
    count,difference,nulls=c.sql(f'''SELECT count(*),max(abs(r.score-s.{feature})),
      sum(((r.score IS NULL)<>(s.{feature} IS NULL))::INT) FROM independent_rank r JOIN ranks s USING(date,code)''').fetchone()
    assert count==2404280 and nulls==0 and difference<2e-14
    rank_values+=count;maximum_rank_error=max(maximum_rank_error,difference)
    c.execute('''CREATE OR REPLACE TEMP TABLE observations AS SELECT o.*,r.score,
      CASE WHEN r.score IS NULL THEN 'missing' WHEN r.score<=.2 THEN 'low20'
      WHEN r.score>=.8 THEN 'high20' ELSE 'middle60' END AS band
      FROM out o JOIN independent_rank r USING(date,code) WHERE o.necessary_tradeable''')
    daily=c.sql(f'SELECT date,half,board,band,{agg} FROM observations GROUP BY date,half,board,band').df()
    daily=daily.merge(baseline[['date','board',*metrics]],on=['date','board'],how='left',validate='many_to_one',suffixes=('','_baseline'))
    for name in ['winner','loser','positive']:
        daily[name+'_lower_delta']=daily[name+'_lower']-daily[name+'_upper_baseline']
        daily[name+'_upper_delta']=daily[name+'_upper']-daily[name+'_lower_baseline']
    daily['net_mean_delta']=daily.net_mean-daily.net_mean_baseline;daily['feature']=feature
    stored=c.execute('SELECT * FROM feature_daily WHERE feature=?',[feature]).df()
    keys=['date','board','band']
    pd.testing.assert_frame_equal(daily.set_index(keys).sort_index()[stored.columns.difference(keys,sort=False)],
        stored.set_index(keys).sort_index(),check_dtype=False,atol=2e-12,rtol=0)
    daily_rows+=len(daily)
    for item in [r for r in report['feature_bands'] if r['feature']==feature]:
        p=daily.loc[daily.board.eq(item['board'])&daily.half.eq(item['half'])&daily.band.eq(item['band'])].sort_values('date')
        verify_summary(item,p)
    conditions={'reference_up5':'original_known_label AND original_winner IS TRUE',
        'economic_winner':"label15='economic_winner'",'economic_loser':"label15='economic_loser'",
        'middle':"label15='middle'",'unknown':"label15='unknown'",'no_trade':"label15='no_trade'"}
    for kind,where in conditions.items():
        inverse=c.sql(f'''SELECT date,half,board,count(*) AS n,count(*) FILTER(WHERE score IS NULL) AS missing,
            avg(score) AS mean_rank FROM observations WHERE {where} GROUP BY date,half,board ORDER BY date''').df()
        for item in [r for r in report['inverse_ranks'] if r['feature']==feature and r['kind']==kind]:
            p=inverse.loc[inverse.board.eq(item['board'])&inverse.half.eq(item['half'])]
            assert item['rows']==p.n.sum() and item['feature_missing']==p.missing.sum() and item['dates']==len(p)
            equal(item['mean_rank'],p.mean_rank.mean());equal(item['mean_rank_week_interval'],interval(p,'mean_rank'))
    print(json.dumps({'feature_checked':feature,'complete':rank_values//2404280}),flush=True)
result={'passed':True,'analysis_report_sha256':sha(root/'analysis_report.json'),'visible_rank_values':rank_values,
    'maximum_rank_error':maximum_rank_error,'baseline_dates':len(baseline),'feature_group_dates':daily_rows,
    'baseline_summaries':len(report['baselines']),'cost_distributions':len(report['distributions']),
    'band_means_bounds_and_intervals':len(report['feature_bands'])*len(metrics)*2,
    'inverse_rank_means_and_intervals':len(report['inverse_ranks']),'new_2026_prices_read':False}
save_json(root/'analysis_verification.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
