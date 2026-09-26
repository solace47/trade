"""Independently aggregate every fixed size/turnover cell and inverse portrait."""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from trade_research.corporate_cash import save_json,sha

root=Path('data/research/float_turnover_winner');label_root=Path('data/research/economic_winner/period_quality')
r=json.loads((root/'analysis_report.json').read_text())
assert r['input_report_sha256']==sha(root/'input_report.json')
assert r['input_verification_sha256']==sha(root/'input_verification.json')
assert json.loads((root/'input_verification.json').read_text())['passed']
assert r['label_report_sha256']==sha(label_root/'label_report.json')
assert r['label_analysis_verification_sha256']==sha(label_root/'analysis_verification.json')
for name,digest in r['outputs_sha256'].items():assert sha(root/(name+'.parquet'))==digest
c=duckdb.connect();c.execute('SET threads=4')
c.read_parquet(str(root/'features.parquet')).create_view('features')
c.read_parquet(str(label_root/'labels.parquet')).create_view('labels')
c.execute('''CREATE VIEW pool AS SELECT f.*,l.label15,l.net_return5,l.net_return15,l.original_known_label,l.original_winner
   FROM features f JOIN labels l USING(date,code) WHERE f.necessary_tradeable''')
agg='''count(*) AS n,count(net_return15) AS known,count(*) FILTER(WHERE label15='unknown') AS unknown,
 count(*) FILTER(WHERE label15='no_trade') AS no_trade,count(*) FILTER(WHERE label15='economic_winner') AS winner_count,
 count(*) FILTER(WHERE label15='economic_loser') AS loser_count,count(*) FILTER(WHERE net_return15>0) AS positive_count,
 avg(net_return15) AS net_mean,avg(net_return5) AS net5,
 avg((label15='economic_winner')::INT) AS winner_lower,avg((label15 IN('economic_winner','unknown'))::INT) AS winner_upper,
 avg((label15='economic_loser')::INT) AS loser_lower,avg((label15 IN('economic_loser','unknown'))::INT) AS loser_upper,
 avg(CASE WHEN net_return15>0 THEN 1 ELSE 0 END) AS positive_lower,
 avg(CASE WHEN net_return15>0 OR label15='unknown' THEN 1 ELSE 0 END) AS positive_upper'''
baseline=c.sql(f'SELECT date,board,half,{agg} FROM pool GROUP BY date,board,half').df()
stored=pd.read_parquet(root/'baseline_daily.parquet');keys=['date','board']
pd.testing.assert_frame_equal(baseline.set_index(keys).sort_index()[stored.set_index(keys).columns],stored.set_index(keys).sort_index(),check_dtype=False,atol=2e-12,rtol=0)
metrics=['winner_lower','winner_upper','loser_lower','loser_upper','positive_lower','positive_upper','net_mean','net5']
draws={};checked=0;group_dates=0


def equal(expected,actual):
    global checked
    if actual is None:assert expected is None or(np.isscalar(expected) and pd.isna(expected))
    else:np.testing.assert_allclose(expected,actual,atol=2e-12,rtol=0)
    checked+=1


def interval(p,key):
    if p.empty or p[key].isna().any():return None
    w=p[key].groupby(pd.to_datetime(p.date).dt.to_period('W-SUN').astype(str)).agg(['sum','size']).sort_index();n=len(w)
    if n<2:return None
    if n not in draws:draws[n]=np.random.default_rng(20260926).integers(0,n,(10000,n))
    ix=draws[n]
    return np.percentile(w['sum'].to_numpy()[ix].sum(axis=1)/w['size'].to_numpy()[ix].sum(axis=1),[2.5,97.5])


def verify(item,p):
    for name,value in {'stock_days':p.n.sum(),'dates':len(p),'known':p.known.sum(),'unknown':p.unknown.sum(),
        'no_trade':p.no_trade.sum(),'winner_cases':p.winner_count.sum(),'loser_cases':p.loser_count.sum(),
        'positive_cases':p.positive_count.sum(),'valid_net_dates':p.net_mean.notna().sum()}.items():assert item[name]==value
    for name in metrics:
        equal(p[name].mean(),item[name]);equal(interval(p,name),item[name+'_week_interval'])
        if name+'_delta' in p:
            equal(p[name+'_delta'].mean(),item[name+'_delta']);equal(interval(p,name+'_delta'),item[name+'_delta_week_interval'])


for item in r['baselines']:
    verify(item,baseline.loc[baseline.board.eq(item['board'])&baseline.half.eq(item['half'])])
all_stored=pd.read_parquet(root/'groups_daily.parquet')
for name in ['float_cap_proxy','turnover_1449_proxy','turnover_tail29_proxy','float_cap_x_tail_turnover']:
    expression=f'CAST({name}_quintile AS VARCHAR)' if name!='float_cap_x_tail_turnover' else "CAST(float_cap_proxy_quintile AS VARCHAR)||':'||CAST(turnover_tail29_proxy_quintile AS VARCHAR)"
    g=c.sql(f'SELECT date,board,half,{expression} AS "group",{agg} FROM pool GROUP BY date,board,half,"group"').df()
    g=g.merge(baseline[['date','board',*metrics]],on=['date','board'],validate='many_to_one',suffixes=('','_baseline'))
    for kind in ['winner','loser','positive']:
        g[kind+'_lower_delta']=g[kind+'_lower']-g[kind+'_upper_baseline']
        g[kind+'_upper_delta']=g[kind+'_upper']-g[kind+'_lower_baseline']
    g['net_mean_delta']=g.net_mean-g.net_mean_baseline;g['net5_delta']=g.net5-g.net5_baseline;g['stratum']=name
    stored=all_stored.loc[all_stored.stratum.eq(name)];keys=['date','board','group']
    pd.testing.assert_frame_equal(g.set_index(keys).sort_index()[stored.set_index(keys).columns],stored.set_index(keys).sort_index(),check_dtype=False,atol=2e-12,rtol=0)
    group_dates+=len(g)
    for item in [p for p in r['groups'] if p['stratum']==name]:
        verify(item,g.loc[g.board.eq(item['board'])&g.half.eq(item['half'])&g['group'].eq(item['group'])])
for feature in ['float_cap_proxy','turnover_1449_proxy','turnover_tail29_proxy']:
    for kind,where in {'reference_up5':'original_known_label AND original_winner IS TRUE','economic_winner':"label15='economic_winner'",
        'economic_loser':"label15='economic_loser'",'middle':"label15='middle'",'unknown':"label15='unknown'",'no_trade':"label15='no_trade'"}.items():
        g=c.sql(f'SELECT date,board,half,count(*) AS n,count(*) FILTER(WHERE {feature}_rank IS NULL) AS missing,avg({feature}_rank) AS mean_rank FROM pool WHERE {where} GROUP BY date,board,half').df()
        for item in [p for p in r['inverse_ranks'] if p['feature']==feature and p['kind']==kind]:
            p=g.loc[g.board.eq(item['board'])&g.half.eq(item['half'])]
            assert item['rows']==p.n.sum() and item['missing']==p.missing.sum() and item['dates']==len(p)
            equal(p.mean_rank.mean(),item['mean_rank']);equal(interval(p,'mean_rank'),item['mean_rank_week_interval'])
result={'passed':True,'analysis_report_sha256':sha(root/'analysis_report.json'),'baseline_dates':len(baseline),
    'group_dates':group_dates,'group_summaries':len(r['groups']),'inverse_summaries':len(r['inverse_ranks']),
    'means_bounds_intervals_checked':checked,'new_2026_prices_read':False}
save_json(root/'analysis_verification.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
