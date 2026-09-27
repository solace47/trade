"""Independent SQL outcome joins, complete group means and conditional tails."""
import json
import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from trade_research.corporate_cash import save_json,sha

root=Path('data/research/external_tail_combo');label_root=Path('data/research/economic_winner/period_quality')
r=json.loads((root/'analysis_report.json').read_text())
for key,path in [('input_report_sha256',root/'input_report.json'),('input_verification_sha256',root/'input_verification.json'),
    ('label_report_sha256',label_root/'label_report.json'),('label_analysis_verification_sha256',label_root/'analysis_verification.json')]:assert sha(path)==r[key]
assert json.loads((root/'input_verification.json').read_text())['passed']
assert json.loads((label_root/'analysis_verification.json').read_text())['passed']
assert json.loads((label_root/'label_report.json').read_text())['labels_sha256']==sha(label_root/'labels.parquet')
for name,digest in r['outputs_sha256'].items():assert sha(root/(name+'.parquet'))==digest
c=duckdb.connect();c.execute('SET threads=4')
c.read_parquet(str(root/'pool.parquet')).create_view('pool');c.read_parquet(str(label_root/'labels.parquet')).create_view('labels')
c.execute('''CREATE VIEW linked AS SELECT p.*,l.label5,l.label15,l.net_return5,l.net_return15,l.base_status,l.original_known_label,l.original_winner
    FROM pool p LEFT JOIN labels l USING(date,code) WHERE p.necessary_tradeable''')


def compare(a,b,keys):
    a=a.set_index(keys).sort_index();b=b.set_index(keys).sort_index()
    assert set(a.columns)==set(b.columns)
    for name in b:
        pd.testing.assert_series_equal(a[name].isna(),b[name].isna(),check_index_type=False)
        mask=b[name].notna()
        pd.testing.assert_series_equal(a.loc[mask,name],b.loc[mask,name],check_dtype=False,check_index_type=False,atol=2e-12,rtol=0)


linked=c.sql('SELECT * FROM linked').df();compare(linked,pd.read_parquet(root/'outcomes.parquet'),['date','code'])
c.execute('''CREATE VIEW long AS SELECT l.*,b.bps AS cost_bps,CASE WHEN bps=5 THEN label5 ELSE label15 END AS label,
    CASE WHEN bps=5 THEN net_return5 ELSE net_return15 END AS net FROM linked l CROSS JOIN(VALUES(5),(15)) b(bps)''')
agg='''count(*) AS n,count(net) AS known,count(*) FILTER(WHERE label='unknown') AS unknown,
 count(*) FILTER(WHERE label='no_trade') AS no_trade,count(*) FILTER(WHERE label='economic_winner') AS winner_count,
 count(*) FILTER(WHERE label='economic_loser') AS loser_count,count(*) FILTER(WHERE net>0) AS positive_count,avg(net) AS net_mean,
 avg((label='economic_winner')::INT) AS winner_lower,avg((label IN('economic_winner','unknown'))::INT) AS winner_upper,
 avg((label='economic_loser')::INT) AS loser_lower,avg((label IN('economic_loser','unknown'))::INT) AS loser_upper,
 avg(CASE WHEN net>0 THEN 1 ELSE 0 END) AS positive_lower,avg(CASE WHEN net>0 OR label='unknown' THEN 1 ELSE 0 END) AS positive_upper'''
market=c.sql(f'SELECT date,half,cost_bps,{agg} FROM long GROUP BY ALL').df()
base=c.sql(f'SELECT date,half,cost_bps,{agg} FROM long WHERE base_combo GROUP BY ALL').df()
compare(market,pd.read_parquet(root/'market_daily.parquet'),['date','cost_bps']);compare(base,pd.read_parquet(root/'base_daily.parquet'),['date','cost_bps'])
metrics=['winner_lower','winner_upper','loser_lower','loser_upper','positive_lower','positive_upper','net_mean']
where={'base':'base_combo','recent_only':'base_combo AND recent_activity','vwap_only':'base_combo AND vwap_support',
    **{n:f'"group"=\'{n}\'' for n in ['0:0','0:1','1:0','1:1','detail_unknown','base_unknown','outside_base']}}
tables=[]
for scope,condition in where.items():
    d=c.sql(f'SELECT date,half,cost_bps,{agg} FROM long WHERE {condition} GROUP BY ALL').df()
    for tag,background in [('base',base),('market',market)]:
        d=d.merge(background[['date','cost_bps',*metrics]].rename(columns={m:m+'_'+tag for m in metrics}),on=['date','cost_bps'],how='left',validate='many_to_one')
        for name in ['winner','loser','positive']:
            d[name+'_lower_'+tag+'_delta']=d[name+'_lower']-d[name+'_upper_'+tag]
            d[name+'_upper_'+tag+'_delta']=d[name+'_upper']-d[name+'_lower_'+tag]
        d['net_mean_'+tag+'_delta']=d.net_mean-d['net_mean_'+tag]
    d['scope']=scope;tables.append(d)
groups=pd.concat(tables,ignore_index=True);compare(groups,pd.read_parquet(root/'groups_daily.parquet'),['date','cost_bps','scope'])
draws={};checked=0


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


def part(p,item):return p.loc[p.half.str.startswith(item['period'])&p.cost_bps.eq(item['cost_bps'])]


def verify(item,p,with_background):
    for name,value in {'stock_days':p.n.sum(),'dates':len(p),'known':p.known.sum(),'unknown':p.unknown.sum(),'no_trade':p.no_trade.sum(),
        'winner_cases':p.winner_count.sum(),'loser_cases':p.loser_count.sum(),'positive_cases':p.positive_count.sum(),'valid_net_dates':p.net_mean.notna().sum()}.items():assert item[name]==value
    fields=metrics+([n+'_'+tag+'_delta' for n in metrics for tag in ['base','market']] if with_background else [])
    for name in fields:
        equal(p[name].mean(),item[name]);equal(interval(p,name),item[name+'_week_interval'])


for item in r['market']:verify(item,part(market,item),False)
for item in r['groups']:verify(item,part(groups.loc[groups.scope.eq(item['scope'])],item),True)
for scope,condition in where.items():
    rows=c.sql(f'SELECT half,cost_bps,label,net FROM long WHERE {condition}').df()
    for item in [x for x in r['distributions'] if x['scope']==scope]:
        p=part(rows,item);values=p.net.dropna().sort_values();wins=values.loc[values.gt(0)];losses=values.loc[values.lt(0)]
        assert item['known']==len(values) and item['unknown']==p.label.eq('unknown').sum() and item['no_trade']==p.label.eq('no_trade').sum()
        for name,value in {'positive_frequency':values.gt(0).mean(),'median':values.median(),'mean_positive':wins.mean(),'mean_negative':losses.mean(),
            'payoff_ratio':wins.mean()/-losses.mean() if len(wins) and len(losses) else None,
            'worst_five_percent_mean':values.iloc[:max(1,math.ceil(len(values)*.05))].mean(),'minimum':values.min()}.items():equal(value,item[name])
cross=c.sql('SELECT half,"group",label15,original_known_label,original_winner,count(*) AS n FROM linked GROUP BY ALL').df()
compare(cross,pd.read_parquet(root/'label_cross.parquet'),['half','group','label15','original_known_label','original_winner'])
result={'passed':True,'analysis_report_sha256':sha(root/'analysis_report.json'),'outcome_rows':len(linked),'market_cost_dates':len(market),
    'base_cost_dates':len(base),'group_cost_dates':len(groups),'label_cross_cells':len(cross),'summary_values_and_intervals':checked,'new_2026_prices_read':False}
save_json(root/'analysis_verification.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
