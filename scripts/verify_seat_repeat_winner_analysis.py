"""Recompute seat outcome linkage, conditional statistics and all matched deltas."""
import json
import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from trade_research.corporate_cash import save_json,sha

root=Path('data/research/seat_repeat_winner');label_root=Path('data/research/economic_winner/period_quality')
r=json.loads((root/'analysis_report.json').read_text())
for key,path in [('input_report_sha256',root/'input_report.json'),('input_verification_sha256',root/'input_verification.json'),
    ('label_report_sha256',label_root/'label_report.json'),('label_analysis_verification_sha256',label_root/'analysis_verification.json')]:assert r[key]==sha(path)
assert json.loads((root/'input_verification.json').read_text())['passed']
assert json.loads((label_root/'analysis_verification.json').read_text())['passed']
assert json.loads((label_root/'label_report.json').read_text())['labels_sha256']==sha(label_root/'labels.parquet')
for name,digest in r['outputs_sha256'].items():assert sha(root/(name+'.parquet'))==digest
c=duckdb.connect();c.execute('SET threads=4')
for name,path in [('pool',root/'pool.parquet'),('labels',label_root/'labels.parquet'),('pairs',root/'pairs.parquet')]:c.read_parquet(str(path)).create_view(name)
c.execute('''CREATE VIEW linked AS SELECT p.*,l.label5,l.label15,l.net_return5,l.net_return15,l.base_status,l.original_known_label,l.original_winner
    FROM pool p LEFT JOIN labels l USING(date,code) WHERE p.necessary_tradeable''')


def compare(expected,saved,keys):
    a=expected.set_index(keys).sort_index();b=saved.set_index(keys).sort_index()
    assert set(a.columns)==set(b.columns)
    for name in b:
        pd.testing.assert_series_equal(a[name].isna(),b[name].isna(),check_index_type=False)
        known=b[name].notna()
        pd.testing.assert_series_equal(a.loc[known,name],b.loc[known,name],check_dtype=False,check_index_type=False,atol=2e-12,rtol=0)


linked=c.sql('SELECT * FROM linked').df();compare(linked,pd.read_parquet(root/'outcomes.parquet'),['date','code'])
c.execute('''CREATE VIEW long AS SELECT l.*,b.bps AS cost_bps,
    CASE WHEN b.bps=5 THEN label5 ELSE label15 END AS label,
    CASE WHEN b.bps=5 THEN net_return5 ELSE net_return15 END AS net
    FROM linked l CROSS JOIN(VALUES(5),(15)) b(bps)''')
agg='''count(*) AS n,count(net) AS known,count(*) FILTER(WHERE label='unknown') AS unknown,
 count(*) FILTER(WHERE label='no_trade') AS no_trade,count(*) FILTER(WHERE label='economic_winner') AS winner_count,
 count(*) FILTER(WHERE label='economic_loser') AS loser_count,count(*) FILTER(WHERE net>0) AS positive_count,avg(net) AS net_mean,
 avg((label='economic_winner')::INT) AS winner_lower,avg((label IN('economic_winner','unknown'))::INT) AS winner_upper,
 avg((label='economic_loser')::INT) AS loser_lower,avg((label IN('economic_loser','unknown'))::INT) AS loser_upper,
 avg(CASE WHEN net>0 THEN 1 ELSE 0 END) AS positive_lower,avg(CASE WHEN net>0 OR label='unknown' THEN 1 ELSE 0 END) AS positive_upper'''
base=c.sql(f"SELECT date,half,cost_bps,{agg} FROM long WHERE seat_class<>'prior_day_unlisted' GROUP BY ALL").df()
compare(base,pd.read_parquet(root/'baseline_daily.parquet'),['date','cost_bps'])
groups=c.sql(f'SELECT date,half,cost_bps,seat_class,{agg} FROM long GROUP BY ALL').df()
metrics=['winner_lower','winner_upper','loser_lower','loser_upper','positive_lower','positive_upper','net_mean']
groups=groups.merge(base[['date','cost_bps',*metrics]],on=['date','cost_bps'],how='left',validate='many_to_one',suffixes=('','_baseline'))
for name in ['winner','loser','positive']:
    groups[name+'_lower_delta']=groups[name+'_lower']-groups[name+'_upper_baseline']
    groups[name+'_upper_delta']=groups[name+'_upper']-groups[name+'_lower_baseline']
groups['net_mean_delta']=groups.net_mean-groups.net_mean_baseline
compare(groups,pd.read_parquet(root/'groups_daily.parquet'),['date','cost_bps','seat_class'])
draws={};checked=0


def equal(a,b):
    global checked
    if b is None:assert a is None or(np.isscalar(a) and pd.isna(a))
    else:np.testing.assert_allclose(a,b,atol=2e-12,rtol=0)
    checked+=1


def interval(p,key):
    if p.empty or p[key].isna().any():return None
    w=p[key].groupby(pd.to_datetime(p.date).dt.to_period('W-SUN').astype(str)).agg(['sum','size']).sort_index();n=len(w)
    if n<2:return None
    if n not in draws:draws[n]=np.random.default_rng(20260926).integers(0,n,(10000,n))
    ix=draws[n]
    return np.percentile(w['sum'].to_numpy()[ix].sum(axis=1)/w['size'].to_numpy()[ix].sum(axis=1),[2.5,97.5])


def part(p,item):
    result=p.loc[p.half.str.startswith(item['period'])&p.cost_bps.eq(item['cost_bps'])]
    if 'seat_class' in item:result=result.loc[result.seat_class.eq(item['seat_class'])]
    return result


def group_check(item,p):
    for name,value in {'stock_days':p.n.sum(),'dates':len(p),'known':p.known.sum(),'unknown':p.unknown.sum(),'no_trade':p.no_trade.sum(),
        'winner_cases':p.winner_count.sum(),'loser_cases':p.loser_count.sum(),'positive_cases':p.positive_count.sum(),'valid_net_dates':p.net_mean.notna().sum()}.items():assert item[name]==value
    for name in metrics:
        equal(p[name].mean(),item[name]);equal(interval(p,name),item[name+'_week_interval'])
        if name+'_delta' in p:
            equal(p[name+'_delta'].mean(),item[name+'_delta']);equal(interval(p,name+'_delta'),item[name+'_delta_week_interval'])
    if 'net_mean_baseline' in p:
        equal(p.net_mean_baseline.mean(),item['same_day_listed_net_mean'])
        equal(interval(p,'net_mean_baseline'),item['same_day_listed_net_mean_week_interval'])


for item in r['baselines']:group_check(item,part(base,item))
for item in r['groups']:group_check(item,part(groups,item))
long=c.sql('SELECT date,half,cost_bps,seat_class,label,net FROM long').df()
for item in r['distributions']:
    p=part(long,item);values=p.net.dropna().sort_values();wins=values.loc[values.gt(0)];losses=values.loc[values.lt(0)]
    assert item['known']==len(values) and item['unknown']==p.label.eq('unknown').sum() and item['no_trade']==p.label.eq('no_trade').sum()
    for name,value in {'positive_frequency':values.gt(0).mean(),'median':values.median(),'mean_positive':wins.mean(),'mean_negative':losses.mean(),
        'payoff_ratio':wins.mean()/-losses.mean() if len(wins) and len(losses) else None,
        'worst_five_percent_mean':values.iloc[:max(1,math.ceil(len(values)*.05))].mean(),'minimum':values.min()}.items():equal(value,item[name])
pair=c.sql('''SELECT p.*,h.label5 AS label5_high,h.label15 AS label15_high,h.net_return5 AS net_return5_high,h.net_return15 AS net_return15_high,
    l.label5 AS label5_control,l.label15 AS label15_control,l.net_return5 AS net_return5_control,l.net_return15 AS net_return15_control,
    h.cost_bps,h.net-l.net AS difference,h.net IS NOT NULL AND l.net IS NOT NULL AS known,
    h.label='unknown' OR l.label='unknown' AS unknown,
    NOT(h.label='unknown' OR l.label='unknown') AND(h.net IS NULL OR l.net IS NULL) AS no_trade,
    (h.label='economic_winner')::INT-(l.label IN('economic_winner','unknown'))::INT AS winner_lower_delta,
    (h.label IN('economic_winner','unknown'))::INT-(l.label='economic_winner')::INT AS winner_upper_delta,
    (h.label='economic_loser')::INT-(l.label IN('economic_loser','unknown'))::INT AS loser_lower_delta,
    (h.label IN('economic_loser','unknown'))::INT-(l.label='economic_loser')::INT AS loser_upper_delta,
    (CASE WHEN h.net>0 THEN 1 ELSE 0 END)-(CASE WHEN l.net>0 OR l.label='unknown' THEN 1 ELSE 0 END) AS positive_lower_delta,
    (CASE WHEN h.net>0 OR h.label='unknown' THEN 1 ELSE 0 END)-(CASE WHEN l.net>0 THEN 1 ELSE 0 END) AS positive_upper_delta
    FROM pairs p JOIN long h ON h.date=p.date AND h.code=p.code JOIN long l ON l.date=p.date AND l.code=p.control_code AND l.cost_bps=h.cost_bps''').df()
compare(pair,pd.read_parquet(root/'paired_outcomes.parquet'),['date','code','cost_bps'])
pair_fields=['difference',*[name+suffix+'_delta' for name in ['winner','loser','positive'] for suffix in ['_lower','_upper']]]
c.register('paired',pair)
pair_daily=c.sql('SELECT date,half,cost_bps,count(*) AS n,sum(known::INT) AS known,sum(unknown::INT) AS unknown,sum(no_trade::INT) AS no_trade,'+
    ','.join(f'avg({name}) AS {name}' for name in pair_fields)+' FROM paired GROUP BY ALL').df()
compare(pair_daily,pd.read_parquet(root/'paired_daily.parquet'),['date','cost_bps'])
for item in r['pairs']:
    p=part(pair_daily,item)
    for name,value in {'pairs':p.n.sum(),'dates':len(p),'known':p.known.sum(),'unknown':p.unknown.sum(),'no_trade':p.no_trade.sum(),
        'valid_net_dates':p.difference.notna().sum()}.items():assert item[name]==value
    for name in pair_fields:
        equal(p[name].mean(),item[name]);equal(interval(p,name),item[name+'_week_interval'])
matches=c.sql('SELECT * FROM pairs').df();unmatched=pd.read_parquet(root/'unmatched.parquet')
for item in r['matching']:
    p=matches.loc[matches.half.str.startswith(item['period'])]
    high=linked.loc[linked.half.str.startswith(item['period'])&linked.seat_class.eq('repeat_buy_list_only')]
    missing=unmatched.merge(high[['date','code']],on=['date','code'],validate='one_to_one')
    assert item['primary']==len(high) and item['matched']==len(p) and item['unmatched']==len(missing)
    assert item['unmatched_reasons']==missing.reason.value_counts().to_dict()
    for field in ['distance','return_1449_gap','prior_day_return_gap','return20_prior_adjusted_gap','float_cap_proxy_gap','amount_1449_gap']:
        for suffix,value in [('mean',p[field].mean()),('median',p[field].median()),('p95',p[field].quantile(.95))]:equal(value,item[field+'_'+suffix])
cross=c.sql('SELECT half,seat_class,label15,original_known_label,original_winner,count(*) AS n FROM linked GROUP BY ALL').df()
compare(cross,pd.read_parquet(root/'label_cross.parquet'),['half','seat_class','label15','original_known_label','original_winner'])
result={'passed':True,'analysis_report_sha256':sha(root/'analysis_report.json'),'outcome_rows':len(linked),'baseline_cost_dates':len(base),
    'group_cost_dates':len(groups),'pair_cost_rows':len(pair),'pair_cost_dates':len(pair_daily),'label_cross_cells':len(cross),
    'summary_values_and_intervals':checked,'new_2026_prices_read':False}
save_json(root/'analysis_verification.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
