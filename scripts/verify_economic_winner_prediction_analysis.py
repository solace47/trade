"""Recompute every frozen prediction group, cost statistic and weekly interval."""
import json
import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from trade_research.corporate_cash import save_json,sha

root=Path('data/research/economic_winner_prediction');source=Path('data/research/economic_winner/period_quality')
r=json.loads((root/'analysis_report.json').read_text())
assert r['input_report_sha256']==sha(root/'input_report.json')
assert r['input_verification_sha256']==sha(root/'input_verification.json')
assert json.loads((root/'input_verification.json').read_text())['passed']
assert r['source_label_report_sha256']==sha(source/'label_report.json')
for name,digest in r['outputs_sha256'].items():assert sha(root/(name+'.parquet'))==digest
c=duckdb.connect();c.execute('SET threads=4')
for name,path in [('scores',root/'scores.parquet'),('signals',root/'signals.parquet'),('labels',source/'labels.parquet')]:
    c.read_parquet(str(path)).create_view(name)
fields='l.label5,l.label15,l.net_return5,l.net_return15,l.base_status'
c.execute(f"CREATE VIEW all_rows AS SELECT s.*,{fields} FROM scores s JOIN labels l USING(date,code) WHERE s.role IN('calibration','test')")
c.execute(f'CREATE VIEW selected AS SELECT s.*,{fields} FROM signals s JOIN labels l USING(date,code)')
classes=['economic_loser','economic_winner','middle','no_trade','unknown']
agg=['count(*) AS n','count(net_return15) AS known']
for kind in classes:
    agg += [f"count(*) FILTER(WHERE label15='{kind}') AS {kind}",
            f"avg((label15='{kind}')::INT) AS frequency_{kind}",f'avg(p_{kind}) AS predicted_{kind}']
agg+=['avg(net_return5) AS net5','avg(net_return15) AS net15']
for field in ['raw_score','calibrated_score']:
    agg += [f'avg({field}) AS {field}',f'avg({field}) FILTER(WHERE net_return15 IS NOT NULL) AS {field}_known']
agg=',\n'.join(agg)
groups={}
for name,table,keys in [('prediction_daily','all_rows',['date','half']),('bins_daily','all_rows',['date','half','score_bin']),('selected_daily','selected',['date','half','policy'])]:
    q=c.sql(f"SELECT {','.join(keys)},{agg} FROM {table} GROUP BY {','.join(keys)}").df()
    if name=='selected_daily':
        q=q.merge(groups['prediction_daily'][['date','net5','net15']],on='date',validate='many_to_one',suffixes=('','_baseline'))
        q['net5_delta']=q.net5-q.net5_baseline;q['net15_delta']=q.net15-q.net15_baseline
    stored=pd.read_parquet(root/(name+'.parquet'))
    pd.testing.assert_frame_equal(q.set_index(keys).sort_index()[stored.set_index(keys).columns],stored.set_index(keys).sort_index(),check_dtype=False,atol=2e-12,rtol=0)
    groups[name]=q
selected=c.sql('SELECT * FROM selected').df();keys=['date','code','policy']
pd.testing.assert_frame_equal(selected.set_index(keys).sort_index(),pd.read_parquet(root/'selected_outcomes.parquet').set_index(keys).sort_index(),check_dtype=False)
draws={};checked=0


def equal(expected,actual):
    global checked
    if actual is None:assert expected is None or(np.isscalar(expected) and pd.isna(expected))
    else:np.testing.assert_allclose(expected,actual,atol=2e-12,rtol=0)
    checked+=1


def interval(frame,name):
    if frame.empty or frame[name].isna().any():return None
    blocks=frame[name].groupby(pd.to_datetime(frame.date).dt.to_period('W-SUN').astype(str)).agg(['sum','size']).sort_index()
    count=len(blocks)
    if count<2:return None
    if count not in draws:draws[count]=np.random.default_rng(20260926).integers(0,count,(10000,count))
    indexes=draws[count]
    values=blocks['sum'].to_numpy()[indexes].sum(axis=1)/blocks['size'].to_numpy()[indexes].sum(axis=1)
    return np.percentile(values,[2.5,97.5])


def verify(item,p):
    assert item['dates']==len(p) and item['stock_days']==p.n.sum() and item['known']==p.known.sum()
    for kind in classes:assert item[kind]==p[kind].sum()
    for name in p:
        if name.startswith(('frequency_','predicted_','net','raw_score','calibrated_score')):
            equal(p[name].mean(),item[name]);equal(interval(p,name),item[name+'_week_interval'])


for item in r['full_pool']:
    verify(item,groups['prediction_daily'].loc[groups['prediction_daily'].half.eq(item['half'])])
for item in r['bins']:
    p=groups['bins_daily'];verify(item,p.loc[p.half.eq(item['half'])&p.score_bin.eq(item['score_bin'])])
for item in r['policies']:
    p=groups['selected_daily'];p=p.loc[p.policy.eq(item['policy'])&(p.half.eq(item['period']) if item['period']!='2025' else True)]
    if item['no_selections']:
        assert p.empty and item['stock_days']==item['dates']==0 and item['mean_net'] is None
    else:
        verify(item,p)
        rows=selected.loc[selected.policy.eq(item['policy'])&(selected.half.eq(item['period']) if item['period']!='2025' else True)]
        assert rows.base_status.value_counts().to_dict()==item['base_status_counts']
for item in r['distributions']:
    p=selected.loc[selected.policy.eq(item['policy'])&(selected.half.eq(item['period']) if item['period']!='2025' else True)]
    cost=item['cost_bps'];v=p[f'net_return{cost}'].dropna();wins=v.loc[v.gt(0)];losses=v.loc[v.lt(0)]
    assert item['known']==len(v) and item['unknown']==p[f'label{cost}'].eq('unknown').sum() and item['no_trade']==p[f'label{cost}'].eq('no_trade').sum()
    expected={'positive_frequency':v.gt(0).mean(),'mean_positive':wins.mean(),'mean_negative':losses.mean(),
        'payoff_ratio':wins.mean()/(-losses.mean()) if len(wins) and len(losses) else None,
        'worst_five_percent_mean':v.sort_values().iloc[:max(1,math.ceil(len(v)*.05))].mean(),'minimum':v.min()}
    for key,value in expected.items():equal(value,item[key])
result={'passed':True,'analysis_report_sha256':sha(root/'analysis_report.json'),
    'prediction_dates':len(groups['prediction_daily']),'score_bin_dates':len(groups['bins_daily']),
    'selected_dates':len(groups['selected_daily']),'selected_outcomes':len(selected),
    'summaries_and_intervals_checked':checked,'no_selection_policy_retained':True,'new_2026_prices_read':False}
save_json(root/'analysis_verification.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
