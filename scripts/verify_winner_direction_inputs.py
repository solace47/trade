"""Rebuild every class label, rank feature, probability, selection and match."""
import argparse
import json
import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.special import softmax
from threadpoolctl import threadpool_limits

from trade_research.corporate_cash import save_json,sha

root=Path('data/research/winner_direction')
p=argparse.ArgumentParser();p.add_argument('stage',choices=['fit','finalize']);args=p.parse_args()
report=json.loads((root/'input_report.json').read_text())
assert sha(Path('data/research/next_day_winner/cohort.parquet'))==report['source_cohort_sha256']
for name,digest in report['outputs_sha256'].items():assert sha(root/name)==digest
for name,info in report['models'].items():assert sha(root/name/'signals.parquet')==info['signals_sha256']
cal=pd.read_parquet('data/baostock/market_2020_2026/metadata/calendar.parquet')
calendar=sorted(cal.loc[cal.is_trading_day.eq('1') & cal.calendar_date.between('2024-01-01','2025-12-31'),'calendar_date'])
position={d:i for i,d in enumerate(calendar)}

if args.stage=='finalize':
 fit=json.loads((root/'independent_fit_checks.json').read_text())
 assert fit['input_report_sha256']==sha(root/'input_report.json')
 cat=json.loads((root/'catalog/coverage_report.json').read_text())
 assert cat['complete'] and sha(root/'catalog/combined_events.parquet')==cat['events_sha256']
 assert sha(root/'catalog/combined_coverage.parquet')==cat['coverage_sha256']
 coverage=pd.read_parquet(root/'catalog/combined_coverage.parquet')
 needed=set()
 for name in report['models']:
  signals=pd.read_parquet(root/name/'signals.parquet')
  assert sha(root/name/'signals.parquet')==fit['models'][name]['signals_sha256']
  needed|={(r.code,str(y)) for r in signals.itertuples() for y in range(int(r.date[:4]),int(calendar[position[r.date]+10][:4])+1)}
 assert needed.issubset(set(zip(coverage.code,coverage.year)))
 fit.update(catalogue_code_years=len(needed),catalogue_events_sha256=cat['events_sha256'])
 save_json(root/'independent_input_checks.json',fit)
 print(json.dumps(fit,ensure_ascii=False,indent=2))
 raise SystemExit(0)

model=json.loads((root/'model.json').read_text())
feature=pd.read_parquet(root/'model_features.parquet')
pool=pd.read_parquet(root/'visible_pool.parquet')
labels=pd.read_parquet(root/'training_labels.parquet')
scores=pd.read_parquet(root/'all_scores.parquet')
names=model['feature_names']
assert feature.columns.tolist()==['date','code',*names]
assert np.array_equal(feature[['date','code']],pool[['date','code']])
assert pool.necessary_tradeable.all() and pool.isST.eq(0).all() and (~pool.reference_gap).all()
assert pool.listing_age_sessions.ge(20).all()
assert (pool.date.le('2024-12-30')|pool.date.between('2025-01-02','2025-12-17')).all()
c=duckdb.connect();c.execute('SET threads=4');c.execute("SET memory_limit='6GB'")
c.read_parquet('data/research/next_day_winner/cohort.parquet').create_view('cohort')
c.register('feature',feature);c.register('pool',pool);c.register('labels',labels)
missing=c.sql("""SELECT count(*) FROM cohort WHERE necessary_tradeable
 AND(date<='2024-12-30' OR date BETWEEN '2025-01-02' AND '2025-12-17')
 AND NOT EXISTS(SELECT 1 FROM feature f WHERE f.date=cohort.date AND f.code=cohort.code)""").fetchone()[0]
assert missing==0
rank_fields=[n for n in names if n.startswith('rank_')]
max_rank_error=0.
for name in rank_fields:
 raw=name[5:]
 row=c.sql(f"""WITH ranked AS(SELECT date,code,CASE WHEN \"{raw}\" IS NULL THEN NULL ELSE
 (rank() OVER(PARTITION BY date,board ORDER BY \"{raw}\" NULLS LAST)
 +(count(*) OVER(PARTITION BY date,board,\"{raw}\")-1)/2.)
 /nullif(count(\"{raw}\") OVER(PARTITION BY date,board),0) END AS value FROM cohort)
 SELECT count(*) FILTER(WHERE(f.\"{name}\" IS NULL)<>(r.value IS NULL)) AS missing_error,
 max(abs(f.\"{name}\"-r.value)) AS numeric_error FROM feature f JOIN ranked r USING(date,code)""").fetchone()
 assert row[0]==0 and row[1]<1e-12,(name,row)
 max_rank_error=max(max_rank_error,row[1])
 print('rank_checked',name,flush=True)
for name,val in {'log_price':np.log(pool.price_1449),'log_amount':np.log(pool.amount_1449),
 'day_range':(pool.high_1449-pool.low_1449)/pool.preclose,
 **{k:pool[k] for k in ['return_1449','return5_prior_adjusted','return20_prior_adjusted','market_return','market_rising']},
 'board_chinext':pool.board.eq('chinext').astype(float),'board_star':pool.board.eq('star').astype(float)}.items():
 assert np.allclose(feature[name],val,rtol=0,atol=1e-12,equal_nan=True),name
train=pool.date.lt('2025-01-01')
assert np.array_equal(pool.loc[train,['date','code']],labels[['date','code']])
assert labels.next_date.lt('2025-01-01').all()
assert (labels.next_date==labels.date.map({d:calendar[position[d]+1] for d in labels.date.unique()})).all()
source=pd.read_parquet('data/research/next_day_winner/labels.parquet',filters=[('date','<=','2024-12-30')])
actual=labels[['date','code','target']].merge(source,on=['date','code'],validate='one_to_one')
close=np.rint(actual.next_close.fillna(0)*100).astype('int64');ref=np.rint(actual.next_preclose.fillna(0)*100).astype('int64')
target=np.full(len(actual),'flat',dtype=object)
target[close*100>=ref*105]='up';target[close*100<=ref*95]='down';target[~actual.known_label]='unknown'
assert np.array_equal(actual.target,target)
raw=feature[names].to_numpy()
median=np.nanmedian(raw[train],axis=0)
filled=np.where(np.isnan(raw),median,raw)
# Compensated column sums avoid row-major accumulation noise at this sample size.
train_values=filled[train]
mean=np.array([math.fsum(train_values[:,i])/len(labels) for i in range(len(names))])
scale=np.maximum(np.sqrt([math.fsum((train_values[:,i]-mean[i])**2)/len(labels) for i in range(len(names))]),1e-12)
assert np.allclose(median,model['median'],rtol=0,atol=1e-12)
assert np.allclose(mean,model['mean'],rtol=0,atol=1e-12)
assert np.allclose(scale,model['scale'],rtol=0,atol=1e-12)
z=(filled-np.array(model['mean']))/np.array(model['scale'])
coef=np.array(model['coefficients']);intercept=np.array(model['intercepts'])
with threadpool_limits(limits=4):
 probabilities=softmax(z@coef.T+intercept,axis=1)
 error=probabilities[train].copy()
 index={name:i for i,name in enumerate(model['classes'])}
 error[np.arange(len(labels)),labels.target.map(index).to_numpy()]-=1
 gradient=error.T@z[train]/len(labels)+coef/(model['C']*len(labels))
 max_gradient=max(float(np.max(np.abs(gradient))),float(np.max(np.abs(error.mean(axis=0)))))
 assert max_gradient<1e-7,max_gradient
assert np.array_equal(pool.loc[~train,['date','code']],scores[['date','code']])
prediction_error=max(float(np.max(np.abs(probabilities[~train,i]-scores['p_'+name]))) for i,name in enumerate(model['classes']))
assert prediction_error<1e-12
assert np.allclose(scores.balanced,scores.p_up-scores.p_down,rtol=0,atol=0)
assert np.array_equal(scores.upside_only,scores.p_up)
model_checks={}
test_pool=pool.loc[~train].copy();c.register('test_pool',test_pool)
for model_name,info in report['models'].items():
 signals=pd.read_parquet(root/model_name/'signals.parquet');high=signals.loc[signals.arm.eq('high')];low=signals.loc[signals.arm.eq('low')]
 selected=[];last={}
 for date,part in scores.loc[scores[model_name].gt(0)].groupby('date',sort=True):
  rank=0
  for row in part.sort_values([model_name,'code'],ascending=[False,True]).itertuples():
   if position[date]-last.get(row.code,-1000)<=5:continue
   rank+=1;last[row.code]=position[date];selected.append((date,row.code,rank))
   if rank==5:break
 assert set(selected)==set(zip(high.date,high.code,high.daily_rank))
 c.register('chosen',high)
 edges=c.sql("""WITH e AS(SELECT h.date,h.code AS event_code,h.daily_rank,p.code AS peer_code,
 abs(p.return20_prior_adjusted-h.return20_prior_adjusted) AS prior_gap,
 abs(p.return_1449-h.return_1449) AS day_gap,p.price_1449/h.price_1449 AS pr,p.amount_1449/h.amount_1449 AS ar
 FROM chosen h JOIN test_pool p ON h.date=p.date AND h.board=p.board
 WHERE NOT EXISTS(SELECT 1 FROM chosen x WHERE x.date=p.date AND x.code=p.code))
 SELECT *,prior_gap/.05+day_gap/.02+abs(ln(pr))/ln(2)+abs(ln(ar))/ln(2) AS distance
 FROM e WHERE prior_gap<=.05 AND day_gap<=.02 AND pr BETWEEN .5 AND 2 AND ar BETWEEN .5 AND 2
 ORDER BY date,daily_rank,distance,peer_code""").df()
 used=set();assigned=set();pairs=set()
 for row in edges.itertuples():
  if (row.date,row.event_code) in assigned or (row.date,row.peer_code) in used:continue
  assigned.add((row.date,row.event_code));used.add((row.date,row.peer_code));pairs.add((row.date,row.peer_code,row.date+':'+row.event_code))
 assert pairs==set(zip(low.date,low.code,low.pair_id))
 values=signals.merge(test_pool,on=['date','code'],validate='one_to_one',suffixes=('_actual','_base'))
 for column in test_pool.columns:
  if column not in ('date','code'):
   pd.testing.assert_series_equal(values[column+'_actual'],values[column+'_base'],check_names=False)
 cents=np.rint(signals.price_1449*100).astype('int64');affordable=2_000_000//cents
 expected=np.where(signals.board.eq('star'),np.where(affordable>=200,affordable,0),affordable//100*100)
 assert np.array_equal(expected,signals.decision_shares)
 model_checks[model_name]={'candidates':len(high),'controls':len(low),'all_matching_edges':len(edges),'signals_sha256':info['signals_sha256']}
result={'training_rows':len(labels),'target_counts':labels.target.value_counts().to_dict(),'next_label_before_2025':True,
 'rank_values_checked':len(feature)*len(rank_fields),'rank_max_error':max_rank_error,
 'softmax_rows_checked':int((~train).sum()),'prediction_max_error':prediction_error,'objective_gradient_max':max_gradient,
 'models':model_checks,'input_report_sha256':sha(root/'input_report.json'),'new_2026_prices_read':False}
save_json(root/'independent_fit_checks.json',result)
print(json.dumps(result,ensure_ascii=False,indent=2))
