"""Check the unchanged feature contract, convex fit, calibration and all orders."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from threadpoolctl import threadpool_limits

from trade_research.corporate_cash import save_json,sha

root=Path('data/research/economic_winner_prediction')
source=Path('data/research/economic_winner/period_quality')
features=Path('data/research/winner_direction')
report=json.loads((root/'input_report.json').read_text())
assert report['protocol_sha256']==sha(Path('config/economic_winner_prediction.json'))
for name in ['label_report','label_verification']:
    assert report['source_'+name+'_sha256']==sha(source/(name+'.json'))
assert report['source_input_report_sha256']==sha(features/'input_report.json')
for name,digest in report['outputs_sha256'].items():
    assert sha(root/(name+('.json' if name=='model' else '.parquet')))==digest
# Those source features already had all 29 cross-sectional ranks and the ten
# additional variables independently rebuilt. Bind that check to the exact file.
old_check=json.loads((features/'independent_fit_checks.json').read_text())
assert old_check['input_report_sha256']==sha(features/'input_report.json')
old_report=json.loads((features/'input_report.json').read_text())
for name in ['model_features.parquet','visible_pool.parquet']:
    assert sha(features/name)==old_report['outputs_sha256'][name]
model=json.loads((root/'model.json').read_text())
encoded=pd.read_parquet(features/'model_features.parquet')
pool=pd.read_parquet(features/'visible_pool.parquet',columns=['date','code','half','board','necessary_tradeable','decision_shares','price_1449'])
scores=pd.read_parquet(root/'scores.parquet');history=pd.read_parquet(root/'historic_labels.parquet')
names=model['feature_names'];classes=model['classes'];ordinary=['economic_loser','economic_winner','middle']
pd.testing.assert_frame_equal(pool[['date','code']],encoded[['date','code']])
pd.testing.assert_frame_equal(pool[['date','code','half','board']],scores[['date','code','half','board']])
assert pool.necessary_tradeable.all() and names==list(encoded.drop(columns=['date','code']))
expected_history=pool.loc[pool.date.lt('2025-01-01'),['date','code']].merge(
    pd.read_parquet(source/'labels.parquet',filters=[('date','<','2025-01-01')],columns=list(history)),on=['date','code'],validate='one_to_one')
pd.testing.assert_frame_equal(history.reset_index(drop=True),expected_history)
cal=pd.read_parquet('data/baostock/market_2020_2026/metadata/calendar.parquet')
days=sorted(cal.loc[cal.is_trading_day.eq('1')&cal.calendar_date.between('2024-01-01','2025-12-31'),'calendar_date'])
position={day:i for i,day in enumerate(days)}
assert history.next_date.eq(history.date.map({d:days[position[d]+1] for d in history.date.unique()})).all()
roles=np.full(len(pool),'purged_boundary',dtype=object)
hist=pool[['date','code']].merge(history,on=['date','code'],how='left',validate='one_to_one')
train=hist.date.lt('2024-07-01')&hist.next_date.lt('2024-07-01')
calibration=hist.date.between('2024-07-01','2024-12-30')&hist.next_date.lt('2025-01-01')
test=hist.date.ge('2025-01-01')
roles[train]='fit';roles[calibration]='calibration';roles[test]='test'
np.testing.assert_array_equal(scores.role,roles)
assert train.sum()==report['training_rows'] and hist.loc[train,'next_date'].max()==report['training_last_label']
assert hist.loc[calibration,'next_date'].max()==report['calibration_last_label']
values=encoded[names].to_numpy();median=np.nanmedian(values[train],axis=0)
train_values=np.where(np.isnan(values[train]),median,values[train]);n=len(train_values)
mean=np.array([math.fsum(train_values[:,i])/n for i in range(len(names))])
scale=np.maximum(np.sqrt([math.fsum((train_values[:,i]-mean[i])**2)/n for i in range(len(names))]),1e-12)
for actual,stored in [(median,model['median']),(mean,model['mean']),(scale,model['scale'])]:
    np.testing.assert_allclose(actual,stored,atol=1e-12,rtol=0)
class_returns={name:float(hist.loc[train&hist.label15.eq(name),'net_return15'].mean()) for name in ordinary}
np.testing.assert_allclose([class_returns[name] for name in ordinary],
    [model['class_returns'][name] for name in ordinary],atol=2e-14,rtol=0)
coef=np.asarray(model['coefficients']);intercept=np.asarray(model['intercepts'])
gradient=np.zeros_like(coef);intercept_gradient=np.zeros(len(classes));maximum_error=0.
target=hist.label15.map({name:i for i,name in enumerate(classes)}).fillna(-1).to_numpy(dtype=int)
raw=np.empty(len(values));class_means=np.array([class_returns[name] for name in ordinary])
ordinary_indices=[classes.index(name) for name in ordinary]
with threadpool_limits(limits=4):
    for first in range(0,len(values),100000):
        last=min(first+100000,len(values));part=values[first:last]
        z=(np.where(np.isnan(part),median,part)-np.asarray(model['mean']))/np.asarray(model['scale'])
        probability=softmax(z@coef.T+intercept,axis=1)
        expected=scores.iloc[first:last][['p_'+name for name in classes]].to_numpy()
        maximum_error=max(maximum_error,float(np.max(np.abs(probability-expected))))
        np.testing.assert_allclose(probability,expected,atol=1e-12,rtol=0)
        ordinary_p=probability[:,ordinary_indices]
        raw[first:last]=(ordinary_p*class_means).sum(axis=1)/ordinary_p.sum(axis=1)
        mask=train.iloc[first:last].to_numpy()
        error=probability[mask].copy();error[np.arange(mask.sum()),target[first:last][mask]]-=1
        gradient+=error.T@z[mask];intercept_gradient+=error.sum(axis=0)
gradient=(gradient+coef/model['C'])/n;intercept_gradient/=n
maximum_gradient=max(float(np.abs(gradient).max()),float(np.abs(intercept_gradient).max()))
assert maximum_gradient<1e-7,maximum_gradient
np.testing.assert_allclose(raw,scores.raw_score,atol=2e-14,rtol=0)
eligible=calibration&hist.net_return15.notna()
known=hist.loc[eligible].copy();weight=1/known.groupby('date').code.transform('size').to_numpy()
matrix=np.column_stack([np.ones(len(known)),raw[eligible]])
unconstrained=np.linalg.lstsq(matrix*np.sqrt(weight)[:,None],known.net_return15.to_numpy()*np.sqrt(weight),rcond=None)[0]
expected_intercept=float(unconstrained[0]);expected_slope=float(unconstrained[1])
if expected_slope<0:
    expected_slope=0.;expected_intercept=float(np.average(known.net_return15,weights=weight))
np.testing.assert_allclose([expected_intercept,expected_slope],[model['calibration']['intercept'],model['calibration']['slope']],atol=2e-12,rtol=0)
np.testing.assert_allclose(scores.calibrated_score,expected_intercept+expected_slope*raw,atol=2e-12,rtol=0)
bounds=[raw[calibration].min(),raw[calibration].max()]
np.testing.assert_allclose(bounds,model['calibration']['raw_score_bounds'],atol=2e-14,rtol=0)
np.testing.assert_array_equal(scores.in_calibration_support,(scores.raw_score>=bounds[0]-2e-14)&(scores.raw_score<=bounds[1]+2e-14))
edges=np.percentile(scores.loc[calibration,'raw_score'],np.arange(10,100,10))
np.testing.assert_allclose(edges,model['calibration']['raw_score_deciles'],atol=2e-14,rtol=0)
np.testing.assert_array_equal(scores.score_bin,np.digitize(scores.raw_score,edges,right=False)+1)
signals=pd.read_parquet(root/'signals.parquet');selected=[]
for name in ['calibrated','raw']:
    p=scores.loc[test&(scores.calibrated_score.gt(0)&scores.in_calibration_support if name=='calibrated' else scores.raw_score.gt(0))]
    previous={}
    for date,g in p.groupby('date',sort=True):
        order=['calibrated_score','raw_score','code'] if name=='calibrated' else ['raw_score','code']
        ascending=[False,False,True] if name=='calibrated' else [False,True]
        count=0
        for row in g.sort_values(order,ascending=ascending).itertuples():
            if position[date]-previous.get(row.code,-1000)<=5:continue
            count+=1;previous[row.code]=position[date];selected.append((date,row.code,count,name))
            if count>=5:break
assert set(selected)==set(zip(signals.date,signals.code,signals.daily_rank,signals.policy))
expected=pd.DataFrame(selected,columns=['date','code','daily_rank','policy']).merge(scores,on=['date','code'],validate='many_to_one').merge(pool[['date','code','decision_shares','price_1449']],on=['date','code'],validate='many_to_one')
keys=['date','code','policy'];pd.testing.assert_frame_equal(signals.set_index(keys).sort_index(),expected.set_index(keys).sort_index())
result={'passed':True,'input_report_sha256':sha(root/'input_report.json'),
    'source_feature_verification_sha256':sha(features/'independent_fit_checks.json'),
    'feature_values_bound_to_verified_source':len(pool)*len(names),'probabilities_checked':len(scores)*len(classes),
    'maximum_probability_error':maximum_error,'maximum_average_gradient':maximum_gradient,
    'unconstrained_calibration_slope':float(unconstrained[1]),'calibration_slope':expected_slope,
    'orders_checked':len(selected),'test_counts':report['test_counts'],
    'test_labels_used':False,'new_2026_prices_read':False}
save_json(root/'input_verification.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
