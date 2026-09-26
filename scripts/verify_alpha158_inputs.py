from pathlib import Path
import json,math
import numpy as np,pandas as pd
from trade_research.corporate_cash import sha,save_json
from trade_research.alpha158_asof import features_for_symbol
from trade_research.alpha158_inputs import DAILY,DAILY_COLUMNS,PREFIX_COLUMNS
p=Path('data/research/alpha158_1449')
report=json.loads((p/'input_report.json').read_text());fr=json.loads((p/'feature_report.json').read_text())
assert sha(Path('src/trade_research/alpha158_asof.py'))==fr['adapter_sha256']
assert sha(Path('config/alpha158_definition.json'))==fr['definition_sha256']
base=pd.read_parquet('data/research/long_history_ridge_1449/features.parquet').set_index(['date','code']).sort_index()
alpha=pd.read_parquet(p/'features.parquet').set_index(['date','code']).sort_index()
pd.testing.assert_index_equal(base.index,alpha.index)
samples=pd.read_parquet(p/'feature_check_samples.parquet');feature_error=0.
for r in samples.itertuples():
 path=DAILY/(r.code.replace('.','_')+'.parquet')
 daily=pd.read_parquet(path,columns=DAILY_COLUMNS,filters=[('date','>=','2019-01-01'),('date','<=',r.date)])
 prefix=samples.loc[samples.date.eq(r.date)&samples.code.eq(r.code),PREFIX_COLUMNS]
 expected=features_for_symbol(daily,prefix).drop(columns='date').iloc[0].to_numpy()
 actual=alpha.loc[(r.date,r.code)].to_numpy()
 np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0,equal_nan=True)
 feature_error=max(feature_error,np.nanmax(abs(actual-expected)))
cal=pd.read_parquet('data/baostock/market_2020_2026/metadata/calendar.parquet')
calendar=sorted(cal.loc[cal.is_trading_day.eq('1')&cal.calendar_date.between('2022-01-01','2025-12-31'),'calendar_date']);indices={d:i for i,d in enumerate(calendar)}
coverage=pd.read_parquet('data/research/cash_dividend_catalog/query_coverage.parquet')
fits=[];lists={};max_score_error=0.
for name,details in report['models'].items():
 scores=pd.read_parquet(p/name/'all_scores.parquet').set_index(['date','code']).sort_index()
 signals=pd.read_parquet(p/name/'signals.parquet');assert sha(p/name/'signals.parquet')==details['signals_sha256']
 source=base if name=='robust18' else alpha;chosen=[]
 for audit in [a for a in report['training'] if a['model']==name]:
  model=json.loads((p/f"{name}_{audit['period']}.json").read_text());assert sha(p/f"{name}_{audit['period']}.json")==audit['model_sha256']
  train=source.loc[source.index.get_level_values('date')<=audit['train_last'],model['columns']]
  assert len(train)==audit['training_rows']
  for i,col in enumerate(model['columns']):
   series=train[col].dropna();center=series.quantile(.5);spread=(series-center).abs().quantile(.5)
   assert abs(center-model['median'][i])<1e-12 and abs((spread+1e-12)*1.4826-model['scale'][i])<1e-12
  test=scores.loc[(scores.index.get_level_values('date')>=audit['test_first'])&(scores.index.get_level_values('date')<=audit['test_last'])].copy()
  predicted=np.full(len(test),model['intercept'])
  for i,col in enumerate(model['columns']):
   z=(source.loc[test.index,col]-model['median'][i])/model['scale'][i]
   z=z.clip(-3,3).fillna(0) if col not in model['all_missing_columns'] else z*0
   predicted+=z.to_numpy()*model['coefficients'][i]
  error=float(np.max(abs(predicted-test.score)));assert error<1e-12;max_score_error=max(max_score_error,error)
  test['score']=predicted;last={}
  for date,group in test.loc[test.score>0].reset_index().groupby('date'):
   rank=0
   for row in group.sort_values(['score','code'],ascending=[False,True]).itertuples():
    if indices[date]-last.get(row.code,-1000)>5:
     rank+=1;last[row.code]=indices[date];chosen.append((date,row.code,rank))
     if rank==5:break
  fits.append({'model':name,'period':audit['period'],'training_rows':len(train),'normalization_columns':len(model['columns']),'predictions':len(test),'maximum_score_error':error})
 high=signals.loc[signals.arm.eq('high')];low=signals.loc[signals.arm.eq('low')]
 assert set(chosen)==set(zip(high.date,high.code,high.daily_rank));matched=[]
 eligible=base.reset_index();eligible['price_signal']=eligible.price_1449.round(2)
 for date,group in high.groupby('date'):
  available=eligible.loc[eligible.date.eq(date)&~eligible.code.isin(group.code)].set_index('code',drop=False)
  for r in group.sort_values('daily_rank').itertuples():
   options=[]
   for z in available.itertuples(index=False):
    ar=z.amount_signal/r.amount_signal;pr=z.price_signal/r.price_signal
    prior=abs(z.return20_prior_adjusted-r.return20_prior_adjusted);day=abs(z.return_1450-r.return_1450)
    if z.board==r.board and .5<=ar<=2 and .5<=pr<=2 and prior<=.05 and day<=.02:
     dist=prior/.05+day/.02+abs(math.log(ar))/math.log(2)+abs(math.log(pr))/math.log(2)
     options.append((dist,z.code))
   if options:
    dist,code=min(options);matched.append((date,code,r.pair_id));available=available.drop(index=code)
 assert set(matched)==set(zip(low.date,low.code,low.pair_id))
 needed=[]
 for r in signals.itertuples():
  for year in range(int(r.date[:4]),int(calendar[indices[r.date]+10][:4])+1):needed.append((r.code,str(year)))
 needed=pd.DataFrame(set(needed),columns=['code','year']);assert needed.merge(coverage,on=['code','year'],how='left',indicator=True)._merge.eq('both').all()
 lists[name]={'candidates':len(chosen),'controls':len(matched),'holding_window_code_years':len(needed),'signals_sha256':sha(p/name/'signals.parquet')}
 print(name,lists[name],flush=True)
result={'saved_real_feature_values':len(samples)*158,'saved_feature_max_difference':feature_error,'features_sha256':sha(p/'features.parquet'),'fits':fits,'models':lists,'maximum_prediction_difference':max_score_error,'new_test_returns_read':False,'holdout_prices_read':False}
save_json(p/'independent_input_checks.json',result);print(json.dumps(result,indent=2))
