from pathlib import Path
import importlib.util,json,hashlib
import pandas as pd,numpy as np
from trade_research.alpha158_asof import FIELDS,Evaluator,features_for_symbol
from trade_research.alpha158_inputs import DAILY,DAILY_COLUMNS,PREFIX_COLUMNS
from trade_research.corporate_cash import save_json,sha
p=Path('data/research/alpha158_1449');base=pd.read_parquet('data/research/long_history_ridge_1449/features.parquet',columns=PREFIX_COLUMNS)
base['half']=base.date.str[:4]+np.where(base.date.str[5:7].le('06'),'H1','H2');base['exchange']=base.code.str[:2]
base['order']=[hashlib.sha256((d+code).encode()).hexdigest() for d,code in zip(base.date,base.code)]
samples=base.sort_values(['half','exchange','order']).groupby(['half','exchange'],sort=True).head(4).copy();samples.to_parquet(p/'feature_check_samples.parquet',index=False)
spec=importlib.util.spec_from_file_location('independent_reference','tests/test_alpha158_asof.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
checks=[];max_difference=0.
for number,r in enumerate(samples.itertuples(),1):
 path=DAILY/(r.code.replace('.','_')+'.parquet')
 past=pd.read_parquet(path,columns=DAILY_COLUMNS,filters=[('date','>=','2019-01-01'),('date','<',r.date)])
 past=past.loc[past.tradestatus.eq(1)].sort_values('date')
 today=pd.read_parquet(path,columns=['date','open','preclose','tradestatus'],filters=[('date','==',r.date)]).iloc[0]
 assert today.tradestatus==1
 previous=None;factor=1.;values=[]
 for row in past.itertuples():
  if previous is not None and abs(row.preclose-previous)>.005:factor*=row.preclose/previous
  values.append({name:(row.volume*factor if name=='volume' else row.amount/row.volume/factor if name=='vwap' and row.volume>0 else np.nan if name=='vwap' else getattr(row,name)/factor) for name in FIELDS})
  previous=row.close
 if previous is not None and abs(today.preclose-previous)>.005:factor*=today.preclose/previous
 values.append({'open':today.open/factor,'high':r.high_1449/factor,'low':r.low_1449/factor,'close':r.price_1449/factor,'volume':r.volume_1449*factor,'vwap':r.amount_1449/r.volume_1449/factor})
 banks={k:np.full((1,61),np.nan) for k in FIELDS}
 for lag,value in enumerate(values[-61:][::-1]):
  for k in FIELDS:banks[k][0,lag]=value[k]
 expected=module.scalar_reference(banks,len(values),0)
 daily=pd.read_parquet(path,columns=DAILY_COLUMNS,filters=[('date','>=','2019-01-01'),('date','<=',r.date)])
 prefix=samples.loc[samples.date.eq(r.date)&samples.code.eq(r.code),PREFIX_COLUMNS].reset_index(drop=True)
 actual=features_for_symbol(daily,prefix).drop(columns='date').iloc[0]
 a=actual.to_numpy();b=np.array(list(expected.values()));diff=np.nanmax(np.abs(a-b));max_difference=max(max_difference,diff)
 np.testing.assert_allclose(a,b,atol=5e-8,rtol=1e-8,equal_nan=True)
 checks.append({'date':r.date,'code':r.code,'max_difference':float(diff),'missing_features':int(actual.isna().sum()),'daily_sha256':sha(path)})
 if number%16==0:print({'checked':number,'total':len(samples),'max_difference':max_difference},flush=True)
result={'samples':len(samples),'factor_values_checked':158*len(samples),'maximum_difference':max_difference,'samples_sha256':sha(p/'feature_check_samples.parquet'),'checks':checks,'new_test_labels_read':False,'holdout_prices_read':False}
save_json(p/'independent_feature_checks.json',result);print({k:v for k,v in result.items() if k!='checks'})
