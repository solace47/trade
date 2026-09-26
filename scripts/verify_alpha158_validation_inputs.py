"""Independent columnwise scores, SQL rankings and exhaustive match edges."""
from pathlib import Path
import json
import duckdb
import numpy as np
import pandas as pd
from trade_research.corporate_cash import save_json,sha

p=Path('data/research/alpha158_pool_2026');report=json.loads((p/'input_report.json').read_text())
policy=json.loads(Path('config/alpha158_pool_2026_protocol.json').read_text())
assert sha(Path(policy['model_path']))==report['model_sha256']==policy['model_sha256']
model=json.loads(Path(policy['model_path']).read_text())
base=pd.read_parquet(p/'base_features.parquet');alpha=pd.read_parquet(p/'features.parquet')
scores=pd.read_parquet(p/'all_scores.parquet');assert sha(p/'all_scores.parquet')==report['all_scores_sha256']
pd.testing.assert_frame_equal(base[['date','code']],alpha[['date','code']]);pd.testing.assert_frame_equal(scores[['date','code']],alpha[['date','code']])
predicted=np.full(len(alpha),model['intercept'])
for i,column in enumerate(model['columns']):
 z=((alpha[column]-model['median'][i])/model['scale'][i]).clip(-3,3).fillna(0)
 if column in model['all_missing_columns']:z[:]=0
 predicted+=z.to_numpy()*model['coefficients'][i]
error=float(abs(predicted-scores.score).max());assert error<1e-12
c=duckdb.connect();c.execute('SET threads=4');c.register('scores',scores)
base['price_signal']=base.price_1449.round(2);c.register('base',base)
cal=pd.read_parquet('data/baostock/market_2020_2026/metadata/calendar.parquet')
calendar=sorted(cal.loc[cal.is_trading_day.eq('1')&cal.calendar_date.between(policy['signal_first'],policy['execution_last']),'calendar_date']);indices={d:i for i,d in enumerate(calendar)}
coverage=pd.read_parquet(p/'catalog/query_coverage.parquet');results={}
for name,details in report['models'].items():
 ordering="sha256('positive-score-pool-v1'||date||code),code" if name=='positive_pool' else 'score DESC,code'
 pool=c.sql(f'SELECT date,code,score FROM scores WHERE score>0 ORDER BY date,{ordering}').df()
 selected=[];last={};date=None;rank=0
 for row in pool.itertuples():
  if row.date!=date:date=row.date;rank=0
  if rank==5 or indices[date]-last.get(row.code,-1000)<=5:continue
  rank+=1;selected.append((date,row.code,rank));last[row.code]=indices[date]
 s=pd.read_parquet(p/name/'signals.parquet');assert sha(p/name/'signals.parquet')==details['signals_sha256']
 h=s.loc[s.arm.eq('high')];lo=s.loc[s.arm.eq('low')]
 assert set(selected)==set(zip(h.date,h.code,h.daily_rank))
 c.register('high',h)
 edges=c.sql('''WITH x AS(SELECT h.date,h.code AS event,h.daily_rank,b.code AS peer,
 abs(b.return20_prior_adjusted-h.return20_prior_adjusted) AS prior_gap,
 abs(b.return_1450-h.return_1450) AS day_gap,b.amount_signal/h.amount_signal AS ar,b.price_signal/h.price_signal AS pr
 FROM high h JOIN base b ON b.date=h.date AND b.board=h.board
 WHERE NOT EXISTS(SELECT 1 FROM high x WHERE x.date=b.date AND x.code=b.code))
 SELECT *,prior_gap/.05+day_gap/.02+abs(ln(ar))/ln(2)+abs(ln(pr))/ln(2) AS distance FROM x
 WHERE prior_gap<=.05 AND day_gap<=.02 AND ar BETWEEN .5 AND 2 AND pr BETWEEN .5 AND 2
 ORDER BY date,daily_rank,distance,peer''').df()
 keyed={(date,event):g for (date,event),g in edges.groupby(['date','event'])};matches=[]
 for date,g in h.groupby('date'):
  used=set()
  for row in g.sort_values('daily_rank').itertuples():
   options=keyed.get((date,row.code))
   if options is None:continue
   for candidate in options.itertuples():
    if candidate.peer not in used:
     matches.append((date,candidate.peer,row.pair_id));used.add(candidate.peer);break
 assert set(matches)==set(zip(lo.date,lo.code,lo.pair_id))
 needed=[]
 for row in s.itertuples():
  for year in range(int(row.date[:4]),int(calendar[indices[row.date]+10][:4])+1):needed.append((row.code,str(year)))
 needed=pd.DataFrame(set(needed),columns=['code','year'])
 assert needed.merge(coverage,on=['code','year'],how='left',indicator=True)._merge.eq('both').all()
 result={'candidates':len(selected),'controls':len(matches),'match_edges':len(edges),'catalogue_code_years':len(needed),'signals_sha256':sha(p/name/'signals.parquet')}
 results[name]=result;print(name,result,flush=True)
save_json(p/'independent_input_checks.json',{'score_rows':len(scores),'columnwise_score_max_error':error,'models':results,'new_holding_results_read':False,'holdout_inputs_read':True})
