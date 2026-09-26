"""Independent SQL reconstruction of both tails and both matching definitions."""
import json
import duckdb
import numpy as np
import pandas as pd
from trade_research.corporate_cash import save_json,sha
from trade_research.next_day_winner import ROOT
from trade_research.next_day_winner_analysis import FEATURES

c=duckdb.connect();c.execute('SET threads=4');c.execute("SET memory_limit='6GB'")
c.read_parquet(str(ROOT/'cohort.parquet')).create_view('cohort')
c.read_parquet(str(ROOT/'labels.parquet')).create_view('labels')
c.execute("""CREATE VIEW f AS SELECT a.*,
 CASE WHEN a.known_label THEN (cast(round(b.next_close*100) AS BIGINT)*100<=cast(round(b.next_preclose*100) AS BIGINT)*95)::DOUBLE END AS loss
 FROM cohort a JOIN labels b USING(date,code)""")
saved=pd.read_parquet(ROOT/'tail_balance.parquet')
original=pd.read_parquet(ROOT/'matched_daily.parquet')
ranged=pd.read_parquet(ROOT/'range_matched_daily.parquet')
cells=0;dailycells=0;largest=0.
for feature in FEATURES:
 c.execute(f"""CREATE OR REPLACE TEMP TABLE ranked AS SELECT *,
 CASE WHEN \"{feature}\" IS NULL THEN NULL ELSE
 (rank() OVER(PARTITION BY date,board ORDER BY \"{feature}\" NULLS LAST)
 +(count(*) OVER(PARTITION BY date,board,\"{feature}\")-1)/2.)
 /nullif(count(\"{feature}\") OVER(PARTITION BY date,board),0) END AS rank_value FROM f""")
 rebuilt=c.sql("""WITH tagged AS(SELECT *,CASE WHEN rank_value IS NULL THEN 'missing'
 WHEN rank_value<=.2 THEN 'low20' WHEN rank_value>=.8 THEN 'high20' ELSE 'middle60' END AS band
 FROM ranked WHERE necessary_tradeable),
 d AS(SELECT date,half,board,band,count(*) n,sum(known_label::INT) known,
 sum(winner_float) winners,sum(loss) losers,avg(winner_float) w,avg(loss) l,avg(next_gain) r
 FROM tagged GROUP BY ALL)
 SELECT half,board,band,sum(n) AS rows,sum(known) AS known,sum(winners) AS winners,sum(losers) AS losers,
 avg(w) winner_probability,avg(l) loser_probability,avg(r) reference_day_return,avg(w-l) tail_balance
 FROM d GROUP BY ALL ORDER BY half,board,band""").df()
 current=saved.loc[saved.feature.eq(feature)].sort_values(['half','board','band']).reset_index(drop=True)
 assert len(current)==len(rebuilt)
 for col in ['half','board','band']:assert np.array_equal(current[col],rebuilt[col]),(feature,col)
 for col in ['rows','known','winners','losers','winner_probability','loser_probability','reference_day_return','tail_balance']:
  assert np.allclose(current[col],rebuilt[col],rtol=0,atol=1e-11,equal_nan=True),(feature,col)
  errors=(current[col]-rebuilt[col]).abs().dropna()
  if len(errors):largest=max(largest,float(errors.max()))
 cells+=len(current)
 for scope,data in [('original',original),('range',ranged)]:
  extra=',floor(((high_1449-low_1449)/preclose)/.02) AS range_bin' if scope=='range' else ''
  rebuilt=c.sql(f"""WITH cells AS(SELECT date,board,
   floor(return_1449/.02) AS day_bin,floor(return20_prior_adjusted/.10) AS prior_bin,
   floor(log2(price_1449/5)) AS price_bin,floor(log2(amount_1449/3e7)) AS amount_bin {extra},
   count(*) FILTER(WHERE winner_float=1) AS w,count(*) FILTER(WHERE winner_float=0) AS n,
   avg(rank_value) FILTER(WHERE winner_float=1) AS mean_w,
   avg(rank_value) FILTER(WHERE winner_float=0) AS mean_n
   FROM ranked WHERE necessary_tradeable AND known_label AND return20_prior_adjusted IS NOT NULL
   AND rank_value IS NOT NULL GROUP BY ALL)
   SELECT date,board,sum(w) weight,sum((mean_w-mean_n)*w)/sum(w) rank_difference
   FROM cells WHERE w>0 AND n>0 GROUP BY ALL ORDER BY date,board""").df()
  target=data.loc[data.feature.eq(feature)].sort_values(['date','board']).reset_index(drop=True)
  assert len(target)==len(rebuilt),(feature,scope,len(target),len(rebuilt))
  for col in ['date','board']:assert np.array_equal(target[col],rebuilt[col]),(feature,scope,col)
  for col in ['weight','rank_difference']:
   assert np.allclose(target[col],rebuilt[col],rtol=0,atol=1e-11),(feature,scope,col)
  dailycells+=len(target)
 print(feature,flush=True)
result={'negative_labels':int(c.sql('select sum(loss) from f').fetchone()[0]),
 'both_tail_probability_cells':cells,'original_and_range_matched_day_cells':dailycells,
 'maximum_probability_or_mean_error':largest,'cohort_sha256':sha(ROOT/'cohort.parquet'),
 'new_2026_prices_read':False}
save_json(ROOT/'independent_balance_checks.json',result)
print(json.dumps(result,ensure_ascii=False,indent=2))
