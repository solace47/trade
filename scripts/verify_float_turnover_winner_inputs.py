"""Rebuild lagged denominators, precise quintiles and fixed raw-minute samples."""
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from trade_research.corporate_cash import save_json,sha

root=Path('data/research/float_turnover_winner')
r=json.loads((root/'input_report.json').read_text())
assert r['protocol_sha256']==sha(Path('config/float_turnover_winner_protocol.json'))
assert r['cohort_sha256']==sha(Path('data/research/next_day_winner/cohort.parquet'))
assert r['calendar_sha256']==sha(Path('data/baostock/market_2020_2026/metadata/calendar.parquet'))
assert r['features_sha256']==sha(root/'features.parquet')
assert r['source_document_manifest_sha256']==sha(root/'source/manifest.json')
for name,digest in json.loads((root/'source/manifest.json').read_text())['files_sha256'].items():assert sha(root/'source'/name)==digest
assert '[指定交易日的成交量(股)/指定交易日的股票的流通股总股数(股)]' in (root/'source/stockKData.md').read_text()
for name,digest in r['daily_files_sha256'].items():assert sha(Path(name))==digest
c=duckdb.connect();c.execute('SET threads=4');c.execute("SET memory_limit='6GB'")
c.read_parquet('data/research/next_day_winner/cohort.parquet').create_view('cohort')
c.read_parquet('data/baostock/market_2020_2026/metadata/calendar.parquet').create_view('calendar')
c.read_parquet(list(r['daily_files_sha256'])).create_view('daily')
c.read_parquet(str(root/'features.parquet')).create_view('stored')
c.execute('''CREATE TABLE independent AS WITH days AS(
  SELECT calendar_date AS date,lag(calendar_date) OVER(ORDER BY calendar_date) AS prior_date
  FROM calendar WHERE is_trading_day='1' AND calendar_date BETWEEN '2023-12-29' AND '2025-12-31')
  SELECT b.date,b.code,b.half,b.board,b.necessary_tradeable,b.price_1449,b.volume_1449,b.volume_last29,
    s.prior_date,d.date AS source_date,CAST(d.volume AS DOUBLE) AS prior_volume,CAST(d.turn AS DOUBLE) AS prior_turn,
    CAST(d.tradestatus AS DOUBLE) AS prior_trade_status,CAST(d.adjustflag AS DOUBLE) AS prior_adjustflag,
    coalesce(d.tradestatus=1 AND d.adjustflag=3 AND isfinite(d.volume) AND d.volume>0 AND isfinite(d.turn) AND d.turn>0,FALSE) AS source_valid
  FROM cohort b JOIN days s USING(date) LEFT JOIN daily d ON d.code=b.code AND d.date=s.prior_date
    AND d.date BETWEEN '2023-12-29' AND '2025-12-29'
''')
original=c.sql('SELECT * FROM independent ORDER BY date,code').df()
stored=pd.read_parquet(root/'features.parquet').sort_values(['date','code']).reset_index(drop=True)
pd.testing.assert_frame_equal(original,stored[original.columns],check_dtype=False)
assert len(stored)==r['stock_days']==2404280 and stored.necessary_tradeable.sum()==r['necessary_tradeable']
status=np.full(len(original),'valid_lagged_proxy',dtype=object)
status[~original.source_valid]='invalid_prior_denominator'
status[~original.prior_trade_status.eq(1)]='prior_not_trading';status[original.source_date.isna()]='missing_prior_day'
np.testing.assert_array_equal(status,stored.source_status)
assert r['source_status_counts']==stored.source_status.value_counts().to_dict()
assert original.prior_date.min()==r['first_prior_date'] and original.prior_date.max()==r['last_prior_date']
assert original.prior_date.lt(original.date).all()
assert (original.prior_turn.dropna()*10000-(original.prior_turn.dropna()*10000).round()).abs().max()<2e-10
features=['float_cap_proxy','turnover_1449_proxy','turnover_tail29_proxy']
rank_checks=0;max_rank_error=0.
for suffix,shift in [('',0.),('_turn_lower',-.0001),('_turn_upper',.0001)]:
    c.execute(f'''CREATE OR REPLACE TEMP TABLE proxy AS WITH a AS(SELECT *,
       CASE WHEN source_valid AND prior_turn+({shift})>0 THEN prior_turn+({shift}) END AS t FROM independent)
      SELECT date,code,board,prior_volume*100./t AS float_shares_proxy{suffix},
        prior_volume*100./t*price_1449 AS float_cap_proxy{suffix},
        volume_1449*t/prior_volume AS turnover_1449_proxy{suffix},
        volume_last29*t/prior_volume AS turnover_tail29_proxy{suffix} FROM a''')
    p=c.sql('SELECT * FROM proxy ORDER BY date,code').df()
    for name in ['float_shares_proxy',*features]:
        np.testing.assert_allclose(p[name+suffix],stored[name+suffix],atol=1e-10,rtol=2e-14,equal_nan=True)
    for name in features:
        col=name+suffix
        row=c.sql(f'''WITH a AS(SELECT date,code,{col},
          2*rank() OVER(PARTITION BY date,board ORDER BY {col} NULLS LAST)
            +count(*) OVER(PARTITION BY date,board,{col})-1 AS twice_rank,
          2*count({col}) OVER(PARTITION BY date,board) AS denominator FROM proxy),
          expected AS(SELECT date,code,CASE WHEN {col} IS NOT NULL THEN twice_rank*1./denominator END AS rank,
            CASE WHEN {col} IS NOT NULL THEN ceil(5.*twice_rank/denominator) ELSE 0 END AS quintile FROM a)
          SELECT count(*) AS n,max(abs(e.rank-s.{col}_rank)) AS rank_error,
            count(*) FILTER(WHERE(e.rank IS NULL)<>(s.{col}_rank IS NULL)) AS missing_error,
            count(*) FILTER(WHERE e.quintile<>s.{col}_quintile) AS group_errors
          FROM expected e JOIN stored s USING(date,code)''').fetchone()
        assert row[0]==2404280 and row[2]==row[3]==0 and row[1]<2e-14,(col,row)
        rank_checks+=row[0];max_rank_error=max(max_rank_error,row[1])
for item in r['precision_group_changes']:
    p=stored.loc[stored.board.eq(item['board'])&stored.half.eq(item['half'])]
    a=item['feature']+'_quintile';b=item['feature']+item['scenario']+'_quintile'
    assert len(p)==item['rows'] and p[a].ne(p[b]).sum()==item['group_changes']
stored['sample_hash']=[hashlib.sha256((date+':'+code).encode()).hexdigest() for date,code in zip(stored.date,stored.code)]
samples=stored.sort_values('sample_hash').groupby(['board','half','source_status']).head(4)
raw_bars=0
for row in samples.itertuples():
    exchange,code=row.code.split('.')
    p=c.execute('''SELECT timestamp,CAST(close AS DOUBLE) AS close,volume FROM read_parquet(?)
       WHERE timestamp>=?::TIMESTAMP AND timestamp<=?::TIMESTAMP ORDER BY timestamp''',
       [f'data/hf/pilot/data/stock_1m/{exchange.upper()}/{code}.parquet',row.date+' 09:30:00',row.date+' 14:49:00']).df()
    assert len(p)==230 and p.timestamp.nunique()==230
    assert p.volume.sum()==row.volume_1449
    assert p.loc[p.timestamp.dt.strftime('%H%M').ge('1421'),'volume'].sum()==row.volume_last29
    assert round(p.close.iloc[-1],2)==row.price_1449
    d=pd.read_parquet(f'data/baostock/market_2020_2026/daily/{exchange}_{code}.parquet',filters=[('date','==',row.prior_date)])
    assert len(d)==1
    for field,expected in [('volume',row.prior_volume),('turn',row.prior_turn)]:
        value=d[field].iloc[0]
        np.testing.assert_allclose(float(value) if pd.notna(value) else np.nan,expected,equal_nan=True)
    raw_bars+=len(p)
samples[['date','code','prior_date','board','half','source_status','sample_hash']].to_parquet(root/'input_samples.parquet',index=False)
result={'passed':True,'input_report_sha256':sha(root/'input_report.json'),'rows':len(stored),
    'rank_and_group_values':rank_checks,'maximum_rank_error':max_rank_error,'raw_samples':len(samples),'raw_prefix_bars':raw_bars,
    'sample_keys_sha256':sha(root/'input_samples.parquet'),'outcomes_read':False,'new_2026_prices_read':False}
save_json(root/'input_verification.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
