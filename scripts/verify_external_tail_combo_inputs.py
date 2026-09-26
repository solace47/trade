"""Independently verify complete external-screen inputs before any outcome join."""
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from trade_research.corporate_cash import save_json,sha,MINUTES

root=Path('data/research/external_tail_combo');r=json.loads((root/'input_report.json').read_text())
for key,path in [('protocol_sha256','config/external_tail_combo_protocol.json'),('cohort_sha256','data/research/next_day_winner/cohort.parquet'),
    ('calendar_sha256','data/baostock/market_2020_2026/metadata/calendar.parquet'),('float_input_report_sha256','data/research/float_turnover_winner/input_report.json'),
    ('float_input_verification_sha256','data/research/float_turnover_winner/input_verification.json'),('external_manifest_sha256','data/research/external_tail_review/manifest.json'),
    ('source_raw_report_sha256','data/research/next_day_winner/raw_report.json')]:assert sha(Path(path))==r[key]
for field in ['daily_files_sha256','minute_files_sha256']:
    for name,digest in r[field].items():assert sha(Path(name))==digest
for name,digest in r['outputs_sha256'].items():assert sha(root/(name+'.parquet'))==digest
c=duckdb.connect();c.execute('SET threads=4');c.execute("SET memory_limit='6GB'")
c.read_parquet(list(r['daily_files_sha256'])).create_view('daily')
daily=c.sql("SELECT date,code,tradestatus,adjustflag,close,preclose,pctChg,volume FROM daily WHERE date BETWEEN '2023-11-01' AND '2025-12-29'").df()
cal=pd.read_parquet('data/baostock/market_2020_2026/metadata/calendar.parquet');days=sorted(cal.loc[cal.is_trading_day.eq('1'),'calendar_date'])
positions={day:i for i,day in enumerate(days)}
dense_days=[d for d in days if '2023-11-01'<=d<='2025-12-30']
pool=pd.read_parquet(root/'pool.parquet');expected=[]
for code,part in daily.groupby('code'):
    assert not part.date.duplicated().any()
    d=part.set_index('date').reindex(dense_days)
    present=d.code.notna()
    valid=(d.adjustflag.eq(3)&(d.tradestatus.eq(0)|(d.tradestatus.eq(1)&d.close.gt(0)&d.preclose.gt(0)&np.isfinite(d.pctChg)))).fillna(False)
    active=(d.tradestatus.eq(1)&d.pctChg.ge(9.9)).fillna(False)
    # Count observations whose values are valid; a missing volume is not valid.
    # The original volume itself stays missing and never becomes a zero trade.
    volume_valid=(d.adjustflag.eq(3)&d.volume.ge(0)&np.isfinite(d.volume)).fillna(False)
    h=pd.DataFrame({'history_rows':present.rolling(20).sum().shift(1),'history_dates':present.rolling(20).sum().shift(1),
        'valid_history_rows':valid.rolling(20).sum().shift(1),'big_up_days':active.rolling(20).sum().shift(1),
        'volume5_rows':volume_valid.rolling(5).sum().shift(1),'volume5_sum':d.volume.rolling(5,min_periods=1).sum().shift(1)})
    p=pool.loc[pool.code.eq(code),['date','code','volume_1449']].set_index('date')
    h=h.reindex(p.index);h['code']=code;h.index.name='date';h=h.reset_index()
    for name,k in [('prior20_start',20),('prior5_start',5),('prior_date',1)]:h[name]=[days[positions[date]-k] for date in h.date]
    h['history_valid']=h.history_rows.eq(20)&h.valid_history_rows.eq(20)
    h['recent_activity']=h.big_up_days.gt(0).astype('boolean').where(h.history_valid,pd.NA)
    h['simple_volume_ratio']=(p.volume_1449.to_numpy()*5/h.volume5_sum).where(h.volume5_rows.eq(5)&h.volume5_sum.gt(0))
    expected.append(h)
history=pd.concat(expected,ignore_index=True)


def compare(a,b,keys):
    a=a.set_index(keys).sort_index();b=b.set_index(keys).sort_index()
    assert set(a.columns)==set(b.columns)
    for name in b:
        if not a[name].isna().equals(b[name].isna()):
            mismatch=a[name].isna().ne(b[name].isna())
            raise AssertionError((name,int(mismatch.sum()),pd.concat([a.loc[mismatch,name].rename('rebuilt'),b.loc[mismatch,name].rename('stored')],axis=1).head(5).to_dict('index')))
        pd.testing.assert_series_equal(a[name].isna(),b[name].isna(),check_index_type=False)
        mask=b[name].notna()
        pd.testing.assert_series_equal(a.loc[mask,name],b.loc[mask,name],check_dtype=False,check_index_type=False,atol=1e-10,rtol=1e-12)


compare(history,pd.read_parquet(root/'history.parquet'),['date','code'])
columns=['date','code','half','board','necessary_tradeable','price_1449','preclose','volume_1449','amount_1449','return_1449',
         'market_return','decision_shares','return20_prior_adjusted']
original=pd.read_parquet('data/research/next_day_winner/cohort.parquet',columns=columns)
original=original.loc[original.code.str.match(r'^sz\.00[23]')]
floats=pd.read_parquet('data/research/float_turnover_winner/features.parquet',columns=['date','code','float_cap_proxy','turnover_1449_proxy'])
rebuilt=original.merge(floats,on=['date','code'],validate='one_to_one').merge(history,on=['date','code'],validate='one_to_one')
c.register('base',rebuilt)
rebuilt=c.sql('''WITH flags AS(SELECT *,
    CAST(round(price_1449*100) AS BIGINT)*100 BETWEEN CAST(round(preclose*100) AS BIGINT)*103 AND CAST(round(preclose*100) AS BIGINT)*105 AS gain_gate,
    CASE WHEN simple_volume_ratio IS NOT NULL THEN volume_1449*5>volume5_sum END AS volume_gate,
    turnover_1449_proxy BETWEEN 5 AND 10 AS turnover_gate,float_cap_proxy BETWEEN 5000000000 AND 20000000000 AS size_gate,
    return_1449>market_return AS market_gate FROM base)
    SELECT *,gain_gate AND volume_gate AND turnover_gate AND size_gate AND market_gate AS base_combo FROM flags''').df()
raw=pd.read_parquet(root/'raw_prefix.parquet');paths=pd.read_parquet(root/'paths.parquet');c.register('raw',raw)
assert raw.date.between('2024-01-01','2025-12-31').all() and raw.label.le('1449').all()
minute_labels=[f'{i//60:02d}{i%60:02d}' for i in list(range(570,691))+list(range(781,890))]
observe=[f'{i//60:02d}{i%60:02d}' for i in list(range(575,691,5))+list(range(785,886,5))]
assert len(minute_labels)==230 and len(observe)==45
c.register('valid_labels',pd.DataFrame({'label':minute_labels}))
c.register('observations',pd.DataFrame({'label':observe,'late':[False]*30+[True]*15}))
condition=' AND '.join(f'isfinite({col})' for col in ['open','high','low','close','volume','turnover'])
condition+=' AND open>0 AND high>0 AND low>0 AND close>0 AND volume>=0 AND turnover>=0'
condition+=' AND high>=greatest(open,low,close) AND low<=least(open,high,close) AND (volume=0)=(turnover=0)'
condition+=' AND '+' AND '.join(f'abs({col}-round({col},2))<=.0001' for col in ['open','high','low','close'])
c.execute(f'''CREATE TABLE ordered AS SELECT *,NOT coalesce({condition},false) AS bad_numeric,
    volume>0 AND(NOT coalesce(turnover/volume BETWEEN low-.0101 AND high+.0101,false) OR NOT isfinite(turnover/volume)) AS bad_amount,
    sum(volume) OVER w AS cumulative_volume,sum(turnover) OVER w AS cumulative_amount
    FROM raw WINDOW w AS(PARTITION BY date,code ORDER BY label ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)''')
rebuilt_paths=c.sql('''WITH a AS(SELECT r.date,r.code,count(*) AS bars,count(*)=230 AND count(DISTINCT r.label)=230 AND count(v.label)=230 AS complete_labels,
    count(*) FILTER(WHERE bad_numeric OR bad_amount) AS bad_bars,count(*) FILTER(WHERE bad_amount) AS amount_bad_bars,
    count(*) FILTER(WHERE o.label IS NOT NULL AND cumulative_volume>0) AS valid_samples,
    count(*) FILTER(WHERE o.label IS NOT NULL AND cumulative_volume>0 AND round(close,2)>=cumulative_amount/cumulative_volume) AS above_samples,
    count(*) FILTER(WHERE o.late AND cumulative_volume>0 AND round(close,2)>=cumulative_amount/cumulative_volume) AS late_above_samples,
    bool_and(cumulative_volume>0 AND round(close,2)>=cumulative_amount/cumulative_volume) FILTER(WHERE r.label='1449') AS current_above
    FROM ordered r LEFT JOIN valid_labels v USING(label) LEFT JOIN observations o USING(label) GROUP BY r.date,r.code),
    b AS(SELECT *,complete_labels AND bad_bars=0 AND valid_samples=45 AS path_source_valid FROM a)
    SELECT *,CASE WHEN path_source_valid THEN above_samples>=34 AND late_above_samples=15 AND current_above END AS vwap_support FROM b''').df()
compare(rebuilt_paths,paths,['date','code'])
raw_totals=raw.groupby(['date','code']).agg(volume=('volume','sum'),amount=('turnover','sum')).reset_index()
matched=raw_totals.merge(original[['date','code','volume_1449','amount_1449']],on=['date','code'],validate='one_to_one')
np.testing.assert_array_equal(matched.volume,matched.volume_1449)
np.testing.assert_allclose(matched.amount,matched.amount_1449,atol=.001,rtol=1e-12)
rebuilt=rebuilt.merge(rebuilt_paths,on=['date','code'],how='left',validate='one_to_one');c.register('combined',rebuilt)
rebuilt=c.sql('''WITH a AS(SELECT *,CASE WHEN base_combo IS NULL THEN 'base_unknown' WHEN NOT base_combo THEN 'outside_base'
    WHEN recent_activity IS NULL OR vwap_support IS NULL THEN 'detail_unknown'
    ELSE CAST(recent_activity::INT AS VARCHAR)||':'||CAST(vwap_support::INT AS VARCHAR) END AS "group" FROM combined)
    SELECT *,necessary_tradeable AND "group"='1:1' AS "primary" FROM a''').df()
compare(rebuilt,pool,['date','code'])
assert len(pool)==r['visible_rows'] and pool.necessary_tradeable.sum()==r['necessary_rows']
assert pool.base_combo.fillna(False).sum()==r['base_rows'] and (pool.base_combo.fillna(False)&pool.necessary_tradeable).sum()==r['necessary_base_rows']
assert pool.primary.sum()==r['primary_rows'] and len(raw)==r['raw_bars']
assert (~paths.path_source_valid).sum()==r['path_source_unknown']
assert pool.loc[pool.necessary_tradeable].groupby(['half','group']).size().rename('n').reset_index().to_dict('records')==r['groups']
sample=paths.merge(pool[['date','code','half']],on=['date','code'],validate='one_to_one')
sample['hash']=[hashlib.sha256((d+code).encode()).hexdigest() for d,code in sample[['date','code']].itertuples(index=False,name=None)]
sample=sample.sort_values('hash').groupby(['half','path_source_valid'],group_keys=False).head(8)
c.register('sample',sample[['date','code']])
original_files=[str(MINUTES/code[:2].upper()/(code[3:]+'.parquet')) for code in sorted(sample.code.unique())]
c.read_parquet(original_files).create_view('source')
sample_raw=c.sql('''SELECT s.date,s.code,strftime(timestamp,'%H%M') AS label,m.open,m.high,m.low,m.close,m.volume,m.turnover
    FROM source m JOIN sample s ON lower(exchange)||'.'||symbol=s.code AND strftime(timestamp,'%Y-%m-%d')=s.date
    WHERE timestamp>=TIMESTAMP '2024-01-01' AND timestamp<TIMESTAMP '2026-01-01'
    AND strftime(timestamp,'%H%M') IN(SELECT label FROM valid_labels)''').df()
expected_raw=raw.merge(sample[['date','code']],on=['date','code'],validate='many_to_one')
compare(sample_raw,expected_raw,['date','code','label'])
result={'passed':True,'input_report_sha256':sha(root/'input_report.json'),'all_history_and_base_rows':len(pool),'raw_prefix_rows':len(raw),
    'all_vwap_paths':len(paths),'independent_source_sample_days':len(sample),'independent_source_sample_minutes':len(sample_raw),
    'base_rows':r['base_rows'],'primary_rows':r['primary_rows'],'outcomes_read':False,'new_2026_prices_read':False}
save_json(root/'input_verification.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
