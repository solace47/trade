"""Independent first-trigger, chronological execution and fee reconstruction."""
from decimal import Decimal, ROUND_HALF_UP
import json
from math import ceil
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from trade_research.corporate_cash import save_json, sha

root=Path('data/research/winner_take_profit');source=Path('data/research/winner_exit_opportunity')
manifest=json.loads((root/'input_report.json').read_text());report=json.loads((root/'analysis_report.json').read_text())
for path,digest in manifest['sha256'].items(): assert sha(Path(path))==digest
assert report['input_report_sha256']==sha(root/'input_report.json')
for name,digest in report['outputs_sha256'].items():assert sha(root/name)==digest
prior_check=json.loads((source/'verification_report.json').read_text())
assert prior_check['passed'] and prior_check['analysis_report_sha256']==sha(source/'analysis_report.json')
raw=json.loads((source/'raw_report.json').read_text())
paths,daily_paths=[],[]
for item in raw['sources']:
    p=source/'raw_parts'/(item['code']+'.parquet');d=p.with_name(item['code']+'.daily.parquet')
    assert sha(p)==item['minute_output_sha256'] and sha(d)==item['daily_output_sha256']
    paths.append(str(p));daily_paths.append(str(d))
c=duckdb.connect();c.execute('SET threads=4')
c.read_parquet(paths).create_view('raw')
c.read_parquet(daily_paths).create_view('daily')
c.read_parquet(str(source/'positions.parquet')).create_view('p')
c.execute('''CREATE TABLE bars AS SELECT timestamp,date,code,open::DOUBLE AS open,high::DOUBLE AS high,
    low::DOUBLE AS low,close::DOUBLE AS close,volume::DOUBLE AS volume,turnover::DOUBLE AS turnover,
    hour(timestamp)*60+minute(timestamp) AS minute FROM raw''')
c.execute('''CREATE TABLE trigger_values AS WITH inputs AS (
    SELECT p.model,p.date,p.code,b.minute,p.shares,p.entry_price/1.0005 raw_buy,b.close,
      p.shares*(b.close-greatest(b.close*.0015,.005)) AS sell_value,
      p.shares*(p.entry_price/1.0005+greatest(p.entry_price/1.0005*.0015,.005)) AS buy_value
    FROM p JOIN bars b ON b.code=p.code AND b.date=p.target_exit_date
      JOIN daily d ON d.code=p.code AND d.date=p.target_exit_date
      JOIN daily previous ON previous.code=p.code AND previous.date=p.date
    WHERE p.entry_status='filled' AND d.tradestatus=1 AND abs(d.preclose-previous.close)<=.005
      AND (b.minute BETWEEN 575 AND 685 OR b.minute BETWEEN 781 AND 889)
      AND b.timestamp=date_trunc('minute',b.timestamp)
      AND isfinite(b.open) AND isfinite(b.high) AND isfinite(b.low) AND isfinite(b.close)
      AND isfinite(b.volume) AND isfinite(b.turnover) AND least(b.open,b.high,b.low,b.close)>0
      AND b.high+.0001>=greatest(b.open,b.close,b.low) AND b.low-.0001<=least(b.open,b.close)
      AND b.volume>=0 AND b.turnover>=0 AND (b.volume=0)=(b.turnover=0)
      AND (b.volume=0 OR b.turnover/b.volume BETWEEN b.low-.0101 AND b.high+.0101))
    SELECT *, (sell_value-greatest(5.,sell_value*.0003)-sell_value*.00051)/
        (buy_value+greatest(5.,buy_value*.0003)+buy_value*.00001)-1 AS estimated_return15 FROM inputs''')
first=c.sql('''SELECT model,date,code,minute AS trigger_minute,estimated_return15 AS trigger_estimated_return15
    FROM trigger_values WHERE estimated_return15>=.01
    QUALIFY row_number() OVER(PARTITION BY model,date,code ORDER BY minute)=1''').df()
keys=['model','date','code']
old=pd.read_parquet(source/'positions.parquet').set_index(keys,drop=False).sort_index()
out=pd.read_parquet(root/'outcomes.parquet').set_index(keys,drop=False).sort_index()
assert old.index.equals(out.index)
check=old[keys].reset_index(drop=True).merge(first,on=keys,how='left',validate='one_to_one').set_index(keys).sort_index()
for name in ['trigger_minute','trigger_estimated_return15']:
    np.testing.assert_allclose(check[name].to_numpy(dtype=float),out[name].to_numpy(dtype=float),atol=2e-12,rtol=0,equal_nan=True)
for name in ['entry_status','entry_price','shares','entry_window_status','entry_window_low','entry_window_high']:
    pd.testing.assert_series_equal(old[name].convert_dtypes(),out[name].convert_dtypes(),check_dtype=False,check_names=False)
assert c.sql('SELECT count(*)-count(DISTINCT (code,timestamp)) FROM bars').fetchone()[0]==0
quotes=pd.read_parquet(source/'window_scenarios.parquet').set_index(keys+['start_minute'])
daily=c.sql('SELECT * FROM daily').df().set_index(['code','date'])
raw_index=c.sql('SELECT * FROM bars').df().set_index(['code','date'])
records=[];selected={}
for key,trigger in first.set_index(keys).iterrows():
    row=old.loc[key];day=daily.loc[(row.code,row.target_exit_date)]
    multiplier=Decimal('.8') if row.code.startswith(('sh.68','sz.30')) else (Decimal('.95') if day.isST else Decimal('.9'))
    lower=float((Decimal(str(day.preclose))*multiplier).quantize(Decimal('.01'),rounding=ROUND_HALF_UP))
    start=int(trigger.trigger_minute)+3
    while True:
        if 688<=start<781:start=781
        if start>888:start=892
        q=quotes.loc[(*key,start)]
        if not np.isfinite(q.volume) or not np.isfinite(q.vwap):status='missing_or_nonfinite_window'
        elif q.vwap<=0 or q.volume<=0 or day.preclose<=0:status='no_liquidity'
        elif row.shares>q.volume*.1:status='volume_cap'
        elif q.vwap*.9995<=lower+.005:status='estimated_lower_limit'
        else:status='filled'
        records.append(dict(zip(keys,key),target_date=row.target_exit_date,trigger_minute=int(trigger.trigger_minute),
            start_minute=start,fill_status=status,source_valid=bool(q.source_valid),queue_unknown=bool(q.queue_unknown)))
        if status=='filled':
            selected[key]=start
            actual=out.loc[key]
            assert actual.exit_date==row.target_exit_date and actual.exit_start==f'{start//60:02d}{start%60:02d}'
            assert actual.early_sale==(start<892) and actual.exit_delay_sessions==0
            assert abs(actual.exit_price-q.vwap*.9995)<1e-12
            bars=raw_index.loc[(row.code,row.target_exit_date)]
            selected_bars=bars.loc[bars.minute.between(start,start+3)&bars.volume.gt(0)]
            assert actual.exit_window_low==selected_bars.low.min() and actual.exit_window_high==selected_bars.high.max()
            assert (actual.exit_window_status=='valid')==q.source_valid
            assert actual.exit_queue_unknown==q.queue_unknown
            assert actual.catalog_sold_shares==row.shares and actual.catalog_dividend_gross==actual.catalog_dividend_tax==0
            break
        if start==892:break
        start+=4
expected_attempts=pd.DataFrame(records).sort_values(keys+['start_minute']).reset_index(drop=True)
actual_attempts=pd.read_parquet(root/'attempts.parquet').sort_values(keys+['start_minute']).reset_index(drop=True)
pd.testing.assert_frame_equal(expected_attempts,actual_attempts,check_dtype=False)
assert (actual_attempts.start_minute>=actual_attempts.trigger_minute+3).all()
for key,g in actual_attempts.groupby(keys):
    starts=g.start_minute.to_numpy();assert (np.diff(starts)>=4).all()
    assert ((starts<=687)|((starts>=781)&(starts<=888))|(starts==892)).all()
    assert len(g)==out.loc[key,'attempts']
unchanged=[key for key in old.index if key not in selected]
for name in ['exit_date','exit_price','exit_window_status','exit_window_low','exit_window_high',
             'catalog_sold_shares','catalog_dividend_gross','catalog_dividend_tax','exit_delay_sessions']:
    pd.testing.assert_series_equal(old.loc[unchanged,name].convert_dtypes(),out.loc[unchanged,name].convert_dtypes(),check_dtype=False,check_names=False)
# Rebuild every original/new point and each four-corner price sensitivity.
def impact(price,bps,side):
    price=np.asarray(price,dtype=float)
    return price+side*np.maximum(.005,price*bps/10000)

def profit(buy_qty,sell_qty,buy,sell,cash):
    b=buy_qty*buy;s=sell_qty*sell
    return (s-np.maximum(5.,s*.0003)-s*.00051+cash)/(b+np.maximum(5.,b*.0003)+b*.00001)-1

bought=out.entry_status.eq('filled');known=bought&out.exit_price.notna()
p=out.loc[known];cash=p.catalog_dividend_gross.to_numpy()-p.catalog_dividend_tax.to_numpy()
cent=lambda values:np.array([float(Decimal(str(x)).quantize(Decimal('.01'),rounding=ROUND_HALF_UP)) for x in values])
fee_checks=0
for bps in (5,15):
    buy=impact(p.entry_price.to_numpy()/1.0005,bps,1);sell=impact(p.exit_price.to_numpy()/.9995,bps,-1)
    point=profit(p.shares.to_numpy(),p.catalog_sold_shares.to_numpy(),buy,sell,cash)
    np.testing.assert_allclose(point,p[f'tick_return{bps}'],atol=2e-12,rtol=0)
    buy_bounds=[np.where(p.entry_window_status.eq('valid'),buy,impact(cent(p[f'entry_window_{edge}']),bps,1)) for edge in ['low','high']]
    sell_bounds=[np.where(p.exit_window_status.eq('valid'),sell,impact(cent(p[f'exit_window_{edge}']),bps,-1)) for edge in ['low','high']]
    corners=np.stack([profit(p.shares.to_numpy(),p.catalog_sold_shares.to_numpy(),bp,sp,cash) for bp in buy_bounds for sp in sell_bounds])
    np.testing.assert_allclose(corners.min(axis=0),p[f'tick_lower{bps}'],atol=2e-12,rtol=0)
    np.testing.assert_allclose(corners.max(axis=0),p[f'tick_upper{bps}'],atol=2e-12,rtol=0)
    assert out.loc[~bought,f'tick_return{bps}'].eq(0).all()
    assert out.loc[bought&~known,f'tick_return{bps}'].isna().all()
    fee_checks+=len(p)*3+int((~bought).sum())
    np.testing.assert_allclose(out[f'original_return{bps}'].to_numpy(dtype=float),old[f'tick_return{bps}'].to_numpy(dtype=float),equal_nan=True,atol=0,rtol=0)
# Original unknowns cannot disappear on a new sale or on a cashless row.
source_ok=out.entry_window_status.eq('valid')&(out.exit_window_status.eq('valid')|~bought)
queue_unknown=bought&(out.entry_queue_unknown|out.exit_queue_unknown)
assert source_ok.eq(out.source_valid).all() and queue_unknown.eq(out.queue_unknown).all()
expected_unknown=bought&(old.unknown_after_buy|~source_ok|queue_unknown|out.action_applied|out.exit_price.isna())
assert expected_unknown.eq(out.unverified_preserving_original).all()
merger=json.loads(Path('data/research/winner_direction/balanced/merger_followup/report.json').read_text())

def period_part(model,period):
    lo,hi=('2025-01-01','2025-12-31') if period=='2025' else (('2025-01-01','2025-06-30') if period=='2025H1' else ('2025-07-01','2025-12-31'))
    return out.loc[out.model.eq(model)&out.date.between(lo,hi)].copy()

def scenario(p,bps,quantity):
    a,b=p[f'tick_return{bps}'].copy(),p[f'original_return{bps}'].copy()
    missing=p.entry_status.eq('filled')&a.isna()
    if missing.any():
        assert missing.sum()==1 and p.loc[missing,'code'].iloc[0]=='sh.601989'
        value=next(v['net_return'] for v in merger['scenarios'] if v['bps']==bps and v['sold_shares']==quantity)
        a.loc[missing]=value;b.loc[missing]=value
    assert a.notna().all() and b.notna().all()
    return a,b

def interval(values):
    blocks={}
    for date,value in values.items():
        week=str(pd.Timestamp(date).to_period('W-SUN'));total,count=blocks.get(week,(0.,0));blocks[week]=(total+value,count+1)
    v=np.array([blocks[k] for k in sorted(blocks)])
    sample=np.random.default_rng(20260926).integers(0,len(v),(10000,len(v)))
    return np.quantile(v[sample,0].sum(axis=1)/v[sample,1].sum(axis=1),[.025,.975])

for r in report['metrics']:
    p=period_part(r['model'],r['period']);new,old_values=scenario(p,r['bps'],r['merger_shares'])
    values={'own':new,'original':old_values,'difference':new-old_values}[r['metric']].groupby(p.date).mean()
    assert len(values)==r['dates'] and abs(values.mean()-r['mean'])<2e-12
    np.testing.assert_allclose(interval(values),r['weekly_interval'],rtol=0,atol=2e-12)
for r in report['distributions']:
    p=period_part(r['model'],r['period']);new,_=scenario(p,r['bps'],r['merger_shares']);values=new[p.entry_status.eq('filled')]
    win,lose=values[values>0],values[values<0]
    expected={'bought':len(values),'win_rate':(values>0).mean(),'mean_win':win.mean(),'mean_loss':lose.mean(),
        'payoff_ratio':win.mean()/-lose.mean(),'worst_five_percent_mean':values.sort_values().iloc[:ceil(.05*len(values))].mean(),
        'worst_trade_return':values.min()}
    for name,value in expected.items():assert abs(value-r[name])<2e-12
for r in report['counts']:
    p=period_part(r['model'],r['period']);b=p.entry_status.eq('filled');t=p.trigger_minute.notna();e=p.early_sale
    expected={'orders':len(p),'bought':b.sum(),'triggered':t.sum(),'early_sales':e.sum(),
        'triggered_unfilled':(t&p.status.str.contains('unfilled')).sum(),'source_invalid_bought':(b&~p.source_valid).sum(),
        'queue_unknown':p.queue_unknown.sum(),'preserved_actual_unknown':p.unverified_preserving_original.sum(),
        'delayed_original_or_new':p.exit_delay_sessions.gt(0).sum(),'unresolved_original_merger':(b&p.exit_price.isna()).sum(),
        'early_sale_return_below_one_percent':(e&p.tick_return15.lt(.01)).sum(),'early_sale_loss':(e&p.tick_return15.lt(0)).sum(),
        'mean_execution_minus_signal_return15':(p.loc[e,'tick_return15']-p.loc[e,'trigger_estimated_return15']).mean()}
    for name,value in expected.items():assert abs(value-r[name])<2e-12
result={'passed':True,'unchanged_orders':len(out),'independent_signal_minutes':c.sql('SELECT count(*) FROM trigger_values').fetchone()[0],
    'first_triggers':len(first),'chronological_attempts':len(actual_attempts),'fee_and_bound_values':fee_checks,
    'daily_means_and_weekly_intervals':len(report['metrics']),'distributions':len(report['distributions']),
    'count_groups':len(report['counts']),'analysis_report_sha256':sha(root/'analysis_report.json'),'new_2026_prices_read':False}
save_json(root/'verification_report.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
