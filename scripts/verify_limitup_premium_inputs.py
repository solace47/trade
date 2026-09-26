"""Independent historical membership and complete 14:49 environment reconstruction."""
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from trade_research.corporate_cash import save_json, sha

root=Path('data/research/limitup_premium_environment')
r=json.loads((root/'input_report.json').read_text())
for path,digest in r['sha256'].items():assert sha(Path(path))==digest
for name,digest in r['outputs_sha256'].items():assert sha(root/(name+'.parquet'))==digest
cal=pd.read_parquet('data/baostock/market_2020_2026/metadata/calendar.parquet')
days=sorted(cal.loc[cal.is_trading_day.eq('1'),'calendar_date']);previous=dict(zip(days[1:],days[:-1]))
base=pd.read_parquet('data/research/next_day_winner/visible_base.parquet')
history=pd.read_parquet(root/'history.parquet').sort_values(['code','date']).reset_index(drop=True)
c=duckdb.connect();c.execute('SET threads=4')
files=[p for p in r['sha256'] if '/daily/' in p]
c.read_parquet(files).create_view('original')
cols=[col for col in history.columns if col!='age']
original=c.execute("SELECT "+','.join('"'+col+'"' for col in cols)+" FROM original WHERE date BETWEEN '2023-01-01' AND ? ORDER BY code,date",[max(previous[d] for d in base.date.unique())]).df()
pd.testing.assert_frame_equal(original.convert_dtypes(),history[cols].convert_dtypes(),check_dtype=False,atol=0,rtol=0)
ages=history.tradestatus.eq(1).groupby(history.code).cumsum();np.testing.assert_array_equal(ages,history.age)
lookup={previous[d]:d for d in base.date.unique()}
p=history.loc[history.date.isin(lookup)].copy();p.rename(columns={'date':'prior_date'},inplace=True);p['date']=p.prior_date.map(lookup)
fields=p[['open','high','low','close','preclose']].to_numpy(dtype=float)
valid=(np.isfinite(fields).all(axis=1)&(fields.min(axis=1)>0)&p.tradestatus.eq(1)&p.isST.eq(0)
    &p.age.ge(20)&p.adjustflag.eq(3)&p.volume.gt(0)
    &(p.high+.0001>=p[['open','close','low']].max(axis=1))&(p.low-.0001<=p[['open','close']].min(axis=1))
    &(p.close-p.close.round(2)).abs().le(.0001)&(p.preclose-p.preclose.round(2)).abs().le(.0001))
p['prior_eligible']=valid
p['closed_limit']=False
p.loc[valid,'closed_limit']=[Decimal(str(close)).quantize(Decimal('.01'))==
    (Decimal(str(pre))*Decimal('1.1')).quantize(Decimal('.01'),rounding=ROUND_HALF_UP)
    for close,pre in zip(p.loc[valid,'close'],p.loc[valid,'preclose'])]
state=pd.read_parquet(root/'prior_state.parquet').set_index(['date','code']).sort_index()
expected_state=p.set_index(['date','code']).sort_index()
assert expected_state.index.equals(state.index)
np.testing.assert_array_equal(expected_state.prior_eligible,state.prior_eligible)
np.testing.assert_array_equal(expected_state.loc[expected_state.prior_eligible,'closed_limit'],state.loc[expected_state.prior_eligible,'closed_limit'])
basket=p.loc[valid&p.closed_limit,['date','prior_date','code']].merge(base[['date','code','return_1449','sealed_quote_1449']],on=['date','code'],how='left',validate='one_to_one')
basket['visible']=basket.return_1449.notna()
stored_members=pd.read_parquet(root/'prior_members.parquet').sort_values(['date','code']).reset_index(drop=True)
pd.testing.assert_frame_equal(basket.sort_values(['date','code']).reset_index(drop=True)[stored_members.columns.intersection(basket.columns)].convert_dtypes(),
    stored_members[stored_members.columns.intersection(basket.columns)].convert_dtypes(),check_dtype=False,atol=1e-14,rtol=0)
daily=[]
for date,main in base.loc[base.board.eq('main')].groupby('date'):
    b=basket.loc[basket.date.eq(date)];v=b.loc[b.visible]
    count=len(b);n=len(v);ok=count>=5 and n/count>=.9
    touched=main.high_1449.ge(main.upper_limit-.005)
    ret=v.return_1449.mean() if ok else np.nan;market=main.return_1449.mean()
    excess=ret-market
    daily.append({'date':date,'prior_date':previous[date],'market_count':len(main),'market_return':market,
      'market_rising':main.return_1449.gt(0).mean(),'touched_limit_count':int(touched.sum()),
      'touched_open_count':int((touched&~main.sealed_quote_1449).sum()),'prior_limit_count':count,
      'visible_members':n,'member_return_sum':v.return_1449.sum(),'rising_members':v.return_1449.gt(0).sum(),
      'resealed_members':v.sealed_quote_1449.eq(True).sum(),'member_coverage':n/count if count else np.nan,
      'environment_valid':ok,'raw_premium':ret,'rising_fraction':v.return_1449.gt(0).mean() if ok else np.nan,
      'resealed_fraction':v.sealed_quote_1449.eq(True).mean() if ok else np.nan,'excess_premium':excess,
      'opened_after_touch_fraction':int((touched&~main.sealed_quote_1449).sum())/int(touched.sum()) if touched.any() else np.nan,
      'environment_bin':'unknown' if not ok else ('strong' if excess>=.01 else ('weak' if excess<=-.01 else 'neutral'))})
daily=pd.DataFrame(daily).set_index('date').sort_index();stored_daily=pd.read_parquet(root/'daily_features.parquet').set_index('date').sort_index()
pd.testing.assert_frame_equal(daily[stored_daily.columns].convert_dtypes(),stored_daily.convert_dtypes(),check_dtype=False,atol=2e-12,rtol=0)
# Rebuild all row-specific sums independently: members drop their own quotes.
f=base[['date','code','board','necessary_tradeable','return_1449','sealed_quote_1449']].merge(
    p[['date','code','prior_eligible','closed_limit']],on=['date','code'],how='left',validate='one_to_one')
f=f.merge(daily.reset_index(),on='date',validate='many_to_one')
member=f.prior_eligible.fillna(False)&f.closed_limit.fillna(False)
self_weight=member.astype(int)
count=f.prior_limit_count-self_weight;n=f.visible_members-self_weight
coverage=n/count.replace(0,np.nan);ok=count.ge(5)&coverage.ge(.9)
raw=(f.member_return_sum-self_weight*f.return_1449)/n.replace(0,np.nan)
benchmark=(f.market_return*f.market_count-self_weight*f.return_1449)/(f.market_count-self_weight)
expected=pd.DataFrame({'date':f.date,'code':f.code,'member_self':member,'context_members':count,
    'context_visible':n,'context_coverage':coverage,'context_valid':ok,
    'raw_premium':raw.where(ok),'excess_premium':(raw-benchmark).where(ok),
    'rising_fraction':((f.rising_members-self_weight*f.return_1449.gt(0))/n.replace(0,np.nan)).where(ok),
    'resealed_fraction':((f.resealed_members-self_weight*f.sealed_quote_1449)/n.replace(0,np.nan)).where(ok)})
expected['environment_bin']=np.select([~ok,expected.excess_premium.ge(.01),expected.excess_premium.le(-.01)],['unknown','strong','weak'],default='neutral')
expected['primary_pool']=f.board.eq('main')&f.necessary_tradeable&f.prior_eligible.eq(True)&f.closed_limit.eq(False)
expected['primary_pool']=expected.primary_pool.astype('boolean')
expected.loc[f.board.eq('main')&f.necessary_tradeable&~f.prior_eligible.fillna(False),'primary_pool']=pd.NA
expected=expected.set_index(['date','code']).sort_index()
stored=pd.read_parquet(root/'features.parquet').set_index(['date','code']).sort_index()
pd.testing.assert_frame_equal(expected.convert_dtypes(),stored[expected.columns].convert_dtypes(),check_dtype=False,atol=2e-14,rtol=0)
assert len(stored)==len(base)==r['stock_days'] and stored.primary_pool.sum()==r['primary_pool_rows']
assert not stored.index.duplicated().any() and (stored.prior_date<stored.index.get_level_values('date')).all()
result={'passed':True,'history_rows':len(history),'prior_states':len(state),'fixed_prior_members':len(basket),
    'days':len(daily),'row_features':len(expected),'checked_feature_values':len(expected)*(len(expected.columns)),
    'max_basket_sum_error':float((daily.member_return_sum-stored_daily.member_return_sum).abs().max()),
    'input_report_sha256':sha(root/'input_report.json'),'new_outcome_labels_accessed':False,'new_2026_prices_read':False}
save_json(root/'input_verification.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
