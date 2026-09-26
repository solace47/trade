"""Freeze first-limit-up support candidates for a next-morning exit test."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .absolute_ridge import match_controls
from .corporate_cash import save_json, sha
from .hf_outcomes import _limit_price
from .limit_down_recovery_1449 import DELIST_NOTICES
from .quote_precision import quote_cents, fixed_quote_shares
from .reference_gain_eval import DAILY
from .risk_removal_1449 import decision_buyable
from .turnover_reference import CALENDAR

ROOT=Path('data/research/first_limitup_overnight')
RULE_COMMIT='b33259b'


def prior_pattern(close1: float, reference1: float, close2: float, reference2: float) -> tuple[bool,bool,bool]:
    try:
        c1,r1,c2,r2=(quote_cents(x) for x in (close1,reference1,close2,reference2))
    except ValueError:
        return False,False,False
    first_limit=c1==quote_cents(_limit_price(r1/100,.1,True))
    earlier_limit=c2==quote_cents(_limit_price(r2/100,.1,True))
    strong_unsealed=(106*r1<=100*c1 and 1000*c1<=1095*r1 and not first_limit)
    return first_limit and not earlier_limit,strong_unsealed and not earlier_limit,True


def select(events: pd.DataFrame, calendar: list[str]) -> pd.DataFrame:
    positions={date:i for i,date in enumerate(calendar)};last={};rows=[]
    for date,group in events.groupby('date',sort=True):
        rank=0
        for row in group.sort_values(['amount_1449','code'],ascending=[False,True]).itertuples():
            if positions[date]-last.get(row.code,-1000)<=5:continue
            rank+=1;rows.append({'date':date,'code':row.code,'daily_rank':rank});last[row.code]=positions[date]
            if rank==5:break
    return pd.DataFrame(rows,columns=['date','code','daily_rank'])


def freeze(output: Path=ROOT) -> dict:
    if (output/'input_report.json').exists():raise ValueError('Cannot replace frozen overnight selections')
    output.mkdir(parents=True,exist_ok=True)
    calendar_table=pd.read_parquet(CALENDAR)
    all_days=sorted(calendar_table.loc[calendar_table.is_trading_day.eq('1') & calendar_table.calendar_date.between('2023-01-01','2025-12-31'),'calendar_date'])
    days=[d for d in all_days[:-10] if d>='2024-01-01'];idx={d:i for i,d in enumerate(all_days)}
    schedule=pd.DataFrame({'date':days,'expected_previous':[all_days[idx[d]-1] for d in days],
        'expected_second':[all_days[idx[d]-2] for d in days]})
    sources=[*sorted(Path('data/research/minute_prefix_1449').glob('202[45]/*.parquet')),
        *sorted(Path('data/research/market_snapshots_ci').glob('*.parquet')),DELIST_NOTICES,CALENDAR]
    daily_sources=sorted([*DAILY.glob('sh_60*.parquet'),*DAILY.glob('sz_00*.parquet')])
    save_json(output/'source_manifest.json',{'rule_commit':RULE_COMMIT,'first':days[0],'last':days[-1],
        'source_sha256':{str(p):sha(p) for p in [*sources,*daily_sources]},'new_holding_results_read':False,'new_2026_prices_read':False})
    c=duckdb.connect();c.execute('SET threads=4');c.register('dates',schedule)
    c.read_parquet('data/research/minute_prefix_1449/202[45]/*.parquet').create_view('prefix')
    c.read_parquet('data/research/market_snapshots_ci/*.parquet').create_view('snapshots')
    c.read_parquet(str(DELIST_NOTICES)).create_view('notices');c.read_parquet([str(p) for p in daily_sources]).create_view('daily')
    frame=c.sql('''WITH past AS(SELECT code,date,
      lag(date,1) OVER w AS previous_date,lag(date,2) OVER w AS second_date,
      lag(close,1) OVER w AS close1,lag(close,2) OVER w AS close2,
      lag(preclose,1) OVER w AS reference1,lag(preclose,2) OVER w AS reference2,
      lag(isST,1) OVER w AS st1,lag(isST,2) OVER w AS st2
      FROM daily WHERE tradestatus=1 AND date BETWEEN '2023-01-01' AND '2025-12-31'
      WINDOW w AS(PARTITION BY code ORDER BY date))
      SELECT p.date,p.code,substr(p.code,1,2) AS exchange,p.price_1449,p.high_1449,p.low_1449,
      p.amount_1449,p.volume_1449,p.price_1420,p.return_last29,s.preclose,s.isST,s.reference_gap,
      s.listing_age_sessions,s.return20_prior_adjusted,x.previous_date,x.second_date,x.close1,x.close2,x.reference1,x.reference2
      FROM prefix p JOIN snapshots s USING(date,code) JOIN dates d USING(date) JOIN past x USING(date,code)
      WHERE (p.code LIKE 'sh.60%' OR p.code LIKE 'sz.00%') AND s.isST=0 AND s.tradestatus=1
      AND s.listing_age_sessions>=20 AND NOT s.reference_gap AND NOT p.quote_outside_traded_range
      AND p.price_1449>=5 AND p.amount_1449>=30000000 AND p.volume_1449>0 AND s.preclose>0
      AND p.return_last29>=0 AND p.price_1449>=p.amount_1449/p.volume_1449
      AND x.previous_date=d.expected_previous AND x.second_date=d.expected_second
      AND x.st1=0 AND x.st2=0 AND abs(x.reference1-x.close2)<=.005
      AND NOT EXISTS(SELECT 1 FROM notices n WHERE n.code=p.code AND n.notice_date<p.date
          AND regexp_matches(n.title,'进入退市整理|退市整理期交易'))
      ORDER BY p.date,p.code''').df();c.close()
    initial=len(frame)
    frame=frame.loc[[decision_buyable(r.code,r.price_1449,r.preclose) for r in frame.itertuples()]].copy()
    frame['price_1449']=frame.price_1449.map(lambda x:quote_cents(x)/100)
    states=[prior_pattern(r.close1,r.reference1,r.close2,r.reference2) for r in frame.itertuples()]
    frame[['first_board','strong_unsealed','prior_quotes_valid']]=pd.DataFrame(states,index=frame.index)
    frame['prior_day_return']=frame.close1/frame.reference1-1
    frame['return_1450']=frame.price_1449/frame.preclose-1
    frame['amount_signal'],frame['price_signal'],frame['board']=frame.amount_1449,frame.price_1449,frame.exchange
    frame['half']=frame.date.str[:4]+np.where(frame.date.str[5:7].le('06'),'H1','H2')
    if frame.duplicated(['date','code']).any():raise ValueError('A visible stock-day was duplicated')
    frame.to_parquet(output/'visible_pool.parquet',index=False,compression='zstd')
    high=select(frame.loc[frame.first_board],all_days)
    high_keys=pd.MultiIndex.from_frame(high[['date','code']])
    matching=frame.loc[frame.strong_unsealed | frame.set_index(['date','code']).index.isin(high_keys)]
    low=match_controls(high,matching)
    high['arm'],high['pair_id']='high',high.code;low['arm']='low'
    members=pd.concat([high,low],ignore_index=True);members['pair_id']=members.date+':'+members.pair_id
    signals=members.merge(frame,on=['date','code'],validate='one_to_one')
    signals['decision_shares']=[fixed_quote_shares(r.code,r.price_1449,20000) for r in signals.itertuples()]
    signals=signals.sort_values(['date','arm','daily_rank','code']).reset_index(drop=True)
    if len(signals)!=len(members):raise ValueError('Selected identities changed')
    signals.to_parquet(output/'signals.parquet',index=False,compression='zstd')
    result={'rule_commit':RULE_COMMIT,'before_buyable_rows':initial,'visible_pool_rows':len(frame),
        'invalid_prior_quotes':int((~frame.prior_quotes_valid).sum()),'first_board_rows':int(frame.first_board.sum()),
        'unsealed_control_pool':int(frame.strong_unsealed.sum()),'candidates':len(high),'controls':len(low),
        'by_half':signals.groupby(['half','arm']).agg(rows=('code','size'),days=('date','nunique')).reset_index().to_dict('records'),
        'signals_sha256':sha(output/'signals.parquet'),'pool_sha256':sha(output/'visible_pool.parquet'),
        'source_manifest_sha256':sha(output/'source_manifest.json'),'new_holding_results_read':False,'new_2026_prices_read':False}
    save_json(output/'input_report.json',result)
    return result


if __name__=='__main__':
    print(json.dumps(freeze(),ensure_ascii=False,indent=2))
