"""PIT context from yesterday's fixed limit-up members, with full missing coverage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .corporate_cash import DAILY, save_json, sha
from .turnover_reference import CALENDAR

ROOT=Path('data/research/limitup_premium_environment')
BASE=Path('data/research/next_day_winner/visible_base.parquet')
PROTOCOL=Path('config/limitup_premium_environment_protocol.json')


def inputs() -> dict:
    if (ROOT/'input_report.json').exists():
        raise ValueError('Do not replace frozen limit-up environment inputs')
    ROOT.mkdir(parents=True,exist_ok=True)
    original=json.loads(BASE.with_name('base_report.json').read_text())
    assert sha(BASE)==original['base_sha256']
    paths=sorted(DAILY.glob('sh_60*.parquet'))+sorted(DAILY.glob('sz_00*.parquet'))
    provenance={str(p):sha(p) for p in [PROTOCOL,BASE,CALENDAR,*paths]}
    c=duckdb.connect();c.execute('SET threads=4');c.execute("SET memory_limit='6GB'")
    c.read_parquet(str(BASE)).create_view('visible')
    c.read_parquet(str(CALENDAR)).create_view('calendar')
    c.read_parquet([str(p) for p in paths]).create_view('raw_daily')
    c.execute('''CREATE TABLE schedule AS WITH previous AS (
      SELECT calendar_date AS date,lag(calendar_date) OVER(ORDER BY calendar_date) AS prior_date
      FROM calendar WHERE is_trading_day='1')
      SELECT * FROM previous WHERE date IN(SELECT DISTINCT date FROM visible)''')
    c.execute('''CREATE TABLE history AS SELECT *,
      count(*) FILTER(WHERE tradestatus=1) OVER(PARTITION BY code ORDER BY date ROWS UNBOUNDED PRECEDING) AS age
      FROM raw_daily WHERE date>='2023-01-01' AND date<=(SELECT max(prior_date) FROM schedule)''')
    c.execute('''CREATE TABLE prior_state AS SELECT s.date,s.prior_date,h.code,h.close,h.preclose,h.age,
      h.tradestatus=1 AND h.isST=0 AND h.age>=20 AND h.adjustflag=3 AND h.volume>0
      AND isfinite(h.open) AND isfinite(h.high) AND isfinite(h.low) AND isfinite(h.close) AND isfinite(h.preclose)
      AND least(h.open,h.high,h.low,h.close,h.preclose)>0
      AND h.high+.0001>=greatest(h.open,h.close,h.low) AND h.low-.0001<=least(h.open,h.close)
      AND abs(h.close-round(h.close,2))<=.0001 AND abs(h.preclose-round(h.preclose,2))<=.0001 AS prior_eligible,
      round(h.close*100)::BIGINT=(round(h.preclose*100)::BIGINT*110+50)//100 AS closed_limit
      FROM schedule s JOIN history h ON h.date=s.prior_date''')
    c.execute('''CREATE TABLE prior_members AS SELECT p.*,v.code IS NOT NULL AS visible,
      v.return_1449,v.sealed_quote_1449 FROM prior_state p
      LEFT JOIN visible v ON p.code=v.code AND p.date=v.date
      WHERE p.prior_eligible AND p.closed_limit''')
    daily=c.sql('''WITH market AS (SELECT date,count(*) AS market_count,avg(return_1449) AS market_return,
        avg((return_1449>0)::INT) AS market_rising,
        count(*) FILTER(WHERE high_1449>=upper_limit-.005) AS touched_limit_count,
        count(*) FILTER(WHERE high_1449>=upper_limit-.005 AND NOT sealed_quote_1449) AS touched_open_count
      FROM visible WHERE board='main' GROUP BY date),
      basket AS (SELECT date,count(*) AS prior_limit_count,sum(visible::INT) AS visible_members,
        sum(return_1449) AS member_return_sum,
        sum((return_1449>0)::INT) FILTER(WHERE visible) AS rising_members,
        sum(sealed_quote_1449::INT) FILTER(WHERE visible) AS resealed_members
      FROM prior_members GROUP BY date)
      SELECT s.*,m.*,b.* EXCLUDE(date) FROM schedule s JOIN market m USING(date) LEFT JOIN basket b USING(date)
      ORDER BY s.date''').df()
    # DuckDB keeps both explicitly selected date columns; retain only the schedule key.
    daily=daily.loc[:,~daily.columns.str.match(r'date_\d+$')]
    for col in ['prior_limit_count','visible_members','member_return_sum','rising_members','resealed_members']:
        daily[col]=daily[col].fillna(0)
    daily['member_coverage']=daily.visible_members/daily.prior_limit_count.replace(0,np.nan)
    daily['environment_valid']=daily.prior_limit_count.ge(5)&daily.member_coverage.ge(.9)
    for name,col in [('raw_premium','member_return_sum'),('rising_fraction','rising_members'),('resealed_fraction','resealed_members')]:
        daily[name]=(daily[col]/daily.visible_members.replace(0,np.nan)).where(daily.environment_valid)
    daily['excess_premium']=(daily.raw_premium-daily.market_return).where(daily.environment_valid)
    daily['opened_after_touch_fraction']=daily.touched_open_count/daily.touched_limit_count.replace(0,np.nan)
    daily['environment_bin']=np.select([~daily.environment_valid,daily.excess_premium.ge(.01),daily.excess_premium.le(-.01)],['unknown','strong','weak'],default='neutral')
    c.register('context',daily)
    features=c.sql('''SELECT v.date,v.code,v.board,v.necessary_tradeable,v.half,
      p.prior_eligible,p.closed_limit,
      (p.prior_eligible AND p.closed_limit) IS TRUE AS member_self,
      CASE WHEN v.board<>'main' THEN false WHEN p.prior_eligible THEN p.closed_limit END AS prior_main_closed_limit,
      d.prior_date,d.prior_limit_count,d.visible_members,d.member_return_sum,d.rising_members,d.resealed_members,
      d.market_count,d.market_return AS main_market_return,d.market_rising AS main_market_rising,
      d.opened_after_touch_fraction,d.environment_bin AS daily_environment_bin,
      v.return_1449,v.sealed_quote_1449
      FROM visible v JOIN context d USING(date) LEFT JOIN prior_state p USING(date,code)
      ORDER BY date,code''').df()
    own=features.member_self.astype(int)
    features['context_members']=features.prior_limit_count-own
    features['context_visible']=features.visible_members-own
    coverage=features.context_visible/features.context_members.replace(0,np.nan)
    features['context_coverage']=coverage
    good=features.context_members.ge(5)&coverage.ge(.9)
    features['context_valid']=good
    denominator=features.context_visible.replace(0,np.nan)
    features['raw_premium']=((features.member_return_sum-own*features.return_1449)/denominator).where(good)
    benchmark=(features.main_market_return*features.market_count-own*features.return_1449)/(features.market_count-own)
    features['excess_premium']=(features.raw_premium-benchmark).where(good)
    features['rising_fraction']=((features.rising_members-own*features.return_1449.gt(0))/denominator).where(good)
    features['resealed_fraction']=((features.resealed_members-own*features.sealed_quote_1449)/denominator).where(good)
    features['environment_bin']=np.select([~good,features.excess_premium.ge(.01),features.excess_premium.le(-.01)],['unknown','strong','weak'],default='neutral')
    features['primary_pool']=features.board.eq('main')&features.necessary_tradeable&features.prior_main_closed_limit.eq(False)
    assert len(features)==2404280 and not features.duplicated(['date','code']).any()
    assert features.prior_date.lt(features.date).all()
    for name in ['history','prior_state','prior_members']:
        c.sql('SELECT * FROM '+name).df().to_parquet(ROOT/(name+'.parquet'),index=False,compression='zstd')
    daily.to_parquet(ROOT/'daily_features.parquet',index=False,compression='zstd')
    features.to_parquet(ROOT/'features.parquet',index=False,compression='zstd')
    result={'protocol_commit':'b637849','history_start':'2023-01-01','stocks':len(paths),
        'history_rows':c.sql('SELECT count(*) FROM history').fetchone()[0],
        'history_end':c.sql('SELECT max(date) FROM history').fetchone()[0],
        'prior_member_stock_days':c.sql('SELECT count(*) FROM prior_members').fetchone()[0],
        'days':len(daily),'stock_days':len(features),'primary_pool_rows':int(features.primary_pool.sum()),
        'previous_main_ineligible_rows':int((features.board.eq('main')&features.prior_main_closed_limit.isna()).sum()),
        'daily_bins':daily.groupby('environment_bin').size().to_dict(),
        'half_coverage':features.groupby('half').agg(rows=('date','size'),primary=('primary_pool','sum'),context_valid=('context_valid','sum')).reset_index().to_dict('records'),
        'sha256':provenance,'outputs_sha256':{name:sha(ROOT/(name+'.parquet')) for name in
            ['history','prior_state','prior_members','daily_features','features']},
        'new_outcome_labels_accessed':False,'new_2026_prices_read':False}
    save_json(ROOT/'input_report.json',result)
    return {k:v for k,v in result.items() if k!='sha256'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['inputs']);args=parser.parse_args()
    print(json.dumps(globals()[args.stage](),ensure_ascii=False,indent=2))
