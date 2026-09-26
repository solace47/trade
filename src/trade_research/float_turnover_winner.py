"""Add only lagged floating-size and absolute-turnover proxies to the portrait."""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .corporate_cash import DAILY,save_json,sha
from .turnover_reference import CALENDAR

ROOT=Path('data/research/float_turnover_winner')
COHORT=Path('data/research/next_day_winner/cohort.parquet')
PROTOCOL=Path('config/float_turnover_winner_protocol.json')
FEATURES=['float_cap_proxy','turnover_1449_proxy','turnover_tail29_proxy']


def rank_columns(frame,names):
    group=frame.groupby(['date','board'])
    rank=group[names].rank(method='average')
    count=group[names].transform('count')
    for name in names:
        frame[name+'_rank']=rank[name]/count[name].replace(0,np.nan)
        twice=rank[name].fillna(0).mul(2).astype('int64')
        denominator=count[name].mul(2).clip(lower=1).astype('int64')
        frame[name+'_quintile']=((5*twice+denominator-1)//denominator).where(frame[name].notna(),0).astype('int8')


def freeze():
    if (ROOT/'input_report.json').exists():raise ValueError('Do not replace inspected size-turnover inputs')
    prior_report=json.loads(Path('data/research/next_day_winner/analysis_report.json').read_text())
    assert sha(COHORT)==prior_report['cohort_sha256']
    source=json.loads((ROOT/'source/manifest.json').read_text())
    for name,digest in source['files_sha256'].items():assert sha(ROOT/'source'/name)==digest
    doc=(ROOT/'source/stockKData.md').read_text()
    assert '流通股' in doc and 'turn' in doc
    cal=pd.read_parquet(CALENDAR)
    days=sorted(cal.loc[cal.is_trading_day.eq('1')&cal.calendar_date.between('2023-12-29','2025-12-31'),'calendar_date'])
    schedule=pd.DataFrame({'date':days[1:],'prior_date':days[:-1]})
    c=duckdb.connect();c.execute('SET threads=4');c.execute("SET memory_limit='6GB'")
    c.register('schedule',schedule);c.read_parquet(str(COHORT)).create_view('cohort')
    c.execute(f"CREATE VIEW prior_daily AS SELECT * FROM read_parquet('{DAILY}/*.parquet') WHERE date BETWEEN '2023-12-29' AND '2025-12-29'")
    frame=c.sql('''SELECT b.date,b.code,b.half,b.board,b.necessary_tradeable,b.price_1449,b.volume_1449,b.volume_last29,
      s.prior_date,d.date AS source_date,d.volume AS prior_volume,d.turn AS prior_turn,
      d.tradestatus AS prior_trade_status,d.adjustflag AS prior_adjustflag
      FROM cohort b JOIN schedule s USING(date) LEFT JOIN prior_daily d ON b.code=d.code AND s.prior_date=d.date
      ORDER BY b.date,b.code''').df();c.close()
    assert len(frame)==2404280 and not frame.duplicated(['date','code']).any()
    for name in ['prior_volume','prior_turn','prior_trade_status','prior_adjustflag']:frame[name]=frame[name].astype('float64')
    valid=(frame.source_date.eq(frame.prior_date)&frame.prior_trade_status.eq(1)&frame.prior_adjustflag.eq(3)
        &np.isfinite(frame.prior_volume)&frame.prior_volume.gt(0)&np.isfinite(frame.prior_turn)&frame.prior_turn.gt(0))
    frame['source_valid']=valid
    frame['source_status']=np.select([frame.source_date.isna(),~frame.prior_trade_status.eq(1),~valid],
        ['missing_prior_day','prior_not_trading','invalid_prior_denominator'],default='valid_lagged_proxy')
    for suffix,shift in [('',0.),('_turn_lower',-.0001),('_turn_upper',.0001)]:
        turn=(frame.prior_turn+shift).where(valid&(frame.prior_turn+shift).gt(0))
        shares=frame.prior_volume*100/turn
        frame['float_shares_proxy'+suffix]=shares
        frame['float_cap_proxy'+suffix]=shares*frame.price_1449
        frame['turnover_1449_proxy'+suffix]=frame.volume_1449*turn/frame.prior_volume
        frame['turnover_tail29_proxy'+suffix]=frame.volume_last29*turn/frame.prior_volume
        rank_columns(frame,[name+suffix for name in FEATURES])
    ROOT.mkdir(exist_ok=True)
    frame.to_parquet(ROOT/'features.parquet',index=False,compression='zstd')
    source_files={str(DAILY/(code.replace('.','_')+'.parquet')):sha(DAILY/(code.replace('.','_')+'.parquet')) for code in sorted(frame.code.unique())}
    precision=[]
    for (board,half),p in frame.groupby(['board','half']):
        for feature in FEATURES:
            for suffix in ['_turn_lower','_turn_upper']:
                precision.append({'board':board,'half':half,'feature':feature,'scenario':suffix,
                    'rows':len(p),'group_changes':int(p[feature+'_quintile'].ne(p[feature+suffix+'_quintile']).sum())})
    report={'protocol_sha256':sha(PROTOCOL),'cohort_sha256':sha(COHORT),'calendar_sha256':sha(CALENDAR),
        'source_document_manifest_sha256':sha(ROOT/'source/manifest.json'),'daily_files_sha256':source_files,
        'features_sha256':sha(ROOT/'features.parquet'),'stock_days':len(frame),
        'necessary_tradeable':int(frame.necessary_tradeable.sum()),'source_status_counts':frame.source_status.value_counts().to_dict(),
        'source_coverage_by_half_board':frame.groupby(['board','half','source_status']).size().rename('rows').reset_index().to_dict('records'),
        'first_prior_date':frame.prior_date.min(),'last_prior_date':frame.prior_date.max(),
        'precision_group_changes':precision,'new_outcome_groups_computed':False,'new_2026_prices_read':False}
    save_json(ROOT/'input_report.json',report)
    return {k:report[k] for k in ['features_sha256','stock_days','necessary_tradeable','source_status_counts','first_prior_date','last_prior_date']}


if __name__=='__main__':print(json.dumps(freeze(),ensure_ascii=False,indent=2))
