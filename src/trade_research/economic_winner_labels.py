"""Strict T1 hypothetical cash labels with explicit no-trade and unknown states."""
import json

import duckdb
import numpy as np
import pandas as pd

from .corporate_cash import save_json,sha
from .economic_winner import ROOT,BASE,LABELS,PROTOCOL


def evaluate():
    if (ROOT/'label_report.json').exists():raise ValueError('Do not overwrite inspected economic labels')
    manifest=json.loads((ROOT/'input_manifest.json').read_text())
    raw=json.loads((ROOT/'raw_report.json').read_text())
    cat=json.loads((ROOT/'catalog/coverage_report.json').read_text())
    assert cat['complete'] and raw['input_manifest_sha256']==sha(ROOT/'input_manifest.json')
    assert sha(BASE)==manifest['base_sha256'] and sha(LABELS)==manifest['labels_sha256'] and sha(PROTOCOL)==manifest['protocol_sha256']
    for path,digest in raw['parts_sha256'].items():assert sha(__import__('pathlib').Path(path))==digest
    for name in ['events','coverage']:assert sha(ROOT/'catalog'/('combined_'+name+'.parquet'))==cat[name+'_sha256']
    c=duckdb.connect();c.execute('SET threads=4');c.execute("SET memory_limit='6GB'")
    c.read_parquet(list(raw['parts_sha256'])).create_view('raw')
    c.read_parquet(str(ROOT/'window_keys.parquet')).create_view('window_keys')
    c.read_parquet(str(BASE)).create_view('base')
    c.read_parquet(str(LABELS)).create_view('labels')
    c.read_parquet(str(ROOT/'catalog/combined_events.parquet')).create_view('events')
    c.read_parquet(str(ROOT/'catalog/combined_coverage.parquet')).create_view('coverage')
    c.execute('''CREATE TABLE windows AS WITH bars AS (
      SELECT *,coalesce(timestamp=date_trunc('minute',timestamp) AND isfinite(open) AND isfinite(high) AND isfinite(low)
        AND isfinite(close) AND isfinite(volume) AND isfinite(amount) AND least(open,high,low,close)>0
        AND high+.0001>=greatest(open,close,low) AND low-.0001<=least(open,close)
        AND volume>=0 AND amount>=0 AND (volume=0)=(amount=0)
        AND (volume=0 OR amount/volume BETWEEN low-.0101 AND high+.0101),FALSE) AS valid,
        coalesce(abs(high-round(high,2))<=.0001 AND abs(low-round(low,2))<=.0001,FALSE) AS cent_bounds
      FROM raw),aggregates AS (
        SELECT date,code,count(*) AS bars,count(DISTINCT minute) AS labels,
          count(*) FILTER(WHERE valid) AS valid_bars,
          count(*) FILTER(WHERE volume>0) AS positive_bars,
          count(*) FILTER(WHERE volume>0 AND NOT cent_bounds) AS invalid_cent_bars,
          sum(volume) AS volume,sum(amount) AS amount,sum(amount)/nullif(sum(volume),0) AS vwap,
          min(low) FILTER(WHERE volume>0) AS positive_low,max(high) FILTER(WHERE volume>0) AS positive_high
        FROM bars GROUP BY date,code)
      SELECT k.date,k.code,a.* EXCLUDE(date,code),
        coalesce(a.bars=4 AND a.labels=4 AND a.valid_bars=4,FALSE) AS source_valid,
        coalesce(a.positive_bars>0 AND a.invalid_cent_bars=0,FALSE) AS queue_bounds_valid
      FROM window_keys k LEFT JOIN aggregates a USING(date,code)''')
    windows=c.sql('SELECT * FROM windows ORDER BY date,code').df()
    assert len(windows)==manifest['windows'] and not windows.duplicated(['date','code']).any()
    windows.to_parquet(ROOT/'windows.parquet',index=False,compression='zstd')
    row=c.sql('''SELECT b.date,b.code,b.half,b.board,b.necessary_tradeable,b.decision_shares,b.price_1449,
        b.preclose,b.upper_limit,l.next_date,l.day_close,l.next_preclose,l.next_trade_status,l.next_isST,
        l.next_adjustflag,l.known_label AS original_known_label,l.winner AS original_winner,l.next_gain AS original_next_gain,
        i.bars AS entry_bars,i.labels AS entry_labels,i.source_valid AS entry_source_valid,
        i.queue_bounds_valid AS entry_bounds_valid,i.volume AS entry_volume,i.vwap AS entry_vwap,
        i.positive_low AS entry_low,i.positive_high AS entry_high,
        o.bars AS exit_bars,o.labels AS exit_labels,o.source_valid AS exit_source_valid,
        o.queue_bounds_valid AS exit_bounds_valid,o.volume AS exit_volume,o.vwap AS exit_vwap,
        o.positive_low AS exit_low,o.positive_high AS exit_high,
        EXISTS(SELECT 1 FROM coverage v WHERE v.code=b.code AND v.year=substr(b.date,1,4))
          AND EXISTS(SELECT 1 FROM coverage v WHERE v.code=b.code AND v.year=substr(l.next_date,1,4)) AS catalog_covered,
        EXISTS(SELECT 1 FROM events e WHERE e.code=b.code AND
          (e.dividRegistDate=b.date OR(e.dividOperateDate>b.date AND e.dividOperateDate<=l.next_date))) AS action_exposure
        FROM base b JOIN labels l USING(date,code)
        LEFT JOIN windows i ON i.date=b.date AND i.code=b.code
        LEFT JOIN windows o ON o.date=l.next_date AND o.code=b.code ORDER BY b.date,b.code''').df()
    c.close();assert len(row)==manifest['stock_days'] and not row.duplicated(['date','code']).any()
    numeric=['next_preclose','next_trade_status','next_isST','next_adjustflag','day_close',
        'entry_volume','entry_vwap','entry_low','entry_high','exit_volume','exit_vwap','exit_low','exit_high']
    for name in numeric:row[name]=row[name].astype('float64')
    for name in ['entry_source_valid','entry_bounds_valid','exit_source_valid','exit_bounds_valid']:
        row[name]=row[name].fillna(False).astype(bool)
    next_ref=row.next_preclose
    valid_reference=np.isfinite(next_ref)&next_ref.gt(0)&(next_ref-next_ref.round(2)).abs().le(.0001)
    valid_day_close=np.isfinite(row.day_close)&row.day_close.gt(0)&(row.day_close-row.day_close.round(2)).abs().le(.0001)
    rates=np.where(row.board.isin(['chinext','star']),20,np.where(row.next_isST.eq(1),5,10))
    pre_cents=np.rint(next_ref.fillna(0)*100).astype('int64')
    row['exit_lower_limit']=((pre_cents*(100-rates)+50)//100)/100
    row['exit_daily_valid']=valid_reference&row.next_trade_status.eq(1)&row.next_isST.isin([0,1])&row.next_adjustflag.eq(3)
    row['reference_gap_or_unknown']=~(valid_reference&valid_day_close)|(next_ref-row.day_close).abs().gt(.005)
    row['corporate_unknown']=~row.catalog_covered|row.action_exposure|row.reference_gap_or_unknown
    shares=row.decision_shares.to_numpy(dtype=float)
    entry_liquid=np.isfinite(row.entry_vwap)&row.entry_vwap.gt(0)&row.entry_volume.gt(0)
    entry_capacity=row.entry_volume.mul(.1).ge(shares)
    entry_at_limit=row.entry_vwap.mul(1.0005).ge(row.upper_limit-.005)
    row['entry_fill_status']=np.select([~row.necessary_tradeable,~entry_liquid,~entry_capacity,entry_at_limit],
        ['not_submitted','no_liquidity','volume_cap','estimated_upper_limit'],default='filled')
    bought=row.entry_fill_status.eq('filled')
    row['entry_recorded']=bought
    row['entry_sealed_limit']=row.entry_bounds_valid&row.entry_low.round(2).eq(row.upper_limit)&row.entry_high.round(2).eq(row.upper_limit)
    row['entry_queue_unknown']=bought&(~row.entry_bounds_valid|row.entry_high.round(2).ge(row.upper_limit))
    exit_liquid=np.isfinite(row.exit_vwap)&row.exit_vwap.gt(0)&row.exit_volume.gt(0)
    exit_capacity=row.exit_volume.mul(.1).ge(shares)
    exit_at_limit=row.exit_vwap.mul(.9995).le(row.exit_lower_limit+.005)
    row['exit_fill_status']=np.select([~bought,~row.exit_daily_valid,~exit_liquid,~exit_capacity,exit_at_limit],
        ['no_recorded_entry','no_trading_bar','no_liquidity','volume_cap','estimated_lower_limit'],default='filled')
    row['exit_queue_unknown']=bought&(~row.exit_bounds_valid|row.exit_low.round(2).le(row.exit_lower_limit))
    no_entry=row.necessary_tradeable&row.entry_source_valid&~bought
    # All diagnostic flags remain present even when an earlier status is primary.
    row['base_status']=np.select([~row.necessary_tradeable,~row.entry_source_valid,no_entry,row.entry_queue_unknown,
        row.corporate_unknown,~row.exit_source_valid,~row.exit_fill_status.eq('filled'),row.exit_queue_unknown],
        ['not_submitted','entry_source_unknown','not_bought','entry_queue_unknown','corporate_action_unknown',
         'exit_source_unknown','unknown_exit','exit_queue_unknown'],default='ordinary_t1')
    for bps in [5,15]:
        buy=row.entry_vwap+np.maximum(row.entry_vwap*bps/10000,.005)
        sell=row.exit_vwap-np.maximum(row.exit_vwap*bps/10000,.005)
        row[f'stress_limit_unknown{bps}']=buy.ge(row.upper_limit-.005)|sell.le(row.exit_lower_limit+.005)
        eligible=row.base_status.eq('ordinary_t1')&~row[f'stress_limit_unknown{bps}']
        buy_value=shares*buy;sell_value=shares*sell
        buy_cash=buy_value+np.maximum(5.,buy_value*.0003)+buy_value*.00001
        sell_cash=sell_value-np.maximum(5.,sell_value*.0003)-sell_value*.00051
        row[f'buy_cash{bps}']=buy_cash.where(eligible)
        row[f'sell_cash{bps}']=sell_cash.where(eligible)
        row[f'net_return{bps}']=(sell_cash/buy_cash-1).where(eligible)
        row[f'known_profit{bps}']=eligible
        row[f'label{bps}']=np.select([~row.necessary_tradeable,no_entry,~eligible,
            row[f'net_return{bps}'].ge(.01),row[f'net_return{bps}'].le(-.01)],
            ['not_submitted','no_trade','unknown','economic_winner','economic_loser'],default='middle')
    row.to_parquet(ROOT/'labels.parquet',index=False,compression='zstd')
    summary=row.groupby(['half','board','base_status']).size().rename('rows').reset_index().to_dict('records')
    result={'input_manifest_sha256':sha(ROOT/'input_manifest.json'),'raw_report_sha256':sha(ROOT/'raw_report.json'),
        'catalog_coverage_sha256':sha(ROOT/'catalog/coverage_report.json'),'windows_sha256':sha(ROOT/'windows.parquet'),
        'labels_sha256':sha(ROOT/'labels.parquet'),'stock_days':len(row),'windows':len(windows),'base_statuses':summary,
        'new_2026_prices_read':False,'complete_real_portfolio_returns':False,
        'label_counts':{str(bps):row[f'label{bps}'].value_counts().to_dict() for bps in [5,15]}}
    save_json(ROOT/'label_report.json',result)
    return {k:v for k,v in result.items() if k!='base_statuses'}


if __name__=='__main__':print(json.dumps(evaluate(),ensure_ascii=False,indent=2))
