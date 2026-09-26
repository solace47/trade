"""Independent raw-window aggregation, source samples and complete cash-label checks."""
import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from trade_research.corporate_cash import MINUTES,save_json,sha

root=Path('data/research/economic_winner')
portrait=Path('data/research/next_day_winner')


def catalog():
    from collections import Counter
    folder=root/'catalog';inputs=json.loads((root/'input_manifest.json').read_text())
    report=json.loads((folder/'coverage_report.json').read_text())
    source=json.loads((folder/'catalog_report.json').read_text())
    assert report['complete'] and report['catalog_report_sha256']==sha(folder/'catalog_report.json')
    expected=[];coverage=[]
    jobs=pd.read_parquet(folder/'jobs.parquet')
    for row in jobs.itertuples():
        path=folder/'vendor'/(row.code+'_'+row.year+'.json')
        assert not path.with_suffix('.error.json').exists() and source['sha256'][str(path)]==sha(path)
        records=json.loads(path.read_text());assert isinstance(records,list)
        for record in records:assert record['code']==row.code and record['dividOperateDate'].startswith(row.year+'-')
        expected.extend(records);coverage.append({'code':row.code,'year':row.year,'events':len(records)})
    original_fields=sorted({name for row in expected for name in row})
    actual=pd.read_parquet(folder/'events.parquet')
    canonical=lambda rows:Counter(json.dumps({k:r.get(k) for k in original_fields},sort_keys=True,ensure_ascii=False) for r in rows)
    assert canonical(expected)==canonical(actual[original_fields].to_dict('records'))
    pd.testing.assert_frame_equal(pd.DataFrame(coverage),pd.read_parquet(folder/'query_coverage.parquet'),check_dtype=False)
    old_events=Path('data/research/winner_direction/catalog/events_reconciled.parquet')
    old_coverage=old_events.with_name('combined_coverage.parquet')
    assert sha(old_events)==inputs['old_events_sha256'] and sha(old_coverage)==inputs['old_coverage_sha256']
    combined=pd.read_parquet(folder/'combined_events.parquet')
    originals=pd.concat([pd.read_parquet(old_events),actual],ignore_index=True)
    assert canonical(combined[original_fields].to_dict('records'))==canonical(originals[original_fields].to_dict('records'))
    checked=pd.concat([pd.read_parquet(old_coverage),pd.DataFrame(coverage)],ignore_index=True).sort_values(['code','year']).reset_index(drop=True)
    pd.testing.assert_frame_equal(checked,pd.read_parquet(folder/'combined_coverage.parquet').sort_values(['code','year']).reset_index(drop=True),check_dtype=False)
    needed=pd.read_parquet(folder/'needed.parquet')
    assert needed.merge(checked[['code','year']],on=['code','year'],how='left',indicator=True)._merge.eq('both').all()
    assert sha(folder/'combined_events.parquet')==report['events_sha256'] and sha(folder/'combined_coverage.parquet')==report['coverage_sha256']
    result={'passed':True,'coverage_report_sha256':sha(folder/'coverage_report.json'),'new_code_years':len(jobs),
        'required_code_years':len(needed),'new_event_rows':len(actual),
        'duplicate_action_dates_retained':int(actual.loc[actual.duplicated(['code','dividOperateDate'],keep=False),['code','dividOperateDate']].drop_duplicates().shape[0]),
        'cash_terms_not_used_for_labels':True,'new_2026_prices_read':False}
    save_json(root/'catalog_verification.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))


def windows():
    manifest=json.loads((root/'input_manifest.json').read_text())
    raw=json.loads((root/'raw_report.json').read_text())
    assert raw['input_manifest_sha256']==sha(root/'input_manifest.json')
    assert sha(root/'window_keys.parquet')==manifest['window_keys_sha256']
    pieces=[];total=0
    for path,digest in raw['parts_sha256'].items():
        assert sha(Path(path))==digest
        p=pd.read_parquet(path);total+=len(p)
        assert p.date.ge('2024-01-01').all() and p.date.le('2025-12-31').all()
        assert p.minute.between(892,895).all() and p.date.eq(p.timestamp.dt.strftime('%Y-%m-%d')).all()
        assert not p.duplicated(['code','timestamp']).any()
        numbers=p[['open','high','low','close','volume','amount']].to_numpy(dtype=float)
        o,h,l,last,vol,amount=numbers.T
        ratio=np.divide(amount,vol,out=np.zeros(len(p)),where=vol>0)
        valid=(np.isfinite(numbers).all(axis=1)&(numbers[:,:4].min(axis=1)>0)&(h+.0001>=np.maximum.reduce([o,l,last]))
            &(l-.0001<=np.minimum(o,last))&(vol>=0)&(amount>=0)&((vol==0)==(amount==0))
            &((vol==0)|((ratio>=l-.0101)&(ratio<=h+.0101)))&(p.timestamp==p.timestamp.dt.floor('min')).to_numpy())
        p['valid']=valid;p['positive']=vol>0
        p['bad_cent']=(vol>0)&((np.abs(h-np.round(h,2))>.0001)|(np.abs(l-np.round(l,2))>.0001)|~np.isfinite(h)|~np.isfinite(l))
        p['positive_low']=np.where(vol>0,l,np.nan);p['positive_high']=np.where(vol>0,h,np.nan)
        g=p.groupby(['date','code']).agg(bars=('timestamp','size'),labels=('minute','nunique'),valid_bars=('valid','sum'),
            positive_bars=('positive','sum'),invalid_cent_bars=('bad_cent','sum'),volume=('volume','sum'),amount=('amount','sum'),
            positive_low=('positive_low','min'),positive_high=('positive_high','max')).reset_index()
        g['vwap']=g.amount/g.volume.where(g.volume.gt(0))
        pieces.append(g)
    assert total==raw['raw_rows']
    aggregates=pd.concat(pieces,ignore_index=True)
    result=pd.read_parquet(root/'window_keys.parquet').merge(aggregates,on=['date','code'],how='left',validate='one_to_one')
    result['source_valid']=result.bars.eq(4)&result.labels.eq(4)&result.valid_bars.eq(4)
    result['queue_bounds_valid']=result.positive_bars.gt(0)&result.invalid_cent_bars.eq(0)
    assert len(result)==manifest['windows']
    result.to_parquet(root/'independent_windows.parquet',index=False,compression='zstd')
    # Four fixed hash samples in each board/half/source-quality group, independent of returns.
    sample=result.copy()
    sample['board']=np.select([sample.code.str.startswith('sh.68'),sample.code.str.startswith('sz.30')],['star','chinext'],default='main')
    sample['half']=sample.date.str[:4]+np.where(sample.date.str[5:7].le('06'),'H1','H2')
    sample['hash']=[hashlib.sha256(('economic-winner-window-v1|'+d+'|'+code).encode()).hexdigest() for d,code in zip(sample.date,sample.code)]
    sample=sample.sort_values('hash').groupby(['board','half','source_valid']).head(4)
    connection=duckdb.connect();connection.read_parquet(list(raw['parts_sha256'])).create_view('stored_raw')
    samples=[]
    for r in sample.itertuples():
        file=MINUTES/r.code[:2].upper()/(r.code[3:]+'.parquet')
        assert sha(file)==manifest['source_sha256'][str(file)]
        first=pd.Timestamp(r.date+' 14:52');last=pd.Timestamp(r.date+' 14:56')
        original=pq.read_table(file,columns=['timestamp','open','high','low','close','volume','turnover'],
            filters=[('timestamp','>=',first.to_pydatetime()),('timestamp','<',last.to_pydatetime())]).to_pandas().rename(columns={'turnover':'amount'})
        original=original.sort_values('timestamp').reset_index(drop=True)
        stored=connection.execute('SELECT timestamp,open,high,low,close,volume,amount FROM stored_raw WHERE date=? AND code=? ORDER BY timestamp',[r.date,r.code]).df()
        pd.testing.assert_frame_equal(original,stored,check_dtype=False,atol=0,rtol=0)
        samples.append({'date':r.date,'code':r.code,'rows':len(original),'source_valid':r.source_valid})
    connection.close()
    report={'passed':True,'raw_report_sha256':sha(root/'raw_report.json'),'raw_rows':total,'windows':len(result),
        'independent_windows_sha256':sha(root/'independent_windows.parquet'),'samples':samples,
        'new_2026_prices_read':False,'economic_labels_computed':False}
    save_json(root/'window_verification.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='samples'},ensure_ascii=False,indent=2))


def labels():
    report=json.loads((root/'label_report.json').read_text())
    audit=json.loads((root/'window_verification.json').read_text())
    assert audit['passed'] and audit['raw_report_sha256']==report['raw_report_sha256']==sha(root/'raw_report.json')
    assert report['labels_sha256']==sha(root/'labels.parquet') and report['windows_sha256']==sha(root/'windows.parquet')
    assert audit['independent_windows_sha256']==sha(root/'independent_windows.parquet')
    stored_windows=pd.read_parquet(root/'windows.parquet').set_index(['date','code']).sort_index()
    independent_windows=pd.read_parquet(root/'independent_windows.parquet').set_index(['date','code']).sort_index()
    pd.testing.assert_frame_equal(independent_windows[stored_windows.columns].convert_dtypes(),stored_windows.convert_dtypes(),
        check_dtype=False,atol=2e-9,rtol=1e-14)
    del independent_windows,stored_windows
    c=duckdb.connect();c.execute('SET threads=4');c.execute("SET memory_limit='6GB'")
    for name,path in [('base',portrait/'visible_base.parquet'),('daily_labels',portrait/'labels.parquet'),
        ('windows',root/'independent_windows.parquet'),('out',root/'labels.parquet')]:c.read_parquet(str(path)).create_view(name)
    c.execute('''CREATE TABLE inputs AS SELECT b.date,b.code,b.board,b.half,b.necessary_tradeable,
        CASE WHEN b.board='star' THEN CASE WHEN 2000000//round(b.price_1449*100)::BIGINT>=200
          THEN 2000000//round(b.price_1449*100)::BIGINT ELSE 0 END
          ELSE(2000000//round(b.price_1449*100)::BIGINT)//100*100 END AS shares,
        b.price_1449,b.preclose,b.upper_limit,l.next_date,l.day_close,l.next_preclose,l.next_trade_status,l.next_isST,l.next_adjustflag,
        i.source_valid AS entry_valid,i.queue_bounds_valid AS entry_bounds,i.volume AS entry_volume,i.vwap AS buy,i.positive_high AS entry_high,
        o.source_valid AS exit_valid,o.queue_bounds_valid AS exit_bounds,o.volume AS exit_volume,o.vwap AS sell,o.positive_low AS exit_low,
        round(l.next_preclose::DECIMAL(18,4)*CASE WHEN b.board IN ('star','chinext') THEN .80
              WHEN l.next_isST=1 THEN .95 ELSE .90 END,2)::DOUBLE AS lower_limit
        FROM base b JOIN daily_labels l USING(date,code)
        LEFT JOIN windows i ON i.date=b.date AND i.code=b.code
        LEFT JOIN windows o ON o.date=l.next_date AND o.code=b.code''')
    c.execute('''CREATE TABLE states AS SELECT *,
        CASE WHEN NOT necessary_tradeable THEN 'not_submitted'
          WHEN NOT coalesce(isfinite(buy) AND buy>0 AND entry_volume>0,FALSE) THEN 'no_liquidity'
          WHEN NOT coalesce(shares<=entry_volume*.1,FALSE) THEN 'volume_cap'
          WHEN buy*1.0005>=upper_limit-.005 THEN 'estimated_upper_limit' ELSE 'filled' END AS entry_state,
        coalesce(isfinite(next_preclose) AND next_preclose>0 AND abs(next_preclose-round(next_preclose,2))<=.0001
          AND next_trade_status=1 AND next_isST IN(0,1) AND next_adjustflag=3,FALSE) AS next_valid
        FROM inputs''')
    expected=c.sql('''SELECT *,CASE WHEN entry_state<>'filled' THEN 'no_recorded_entry'
        WHEN NOT next_valid THEN 'no_trading_bar'
        WHEN NOT coalesce(isfinite(sell) AND sell>0 AND exit_volume>0,FALSE) THEN 'no_liquidity'
        WHEN NOT coalesce(shares<=exit_volume*.1,FALSE) THEN 'volume_cap'
        WHEN sell*.9995<=lower_limit+.005 THEN 'estimated_lower_limit' ELSE 'filled' END AS exit_state
        FROM states ORDER BY date,code''').df()
    out=pd.read_parquet(root/'labels.parquet').sort_values(['date','code']).reset_index(drop=True)
    assert expected[['date','code']].equals(out[['date','code']]) and len(out)==report['stock_days']
    np.testing.assert_array_equal(expected.shares,out.decision_shares)
    np.testing.assert_array_equal(expected.entry_state,out.entry_fill_status)
    np.testing.assert_array_equal(expected.exit_state,out.exit_fill_status)
    np.testing.assert_array_equal(expected.next_valid,out.exit_daily_valid)
    valid=out.next_preclose.notna()&out.next_preclose.gt(0)
    np.testing.assert_allclose(expected.loc[valid,'lower_limit'].to_numpy(dtype=float),out.loc[valid,'exit_lower_limit'],atol=0,rtol=0)
    # Use event-date unions independently; conflicting money amounts never enter the returns.
    events=pd.read_parquet(root/'catalog/combined_events.parquet');coverage=pd.read_parquet(root/'catalog/combined_coverage.parquet')
    event_groups={code:p for code,p in events.groupby('code')};covered=set(map(tuple,coverage[['code','year']].itertuples(index=False,name=None)))
    action=np.zeros(len(out),dtype=bool)
    for code,g in out.groupby('code',sort=False):
        e=event_groups.get(code)
        if e is None:continue
        d=g.date.to_numpy()[:,None];n=g.next_date.to_numpy()[:,None]
        op=e.dividOperateDate.to_numpy()[None,:];reg=e.dividRegistDate.to_numpy()[None,:]
        action[g.index]=((d==reg)|((d<op)&(n>=op))).any(axis=1)
    catalogue=np.fromiter(((code,date[:4]) in covered and (code,next_date[:4]) in covered
        for code,date,next_date in zip(out.code,out.date,out.next_date)),dtype=bool,count=len(out))
    refs=out[['day_close','next_preclose']].to_numpy(dtype=float)
    good_ref=np.isfinite(refs).all(axis=1)&(refs.min(axis=1)>0)&(np.abs(refs-np.round(refs,2))<=.0001).all(axis=1)
    gap=~good_ref|(np.abs(refs[:,0]-refs[:,1])>.005)
    np.testing.assert_array_equal(action,out.action_exposure);np.testing.assert_array_equal(catalogue,out.catalog_covered)
    np.testing.assert_array_equal(gap,out.reference_gap_or_unknown)
    corporate=~catalogue|action|gap;np.testing.assert_array_equal(corporate,out.corporate_unknown)
    bought=expected.entry_state.eq('filled').to_numpy()
    entry_queue=bought&(~expected.entry_bounds.fillna(False).to_numpy(dtype=bool)|
        (np.round(expected.entry_high.to_numpy(dtype=float),2)>=expected.upper_limit.to_numpy(dtype=float)))
    exit_queue=bought&(~expected.exit_bounds.fillna(False).to_numpy(dtype=bool)|
        (np.round(expected.exit_low.to_numpy(dtype=float),2)<=out.exit_lower_limit.to_numpy(dtype=float)))
    np.testing.assert_array_equal(entry_queue,out.entry_queue_unknown);np.testing.assert_array_equal(exit_queue,out.exit_queue_unknown)
    ordered=expected.necessary_tradeable.to_numpy(dtype=bool)
    entry_valid=expected.entry_valid.fillna(False).to_numpy(dtype=bool);exit_valid=expected.exit_valid.fillna(False).to_numpy(dtype=bool)
    no_trade=ordered&entry_valid&~bought
    ordinary=ordered&entry_valid&bought&~entry_queue&~corporate&exit_valid&expected.exit_state.eq('filled').to_numpy()&~exit_queue
    np.testing.assert_array_equal(ordinary,out.base_status.eq('ordinary_t1'))
    fee_values=0
    for bps in [5,15]:
        buy=expected.buy.to_numpy(dtype=float);sell=expected.sell.to_numpy(dtype=float)
        buy=buy+np.maximum(buy*bps/10000,.005);sell=sell-np.maximum(sell*bps/10000,.005)
        beyond=(buy>=expected.upper_limit.to_numpy(dtype=float)-.005)|(sell<=out.exit_lower_limit.to_numpy()+.005)
        np.testing.assert_array_equal(beyond,out[f'stress_limit_unknown{bps}'])
        known=ordinary&~beyond;np.testing.assert_array_equal(known,out[f'known_profit{bps}'])
        buy_value=expected.shares.to_numpy()*buy;sell_value=expected.shares.to_numpy()*sell
        pay=buy_value*(1+.00001)+np.maximum(5.,buy_value*.0003)
        receive=sell_value*(1-.00051)-np.maximum(5.,sell_value*.0003)
        pnl=receive/pay-1
        for field,values in [('buy_cash',pay),('sell_cash',receive),('net_return',pnl)]:
            np.testing.assert_allclose(out.loc[known,field+str(bps)],values[known],atol=1e-9 if field!='net_return' else 2e-12,rtol=1e-13)
            assert out.loc[~known,field+str(bps)].isna().all()
            fee_values+=int(known.sum())
        label=np.select([~ordered,no_trade,~known,pnl>=.01,pnl<=-.01],
            ['not_submitted','no_trade','unknown','economic_winner','economic_loser'],default='middle')
        np.testing.assert_array_equal(label,out[f'label{bps}'])
    result={'passed':True,'window_verification_sha256':sha(root/'window_verification.json'),
        'label_report_sha256':sha(root/'label_report.json'),'stock_days':len(out),'windows':audit['windows'],
        'raw_minutes':audit['raw_rows'],'source_window_samples':len(audit['samples']),'cash_values_checked':fee_values,
        'new_2026_prices_read':False,'complete_real_portfolio_returns':False}
    save_json(root/'label_verification.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['catalog','windows','labels']);args=p.parse_args()
    globals()[args.stage]()
