"""Reconstruct only historical training trades, with resumable per-stock outputs."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .corporate_cash import save_json,sha
from .downside_ridge_inputs import training_scores,window_summary,ROOT as RECENT
from .hf_outcomes import Assumptions,outcomes_for_symbol
from .long_history_inputs import ROOT,RULE_COMMIT
from .quote_precision import quote_cents,fixed_quote_shares
from .reference_gain_eval import DAILY,MINUTES
from .turnover_reference import CALENDAR


def raw_labels(output: Path = ROOT) -> dict:
    if (output/'long/repriced.parquet').exists():
        raise ValueError('Cannot rebuild historical labels after new test outcomes exist')
    report=json.loads((output/'feature_report.json').read_text())
    if sha(output/'features_2022_2023.parquet')!=report['historical_features_sha256']:
        raise ValueError('The frozen historical candidate features changed')
    signals=pd.read_parquet(output/'features_2022_2023.parquet')
    signals['price_1449']=signals.price_1449.map(lambda p:quote_cents(p)/100)
    signals['isST'],signals['reference_gap'],signals['listing_age_sessions']=0,False,20
    signals.to_parquet(output/'historical_training_signals.parquet',index=False,compression='zstd')
    table=pd.read_parquet(CALENDAR)
    calendar=sorted(table.loc[table.is_trading_day.eq('1') & table.calendar_date.between('2022-01-01','2024-06-30'),'calendar_date'])
    index={d:i for i,d in enumerate(calendar)}
    source_fingerprints=json.loads((output/'prefix_manifest.json').read_text())['minute_sha256']
    parts=output/'historical_parts';parts.mkdir(exist_ok=True)
    code_files=[Path(__file__),Path('src/trade_research/hf_outcomes.py'),Path('src/trade_research/downside_ridge_inputs.py')]
    code_sha={str(p):sha(p) for p in code_files}
    def one(item):
        code,group=item
        exchange,symbol=code.split('.')
        minute_path=MINUTES/exchange.upper()/(symbol+'.parquet');daily_path=DAILY/(code.replace('.','_')+'.parquet')
        job_hash=hashlib.sha256(group[['date','code','price_1449']].to_json(orient='records',double_precision=15).encode()).hexdigest()
        fingerprint={'code':code,'signal_slice_sha256':job_hash,'minute_sha256':sha(minute_path),'daily_sha256':sha(daily_path),'code_sha256':code_sha}
        if fingerprint['minute_sha256']!=source_fingerprints[str(minute_path)]:
            raise ValueError('Minute source changed after the prefix was frozen')
        source_file=parts/(code+'.json');raw_file=parts/(code+'.trades.parquet');window_file=parts/(code+'.windows.parquet')
        if source_file.exists():
            prior=json.loads(source_file.read_text())
            if any(prior[k]!=v for k,v in fingerprint.items()) or sha(raw_file)!=prior['raw_sha256'] or sha(window_file)!=prior['windows_sha256']:
                raise ValueError('A checkpoint no longer matches its frozen sources or code')
            return raw_file,window_file,prior
        dates=sorted({d for signal in group.date for d in calendar[index[signal]:index[signal]+11]})
        if dates[-1]>='2024-07-01' or any(index[d]+10>=len(calendar) for d in group.date):
            raise ValueError('A historical training observation exceeds its fit boundary')
        c=duckdb.connect();c.execute('SET threads=1')
        c.read_parquet(str(minute_path)).create_view('minutes');c.register('needed',pd.DataFrame({'date':dates}))
        c.execute('''CREATE TEMP TABLE selected_bars AS SELECT timestamp,open,high,low,close,volume,turnover FROM minutes
            WHERE timestamp>=?::TIMESTAMP AND timestamp<?::DATE+INTERVAL 1 DAY
              AND strftime(timestamp,'%H%M') IN ('1452','1453','1454','1455')
              AND strftime(timestamp,'%Y-%m-%d') IN(SELECT date FROM needed)''',[dates[0],dates[-1]])
        minute=c.execute('SELECT * FROM selected_bars ORDER BY timestamp').df()
        windows=pd.DataFrame({'date':dates}).merge(window_summary(c),on='date',how='left',validate='one_to_one')
        windows['window_status'],windows['code']=windows.window_status.fillna('missing_window'),code
        c.read_parquet(str(daily_path)).create_view('daily')
        prior=c.execute('SELECT max(date) FROM daily WHERE date<? AND tradestatus=1',[dates[0]]).fetchone()[0]
        if prior is None:
            raise ValueError('Historical trading state needs a prior normal trading day')
        daily=c.execute('SELECT * FROM daily WHERE date BETWEEN ? AND ? ORDER BY date',[prior,dates[-1]]).df();c.close()
        minute['date'],minute['label']=minute.timestamp.dt.strftime('%Y-%m-%d'),minute.timestamp.dt.strftime('%H%M')
        trades=outcomes_for_symbol(group,minute,daily,calendar,Assumptions(target_notional=20000),horizons=(5,),sizing_price_column='price_1449')
        trades['target_notional'],trades['entry_window'],trades['exit_window']=20000,'baseline','close'
        sizing=group[['date','code','price_1449']].merge(trades[['date','code','entry_status','shares']],on=['date','code'],validate='one_to_one')
        for row in sizing.loc[sizing.entry_status.eq('filled')].itertuples():
            if row.shares!=fixed_quote_shares(row.code,row.price_1449,20000):
                raise ValueError('Historical sizing differs from the fixed cent-based decision')
        trades.to_parquet(raw_file,index=False,compression='zstd');windows.to_parquet(window_file,index=False,compression='zstd')
        source={**fingerprint,'first':dates[0],'last':dates[-1],'raw_sha256':sha(raw_file),'windows_sha256':sha(window_file),'rows':len(trades)}
        save_json(source_file,source)
        return raw_file,window_file,source
    groups=list(signals.groupby('code',sort=True));raw_files,window_files,sources=[],[],[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,(r,w,s) in enumerate(pool.map(one,groups),1):
            raw_files.append(r);window_files.append(w);sources.append(s)
            if i%200==0 or i==len(groups):
                save_json(output/'historical_raw_progress.json',{'stocks':i,'total':len(groups)})
                print(f'Historical raw training {i}/{len(groups)} stocks',flush=True)
    trades=pd.concat([pd.read_parquet(p) for p in raw_files],ignore_index=True).sort_values(['date','code'])
    windows=pd.concat([pd.read_parquet(p) for p in window_files],ignore_index=True).sort_values(['date','code'])
    if len(trades)!=len(signals) or trades.duplicated(['date','code']).any():
        raise ValueError('Historical raw training grid changed')
    trades.to_parquet(output/'historical_training_repriced.parquet',index=False,compression='zstd')
    windows.to_parquet(output/'historical_training_windows.parquet',index=False,compression='zstd')
    save_json(output/'historical_training_sources.json',sources)
    result={'rule_commit':RULE_COMMIT,'rows':len(trades),'stocks':len(groups),'first':signals.date.min(),'last_signal':signals.date.max(),
        'last_source_date':max(s['last'] for s in sources),'entry_statuses':trades.entry_status.value_counts().to_dict(),
        'exit_statuses':trades.exit_status.value_counts().to_dict(),'new_test_returns_read':False,'holdout_read':False,
        'signals_sha256':sha(output/'historical_training_signals.parquet'),'raw_sha256':sha(output/'historical_training_repriced.parquet'),
        'windows_sha256':sha(output/'historical_training_windows.parquet'),'sources_sha256':sha(output/'historical_training_sources.json')}
    save_json(output/'historical_raw_report.json',result)
    return result


def assemble(output: Path = ROOT) -> dict:
    raw_report=json.loads((output/'historical_raw_report.json').read_text())
    for name,key in [('historical_training_repriced','raw_sha256'),('historical_training_windows','windows_sha256')]:
        if sha(output/(name+'.parquet'))!=raw_report[key]:
            raise ValueError('Frozen historical raw execution changed')
    catalog_folder=output/'historical_catalog'
    cat_report=json.loads((catalog_folder/'catalog_report.json').read_text())
    if sha(catalog_folder/'events.parquet')!=cat_report['events_sha256']:
        raise ValueError('Historical catalogue changed')
    old_catalog=Path('data/research/cash_dividend_catalog/events_augmented.parquet')
    catalog=pd.concat([pd.read_parquet(catalog_folder/'events.parquet'),pd.read_parquet(old_catalog)],ignore_index=True)
    if catalog.duplicated(['code','dividOperateDate']).any():
        raise ValueError('Conflicting historical/recent distributions')
    catalog.to_parquet(output/'training_catalog.parquet',index=False,compression='zstd')
    labels,events=training_scores(pd.read_parquet(output/'historical_training_repriced.parquet'),pd.read_parquet(output/'historical_training_windows.parquet'),
        catalog_path=output/'training_catalog.parquet',last_catalog_pay_date='2024-06-30')
    if labels.exit_date.dropna().ge('2024-07-01').any() or not labels.date.between('2022-01-01','2023-12-31').all():
        raise ValueError('Historical training labels cross the original test period')
    labels.sort_values(['date','code']).to_parquet(output/'historical_training_labels.parquet',index=False,compression='zstd')
    events.to_parquet(output/'historical_training_actions.parquet',index=False,compression='zstd')
    result={'rule_commit':RULE_COMMIT,'labels':len(labels),'labels_sha256':sha(output/'historical_training_labels.parquet'),
        'raw_report_sha256':sha(output/'historical_raw_report.json'),'catalog_sha256':sha(output/'training_catalog.parquet'),
        'recent_catalog_source_sha256':sha(old_catalog),'historical_catalog_report_sha256':sha(catalog_folder/'catalog_report.json'),
        'score_origins':labels.score_origin.value_counts().to_dict(),'action_statuses':labels.action_status.value_counts().to_dict(),
        'new_test_returns_read':False,'holdout_read':False}
    save_json(output/'historical_label_report.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['raw','assemble']);args=parser.parse_args()
    print(json.dumps((raw_labels if args.stage=='raw' else assemble)(),ensure_ascii=False,indent=2))
