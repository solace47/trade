"""One frozen T1 profit trigger with lagged execution on unchanged candidates."""
from __future__ import annotations

import argparse
import json
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd

from .corporate_cash import save_json, sha
from .hf_outcomes import Assumptions, _fill
from .price_limit_queue_audit import window_exposure
from .hf_outcomes import _board_limit_rate, _limit_price
from .reference_gain_accounting import weekly_interval
from .risk_removal_eval import add_cost_scenarios, cost_price
from .round_number_entry import validate_window
from .winner_exit_opportunity import ROOT as OPPORTUNITY, fixed_inputs, first_day_entitlement

ROOT = Path('data/research/winner_take_profit')
PROTOCOL = Path('config/winner_take_profit_protocol.json')
MERGER = Path('data/research/winner_direction/balanced/merger_followup/report.json')
MODELS = ('balanced', 'upside_only')
STARTS = [*range(575, 688), *range(781, 893)]
SIGNAL_MINUTES = [*range(575, 686), *range(781, 890)]
ECONOMIC_COLUMNS = ['entry_status','entry_price','shares','exit_price','entry_window_status',
    'entry_window_low','entry_window_high','exit_window_status','exit_window_low','exit_window_high',
    'catalog_sold_shares','catalog_dividend_gross','catalog_dividend_tax']


def freeze() -> dict:
    if (ROOT/'input_report.json').exists():
        raise ValueError('Do not replace frozen take-profit inputs')
    fixed_inputs()
    check = json.loads((OPPORTUNITY/'verification_report.json').read_text())
    assert check['passed'] and check['analysis_report_sha256'] == sha(OPPORTUNITY/'analysis_report.json')
    files = [PROTOCOL, MERGER, *[OPPORTUNITY/name for name in
        ('input_report.json','raw_report.json','positions.parquet','analysis_report.json',
         'verification_report.json','window_scenarios.parquet')]]
    files += [Path('data/research/winner_direction')/model/'continued/execution_queue_audit.parquet' for model in MODELS]
    report = {'protocol_commit':'16684db','orders':2330,'sha256':{str(p):sha(p) for p in files},
              'previously_exposed_2025':True,'new_2026_prices_read':False}
    ROOT.mkdir(parents=True,exist_ok=True)
    save_json(ROOT/'input_report.json',report)
    return report


def verify() -> None:
    fixed_inputs()
    report=json.loads((ROOT/'input_report.json').read_text())
    for path,digest in report['sha256'].items():
        if sha(Path(path)) != digest:
            raise ValueError(f'Frozen input changed: {path}')


def signal_prices(bars: pd.DataFrame) -> pd.DataFrame:
    p=bars.copy();p['minute']=p.timestamp.dt.hour*60+p.timestamp.dt.minute
    p=p.loc[p.minute.isin(SIGNAL_MINUTES)].sort_values('timestamp')
    duplicate=p.duplicated('timestamp',keep=False)
    v=p[['open','high','low','close','volume','turnover']].to_numpy(dtype=float)
    o,h,l,close,volume,amount=v.T
    ok=(np.isfinite(v).all(axis=1)&(np.minimum.reduce([o,h,l,close])>0)
        &(h+.0001>=np.maximum.reduce([o,close,l]))&(l-.0001<=np.minimum(o,close))
        &(volume>=0)&(amount>=0)&((volume==0)==(amount==0)))
    vw=np.divide(amount,volume,out=np.zeros_like(amount),where=volume>0)
    ok&=(volume==0)|((vw>=l-.0101)&(vw<=h+.0101))
    p['signal_valid']=ok&~duplicate&p.timestamp.eq(p.timestamp.dt.floor('min'))
    return p


def close_value(price, quantity: int, raw_buy: float) -> np.ndarray:
    buy_value=quantity*cost_price(raw_buy,15,'buy')
    sell_value=quantity*cost_price(price,15,'sell')
    return (sell_value-np.maximum(5.,sell_value*.0003)-sell_value*.00051)/(buy_value+max(5.,buy_value*.0003)+buy_value*.00001)-1


def scheduled_windows(trigger_minute: int) -> list[int]:
    # Three labels leave at least two minutes after completion even if labels
    # denote minute starts. Finish early attempts before the 14:52 fallback.
    result=[];minimum=trigger_minute+3
    for start in STARTS:
        if start>=minimum and start+3<892:
            result.append(start);minimum=start+4
    result.append(892)
    return result


def evaluate() -> dict:
    if (ROOT/'analysis_report.json').exists():
        raise ValueError('Do not overwrite inspected take-profit results')
    verify()
    positions=pd.read_parquet(OPPORTUNITY/'positions.parquet')
    quotes=pd.read_parquet(OPPORTUNITY/'window_scenarios.parquet')
    grouped={key:part.set_index('start_minute') for key,part in quotes.groupby(['model','date','code'])}
    catalog=pd.read_parquet('data/research/winner_direction/catalog/events_reconciled.parquet')
    queue={}
    for model in MODELS:
        q=pd.read_parquet(Path('data/research/winner_direction')/model/'continued/execution_queue_audit.parquet')
        for row in q.itertuples():queue[(model,row.date,row.code)]=row
    raw_sources={x['code']:x for x in json.loads((OPPORTUNITY/'raw_report.json').read_text())['sources']}
    records,attempts=[],[]
    for code,part in positions.groupby('code',sort=True):
        day_groups,daily={},pd.DataFrame()
        if part.entry_status.eq('filled').any():
            path=OPPORTUNITY/'raw_parts'/(code+'.parquet');daily_path=path.with_name(code+'.daily.parquet')
            assert sha(path)==raw_sources[code]['minute_output_sha256']
            assert sha(daily_path)==raw_sources[code]['daily_output_sha256']
            day_groups={date:group for date,group in pd.read_parquet(path).groupby('date')}
            daily=pd.read_parquet(daily_path).set_index('date',drop=False)
        for _,row in part.iterrows():
            key=(row.model,row.date,code);original_queue=queue[key]
            r={name:row[name] for name in ECONOMIC_COLUMNS}
            r.update(model=row.model,date=row.date,code=code,target_date=row.target_exit_date,
                original_exit_date=row.exit_date,exit_date=row.exit_date,exit_start='1452',
                exit_delay_sessions=row.exit_delay_sessions,original_actual_unknown=bool(row.unknown_after_buy),
                entry_queue_unknown=bool(pd.isna(row.entry_limit_touched) or row.entry_limit_touched),
                exit_queue_unknown=bool(pd.isna(original_queue.exit_limit_touched) or original_queue.exit_limit_touched),
                original_action_applied=bool(row.catalog_action_applied),action_applied=bool(row.catalog_action_applied),
                trigger_minute=None,trigger_estimated_return15=None,early_sale=False,attempts=0,
                status='not_bought' if row.entry_status!='filled' else 'no_trigger')
            for bps in (5,15):
                r[f'original_return{bps}']=row[f'tick_return{bps}']
            if row.entry_status!='filled':records.append(r);continue
            target=row.target_exit_date
            if target not in daily.index or row.date not in daily.index or target not in day_groups:
                records.append(dict(r,status='no_target_data_or_suspended'));continue
            day=daily.loc[target]
            if day.tradestatus!=1:
                records.append(dict(r,status='no_target_data_or_suspended'));continue
            if abs(day.preclose-daily.loc[row.date,'close'])>.005:
                records.append(dict(r,status='reference_gap_keep_original'));continue
            signal=signal_prices(day_groups[target]);valid=signal.loc[signal.signal_valid].copy()
            if valid.empty:records.append(dict(r,status='no_valid_signal_minutes'));continue
            valid['estimated_return15']=close_value(valid.close.to_numpy(dtype=float),int(row.shares),row.entry_price/1.0005)
            hits=valid.loc[valid.estimated_return15.ge(.01)]
            if hits.empty:records.append(r);continue
            first=hits.iloc[0];r.update(trigger_minute=int(first.minute),
                trigger_estimated_return15=float(first.estimated_return15),status='trigger_unfilled_keep_original')
            entitlement=first_day_entitlement(row,daily,catalog)
            if entitlement['entitlement_status'] not in ('none','catalogue_scenario'):
                records.append(dict(r,status='trigger_entitlement_unknown_keep_original'));continue
            windows=grouped.get(key)
            if windows is None:
                records.append(r);continue
            for start in scheduled_windows(int(first.minute)):
                q=windows.loc[start]
                if not (np.isfinite(q.volume) and np.isfinite(q.vwap)):
                    price,status=None,'missing_or_nonfinite_window'
                else:
                    price,status=_fill(q,day,code,'sell',entitlement['shares_at_target'],Assumptions(target_notional=20000))
                attempt={'model':row.model,'date':row.date,'code':code,'target_date':target,
                    'trigger_minute':int(first.minute),'start_minute':start,'fill_status':status,
                    'source_valid':bool(q.source_valid),'queue_unknown':bool(q.queue_unknown)}
                attempts.append(attempt);r['attempts']+=1
                if status!='filled':continue
                bars=day_groups[target]
                minute=bars.timestamp.dt.hour*60+bars.timestamp.dt.minute
                window=bars.loc[minute.between(start,start+3)]
                labels=tuple(f'{t//60:02d}{t%60:02d}' for t in range(start,start+4))
                source_status,_=validate_window(window,expected_labels=labels)
                positive=window.loc[window.volume.gt(0)]
                lower=_limit_price(day.preclose,_board_limit_rate(code,int(day.isST),target),False)
                exposure=window_exposure(window,lower,'sell')
                cash=entitlement['net_receivable']/.8
                r.update(exit_price=price,exit_date=target,exit_start=labels[0],exit_delay_sessions=0,
                    exit_window_status=source_status,exit_window_low=float(positive.low.min()),
                    exit_window_high=float(positive.high.max()),catalog_sold_shares=entitlement['shares_at_target'],
                    catalog_dividend_gross=cash,catalog_dividend_tax=cash*.2,
                    action_applied=entitlement['entitlement_status']=='catalogue_scenario',
                    exit_queue_unknown=exposure['limit_touched'] is not False,
                    early_sale=start<892,status='triggered_early_sale' if start<892 else 'triggered_tail_sale')
                break
            records.append(r)
    outcomes=add_cost_scenarios(pd.DataFrame(records))
    outcomes['source_valid']=outcomes.entry_window_status.eq('valid')&(outcomes.exit_window_status.eq('valid')|outcomes.entry_status.ne('filled'))
    outcomes['queue_unknown']=outcomes.entry_status.eq('filled')&(outcomes.entry_queue_unknown|outcomes.exit_queue_unknown)
    outcomes['unverified_preserving_original']=outcomes.entry_status.eq('filled')&(outcomes.original_actual_unknown|~outcomes.source_valid|outcomes.queue_unknown|outcomes.action_applied|outcomes.exit_price.isna())
    outcomes.to_parquet(ROOT/'outcomes.parquet',index=False,compression='zstd')
    pd.DataFrame(attempts).to_parquet(ROOT/'attempts.parquet',index=False,compression='zstd')
    result=summarize(outcomes)
    result.update(input_report_sha256=sha(ROOT/'input_report.json'),
        outputs_sha256={name:sha(ROOT/name) for name in ('outcomes.parquet','attempts.parquet')},
        new_2026_prices_read=False,previously_exposed_2025=True)
    save_json(ROOT/'analysis_report.json',result)
    return result


def summarize(outcomes: pd.DataFrame) -> dict:
    merger=json.loads(MERGER.read_text())
    scenarios=merger['scenarios']
    metrics,distribution,counts=[],[],[]
    for model,g in outcomes.groupby('model'):
        for period,start,end in [('2025','2025-01-01','2025-12-31'),('2025H1','2025-01-01','2025-06-30'),('2025H2','2025-07-01','2025-12-31')]:
            p=g.loc[g.date.between(start,end)].copy();bought=p.entry_status.eq('filled')
            trigger=p.trigger_minute.notna();early=p.early_sale
            counts.append({'model':model,'period':period,'orders':len(p),'bought':int(bought.sum()),
                'triggered':int(trigger.sum()),'early_sales':int(early.sum()),
                'triggered_unfilled':int((trigger&p.status.str.contains('unfilled')).sum()),
                'source_invalid_bought':int((bought&~p.source_valid).sum()),
                'queue_unknown':int(p.queue_unknown.sum()),'preserved_actual_unknown':int(p.unverified_preserving_original.sum()),
                'delayed_original_or_new':int(p.exit_delay_sessions.gt(0).sum()),
                'unresolved_original_merger':int((bought&p.exit_price.isna()).sum()),
                'early_sale_return_below_one_percent':int((early&p.tick_return15.lt(.01)).sum()),
                'early_sale_loss':int((early&p.tick_return15.lt(0)).sum()),
                'mean_execution_minus_signal_return15':float((p.loc[early,'tick_return15']-p.loc[early,'trigger_estimated_return15']).mean()) if early.any() else None})
            for bps in (5,15):
                quantity_scenarios=(522,523) if model=='balanced' else (None,)
                for quantity in quantity_scenarios:
                    new,old=p[f'tick_return{bps}'].copy(),p[f'original_return{bps}'].copy()
                    unresolved=bought&new.isna()
                    if unresolved.any():
                        assert quantity and unresolved.sum()==1
                        assert p.loc[unresolved,'code'].iloc[0]=='sh.601989' and not p.loc[unresolved,'early_sale'].iloc[0]
                        value=next(v['net_return'] for v in scenarios if v['sold_shares']==quantity and v['bps']==bps)
                        new.loc[unresolved]=value;old.loc[unresolved]=value
                    assert new.notna().all() and old.notna().all()
                    for name,value in [('own',new),('original',old),('difference',new-old)]:
                        by_date=value.groupby(p.date).mean()
                        metrics.append({'model':model,'period':period,'bps':bps,'merger_shares':quantity,'metric':name,
                            'dates':len(by_date),'mean':float(by_date.mean()),'weekly_interval':weekly_interval(by_date)})
                    trade=new.loc[bought];wins,losses=trade.loc[trade.gt(0)],trade.loc[trade.lt(0)]
                    distribution.append({'model':model,'period':period,'bps':bps,'merger_shares':quantity,
                        'bought':len(trade),'win_rate':float(trade.gt(0).mean()),
                        'mean_win':float(wins.mean()),'mean_loss':float(losses.mean()),
                        'payoff_ratio':float(wins.mean()/-losses.mean()),
                        'worst_five_percent_mean':float(trade.nsmallest(ceil(.05*len(trade))).mean()),
                        'worst_trade_return':float(trade.min())})
    return {'interpretation':'lagged_profit_trigger_catalogue_and_recorded_fill_scenario_not_complete_actual_portfolio',
        'counts':counts,'metrics':metrics,'distributions':distribution,
        'statuses':outcomes.groupby(['model','status']).size().rename('rows').reset_index().to_dict('records')}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','evaluate']);args=parser.parse_args()
    print(json.dumps(globals()[args.stage](),ensure_ascii=False,indent=2))
