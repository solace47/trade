"""Execute and compare the two already frozen temporal-validation lists."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .alpha158_eval import daily_candidates
from .alpha158_pool_2026 import ROOT, PROTOCOL, RULE_COMMIT, policy
from .corporate_cash import save_json, sha
from .fixed_capital_1449 import prepare as prepare_capital, evaluate as capital
from .quality_period import audit_period
from .reference_gain_eval import reprice, MINUTES, DAILY
from .reference_gain_accounting import evaluate as account, weekly_interval
from .risk_removal_eval import evaluate as tick
from .shallow_tree_continuation import continue_model


def execute() -> dict:
    p = policy(); selection=json.loads((ROOT/'input_report.json').read_text())
    checks=json.loads((ROOT/'independent_input_checks.json').read_text())
    if sha(PROTOCOL)!=selection['protocol_sha256'] or checks['columnwise_score_max_error']>1e-12:
        raise ValueError('The fixed validation protocol or score checks changed')
    catalogue=ROOT/'catalog/events.parquet'
    catalog_report=json.loads((ROOT/'catalog/catalog_report.json').read_text())
    if sha(catalogue)!=catalog_report['events_sha256']:
        raise ValueError('The checked implementation-year catalogue changed')
    quality=ROOT/'quality_period.json'
    if not quality.exists():
        audit_period(Path('data/research/market_issues_ci'),MINUTES,DAILY,p['signal_first'],p['execution_last'],quality)
    result={}
    for name in p['variants']:
        folder=ROOT/name; continued=ROOT/'continued'/name
        expected=selection['models'][name]['signals_sha256']
        if sha(folder/'signals.parquet')!=expected or checks['models'][name]['signals_sha256']!=expected:
            raise ValueError('Validation list changed after independent verification')
        if not (folder/'execution_report.json').exists():
            reprice(folder,expected_signal_sha=expected,rule_commit=RULE_COMMIT,notional=p['notional'],
                first_date=p['signal_first'],last_date=p['execution_last'],quality_report=quality)
        if not (continued/'execution_report.json').exists():
            continue_model(folder,continued)
        if not (continued/'catalog_scenario_report.json').exists():
            account(continued,catalog_path=catalogue,allow_unsettled_payments=True)
        if not (continued/'tick_report.json').exists():
            tick(continued)
        result[name]={'execution':json.loads((continued/'execution_report.json').read_text()),
            'primary':[x for x in json.loads((continued/'tick_report.json').read_text())['annual']
                       if x['horizon']==5 and x['bps']==15]}
        print(name,result[name]['primary'],flush=True)
    return result


def statistics(series: pd.Series) -> dict:
    series=series.sort_index()
    return {'dates':len(series),'unknown_dates':int(series.isna().sum()),
        'mean':None if series.empty or series.isna().any() else float(series.mean()),
        'weekly_interval':weekly_interval(series),
        'weeks':pd.to_datetime(series.index).to_period('W-SUN').nunique()}


def compare() -> dict:
    p=policy(); selection=json.loads((ROOT/'input_report.json').read_text())
    frames={};hashes={};tables=[];metrics=[]
    periods=[['full',p['signal_first'],p['signal_last']],*p['descriptive_periods']]
    for name in p['variants']:
        folder=ROOT/'continued'/name
        report=json.loads((folder/'tick_report.json').read_text())
        if (sha(folder/'signals.parquet')!=selection['models'][name]['signals_sha256']
                or sha(folder/'tick_cost_scenario.parquet')!=report['tick_scenario_sha256']):
            raise ValueError('Frozen validation economic ledger changed')
        frames[name]=pd.read_parquet(folder/'tick_cost_scenario.parquet');hashes[name]=sha(folder/'tick_report.json')
        for horizon,group in frames[name].groupby('horizon'):
            high=group.loc[group.arm.eq('high')];low=group.loc[group.arm.eq('low')]
            paired=high.merge(low,on=['date','pair_id'],suffixes=('_high','_low'),validate='one_to_one')
            for bps in p['cost_bps']:
                own=high[['date',f'tick_return{bps}']].rename(columns={f'tick_return{bps}':'value'})
                edge=paired[['date']].assign(value=paired[f'tick_return{bps}_high']-paired[f'tick_return{bps}_low'])
                for metric,part in [('own',own),('same_day_edge',edge)]:
                    daily=part.groupby('date').value.agg(lambda x: x.mean() if x.notna().all() else np.nan)
                    for label,first,last in periods:
                        values=daily.loc[(daily.index>=first)&(daily.index<=last)]
                        metrics.append({'model':name,'horizon':int(horizon),'bps':bps,'metric':metric,'period':label,**statistics(values)})
    for bps in p['cost_bps']:
        left=daily_candidates(frames['highest_score'],f'tick_return{bps}')
        right=daily_candidates(frames['positive_pool'],f'tick_return{bps}')
        common=left.merge(right,on=['date','horizon'],validate='one_to_one',suffixes=('_highest','_pool'))
        common['difference']=common.value_pool-common.value_highest;common['bps']=bps;tables.append(common)
        for horizon,part in common.groupby('horizon'):
            daily=part.set_index('date').difference
            for label,first,last in periods:
                values=daily.loc[(daily.index>=first)&(daily.index<=last)]
                metrics.append({'model':'pool_minus_highest','horizon':int(horizon),'bps':bps,
                    'metric':'common_date_difference','period':label,**statistics(values)})
    pd.concat(tables,ignore_index=True).to_parquet(ROOT/'common_dates.parquet',index=False,compression='zstd')
    cash=ROOT/'capital'
    if not (cash/'report.json').exists():
        prepare_capital(cash,sources={name:ROOT/'continued'/name for name in p['variants']},
            first_date=p['signal_first'],last_date=p['execution_last'],allow_unsettled_payments=True)
        capital(cash)
    cash_report=json.loads((cash/'report.json').read_text())
    for key,path in [('manifest_sha256','manifest.json'),('ledger_sha256','cash_ledger.parquet'),('decisions_sha256','order_decisions.parquet')]:
        if sha(cash/path)!=cash_report[key]:raise ValueError('Validation capital ledger changed')
    primary=[x for x in metrics if x['horizon']==5 and x['bps']==15 and x['period']=='full']
    statistical={x['metric']:bool(x['mean'] is not None and x['mean']>0 and x['weekly_interval'] is not None and x['weekly_interval'][0]>0)
        for x in primary if x['model'] in ('positive_pool','pool_minus_highest')}
    cash_high=next(x for x in cash_report['books'] if x['model']=='positive_pool' and x['arm']=='high' and x['bps']==15)
    cash_low=next(x for x in cash_report['books'] if x['model']=='positive_pool' and x['arm']=='low' and x['bps']==15)
    hi,lo=cash_high['conditional_capital_return'],cash_low['conditional_capital_return']
    capital_pass=hi is not None and lo is not None and hi>0 and hi>lo
    rows=frames['positive_pool'];actual_unknown=int((rows.entry_status.eq('filled')&rows.known_return5.isna()).sum())
    result={'interpretation':'frozen_2026_temporal_validation_after_exposed_2024_2025_selection_not_pristine_blind',
        'rule_commit':RULE_COMMIT,'protocol_sha256':sha(PROTOCOL),'metrics':metrics,
        'primary':primary,'statistical_checks':statistical,'capital_check':capital_pass,
        'original_actual_unknown_rows':actual_unknown,
        'candidate_passes_numeric_gates':bool(all(statistical.values()) and capital_pass),
        'publishable_formula':False,'publication_requires_independent_checks_and_explained_execution_sources':True,
        'sources':hashes,'common_dates_sha256':sha(ROOT/'common_dates.parquet'),
        'capital_report_sha256':sha(cash/'report.json'),'holdout_prices_read':True,'strict_blind':False}
    save_json(ROOT/'comparison_report.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['execute','compare'])
    args=parser.parse_args();result=globals()[args.stage]()
    print(json.dumps(result,ensure_ascii=False,indent=2))
