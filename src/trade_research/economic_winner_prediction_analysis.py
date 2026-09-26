"""Evaluate the frozen economic predictor, including its no-selection outcome."""
import json
import math

import numpy as np
import pandas as pd

from .corporate_cash import save_json,sha
from .economic_winner_prediction import ROOT,SOURCE
from .reference_gain_accounting import weekly_interval

CLASSES=['economic_loser','economic_winner','middle','no_trade','unknown']


def number(value):return float(value) if pd.notna(value) and np.isfinite(value) else None


def groups(frame,keys):
    records=[]
    for group,p in frame.groupby(keys,sort=True,dropna=False):
        if not isinstance(group,tuple):group=(group,)
        row=dict(zip(keys,group));row.update(n=len(p),known=int(p.net_return15.notna().sum()))
        for kind in CLASSES:
            row[kind]=int(p.label15.eq(kind).sum())
            row['frequency_'+kind]=float(p.label15.eq(kind).mean())
            row['predicted_'+kind]=float(p['p_'+kind].mean())
        for bps in [5,15]:row[f'net{bps}']=number(p[f'net_return{bps}'].mean())
        for field in ['raw_score','calibrated_score']:
            row[field]=float(p[field].mean())
            row[field+'_known']=number(p.loc[p.net_return15.notna(),field].mean())
        records.append(row)
    return pd.DataFrame(records)


def summarize(daily,identifiers):
    result=dict(identifiers,dates=len(daily),stock_days=int(daily.n.sum()),known=int(daily.known.sum()))
    for kind in CLASSES:result[kind]=int(daily[kind].sum())
    fields=[col for col in daily if col.startswith(('frequency_','predicted_','net','raw_score','calibrated_score'))]
    for field in fields:
        series=daily.set_index('date')[field].sort_index()
        result[field]=number(series.mean())
        # Means with a wholly unknown daily P&L keep an undefined interval.
        result[field+'_week_interval']=weekly_interval(series)
    return result


def evaluate():
    if (ROOT/'analysis_report.json').exists():raise ValueError('Do not overwrite prediction outcomes')
    check=json.loads((ROOT/'input_verification.json').read_text())
    assert check['passed'] and check['input_report_sha256']==sha(ROOT/'input_report.json')
    report=json.loads((ROOT/'input_report.json').read_text())
    for name,suffix in [('scores','.parquet'),('signals','.parquet'),('model','.json')]:
        assert sha(ROOT/(name+suffix))==report['outputs_sha256'][name]
    label_report=json.loads((SOURCE/'label_report.json').read_text())
    assert report['source_label_report_sha256']==sha(SOURCE/'label_report.json')
    assert sha(SOURCE/'labels.parquet')==label_report['labels_sha256']
    scores=pd.read_parquet(ROOT/'scores.parquet')
    scores=scores.loc[scores.role.isin(['calibration','test'])].copy()
    outcomes=pd.read_parquet(SOURCE/'labels.parquet',columns=['date','code','label5','label15','net_return5','net_return15','base_status'])
    frame=scores.merge(outcomes,on=['date','code'],how='left',validate='one_to_one')
    assert len(frame)==len(scores) and frame.label15.isin(CLASSES).all()
    daily=groups(frame,['date','half']);daily.to_parquet(ROOT/'prediction_daily.parquet',index=False,compression='zstd')
    bins=groups(frame,['date','half','score_bin']);bins.to_parquet(ROOT/'bins_daily.parquet',index=False,compression='zstd')
    selected=pd.read_parquet(ROOT/'signals.parquet').merge(outcomes,on=['date','code'],how='left',validate='many_to_one')
    assert selected.label15.isin(CLASSES).all()
    selected.to_parquet(ROOT/'selected_outcomes.parquet',index=False,compression='zstd')
    orders=groups(selected,['date','half','policy']) if len(selected) else pd.DataFrame()
    if len(orders):
        orders=orders.merge(daily[['date','net5','net15']],on='date',validate='many_to_one',suffixes=('','_baseline'))
        for bps in [5,15]:orders[f'net{bps}_delta']=orders[f'net{bps}']-orders[f'net{bps}_baseline']
    orders.to_parquet(ROOT/'selected_daily.parquet',index=False,compression='zstd')
    full=[summarize(p,dict(half=half)) for half,p in daily.groupby('half')]
    band=[summarize(p,dict(half=half,score_bin=int(score_bin))) for (half,score_bin),p in bins.groupby(['half','score_bin'])]
    policies=[];distributions=[]
    for policy in ['calibrated','raw']:
        for period in ['2025H1','2025H2','2025']:
            part=selected.loc[selected.policy.eq(policy)&(selected.half.eq(period) if period!='2025' else True)]
            if part.empty:
                policies.append({'policy':policy,'period':period,'stock_days':0,'dates':0,'no_selections':True,
                    'mean_net':None,'interpretation':'No trade signals; not evidence of a profitable selection formula.'})
                continue
            d=orders.loc[orders.policy.eq(policy)&(orders.half.eq(period) if period!='2025' else True)]
            item=summarize(d,dict(policy=policy,period=period,no_selections=False))
            item['base_status_counts']=part.base_status.value_counts().to_dict();policies.append(item)
            for bps in [5,15]:
                value=part[f'net_return{bps}'].dropna();positive=value.loc[value.gt(0)];negative=value.loc[value.lt(0)]
                distributions.append({'policy':policy,'period':period,'cost_bps':bps,'known':len(value),
                    'unknown':int(part[f'label{bps}'].eq('unknown').sum()),'no_trade':int(part[f'label{bps}'].eq('no_trade').sum()),
                    'positive_frequency':number(value.gt(0).mean()),'mean_positive':number(positive.mean()),'mean_negative':number(negative.mean()),
                    'payoff_ratio':number(positive.mean()/(-negative.mean())) if len(positive) and len(negative) else None,
                    'worst_five_percent_mean':number(value.nsmallest(max(1,math.ceil(len(value)*.05))).mean()),'minimum':number(value.min())})
    result={'interpretation':'exploratory_conditional_strict_T1_profit_not_full_portfolio_or_investor_identity',
        'input_report_sha256':sha(ROOT/'input_report.json'),'input_verification_sha256':sha(ROOT/'input_verification.json'),
        'source_label_report_sha256':sha(SOURCE/'label_report.json'),'full_pool':full,'bins':band,'policies':policies,'distributions':distributions,
        'outputs_sha256':{name:sha(ROOT/(name+'.parquet')) for name in ['prediction_daily','bins_daily','selected_outcomes','selected_daily']},
        'new_2026_prices_read':False,'complete_portfolio_returns':False}
    save_json(ROOT/'analysis_report.json',result)
    return {'policies':policies,'distributions':distributions}


if __name__=='__main__':print(json.dumps(evaluate(),ensure_ascii=False,indent=2))
