"""Evaluate all frozen seat classes and same-day, same-reason comparisons."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .corporate_cash import save_json,sha
from .economic_winner_analysis import daily_groups,summarize,number,METRICS
from .reference_gain_accounting import weekly_interval
from .seat_repeat_winner import ROOT,COVARIATES

LABELS=Path('data/research/economic_winner/period_quality')
CLASSES=['prior_day_unlisted','current_ambiguous','history_unknown','no_named_buyers',
         'fresh_named_buyers','repeat_and_sell','repeat_buy_list_only']
PERIODS=['2024H1','2024H2','2025H1','2025H2','2024','2025']


def period(frame,name):return frame.loc[frame.half.eq(name) if len(name)==6 else frame.half.str.startswith(name)]


def costs(frame):
    parts=[]
    for bps in [5,15]:
        p=frame.copy();p['cost_bps']=bps;p['net_return15']=frame[f'net_return{bps}'];p['label']=frame[f'label{bps}']
        p['known']=p.net_return15.notna();p['unknown']=p.label.eq('unknown');p['no_trade']=p.label.eq('no_trade')
        p['winner']=p.label.eq('economic_winner');p['loser']=p.label.eq('economic_loser');p['positive']=p.net_return15.gt(0)
        assert (p.known.astype(int)+p.unknown.astype(int)+p.no_trade.astype(int)).eq(1).all()
        parts.append(p)
    return pd.concat(parts,ignore_index=True)


def distribution(p,identifiers):
    values=p.net_return15.dropna();wins=values.loc[values.gt(0)];losses=values.loc[values.lt(0)]
    return dict(identifiers,known=len(values),unknown=int(p.unknown.sum()),no_trade=int(p.no_trade.sum()),
        positive_frequency=number(values.gt(0).mean()),median=number(values.median()),
        mean_positive=number(wins.mean()),mean_negative=number(losses.mean()),
        payoff_ratio=number(wins.mean()/-losses.mean()) if len(wins) and len(losses) else None,
        worst_five_percent_mean=number(values.nsmallest(max(1,math.ceil(len(values)*.05))).mean()),minimum=number(values.min()))


def group_summary(p,identifiers):
    result=summarize(p,identifiers)
    if 'net_mean_baseline' in p:
        values=p.set_index('date').net_mean_baseline.sort_index()
        result['same_day_listed_net_mean']=number(values.mean())
        result['same_day_listed_net_mean_week_interval']=weekly_interval(values)
    return result


def evaluate():
    if (ROOT/'analysis_report.json').exists():raise ValueError('Do not overwrite seat outcomes')
    check=json.loads((ROOT/'input_verification.json').read_text())
    assert check['passed'] and check['input_report_sha256']==sha(ROOT/'input_report.json')
    inputs=json.loads((ROOT/'input_report.json').read_text())
    for name,digest in inputs['outputs_sha256'].items():assert sha(ROOT/(name+'.parquet'))==digest
    quality=json.loads((LABELS/'analysis_verification.json').read_text())
    assert quality['passed'] and quality['analysis_report_sha256']==sha(LABELS/'analysis_report.json')
    assert json.loads((LABELS/'label_report.json').read_text())['labels_sha256']==sha(LABELS/'labels.parquet')
    pool=pd.read_parquet(ROOT/'pool.parquet');pool=pool.loc[pool.necessary_tradeable].copy()
    labels=pd.read_parquet(LABELS/'labels.parquet',columns=['date','code','label5','label15','net_return5','net_return15',
        'base_status','original_known_label','original_winner'])
    frame=pool.merge(labels,on=['date','code'],how='left',validate='one_to_one')
    assert len(frame)==inputs['necessary_tradeable'] and frame.label15.notna().all()
    frame.to_parquet(ROOT/'outcomes.parquet',index=False,compression='zstd')
    p=costs(frame)
    baseline=daily_groups(p.loc[~p.seat_class.eq('prior_day_unlisted')],['date','half','cost_bps'])
    baseline.to_parquet(ROOT/'baseline_daily.parquet',index=False,compression='zstd')
    groups=daily_groups(p,['date','half','cost_bps','seat_class'])
    groups=groups.merge(baseline[['date','cost_bps',*METRICS]],on=['date','cost_bps'],how='left',validate='many_to_one',suffixes=('','_baseline'))
    for name in ['winner','loser','positive']:
        groups[name+'_lower_delta']=groups[name+'_lower']-groups[name+'_upper_baseline']
        groups[name+'_upper_delta']=groups[name+'_upper']-groups[name+'_lower_baseline']
    groups['net_mean_delta']=groups.net_mean-groups.net_mean_baseline
    groups.to_parquet(ROOT/'groups_daily.parquet',index=False,compression='zstd')
    base_summary=[];group_summaries=[];distributions=[]
    for bps in [5,15]:
        for name in PERIODS:
            base_summary.append(group_summary(period(baseline.loc[baseline.cost_bps.eq(bps)],name),dict(cost_bps=bps,period=name)))
            for kind in CLASSES:
                identifiers=dict(cost_bps=bps,period=name,seat_class=kind)
                group_summaries.append(group_summary(period(groups.loc[groups.cost_bps.eq(bps)&groups.seat_class.eq(kind)],name),identifiers))
                distributions.append(distribution(period(p.loc[p.cost_bps.eq(bps)&p.seat_class.eq(kind)],name),identifiers))
    keys=['date','code'];pairs=pd.read_parquet(ROOT/'pairs.parquet');unmatched=pd.read_parquet(ROOT/'unmatched.parquet')
    columns=['date','code','label5','label15','net_return5','net_return15']
    paired=pairs.merge(frame[columns],on=keys,validate='one_to_one').merge(
        frame[columns].rename(columns={'code':'control_code'}),on=['date','control_code'],validate='many_to_one',suffixes=('_high','_control'))
    parts=[]
    for bps in [5,15]:
        q=paired.copy();q['cost_bps']=bps
        q['difference']=q[f'net_return{bps}_high']-q[f'net_return{bps}_control']
        q['known']=q.difference.notna()
        q['unknown']=q[f'label{bps}_high'].eq('unknown')|q[f'label{bps}_control'].eq('unknown')
        q['no_trade']=~q.unknown&~q.known
        assert (q.known.astype(int)+q.unknown.astype(int)+q.no_trade.astype(int)).eq(1).all()
        for kind in ['winner','loser','positive']:
            if kind=='positive':
                h=q[f'net_return{bps}_high'].gt(0);l=q[f'net_return{bps}_control'].gt(0)
            else:
                h=q[f'label{bps}_high'].eq('economic_'+kind);l=q[f'label{bps}_control'].eq('economic_'+kind)
            hu=q[f'label{bps}_high'].eq('unknown');lu=q[f'label{bps}_control'].eq('unknown')
            q[kind+'_lower_delta']=h.astype(int)-l.astype(int)-lu.astype(int)
            q[kind+'_upper_delta']=h.astype(int)+hu.astype(int)-l.astype(int)
        parts.append(q)
    paired=pd.concat(parts,ignore_index=True)
    paired.to_parquet(ROOT/'paired_outcomes.parquet',index=False,compression='zstd')
    probability_fields=[n+s+'_delta' for n in ['winner','loser','positive'] for s in ['_lower','_upper']]
    pair_daily=paired.groupby(['date','half','cost_bps']).agg(n=('code','size'),known=('known','sum'),unknown=('unknown','sum'),
        no_trade=('no_trade','sum'),difference=('difference','mean'),**{k:(k,'mean') for k in probability_fields}).reset_index()
    pair_daily.to_parquet(ROOT/'paired_daily.parquet',index=False,compression='zstd')
    pair_summaries=[];match_summaries=[]
    for name in PERIODS:
        match=period(pairs,name);high=period(frame.loc[frame.seat_class.eq('repeat_buy_list_only')],name)
        missing=unmatched.merge(high[keys],on=keys,validate='one_to_one')
        item={'period':name,'primary':len(high),'matched':len(match),'unmatched':len(missing),'unmatched_reasons':missing.reason.value_counts().to_dict()}
        for field in ['distance',*[n+'_gap' for n in COVARIATES]]:
            item[field+'_mean']=number(match[field].mean());item[field+'_median']=number(match[field].median());item[field+'_p95']=number(match[field].quantile(.95))
        match_summaries.append(item)
        for bps in [5,15]:
            d=period(pair_daily.loc[pair_daily.cost_bps.eq(bps)],name)
            item={'cost_bps':bps,'period':name,'pairs':int(d.n.sum()),'dates':len(d),'known':int(d.known.sum()),
                'unknown':int(d.unknown.sum()),'no_trade':int(d.no_trade.sum()),'valid_net_dates':int(d.difference.notna().sum())}
            for field in ['difference',*probability_fields]:
                values=d.set_index('date')[field].sort_index()
                item[field]=number(values.mean());item[field+'_week_interval']=weekly_interval(values)
            pair_summaries.append(item)
    cross=frame.groupby(['half','seat_class','label15','original_known_label','original_winner'],dropna=False).size().rename('n').reset_index()
    cross.to_parquet(ROOT/'label_cross.parquet',index=False,compression='zstd')
    result={'interpretation':'exact_named_brokerage_list_repetition_not_investor_identity_or_complete_portfolio',
        'input_report_sha256':sha(ROOT/'input_report.json'),'input_verification_sha256':sha(ROOT/'input_verification.json'),
        'label_report_sha256':sha(LABELS/'label_report.json'),'label_analysis_verification_sha256':sha(LABELS/'analysis_verification.json'),
        'baselines':base_summary,'groups':group_summaries,'distributions':distributions,'pairs':pair_summaries,'matching':match_summaries,
        'outputs_sha256':{n:sha(ROOT/(n+'.parquet')) for n in ['outcomes','baseline_daily','groups_daily','paired_outcomes','paired_daily','label_cross']},
        'new_2026_prices_read':False,'new_strategy_selected':False}
    save_json(ROOT/'analysis_report.json',result)
    return {'primary':[x for x in group_summaries if x['seat_class']=='repeat_buy_list_only'],'pairs':pair_summaries,
        'primary_distributions':[x for x in distributions if x['seat_class']=='repeat_buy_list_only']}


if __name__=='__main__':print(json.dumps(evaluate(),ensure_ascii=False,indent=2))
