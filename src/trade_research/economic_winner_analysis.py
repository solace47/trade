"""Frozen 29-feature descriptions against actual-window conditional net labels."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .corporate_cash import save_json,sha
from .economic_winner import ROOT,PORTRAIT
from .next_day_winner_analysis import FEATURES
from .reference_gain_accounting import weekly_interval

METRICS=['winner_lower','winner_upper','loser_lower','loser_upper','positive_lower','positive_upper','net_mean']
HALVES=['2024H1','2024H2','2025H1','2025H2']
BOARDS=['main','chinext','star']


def number(x):return float(x) if pd.notna(x) and np.isfinite(x) else None


def daily_groups(p,keys):
    result=p.groupby(keys,dropna=False).agg(n=('code','size'),known=('known','sum'),unknown=('unknown','sum'),
        no_trade=('no_trade','sum'),winner_count=('winner','sum'),loser_count=('loser','sum'),positive_count=('positive','sum'),net_mean=('net_return15','mean')).reset_index()
    for name in ['winner','loser','positive']:
        result[name+'_lower']=result[name+'_count']/result.n
        result[name+'_upper']=(result[name+'_count']+result.unknown)/result.n
    return result


def summarize(p,identifiers):
    item=dict(identifiers,stock_days=int(p.n.sum()),dates=len(p),known=int(p.known.sum()),unknown=int(p.unknown.sum()),
        no_trade=int(p.no_trade.sum()),winner_cases=int(p.winner_count.sum()),loser_cases=int(p.loser_count.sum()),
        positive_cases=int(p.positive_count.sum()),valid_net_dates=int(p.net_mean.notna().sum()))
    for name in METRICS:
        series=p.set_index('date')[name].sort_index()
        item[name]=number(series.mean());item[name+'_week_interval']=weekly_interval(series)
        if name+'_delta' in p:
            delta=p.set_index('date')[name+'_delta'].sort_index()
            item[name+'_delta']=number(delta.mean());item[name+'_delta_week_interval']=weekly_interval(delta)
    return item


def evaluate(ROOT:Path=ROOT):
    if (ROOT/'analysis_report.json').exists():raise ValueError('Do not replace economic feature results')
    label_report=json.loads((ROOT/'label_report.json').read_text())
    check=json.loads((ROOT/'label_verification.json').read_text())
    assert check['passed'] and check['label_report_sha256']==sha(ROOT/'label_report.json')
    assert sha(ROOT/'labels.parquet')==label_report['labels_sha256']
    portrait=json.loads((PORTRAIT/'analysis_report.json').read_text())
    assert sha(PORTRAIT/'cohort.parquet')==portrait['cohort_sha256']
    features=pd.read_parquet(PORTRAIT/'cohort.parquet',columns=['date','code','board',*FEATURES])
    ranks=features.groupby(['date','board'])[list(FEATURES)].rank(method='average',pct=True)
    ranks=pd.concat([features[['date','code']],ranks],axis=1)
    ranks.to_parquet(ROOT/'feature_ranks.parquet',index=False,compression='zstd')
    frame=pd.read_parquet(ROOT/'labels.parquet')
    frame['known']=frame.net_return15.notna();frame['unknown']=frame.label15.eq('unknown')
    frame['no_trade']=frame.label15.eq('no_trade');frame['winner']=frame.label15.eq('economic_winner')
    frame['loser']=frame.label15.eq('economic_loser');frame['positive']=frame.net_return15.gt(0)
    frame['reference_class']=np.select([~frame.original_known_label,frame.original_winner.fillna(False).astype(bool)],
        ['unknown','up5'],default='not_up5')
    cross=frame.groupby(['half','board','necessary_tradeable','reference_class','label15'],dropna=False).size().rename('rows').reset_index()
    cross.to_parquet(ROOT/'reference_cross.parquet',index=False,compression='zstd')
    eligible=frame.loc[frame.necessary_tradeable].copy()
    assert (eligible.known.astype(int)+eligible.unknown.astype(int)+eligible.no_trade.astype(int)).eq(1).all()
    baseline=daily_groups(eligible,['date','half','board']).sort_values(['board','date'])
    baseline.to_parquet(ROOT/'baseline_daily.parquet',index=False,compression='zstd')
    summaries=[];distributions=[]
    for board in BOARDS:
        for half in HALVES:
            p=baseline.loc[baseline.board.eq(board)&baseline.half.eq(half)]
            summaries.append(summarize(p,dict(board=board,half=half)))
            rows=eligible.loc[eligible.board.eq(board)&eligible.half.eq(half)]
            for bps in [5,15]:
                values=rows[f'net_return{bps}'].dropna();wins=values.loc[values.gt(0)];losses=values.loc[values.lt(0)]
                cost_daily=rows.groupby('date')[f'net_return{bps}'].mean().sort_index()
                distributions.append(dict(board=board,half=half,cost_bps=bps,valid_rows=len(values),
                    unknown_rows=int(rows[f'label{bps}'].eq('unknown').sum()),no_trade=int(rows[f'label{bps}'].eq('no_trade').sum()),
                    mean_signal_day=number(cost_daily.mean()),mean_signal_day_week_interval=weekly_interval(cost_daily),
                    positive_frequency=number(values.gt(0).mean()),median=number(values.median()),
                    mean_positive=number(wins.mean()),mean_negative=number(losses.mean()),
                    payoff_ratio=number(wins.mean()/(-losses.mean())) if len(losses) and len(wins) else None,
                    worst_five_percent_mean=number(values.nsmallest(max(1,math.ceil(len(values)*.05))).mean()),minimum=number(values.min())))
    inverse=[];tables=[];bands=[]
    # Every band is formed from the entire visible input pool, before outcome masks.
    rank_index=ranks.set_index(['date','code'])
    simple=['date','code','half','board','known','unknown','no_trade','winner','loser','positive','net_return15','reference_class']
    for feature,title in FEATURES.items():
        p=eligible[simple].copy()
        p['rank']=rank_index[feature].reindex(pd.MultiIndex.from_frame(p[['date','code']])).to_numpy()
        p['band']=np.select([p['rank'].isna(),p['rank'].le(.2),p['rank'].ge(.8)],['missing','low20','high20'],default='middle60')
        daily=daily_groups(p,['date','half','board','band'])
        daily=daily.merge(baseline[['date','board',*METRICS]],on=['date','board'],how='left',validate='many_to_one',suffixes=('','_baseline'))
        for name in ['winner','loser','positive']:
            # Conservative difference bounds; missing band and baseline returns need not agree.
            daily[name+'_lower_delta']=daily[name+'_lower']-daily[name+'_upper_baseline']
            daily[name+'_upper_delta']=daily[name+'_upper']-daily[name+'_lower_baseline']
        daily['net_mean_delta']=daily.net_mean-daily.net_mean_baseline
        daily['feature']=feature;tables.append(daily)
        for board in BOARDS:
            for half in HALVES:
                for band in ['low20','middle60','high20','missing']:
                    g=daily.loc[daily.board.eq(board)&daily.half.eq(half)&daily.band.eq(band)]
                    bands.append(summarize(g,dict(feature=feature,title=title,board=board,half=half,band=band)))
        # Compare input ranks in the former daily-gain and new cash-based classes.
        classes={'reference_up5':p.reference_class.eq('up5'),'economic_winner':p.winner,
                 'economic_loser':p.loser,'middle':p.known&~p.winner&~p.loser,'unknown':p.unknown,'no_trade':p.no_trade}
        for kind,mask in classes.items():
            for (board,half),g in p.loc[mask].groupby(['board','half']):
                means=g.groupby('date')['rank'].mean().sort_index()
                inverse.append(dict(feature=feature,title=title,kind=kind,board=board,half=half,rows=len(g),
                    feature_missing=int(g['rank'].isna().sum()),dates=len(means),mean_rank=number(means.mean()),
                    mean_rank_week_interval=weekly_interval(means)))
        print(json.dumps({'feature':feature,'completed':len(tables),'total':len(FEATURES)}),flush=True)
    pd.concat(tables,ignore_index=True).to_parquet(ROOT/'feature_daily.parquet',index=False,compression='zstd')
    result={'interpretation':'catalogue_conditional_strict_T1_cash_scenarios_not_complete_real_portfolio_or_investor_identity',
        'label_report_sha256':sha(ROOT/'label_report.json'),'label_verification_sha256':sha(ROOT/'label_verification.json'),
        'source_cohort_sha256':sha(PORTRAIT/'cohort.parquet'),'baselines':summaries,'distributions':distributions,
        'feature_bands':bands,'inverse_ranks':inverse,
        'outputs_sha256':{name:sha(ROOT/(name+'.parquet')) for name in ['feature_ranks','reference_cross','baseline_daily','feature_daily']},
        'new_2026_prices_read':False}
    save_json(ROOT/'analysis_report.json',result)
    return {'baselines':summaries,'distributions':distributions,'features':len(FEATURES)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    print(json.dumps(evaluate(parser.parse_args().root),ensure_ascii=False,indent=2))
