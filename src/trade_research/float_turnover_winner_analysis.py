"""Compare every fixed floating-size/turnover group against conditional T1 cash."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .corporate_cash import save_json,sha
from .economic_winner_analysis import daily_groups,summarize,number,METRICS,HALVES,BOARDS
from .float_turnover_winner import ROOT,FEATURES
from .reference_gain_accounting import weekly_interval

LABELS=Path('data/research/economic_winner/period_quality')


def aggregate(p,keys):
    result=daily_groups(p,keys)
    cost5=p.groupby(keys,dropna=False).net_return5.mean().rename('net5').reset_index()
    return result.merge(cost5,on=keys,validate='one_to_one')


def summary(p,identifiers):
    row=summarize(p,identifiers)
    for field in ['net5','net5_delta']:
        if field in p:
            values=p.set_index('date')[field].sort_index()
            row[field]=number(values.mean());row[field+'_week_interval']=weekly_interval(values)
    return row


def evaluate():
    if (ROOT/'analysis_report.json').exists():raise ValueError('Do not replace size-turnover groups')
    check=json.loads((ROOT/'input_verification.json').read_text())
    assert check['passed'] and check['input_report_sha256']==sha(ROOT/'input_report.json')
    inputs=json.loads((ROOT/'input_report.json').read_text())
    assert inputs['features_sha256']==sha(ROOT/'features.parquet')
    verified=json.loads((LABELS/'analysis_verification.json').read_text())
    assert verified['passed'] and verified['analysis_report_sha256']==sha(LABELS/'analysis_report.json')
    assert json.loads((LABELS/'label_report.json').read_text())['labels_sha256']==sha(LABELS/'labels.parquet')
    columns=['date','code','half','board','necessary_tradeable',*[n+s for n in FEATURES for s in ['_rank','_quintile']]]
    frame=pd.read_parquet(ROOT/'features.parquet',columns=columns)
    labels=pd.read_parquet(LABELS/'labels.parquet',columns=['date','code','label15','net_return15','net_return5','original_known_label','original_winner'])
    frame=frame.merge(labels,on=['date','code'],validate='one_to_one')
    assert len(frame)==2404280
    frame=frame.loc[frame.necessary_tradeable].copy()
    frame['known']=frame.net_return15.notna();frame['unknown']=frame.label15.eq('unknown')
    frame['no_trade']=frame.label15.eq('no_trade');frame['winner']=frame.label15.eq('economic_winner')
    frame['loser']=frame.label15.eq('economic_loser');frame['positive']=frame.net_return15.gt(0)
    baseline=aggregate(frame,['date','half','board']).sort_values(['date','board'])
    baseline.to_parquet(ROOT/'baseline_daily.parquet',index=False,compression='zstd')
    baselines=[summary(p,dict(board=board,half=half)) for (board,half),p in baseline.groupby(['board','half'])]
    tables=[];groups=[];inverse=[]
    for name in [*FEATURES,'float_cap_x_tail_turnover']:
        p=frame.copy()
        if name in FEATURES:
            p['group']=p[name+'_quintile'].astype(str)
            possibilities=[str(i) for i in range(6)]
        else:
            p['group']=p.float_cap_proxy_quintile.astype(str)+':'+p.turnover_tail29_proxy_quintile.astype(str)
            possibilities=['0:0',*[f'{i}:{j}' for i in range(1,6) for j in range(1,6)]]
            assert p['group'].isin(possibilities).all()
        daily=aggregate(p,['date','half','board','group'])
        daily=daily.merge(baseline[['date','board',*METRICS,'net5']],on=['date','board'],validate='many_to_one',suffixes=('','_baseline'))
        for kind in ['winner','loser','positive']:
            daily[kind+'_lower_delta']=daily[kind+'_lower']-daily[kind+'_upper_baseline']
            daily[kind+'_upper_delta']=daily[kind+'_upper']-daily[kind+'_lower_baseline']
        daily['net_mean_delta']=daily.net_mean-daily.net_mean_baseline
        daily['net5_delta']=daily.net5-daily.net5_baseline
        daily['stratum']=name;tables.append(daily)
        for board in BOARDS:
            for half in HALVES:
                for group in possibilities:
                    part=daily.loc[daily.board.eq(board)&daily.half.eq(half)&daily['group'].eq(group)]
                    groups.append(summary(part,dict(stratum=name,board=board,half=half,group=group)))
        if name in FEATURES:
            classes={'reference_up5':p.original_known_label&p.original_winner.fillna(False).astype(bool),
                'economic_winner':p.winner,'economic_loser':p.loser,'middle':p.known&~p.winner&~p.loser,
                'unknown':p.unknown,'no_trade':p.no_trade}
            for kind,mask in classes.items():
                for (board,half),part in p.loc[mask].groupby(['board','half']):
                    values=part.groupby('date')[name+'_rank'].mean().sort_index()
                    inverse.append({'feature':name,'kind':kind,'board':board,'half':half,'rows':len(part),
                        'missing':int(part[name+'_rank'].isna().sum()),'dates':len(values),'mean_rank':number(values.mean()),
                        'mean_rank_week_interval':weekly_interval(values)})
        print(json.dumps({'stratum':name,'completed':len(tables),'total':4}),flush=True)
    pd.concat(tables,ignore_index=True).to_parquet(ROOT/'groups_daily.parquet',index=False,compression='zstd')
    report={'interpretation':'lagged_denominator_proxy_exploration_not_actual_free_float_or_investor_flow',
        'input_report_sha256':sha(ROOT/'input_report.json'),'input_verification_sha256':sha(ROOT/'input_verification.json'),
        'label_report_sha256':sha(LABELS/'label_report.json'),'label_analysis_verification_sha256':sha(LABELS/'analysis_verification.json'),
        'baselines':baselines,'groups':groups,'inverse_ranks':inverse,
        'outputs_sha256':{name:sha(ROOT/(name+'.parquet')) for name in ['baseline_daily','groups_daily']},
        'new_2026_prices_read':False,'new_strategy_selected':False}
    save_json(ROOT/'analysis_report.json',report)
    return {'baselines':baselines,'strata':4,'group_summaries':len(groups),'inverse_summaries':len(inverse)}


if __name__=='__main__':print(json.dumps(evaluate(),ensure_ascii=False,indent=2))
