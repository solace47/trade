"""Report the whole external conjunction and its prespecified component groups."""
import json
from pathlib import Path

import pandas as pd

from .corporate_cash import save_json,sha
from .economic_winner_analysis import daily_groups,summarize,number,METRICS
from .seat_repeat_winner_analysis import costs,distribution,period,PERIODS
from .reference_gain_accounting import weekly_interval
from .external_tail_combo import ROOT

LABELS=Path('data/research/economic_winner/period_quality')
SCOPES=['base','recent_only','vwap_only','0:0','0:1','1:0','1:1','detail_unknown','base_unknown','outside_base']


def evaluate():
    if (ROOT/'analysis_report.json').exists():raise ValueError('Do not replace external-conjunction outcomes')
    check=json.loads((ROOT/'input_verification.json').read_text())
    assert check['passed'] and check['input_report_sha256']==sha(ROOT/'input_report.json')
    inputs=json.loads((ROOT/'input_report.json').read_text())
    for name,digest in inputs['outputs_sha256'].items():assert sha(ROOT/(name+'.parquet'))==digest
    checked=json.loads((LABELS/'analysis_verification.json').read_text())
    assert checked['passed'] and checked['analysis_report_sha256']==sha(LABELS/'analysis_report.json')
    assert json.loads((LABELS/'label_report.json').read_text())['labels_sha256']==sha(LABELS/'labels.parquet')
    frame=pd.read_parquet(ROOT/'pool.parquet');frame=frame.loc[frame.necessary_tradeable].copy()
    labels=pd.read_parquet(LABELS/'labels.parquet',columns=['date','code','label5','label15','net_return5','net_return15','base_status','original_known_label','original_winner'])
    frame=frame.merge(labels,on=['date','code'],how='left',validate='one_to_one')
    assert len(frame)==inputs['necessary_rows'] and frame.label15.notna().all()
    frame.to_parquet(ROOT/'outcomes.parquet',index=False,compression='zstd')
    long=costs(frame)
    market=daily_groups(long,['date','half','cost_bps'])
    base=daily_groups(long.loc[long.base_combo.fillna(False)],['date','half','cost_bps'])
    market.to_parquet(ROOT/'market_daily.parquet',index=False,compression='zstd')
    base.to_parquet(ROOT/'base_daily.parquet',index=False,compression='zstd')
    definitions={name:long['group'].eq(name) for name in SCOPES if name not in ['base','recent_only','vwap_only']}
    definitions['base']=long.base_combo.fillna(False)
    definitions['recent_only']=long.base_combo.fillna(False)&long.recent_activity.fillna(False)
    definitions['vwap_only']=long.base_combo.fillna(False)&long.vwap_support.fillna(False)
    summaries=[];distributions=[];tables=[]
    for name in SCOPES:
        part=long.loc[definitions[name]]
        d=daily_groups(part,['date','half','cost_bps'])
        for tag,background in [('base',base),('market',market)]:
            renamed=background[['date','cost_bps',*METRICS]].rename(columns={m:m+'_'+tag for m in METRICS})
            d=d.merge(renamed,on=['date','cost_bps'],how='left',validate='many_to_one')
            for field in ['winner','loser','positive']:
                d[field+'_lower_'+tag+'_delta']=d[field+'_lower']-d[field+'_upper_'+tag]
                d[field+'_upper_'+tag+'_delta']=d[field+'_upper']-d[field+'_lower_'+tag]
            d['net_mean_'+tag+'_delta']=d.net_mean-d['net_mean_'+tag]
        d['scope']=name;tables.append(d)
        for bps in [5,15]:
            for time in PERIODS:
                daily=period(d.loc[d.cost_bps.eq(bps)],time);ids=dict(scope=name,period=time,cost_bps=bps)
                item=summarize(daily,ids)
                for tag in ['base','market']:
                    for field in METRICS:
                        col=field+'_'+tag+'_delta';values=daily.set_index('date')[col].sort_index()
                        item[col]=number(values.mean());item[col+'_week_interval']=weekly_interval(values)
                summaries.append(item)
                distributions.append(distribution(period(part.loc[part.cost_bps.eq(bps)],time),ids))
    pd.concat(tables,ignore_index=True).to_parquet(ROOT/'groups_daily.parquet',index=False,compression='zstd')
    baselines=[summarize(period(market.loc[market.cost_bps.eq(bps)],time),dict(period=time,cost_bps=bps)) for bps in [5,15] for time in PERIODS]
    cross=frame.groupby(['half','group','label15','original_known_label','original_winner'],dropna=False).size().rename('n').reset_index()
    cross.to_parquet(ROOT/'label_cross.parquet',index=False,compression='zstd')
    report={'interpretation':'historically_visible_adaptation_not_original_project_backtest_or_investor_identity',
        'input_report_sha256':sha(ROOT/'input_report.json'),'input_verification_sha256':sha(ROOT/'input_verification.json'),
        'label_report_sha256':sha(LABELS/'label_report.json'),'label_analysis_verification_sha256':sha(LABELS/'analysis_verification.json'),
        'market':baselines,'groups':summaries,'distributions':distributions,
        'outputs_sha256':{name:sha(ROOT/(name+'.parquet')) for name in ['outcomes','market_daily','base_daily','groups_daily','label_cross']},
        'new_2026_prices_read':False,'new_strategy_selected':False}
    save_json(ROOT/'analysis_report.json',report)
    return {'groups':[x for x in summaries if x['scope'] in ['base','recent_only','vwap_only','1:1']],
        'primary_distributions':[x for x in distributions if x['scope']=='1:1']}


if __name__=='__main__':print(json.dumps(evaluate(),ensure_ascii=False,indent=2))
