"""Complete reverse portrait and fixed two-exit opening-cash study."""
import argparse
import json
from pathlib import Path
import shutil

import duckdb
import pandas as pd

from .corporate_cash import save_json, sha
from .economic_winner_analysis import number
from .opening_cash_history import ROOT, PROTOCOL, GROUPS
from .tick_flow_winner_analysis import aggregate, in_period, summary, PERIODS, LABELS

CONTRACT=Path('config/opening_cash_history_analysis_contract.json')
STRATEGY=ROOT/'strategy'
QUALITIES=['original_labels','input_quality_sensitivity']


def verified_inputs():
    report=json.loads((ROOT/'input_report.json').read_text())
    gate=json.loads((ROOT/'input_verification.json').read_text())
    assert gate['passed'] and gate['input_report_sha256']==sha(ROOT/'input_report.json')
    assert report['protocol_sha256']==sha(PROTOCOL)
    for name,h in report['output_sha256'].items():
        assert sha(ROOT/name)==h
    return report


def strategy_inputs():
    inputs=verified_inputs()
    if (STRATEGY/'input_report.json').exists():
        raise ValueError('Do not replace frozen strategy inputs')
    STRATEGY.mkdir(exist_ok=True)
    shutil.copyfile(ROOT/'strategy_features.parquet',STRATEGY/'features.parquet')
    shutil.copyfile(ROOT/'primary_pairs.parquet',STRATEGY/'primary_pairs.parquet')
    report=dict(protocol_sha256=sha(PROTOCOL),contract_sha256=sha(CONTRACT),
        parent_input_report_sha256=sha(ROOT/'input_report.json'),
        parent_input_verification_sha256=sha(ROOT/'input_verification.json'),
        rows=inputs['strategy_rows'],primary=inputs['primary'],paired=inputs['paired'],
        output_sha256={n:sha(STRATEGY/n) for n in ['features.parquet','primary_pairs.parquet']},
        outcomes_read=False,new_2026_prices_read=False)
    save_json(STRATEGY/'input_report.json',report)
    return report


def exits():
    from .touch_sequence import prepare_exits
    return prepare_exits(ROOT=STRATEGY,PROTOCOL=PROTOCOL)


def raw():
    from .tick_morning_exit import raw as extract
    return extract(ROOT=STRATEGY/'morning',PROTOCOL=PROTOCOL)


def labels():
    from .tick_morning_exit import labels as classify
    return classify(ROOT=STRATEGY/'morning')


def analyze():
    from .touch_sequence_analysis import analyze as evaluate
    return evaluate(ROOT=STRATEGY,GROUPS=['high_held'],event_column='event',
        interpretation='Frozen historical opening-cash anomaly and unchanged nearest controls; 2024/2025 exploratory, independent orders, no investor identity or 2026 prices')


def portrait():
    if (ROOT/'portrait_report.json').exists():
        raise ValueError('Do not replace inspected opening-cash portraits')
    inputs=verified_inputs()
    proof=json.loads((LABELS/'analysis_verification.json').read_text())
    assert proof['passed'] and proof['analysis_report_sha256']==sha(LABELS/'analysis_report.json')
    assert sha(LABELS/'labels.parquet')==json.loads((LABELS/'label_report.json').read_text())['labels_sha256']
    features=pd.read_parquet(ROOT/'features.parquet',columns=['date','code','half','group','primary','source_valid','decision_shares'])
    c=duckdb.connect()
    c.register('keys',features[['date','code']])
    labels=c.execute('''SELECT l.date,l.code,l.necessary_tradeable,l.decision_shares,l.label5,l.label15,
        l.net_return5,l.net_return15 FROM read_parquet(?) l JOIN keys USING(date,code)''',
        [str(LABELS/'labels.parquet')]).df()
    f=features.merge(labels,on=['date','code'],validate='one_to_one',suffixes=('','_label'))
    assert len(f)==inputs['rows'] and f.necessary_tradeable.all()
    assert f.decision_shares.eq(f.decision_shares_label).all()
    f.to_parquet(ROOT/'portrait_joined.parquet',index=False,compression='zstd')
    groups,reverse,day_tables,parts=[],[],[],{}
    for quality in QUALITIES:
        for cost in [5,15]:
            p=f[['date','code','half','group','primary','source_valid']].copy()
            p['original_label']=f[f'label{cost}']
            p['original_net_return']=f[f'net_return{cost}']
            p['label']=p.original_label
            p['net_return']=p.original_net_return
            if quality=='input_quality_sensitivity':
                p['label']=p.label.where(p.source_valid | p.label.eq('no_trade'),'unknown')
                p['net_return']=p.net_return.where(p.source_valid)
            p['known']=p.net_return.notna()
            p['unknown']=p.label.eq('unknown')
            p['no_trade']=p.label.eq('no_trade')
            p['winner']=p.net_return.ge(.01)
            p['loser']=p.net_return.le(-.01)
            p['positive']=p.net_return.gt(0)
            assert (p.known.astype(int)+p.unknown.astype(int)+p.no_trade.astype(int)).eq(1).all()
            p['quality']=quality
            p['cost_bps']=cost
            path=ROOT/f'portrait_{quality}_{cost}.parquet'
            p.to_parquet(path,index=False,compression='zstd')
            parts[path.name]=sha(path)
            for group in ['all',*GROUPS]:
                rows=p if group=='all' else p.loc[p.group.eq(group)]
                day=aggregate(rows,['date','half'])
                day['quality']=quality
                day['cost_bps']=cost
                day['group']=group
                day_tables.append(day)
                for period in PERIODS:
                    groups.append(summary(in_period(day,period),in_period(rows,period),
                        dict(quality=quality,cost_bps=cost,group=group,period=period)))
            if cost==15:
                masks={label:p.label.eq(label) for label in ['economic_winner','economic_loser','middle','unknown','no_trade']}
                masks.update(big_winner=p.net_return.ge(.03),big_loser=p.net_return.le(-.03))
                for period in PERIODS:
                    for label,mask in masks.items():
                        rows=in_period(p.loc[mask],period)
                        counts=rows.groupby(['date','group']).size().unstack(fill_value=0).reindex(columns=GROUPS,fill_value=0)
                        fractions=counts.div(counts.sum(axis=1),axis=0)
                        reverse.append(dict(quality=quality,period=period,label=label,rows=len(rows),dates=len(counts),
                            counts=rows.group.value_counts().to_dict(),
                            mean_daily_group_fraction={k:number(v) for k,v in fractions.mean().items()}))
            print(json.dumps(dict(quality=quality,cost_bps=cost,groups=len(groups))),flush=True)
    pd.concat(day_tables,ignore_index=True).to_parquet(ROOT/'portrait_daily.parquet',index=False,compression='zstd')
    parts.update({name:sha(ROOT/name) for name in ['portrait_joined.parquet','portrait_daily.parquet']})
    result=dict(input_report_sha256=sha(ROOT/'input_report.json'),input_verification_sha256=sha(ROOT/'input_verification.json'),
        contract_sha256=sha(CONTRACT),label_report_sha256=sha(LABELS/'label_report.json'),
        labels_sha256=sha(LABELS/'labels.parquet'),rows=len(f),groups=groups,reverse=reverse,output_sha256=parts,
        interpretation='Complete retrospective composition keeps unknown features separate from unknown profits; sensitivity also treats unusable signal histories as unknown. No other group promoted or morning sample changed.',
        new_2026_prices_read=False)
    save_json(ROOT/'portrait_report.json',result)
    return dict(rows=len(f),groups=len(groups),reverse=len(reverse))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['strategy_inputs','exits','raw','labels','portrait','analyze'])
    print(json.dumps(globals()[p.parse_args().stage](),ensure_ascii=False,indent=2))
