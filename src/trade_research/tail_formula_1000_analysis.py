"""Learn a native condition from H1 only; evaluate price opportunities, not exits."""
import argparse
import json
import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from .corporate_cash import save_json, sha
from .reference_gain_accounting import weekly_interval
from .tail_formula_1000 import ROOT, PROTOCOL, OLD, EXPRESSIONS

BUY_COLUMNS=['date','code','half','board','necessary_tradeable','decision_shares','price_1449',
    'preclose','upper_limit','next_date','day_close','next_preclose','next_trade_status','next_isST','next_adjustflag',
    'entry_bars','entry_labels','entry_source_valid','entry_bounds_valid','entry_volume','entry_vwap','entry_low','entry_high',
    'entry_fill_status','entry_recorded','entry_queue_unknown','catalog_covered','action_exposure',
    'period_entry_bad_day','period_exit_bad_day','period_bad_symbol']
OBS_COLUMNS=['date','code','next_date','source_valid','active_minutes','max_close','sustained_close','min_low','price_1000']
PERIODS=['2024H1','2024H2','2025H1','2025H2','2024','2025','2024H2_2025']


def number(x):
    return float(x) if pd.notna(x) and np.isfinite(x) else None


def gates():
    for name,parent,key in [('feature_verification.json','feature_report.json','feature_report_sha256'),
                            ('observation_verification.json','observation_report.json','observation_report_sha256')]:
        gate=json.loads((ROOT/name).read_text())
        assert gate['passed'] and gate[key]==sha(ROOT/parent)
    report=json.loads((ROOT/'feature_report.json').read_text())
    assert report['protocol_sha256']==sha(PROTOCOL)
    assert report['output_sha256']['features.parquet']==sha(ROOT/'features.parquet')
    assert json.loads((ROOT/'observation_report.json').read_text())['observations_sha256']==sha(ROOT/'observations.parquet')
    assert json.loads((OLD/'label_report.json').read_text())['labels_sha256']==sha(OLD/'labels.parquet')


def observations_for(scope):
    gates()
    c=duckdb.connect()
    c.execute('SET threads=4')
    c.execute("SET memory_limit='4GB'")
    cutoff="WHERE next_date<'2024-07-01'" if scope=='training' else ''
    # Projection deliberately omits every old exit price, exit quality, and P&L.
    old=c.execute(f"SELECT {','.join(BUY_COLUMNS)} FROM read_parquet(?) {cutoff}",[str(OLD/'labels.parquet')]).df()
    obs=c.execute(f"SELECT {','.join(OBS_COLUMNS)} FROM read_parquet(?) {cutoff}",[str(ROOT/'observations.parquet')]).df()
    c.close()
    return obs.merge(old,on=['date','code','next_date'],validate='one_to_one')


def classify(r):
    r=r.copy()
    valid_ref=np.isfinite(r.next_preclose)&r.next_preclose.gt(0)&(r.next_preclose-r.next_preclose.round(2)).abs().le(.0001)
    valid_close=np.isfinite(r.day_close)&r.day_close.gt(0)&(r.day_close-r.day_close.round(2)).abs().le(.0001)
    r['corporate_unknown']=~r.catalog_covered|r.action_exposure|~valid_ref|~valid_close|(r.next_preclose-r.day_close).abs().gt(.005)
    r['next_daily_valid']=valid_ref&r.next_trade_status.eq(1)&r.next_isST.isin([0,1])&r.next_adjustflag.eq(3)
    r['entry_source_unknown']=~r.entry_source_valid|r.period_entry_bad_day|r.period_bad_symbol
    r['known_no_trade']=~r.entry_source_unknown&~r.entry_recorded
    r['observation_status']=np.select([r.entry_source_unknown,r.known_no_trade,r.entry_queue_unknown,
        r.corporate_unknown,~r.next_daily_valid,~r.source_valid],
        ['entry_source_unknown','no_trade','entry_queue_unknown','corporate_unknown','next_daily_unknown','morning_source_unknown'],
        default='known')
    for bps in [5,15]:
        buy_price=r.entry_vwap+np.maximum(r.entry_vwap*bps/10000,.005)
        value=r.decision_shares*buy_price
        cash=value+np.maximum(5,value*.0003)+value*.00001
        r[f'buy_cash{bps}']=cash
        r[f'entry_stress_unknown{bps}']=r.entry_recorded&buy_price.ge(r.upper_limit-.005)
        known=r.observation_status.eq('known')&~r[f'entry_stress_unknown{bps}']
        r[f'known{bps}']=known
        for name,price in [('sustained',r.sustained_close),('any_close',r.max_close),('mark_1000',r.price_1000),('adverse',r.min_low)]:
            sell=price-np.maximum(price*bps/10000,.005)
            v=r.decision_shares*sell
            mark=(v-np.maximum(5,v*.0003)-v*.00051)/cash-1
            r[f'{name}_return{bps}']=mark.where(known)
        r[f'opportunity{bps}']=r[f'sustained_return{bps}'].gt(0).astype(float).where(known)
        r[f'any_opportunity{bps}']=r[f'any_close_return{bps}'].gt(0).astype(float).where(known)
        r[f'one_percent{bps}']=r[f'sustained_return{bps}'].ge(.01).astype(float).where(known)
        r[f'unknown{bps}']=~known&~r.known_no_trade
        # A whole-day source warning remains available as a conservative sensitivity.
        # It is never replaced by old tail-window exit validity or fill status.
        r[f'sensitive_known{bps}']=known&~r.period_exit_bad_day
    return r


def labels(scope):
    report_path=ROOT/f'{scope}_label_report.json'
    if report_path.exists():
        raise ValueError('Do not replace opportunity labels')
    if scope=='full':
        proof=json.loads((ROOT/'selection_verification.json').read_text())
        assert proof['passed'] and proof['selection_report_sha256']==sha(ROOT/'selection_report.json')
    r=classify(observations_for(scope)).sort_values(['date','code']).reset_index(drop=True)
    if scope=='training':
        assert r.next_date.lt('2024-07-01').all()
    r.to_parquet(ROOT/f'{scope}_labels.parquet',index=False,compression='zstd')
    report=dict(scope=scope,rows=len(r),first_signal=r.date.min(),last_signal=r.date.max(),last_observation=r.next_date.max(),
        protocol_sha256=sha(PROTOCOL),feature_report_sha256=sha(ROOT/'feature_report.json'),
        observation_verification_sha256=sha(ROOT/'observation_verification.json'),
        old_labels_sha256=sha(OLD/'labels.parquet'),labels_sha256=sha(ROOT/f'{scope}_labels.parquet'),
        old_columns=BUY_COLUMNS,no_old_exit_fields=True,no_exit_rules=True,new_2026_prices_read=False)
    if scope=='full':
        report['selection_report_sha256']=sha(ROOT/'selection_report.json')
    save_json(report_path,report)
    return report


def fit_tree(train):
    names=list(EXPRESSIONS)
    known=train.loc[train.known15&train.formula_input_valid].sort_values(['date','code']).copy()
    weights=1/known.groupby('date').code.transform('size')
    tree=DecisionTreeClassifier(max_depth=3,min_samples_leaf=2000,criterion='gini',random_state=20260927)
    tree.fit(known[names],known.opportunity15.astype(int),sample_weight=weights)
    leaves=tree.apply(known[names])
    candidates=[]
    for leaf in np.unique(leaves):
        mask=leaves==leaf
        candidates.append(dict(node=int(leaf),rows=int(mask.sum()),
            weighted_probability=float(np.average(known.loc[mask,'opportunity15'],weights=weights[mask]))))
    chosen=sorted(candidates,key=lambda x:(-x['weighted_probability'],-x['rows'],x['node']))[0]
    paths={0:[]}
    for i in range(tree.tree_.node_count):
        left=tree.tree_.children_left[i]
        if left<0:
            continue
        feature=names[tree.tree_.feature[i]]
        threshold=float(tree.tree_.threshold[i])
        paths[left]=paths[i]+[dict(feature=feature,op='<=',raw_threshold=threshold,threshold=math.floor(threshold*100)/100)]
        right=tree.tree_.children_right[i]
        paths[right]=paths[i]+[dict(feature=feature,op='>',raw_threshold=threshold,threshold=math.ceil(threshold*100)/100)]
    return tree,known,chosen,candidates,paths[chosen['node']]


def select(f,conditions):
    match=f.formula_input_valid.copy()
    for condition in conditions:
        x=f[condition['feature']]
        match &= x.le(condition['threshold']) if condition['op']=='<=' else x.gt(condition['threshold'])
    return match


def freeze():
    if (ROOT/'selection_report.json').exists():
        raise ValueError('Do not refit frozen native formula')
    if (ROOT/'full_labels.parquet').exists():
        raise ValueError('Evaluation labels must follow frozen selection')
    gate=json.loads((ROOT/'training_label_verification.json').read_text())
    assert gate['passed'] and gate['label_report_sha256']==sha(ROOT/'training_label_report.json')
    f=pd.read_parquet(ROOT/'features.parquet')
    train=pd.read_parquet(ROOT/'training_labels.parquet').merge(f[['date','code','formula_input_valid',*EXPRESSIONS]],
        on=['date','code'],validate='one_to_one')
    tree,known,chosen,leaves,conditions=fit_tree(train)
    selection=f[['date','code','half','board','decision_shares']].copy()
    selection['selected']=select(f,conditions)
    selection.to_parquet(ROOT/'selection.parquet',index=False,compression='zstd')
    used=list(dict.fromkeys(c['feature'] for c in conditions))
    expression=' AND '.join(f"{c['feature']}{c['op']}{c['threshold']:.2f}" for c in conditions)
    # This records the numeric core, not a claim of native compiler verification.
    core='\n'.join(f'{name}:={EXPRESSIONS[name]};' for name in used)+'\nCORE:'+expression+';\n'
    (ROOT/'frozen_numeric_core.tdx').write_text(core)
    report=dict(protocol_sha256=sha(PROTOCOL),training_label_report_sha256=sha(ROOT/'training_label_report.json'),
        feature_report_sha256=sha(ROOT/'feature_report.json'),training_rows=len(known),training_days=known.date.nunique(),
        training_last_signal=known.date.max(),training_last_observation=known.next_date.max(),chosen_leaf=chosen,leaves=leaves,
        conditions=conditions,expressions={x:EXPRESSIONS[x] for x in used},numeric_core=core,
        tree=dict(feature=tree.tree_.feature.tolist(),threshold=tree.tree_.threshold.tolist(),
            children_left=tree.tree_.children_left.tolist(),children_right=tree.tree_.children_right.tolist(),
            n_node_samples=tree.tree_.n_node_samples.tolist()),
        selection_sha256=sha(ROOT/'selection.parquet'),core_sha256=sha(ROOT/'frozen_numeric_core.tdx'),
        selected=int(selection.selected.sum()),by_half=selection.groupby('half').selected.agg(['size','sum']).reset_index().to_dict('records'),
        evaluation_labels_read=False,software_compilation_verified=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'selection_report.json',report)
    return {k:v for k,v in report.items() if k not in ['tree','leaves']}


def period(frame,p):
    if p=='2024H2_2025':
        return frame.loc[frame.date.ge('2024-07-01')]
    return frame.loc[frame.half.eq(p)] if 'H' in p else frame.loc[frame.date.str.startswith(p)]


def daily_summary(frame,bps,sensitive=False):
    r=frame.copy()
    known=r[f'sensitive_known{bps}'] if sensitive else r[f'known{bps}']
    r['known']=known.astype(int)
    r['success']=r[f'opportunity{bps}'].eq(1)&known
    r['unknown']=~known&~r.known_no_trade
    r['one_percent']=r[f'one_percent{bps}'].eq(1)&known
    r['any_success']=r[f'any_opportunity{bps}'].eq(1)&known
    r['mark1000']=r[f'mark_1000_return{bps}'].where(known)
    r['adverse']=r[f'adverse_return{bps}'].where(known)
    r['negative1000']=r.mark1000.lt(0).astype(float).where(r.mark1000.notna())
    r['bad3']=r.adverse.le(-.03).astype(float).where(r.adverse.notna())
    g=r.groupby('date',sort=True)
    d=g.agg(rows=('code','size'),known=('known','sum'),success=('success','sum'),unknown=('unknown','sum'),
        no_trade=('known_no_trade','sum'),one_percent=('one_percent','sum'),any_success=('any_success','sum'),
        mean1000=('mark1000','mean'),negative1000=('negative1000','mean'),adverse_mean=('adverse','mean'),bad3=('bad3','mean'))
    d['rate']=d.success/d.known.replace(0,np.nan)
    d['lower']=d.success/d.rows
    d['upper']=(d.success+d.unknown)/d.rows
    d['one_percent_rate']=d.one_percent/d.known.replace(0,np.nan)
    d['any_rate']=d.any_success/d.known.replace(0,np.nan)
    return d


def analyze():
    if (ROOT/'analysis_report.json').exists():
        raise ValueError('Do not replace analysis')
    check=json.loads((ROOT/'full_label_verification.json').read_text())
    assert check['passed'] and check['label_report_sha256']==sha(ROOT/'full_label_report.json')
    selection=json.loads((ROOT/'selection_report.json').read_text())
    assert selection['selection_sha256']==sha(ROOT/'selection.parquet')
    labels=pd.read_parquet(ROOT/'full_labels.parquet')
    chosen=pd.read_parquet(ROOT/'selection.parquet',columns=['date','code','selected'])
    r=labels.merge(chosen,on=['date','code'],validate='one_to_one')
    summaries=[]
    daily=[]
    for bps in [5,15]:
        for sensitive in [False,True]:
            candidate=r.loc[r.selected]
            selected_dates=candidate.date.unique()
            base=r.loc[r.date.isin(selected_dates)]
            a=daily_summary(candidate,bps,sensitive)
            b=daily_summary(base,bps,sensitive)
            assert a.index.equals(b.index)
            for arm,df in [('formula',a),('base_same_dates',b)]:
                d=df.reset_index().assign(half=lambda x:x.date.str[:4]+np.where(x.date.str[5:7].le('06'),'H1','H2'))
                d['arm']=arm;d['bps']=bps;d['sensitive']=sensitive
                daily.append(d)
                for p in PERIODS:
                    q=period(d,p).set_index('date')
                    values={name:number(q[name].mean()) for name in ['rate','lower','upper','one_percent_rate','any_rate',
                        'mean1000','negative1000','adverse_mean','bad3']}
                    summaries.append(dict(arm=arm,bps=bps,sensitive=sensitive,period=p,days=len(q),
                        rows=int(q.rows.sum()),known=int(q.known.sum()),success=int(q.success.sum()),unknown=int(q.unknown.sum()),
                        no_trade=int(q.no_trade.sum()),mean_selected_per_day=number(q.rows.mean()),
                        pooled_rate=number(q.success.sum()/q.known.sum()) if q.known.sum() else None,
                        **values,rate_ci=weekly_interval(q.rate),lower_ci=weekly_interval(q.lower),upper_ci=weekly_interval(q.upper)))
            delta=pd.DataFrame({'date':a.index,'rate_delta':(a.rate-b.rate).values,
                'lower_delta':(a.lower-b.upper).values,'upper_delta':(a.upper-b.lower).values,
                'mark1000_delta':(a.mean1000-b.mean1000).values})
            delta['half']=delta.date.str[:4]+np.where(delta.date.str[5:7].le('06'),'H1','H2')
            for p in PERIODS:
                q=period(delta,p).set_index('date')
                summaries.append(dict(arm='same_day_difference',bps=bps,sensitive=sensitive,period=p,days=len(q),
                    **{k:number(q[k].mean()) for k in ['rate_delta','lower_delta','upper_delta','mark1000_delta']},
                    rate_delta_ci=weekly_interval(q.rate_delta),lower_delta_ci=weekly_interval(q.lower_delta),
                    upper_delta_ci=weekly_interval(q.upper_delta)))
    pd.concat(daily,ignore_index=True).to_parquet(ROOT/'daily_summary.parquet',index=False,compression='zstd')
    report=dict(protocol_sha256=sha(PROTOCOL),selection_report_sha256=sha(ROOT/'selection_report.json'),
        full_label_verification_sha256=sha(ROOT/'full_label_verification.json'),summaries=summaries,
        daily_summary_sha256=sha(ROOT/'daily_summary.parquet'),no_exit_rules=True,new_2026_prices_read=False,
        opportunity_is_not_realized_profit=True,year_2025_is_exploratory=True,software_compilation_verified=False)
    save_json(ROOT/'analysis_report.json',report)
    return {k:v for k,v in report.items() if k!='summaries'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['training','freeze','full','analyze'])
    stage=parser.parse_args().stage
    result=labels(stage) if stage in ['training','full'] else globals()[stage]()
    print(json.dumps(result,ensure_ascii=False,indent=2))
