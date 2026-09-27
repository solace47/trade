"""Discover a selective native rule in H1 and require absolute quality in H2."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from .corporate_cash import save_json, sha
from .reference_gain_accounting import weekly_interval
from .tail_formula_intraday import ROOT as FEATURE_ROOT, SOURCE, EXPRESSIONS, HEADER, conn
from .tail_formula_intraday_study import inputs as training_inputs, paths
from .tail_formula_1000_analysis import select, number
from .tail_formula_1000_daily import analyze as common_analysis

ROOT=Path('data/research/tail_formula_selective')
PROTOCOL=Path('config/tail_formula_selective_protocol.json')


def model(ROOT=ROOT,PROTOCOL=PROTOCOL,FEATURE_ROOT=FEATURE_ROOT,loader=training_inputs,expressions=EXPRESSIONS):
    if (ROOT/'model_report.json').exists():
        raise ValueError('Do not refit the selective formula model')
    ROOT.mkdir(parents=True,exist_ok=True)
    _,train,_,_=loader()
    weights=1/train.groupby('date').code.transform('size')
    reg=DecisionTreeClassifier(criterion='entropy',max_depth=7,min_samples_leaf=300,random_state=20260927)
    reg.fit(train[list(expressions)],train.opportunity15,sample_weight=weights)
    t=reg.tree_
    report=dict(protocol_sha256=sha(PROTOCOL),feature_report_sha256=sha(FEATURE_ROOT/'feature_report.json'),
        training_label_report_sha256=sha(SOURCE/'training_label_report.json'),rows=len(train),days=train.date.nunique(),
        last_observation=train.next_date.max(),parameters=reg.get_params(),classes=reg.classes_.tolist(),
        feature_names=list(expressions),tree=dict(feature=t.feature.tolist(),threshold=t.threshold.tolist(),
            children_left=t.children_left.tolist(),children_right=t.children_right.tolist(),
            n_node_samples=t.n_node_samples.tolist(),weighted_n_node_samples=t.weighted_n_node_samples.tolist(),
            value=t.value.reshape(-1,2).tolist(),impurity=t.impurity.tolist()),
        new_calibration_groups_read=False,new_2025_groups_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'model_report.json',report)
    return {k:v for k,v in report.items() if k!='tree'}


def verify_model(ROOT=ROOT,PROTOCOL=PROTOCOL,FEATURE_ROOT=FEATURE_ROOT,loader=training_inputs,expressions=EXPRESSIONS):
    r=json.loads((ROOT/'model_report.json').read_text())
    assert r['protocol_sha256']==sha(PROTOCOL)
    assert r['feature_report_sha256']==sha(FEATURE_ROOT/'feature_report.json')
    assert r['training_label_report_sha256']==sha(SOURCE/'training_label_report.json')
    assert r['classes']==[0.,1.]
    assert r['parameters']['max_depth']==7 and r['parameters']['min_samples_leaf']==300
    assert r['parameters']['criterion']=='entropy' and r['parameters']['random_state']==20260927
    _,train,_,_=loader()
    c=conn()
    c.register('training',train)
    e=c.sql('SELECT date,code,opportunity15 AS y,1./count(*) OVER(PARTITION BY date) AS w FROM training ORDER BY date,code').df()
    assert e[['date','code']].equals(train[['date','code']])
    assert r['feature_names']==list(expressions)
    x=train[list(expressions)].to_numpy(dtype='float32')
    t=r['tree']
    masks={0:np.ones(len(train),dtype=bool)}
    depth={0:0}
    for i in range(len(t['feature'])):
        m=masks[i]
        assert int(m.sum())==t['n_node_samples'][i]
        w=e.w[m]
        positive=float(np.average(e.y[m],weights=w))
        p=np.array([1-positive,positive])
        entropy=-np.sum(p[p>0]*np.log2(p[p>0]))
        np.testing.assert_allclose(w.sum(),t['weighted_n_node_samples'][i],atol=1e-9,rtol=0)
        np.testing.assert_allclose(p,t['value'][i],atol=2e-11,rtol=0)
        np.testing.assert_allclose(entropy,t['impurity'][i],atol=2e-11,rtol=0)
        left,right=t['children_left'][i],t['children_right'][i]
        if left<0:
            assert m.sum()>=300 and depth[i]<=7
            continue
        lower=x[:,t['feature'][i]]<=t['threshold'][i]
        masks[left],masks[right]=m&lower,m&~lower
        depth[left]=depth[right]=depth[i]+1
    result=dict(passed=True,model_report_sha256=sha(ROOT/'model_report.json'),rows=len(train),
        all_node_counts_probabilities_entropy_rebuilt=True,new_calibration_groups_read=False,
        new_2025_groups_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'model_verification.json',result)
    return result


def calibration_inputs(loader=training_inputs):
    f,_,_,_=loader()
    report=json.loads((SOURCE/'full_label_report.json').read_text())
    proof=json.loads((SOURCE/'full_label_verification.json').read_text())
    assert proof['passed'] and proof['label_report_sha256']==sha(SOURCE/'full_label_report.json')
    assert report['labels_sha256']==sha(SOURCE/'full_labels.parquet')
    c=conn()
    # Do not read 2025 labels to choose among the new leaf groups.
    l=c.execute('''SELECT date,code,next_date,known15,opportunity15,known_no_trade
        FROM read_parquet(?) WHERE next_date<'2025-01-01'
        AND (next_date<'2024-07-01' OR date>='2024-07-01')''',[str(SOURCE/'full_labels.parquet')]).df()
    assert l.next_date.lt('2025-01-01').all()
    l['phase']=np.where(l.next_date.lt('2024-07-01'),'discovery','calibration')
    r=f.merge(l,on=['date','code'],validate='one_to_one')
    base=l.groupby('date').opportunity15.mean()
    return f,r,base


def statistics(d):
    interval=weekly_interval(d.set_index('date').delta)
    result=dict(rows=int(d.rows.sum()),known=int(d.known.sum()),unknown=int(d.unknown.sum()),no_trade=int(d.no_trade.sum()),
        days=len(d),rate=number(d.rate.mean()),lower=number(d.lower.mean()),delta=number(d.delta.mean()),
        mean_selected_per_day=number(d.rows.mean()),p95_selected=number(d.rows.quantile(.95)),delta_ci=interval)
    flags=dict(enough_days=len(d)>=40,enough_known=d.known.sum()>=300,
        rate=d.rate.mean()>=.6,conservative_rate=d.lower.mean()>=.55,delta=d.delta.mean()>=.08,
        mean_count=d.rows.mean()<=20,p95_count=d.rows.quantile(.95)<=50,
        delta_ci_positive=interval is not None and interval[0]>0)
    result['gates']={k:bool(v) for k,v in flags.items()}
    result['eligible']=all(v for k,v in flags.items() if k!='delta_ci_positive')
    return result


def calibrate(ROOT=ROOT,PROTOCOL=PROTOCOL,FEATURE_ROOT=FEATURE_ROOT,loader=training_inputs,expressions=EXPRESSIONS,header=HEADER):
    if (ROOT/'calibration_report.json').exists():
        raise ValueError('Do not recalibrate the frozen selective formula')
    check=json.loads((ROOT/'model_verification.json').read_text())
    assert check['passed'] and check['model_report_sha256']==sha(ROOT/'model_report.json')
    tree=json.loads((ROOT/'model_report.json').read_text())['tree']
    f,r,base=calibration_inputs(loader)
    leaves=paths(tree,list(expressions))
    days=[]
    scores=[]
    for node,conditions in leaves.items():
        selected=r.loc[select(r,conditions)].copy()
        selected['unknown']=~selected.known15&~selected.known_no_trade
        d=selected.groupby(['phase','date']).agg(rows=('code','size'),known=('known15','sum'),
            success=('opportunity15','sum'),unknown=('unknown','sum'),no_trade=('known_no_trade','sum')).reset_index()
        d['rate']=d.success/d.known.replace(0,np.nan)
        d['lower']=d.success/d.rows
        d['base_rate']=d.date.map(base)
        d['delta']=d.rate-d.base_rate
        d['node']=node
        days.append(d)
        score=dict(node=node,conditions=conditions)
        for phase in ['discovery','calibration']:
            score[phase]=statistics(d.loc[d.phase.eq(phase)])
        score['eligible']=score['discovery']['eligible'] and score['calibration']['eligible'] and score['calibration']['gates']['delta_ci_positive']
        scores.append(score)
    eligible=[x for x in scores if x['eligible']]
    best=sorted(eligible,key=lambda x:(-x['calibration']['lower'],-x['calibration']['delta'],-x['calibration']['known'],x['node']))[0] if eligible else None
    daily=pd.concat(days,ignore_index=True)
    daily.to_parquet(ROOT/'calibration_leaf_days.parquet',index=False,compression='zstd')
    selected=f[['date','code','half','board','decision_shares']].copy()
    selected['selected']=select(f,best['conditions']) if best else False
    selected.to_parquet(ROOT/'selection.parquet',index=False,compression='zstd')
    report=dict(protocol_sha256=sha(PROTOCOL),model_report_sha256=sha(ROOT/'model_report.json'),
        feature_report_sha256=sha(FEATURE_ROOT/'feature_report.json'),full_label_report_sha256=sha(SOURCE/'full_label_report.json'),
        calibration_leaf_days_sha256=sha(ROOT/'calibration_leaf_days.parquet'),selection_sha256=sha(ROOT/'selection.parquet'),
        scores=scores,qualified_leaves=len(eligible),chosen_leaf=best,conditions=best['conditions'] if best else None,
        selected=int(selected.selected.sum()),by_half=selected.groupby('half').selected.agg(['size','sum']).reset_index().to_dict('records'),
        last_observation=r.next_date.max(),new_2025_groups_read=False,prior_2025_results_seen=True,new_2026_prices_read=False,
        no_exit_rules=True,software_compilation_verified=False)
    if best:
        used=list(dict.fromkeys(x['feature'] for x in best['conditions']))
        core=header+'\n'.join(f'{n}:={expressions[n]};' for n in used)+'\nCORE:'+' AND '.join(
            f"{x['feature']}{x['op']}{x['threshold']:.2f}" for x in best['conditions'])+';\n'
        (ROOT/'frozen_numeric_core.tdx').write_text(core)
        report['numeric_core']=core
        report['core_sha256']=sha(ROOT/'frozen_numeric_core.tdx')
    save_json(ROOT/'calibration_report.json',report)
    save_json(ROOT/'selection_report.json',report)
    return {k:v for k,v in report.items() if k!='scores'}


def analyze(ROOT=ROOT,PROTOCOL=PROTOCOL):
    report=json.loads((ROOT/'selection_report.json').read_text())
    if not report['qualified_leaves']:
        raise ValueError('No qualifying formula: do not inspect new 2025 leaf results')
    return common_analysis(ROOT,PROTOCOL)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['model','verify_model','calibrate','analyze'])
    print(json.dumps(globals()[parser.parse_args().stage](),ensure_ascii=False,indent=2))
