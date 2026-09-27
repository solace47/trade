"""Freeze and assess one afternoon-path formula; only training labels choose it."""
import argparse
import json
import math

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from .corporate_cash import save_json, sha
from .tail_formula_intraday import ROOT, SOURCE, PROTOCOL, EXPRESSIONS, HEADER, conn
from .tail_formula_1000_analysis import select
from .tail_formula_1000_daily import analyze as common_analysis


def inputs():
    for root,prefix in [(ROOT,'feature'),(SOURCE,'training_label')]:
        proof=json.loads((root/f'{prefix}_verification.json').read_text())
        report=root/f'{prefix}_report.json'
        key='feature_report_sha256' if prefix=='feature' else 'label_report_sha256'
        assert proof['passed'] and proof[key]==sha(report)
    fr=json.loads((ROOT/'feature_report.json').read_text())
    lr=json.loads((SOURCE/'training_label_report.json').read_text())
    assert fr['features_sha256']==sha(ROOT/'features.parquet')
    assert lr['labels_sha256']==sha(SOURCE/'training_labels.parquet')
    f=pd.read_parquet(ROOT/'features.parquet')
    labels=pd.read_parquet(SOURCE/'training_labels.parquet',columns=['date','code','next_date','known15','opportunity15'])
    assert labels.next_date.lt('2024-07-01').all()
    known=labels.loc[labels.known15].copy()
    base=known.groupby('date').opportunity15.mean()
    train=f.loc[f.formula_input_valid].merge(known,on=['date','code'],validate='one_to_one')
    train=train.sort_values(['date','code']).reset_index(drop=True)
    return f,train,known,base


def model():
    if (ROOT/'model_report.json').exists():
        raise ValueError('Do not refit the frozen intraday model')
    _,train,known,base=inputs()
    target=train.opportunity15-train.date.map(base)
    weights=1/train.groupby('date').code.transform('size')
    reg=DecisionTreeRegressor(max_depth=4,min_samples_leaf=2000,criterion='squared_error',random_state=20260927)
    reg.fit(train[list(EXPRESSIONS)],target,sample_weight=weights)
    t=reg.tree_
    report=dict(protocol_sha256=sha(PROTOCOL),feature_report_sha256=sha(ROOT/'feature_report.json'),
        training_label_report_sha256=sha(SOURCE/'training_label_report.json'),rows=len(train),days=train.date.nunique(),
        baseline_rows=len(known),last_observation=train.next_date.max(),parameters=reg.get_params(),
        feature_names=list(EXPRESSIONS),tree=dict(feature=t.feature.tolist(),threshold=t.threshold.tolist(),
            children_left=t.children_left.tolist(),children_right=t.children_right.tolist(),
            n_node_samples=t.n_node_samples.tolist(),weighted_n_node_samples=t.weighted_n_node_samples.tolist(),
            value=t.value.reshape(-1).tolist(),impurity=t.impurity.tolist()),
        baseline_includes_invalid_input_rows=True,new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'model_report.json',report)
    return {k:v for k,v in report.items() if k!='tree'}


def verify_model():
    r=json.loads((ROOT/'model_report.json').read_text())
    assert r['protocol_sha256']==sha(PROTOCOL)
    assert r['feature_report_sha256']==sha(ROOT/'feature_report.json')
    assert r['training_label_report_sha256']==sha(SOURCE/'training_label_report.json')
    _,train,known,_=inputs()
    c=conn()
    c.register('training',train)
    c.register('known',known)
    expected=c.sql('''WITH b AS(SELECT date,avg(opportunity15) AS base FROM known GROUP BY date)
        SELECT date,code,opportunity15-base AS target,1./count(*) OVER(PARTITION BY date) AS weight
        FROM training JOIN b USING(date) ORDER BY date,code''').df()
    assert expected[['date','code']].equals(train[['date','code']])
    np.testing.assert_allclose(expected.groupby('date').weight.sum(),1,atol=2e-14,rtol=0)
    x=train[list(EXPRESSIONS)].to_numpy(dtype='float32')
    t=r['tree']
    masks={0:np.ones(len(train),dtype=bool)}
    depths={0:0}
    for i in range(len(t['feature'])):
        m=masks[i]
        assert int(m.sum())==t['n_node_samples'][i]
        w=expected.weight[m]
        y=expected.target[m]
        mean=float(np.average(y,weights=w))
        variance=float(np.average((y-mean)**2,weights=w))
        np.testing.assert_allclose(w.sum(),t['weighted_n_node_samples'][i],atol=1e-9,rtol=0)
        np.testing.assert_allclose(mean,t['value'][i],atol=2e-11,rtol=0)
        np.testing.assert_allclose(variance,t['impurity'][i],atol=2e-11,rtol=0)
        left,right=t['children_left'][i],t['children_right'][i]
        if left<0:
            assert m.sum()>=2000 and depths[i]<=4
            continue
        lower=x[:,t['feature'][i]]<=t['threshold'][i]
        masks[left],masks[right]=m&lower,m&~lower
        depths[left]=depths[right]=depths[i]+1
    result=dict(passed=True,model_report_sha256=sha(ROOT/'model_report.json'),rows=len(train),
        full_same_day_baseline_rebuilt=True,all_node_counts_means_variances_rebuilt=True,
        equal_training_day_weights=True,new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'model_verification.json',result)
    return result


def paths(tree,names=None):
    all_paths={0:[]}
    leaves={}
    names=list(EXPRESSIONS) if names is None else names
    for i,feature in enumerate(tree['feature']):
        left=tree['children_left'][i]
        if left<0:
            leaves[i]=all_paths[i]
            continue
        threshold=tree['threshold'][i]
        all_paths[left]=all_paths[i]+[dict(feature=names[feature],op='<=',threshold=math.floor(threshold*100)/100)]
        all_paths[tree['children_right'][i]]=all_paths[i]+[dict(feature=names[feature],op='>',threshold=math.ceil(threshold*100)/100)]
    return leaves


def freeze():
    if (ROOT/'selection_report.json').exists():
        raise ValueError('Do not replace frozen intraday selection')
    check=json.loads((ROOT/'model_verification.json').read_text())
    assert check['passed'] and check['model_report_sha256']==sha(ROOT/'model_report.json')
    model_report=json.loads((ROOT/'model_report.json').read_text())
    f,train,_,base=inputs()
    candidates=paths(model_report['tree'])
    scores=[]
    days=[]
    for node,conditions in candidates.items():
        chosen=train.loc[select(train,conditions)]
        d=chosen.groupby('date').agg(rows=('code','size'),rate=('opportunity15','mean'))
        d['base_rate']=base.reindex(d.index)
        d['delta']=d.rate-d.base_rate
        d['node']=node
        days.append(d.reset_index())
        scores.append(dict(node=node,rows=len(chosen),days=len(d),daily_rate=float(d.rate.mean()),daily_delta=float(d.delta.mean()),
            eligible=bool(len(chosen)>=2000 and len(d)>=40 and d.delta.mean()>0)))
    pd.concat(days,ignore_index=True).to_parquet(ROOT/'training_leaf_days.parquet',index=False,compression='zstd')
    save_json(ROOT/'training_scores.json',scores)
    eligible=[s for s in scores if s['eligible']]
    if not eligible:
        raise ValueError('No positive training leaf; preserve scores without a new evaluation')
    best=sorted(eligible,key=lambda s:(-s['daily_delta'],-s['rows'],s['node']))[0]
    conditions=candidates[best['node']]
    selection=f[['date','code','half','board','decision_shares']].copy()
    selection['selected']=select(f,conditions)
    selection.to_parquet(ROOT/'selection.parquet',index=False,compression='zstd')
    used=list(dict.fromkeys(x['feature'] for x in conditions))
    core=HEADER+'\n'.join(f'{name}:={EXPRESSIONS[name]};' for name in used)+'\nCORE:'+' AND '.join(
        f"{x['feature']}{x['op']}{x['threshold']:.2f}" for x in conditions)+';\n'
    (ROOT/'frozen_numeric_core.tdx').write_text(core)
    report=dict(protocol_sha256=sha(PROTOCOL),model_report_sha256=sha(ROOT/'model_report.json'),
        feature_report_sha256=sha(ROOT/'feature_report.json'),training_label_report_sha256=sha(SOURCE/'training_label_report.json'),
        training_leaf_days_sha256=sha(ROOT/'training_leaf_days.parquet'),training_scores=scores,chosen_leaf=best,
        conditions=conditions,numeric_core=core,selection_sha256=sha(ROOT/'selection.parquet'),core_sha256=sha(ROOT/'frozen_numeric_core.tdx'),
        selected=int(selection.selected.sum()),by_half=selection.groupby('half').selected.agg(['size','sum']).reset_index().to_dict('records'),
        new_group_outcomes_read=False,prior_version_outcomes_seen=True,new_2026_prices_read=False,no_exit_rules=True,
        software_compilation_verified=False)
    save_json(ROOT/'selection_report.json',report)
    return {k:v for k,v in report.items() if k!='training_scores'}


def verify():
    report=json.loads((ROOT/'selection_report.json').read_text())
    for field,path in [('protocol_sha256',PROTOCOL),('model_report_sha256',ROOT/'model_report.json'),
                       ('feature_report_sha256',ROOT/'feature_report.json'),('selection_sha256',ROOT/'selection.parquet'),
                       ('training_leaf_days_sha256',ROOT/'training_leaf_days.parquet'),('core_sha256',ROOT/'frozen_numeric_core.tdx')]:
        assert report[field]==sha(path)
    check=json.loads((ROOT/'model_verification.json').read_text())
    assert check['passed'] and check['model_report_sha256']==report['model_report_sha256']
    f,train,known,_=inputs()
    tree=json.loads((ROOT/'model_report.json').read_text())['tree']
    # Parent-pointer reconstruction is independent of the forward path builder.
    all_paths={}
    names=list(EXPRESSIONS)
    for leaf,left in enumerate(tree['children_left']):
        if left>=0:
            continue
        node=leaf
        reverse=[]
        while node:
            if node in tree['children_left']:
                parent=tree['children_left'].index(node)
                op,threshold='<=',math.floor(tree['threshold'][parent]*100)/100
            else:
                parent=tree['children_right'].index(node)
                op,threshold='>',math.ceil(tree['threshold'][parent]*100)/100
            reverse.append(dict(feature=names[tree['feature'][parent]],op=op,threshold=threshold))
            node=parent
        all_paths[leaf]=list(reversed(reverse))
    c=conn()
    c.register('training',train)
    c.register('known',known)
    c.register('features',f)
    daily=[]
    scores=[]
    for node,conditions in all_paths.items():
        condition=' AND '.join(f"{x['feature']}{x['op']}{x['threshold']}" for x in conditions)
        d=c.sql(f'''WITH base AS(SELECT date,avg(opportunity15) AS base_rate FROM known GROUP BY date),
            chosen AS(SELECT date,count(*) AS rows,avg(opportunity15) AS rate FROM training WHERE {condition} GROUP BY date)
            SELECT chosen.*,base_rate,rate-base_rate AS delta,{node} AS node FROM chosen JOIN base USING(date) ORDER BY date''').df()
        daily.append(d)
        scores.append(dict(node=node,rows=int(d.rows.sum()),days=len(d),daily_rate=float(d.rate.mean()),daily_delta=float(d.delta.mean()),
            eligible=bool(d.rows.sum()>=2000 and len(d)>=40 and d.delta.mean()>0)))
    pd.testing.assert_frame_equal(pd.read_parquet(ROOT/'training_leaf_days.parquet'),pd.concat(daily,ignore_index=True),
        check_dtype=False,rtol=0,atol=2e-12)
    for a,b in zip(report['training_scores'],scores):
        for key in a:
            np.testing.assert_allclose(a[key],b[key],rtol=0,atol=2e-12)
    best=sorted([x for x in scores if x['eligible']],key=lambda x:(-x['daily_delta'],-x['rows'],x['node']))[0]
    assert best['node']==report['chosen_leaf']['node']
    assert all_paths[best['node']]==report['conditions']
    condition=' AND '.join(f"{x['feature']}{x['op']}{x['threshold']}" for x in report['conditions'])
    expected=c.sql('SELECT date,code,half,board,decision_shares,formula_input_valid AND '+condition+
        ' AS selected FROM features ORDER BY date,code').df()
    selected=pd.read_parquet(ROOT/'selection.parquet')
    pd.testing.assert_frame_equal(selected,expected,check_exact=True,check_dtype=False)
    result=dict(passed=True,selection_report_sha256=sha(ROOT/'selection_report.json'),
        all_parent_paths_and_daily_scores_rebuilt=True,all_selection_rows_rebuilt=True,selected=int(selected.selected.sum()),
        new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'selection_verification.json',result)
    return result


def analyze():
    return common_analysis(ROOT,PROTOCOL)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['model','verify_model','freeze','verify','analyze'])
    print(json.dumps(globals()[parser.parse_args().stage](),ensure_ascii=False,indent=2))
