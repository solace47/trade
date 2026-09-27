"""Reuse the fixed native tree; choose its leaf by training-day relative opportunity."""
import argparse
import json
import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .corporate_cash import save_json, sha
from .tail_formula_1000 import ROOT as SOURCE, EXPRESSIONS
from .tail_formula_1000_analysis import analyze as common_analysis, select

ROOT=Path('data/research/tail_formula_1000_daily')
PROTOCOL=Path('config/tail_formula_1000_daily_protocol.json')


def leaf_paths(tree):
    paths={0:[]}
    names=list(EXPRESSIONS)
    leaves={}
    for i,feature in enumerate(tree['feature']):
        left=tree['children_left'][i]
        if left<0:
            leaves[i]=paths[i]
            continue
        t=tree['threshold'][i]
        paths[left]=paths[i]+[dict(feature=names[feature],op='<=',threshold=math.floor(t*100)/100)]
        paths[tree['children_right'][i]]=paths[i]+[dict(feature=names[feature],op='>',threshold=math.ceil(t*100)/100)]
    return leaves


def inputs():
    source_report=json.loads((SOURCE/'selection_report.json').read_text())
    assert json.loads((SOURCE/'analysis_verification.json').read_text())['analysis_report_sha256']==sha(SOURCE/'analysis_report.json')
    f_report=json.loads((SOURCE/'feature_report.json').read_text())
    assert f_report['output_sha256']['features.parquet']==sha(SOURCE/'features.parquet')
    l_report=json.loads((SOURCE/'training_label_report.json').read_text())
    assert l_report['labels_sha256']==sha(SOURCE/'training_labels.parquet')
    f=pd.read_parquet(SOURCE/'features.parquet')
    l=pd.read_parquet(SOURCE/'training_labels.parquet',columns=['date','code','next_date','known15','opportunity15'])
    assert l.next_date.lt('2024-07-01').all()
    train=f.merge(l,on=['date','code'],validate='one_to_one')
    train=train.loc[train.known15&train.formula_input_valid].sort_values(['date','code']).reset_index(drop=True)
    return source_report,f,train


def freeze():
    if (ROOT/'selection_report.json').exists():
        raise ValueError('Do not replace frozen daily-relative choice')
    ROOT.mkdir(parents=True,exist_ok=True)
    source_report,f,train=inputs()
    paths=leaf_paths(source_report['tree'])
    base=train.groupby('date').opportunity15.mean()
    scores=[]
    daily=[]
    for node,conditions in paths.items():
        chosen=train.loc[select(train,conditions)]
        d=chosen.groupby('date').agg(rows=('code','size'),rate=('opportunity15','mean'))
        d['base_rate']=base.reindex(d.index)
        d['delta']=d.rate-d.base_rate
        d['node']=node
        daily.append(d.reset_index())
        scores.append(dict(node=node,rows=len(chosen),days=len(d),daily_rate=float(d.rate.mean()),
            daily_delta=float(d.delta.mean()),eligible=bool(len(chosen)>=2000 and len(d)>=40 and d.delta.mean()>0)))
    eligible=[s for s in scores if s['eligible']]
    best=sorted(eligible,key=lambda s:(-s['daily_delta'],-s['rows'],s['node']))[0] if eligible else None
    if best is None:
        raise ValueError('No positive daily-relative leaf; preserve training scores before any new evaluation')
    conditions=paths[best['node']]
    selection=f[['date','code','half','board','decision_shares']].copy()
    selection['selected']=select(f,conditions)
    selection.to_parquet(ROOT/'selection.parquet',index=False,compression='zstd')
    pd.concat(daily,ignore_index=True).to_parquet(ROOT/'training_leaf_days.parquet',index=False,compression='zstd')
    used=list(dict.fromkeys(c['feature'] for c in conditions))
    core='\n'.join(f'{name}:={EXPRESSIONS[name]};' for name in used)+'\nCORE:'+' AND '.join(
        f"{c['feature']}{c['op']}{c['threshold']:.2f}" for c in conditions)+';\n'
    (ROOT/'frozen_numeric_core.tdx').write_text(core)
    report=dict(protocol_sha256=sha(PROTOCOL),source_selection_report_sha256=sha(SOURCE/'selection_report.json'),
        source_analysis_report_sha256=sha(SOURCE/'analysis_report.json'),training_label_report_sha256=sha(SOURCE/'training_label_report.json'),
        source_feature_report_sha256=sha(SOURCE/'feature_report.json'),training_scores=scores,chosen_leaf=best,
        conditions=conditions,numeric_core=core,selection_sha256=sha(ROOT/'selection.parquet'),
        training_leaf_days_sha256=sha(ROOT/'training_leaf_days.parquet'),core_sha256=sha(ROOT/'frozen_numeric_core.tdx'),
        selected=int(selection.selected.sum()),by_half=selection.groupby('half').selected.agg(['size','sum']).reset_index().to_dict('records'),
        new_group_outcomes_read=False,prior_version_outcomes_seen=True,tree_refitted=False,
        new_2026_prices_read=False,no_exit_rules=True,software_compilation_verified=False)
    save_json(ROOT/'selection_report.json',report)
    return {k:v for k,v in report.items() if k!='training_scores'}


def verify():
    report=json.loads((ROOT/'selection_report.json').read_text())
    assert report['protocol_sha256']==sha(PROTOCOL)
    assert report['selection_sha256']==sha(ROOT/'selection.parquet')
    assert report['training_leaf_days_sha256']==sha(ROOT/'training_leaf_days.parquet')
    source_report,f,train=inputs()
    assert report['source_selection_report_sha256']==sha(SOURCE/'selection_report.json')
    tree=source_report['tree']
    names=list(EXPRESSIONS)
    # Reconstruct each path from parent pointers, independently of leaf_paths.
    paths={}
    for leaf,left in enumerate(tree['children_left']):
        if left>=0:
            continue
        node=leaf
        reverse=[]
        while node:
            if node in tree['children_left']:
                parent=tree['children_left'].index(node)
                op='<='
                threshold=math.floor(tree['threshold'][parent]*100)/100
            else:
                parent=tree['children_right'].index(node)
                op='>'
                threshold=math.ceil(tree['threshold'][parent]*100)/100
            reverse.append(dict(feature=names[tree['feature'][parent]],op=op,threshold=threshold))
            node=parent
        paths[leaf]=list(reversed(reverse))
    c=duckdb.connect()
    c.register('train',train)
    c.register('features',f)
    parts=[]
    scores=[]
    for node,conditions in paths.items():
        condition=' AND '.join(f"{x['feature']}{x['op']}{x['threshold']}" for x in conditions)
        d=c.sql(f'''WITH base AS(SELECT date,avg(opportunity15) AS base_rate FROM train GROUP BY date),
            chosen AS(SELECT date,count(*) AS rows,avg(opportunity15) AS rate FROM train WHERE {condition} GROUP BY date)
            SELECT chosen.*,base_rate,rate-base_rate AS delta,{node} AS node FROM chosen JOIN base USING(date) ORDER BY date''').df()
        parts.append(d)
        scores.append(dict(node=node,rows=int(d.rows.sum()),days=len(d),daily_rate=float(d.rate.mean()),
            daily_delta=float(d.delta.mean()),eligible=d.rows.sum()>=2000 and len(d)>=40 and d.delta.mean()>0))
    actual=pd.read_parquet(ROOT/'training_leaf_days.parquet')
    pd.testing.assert_frame_equal(actual,pd.concat(parts,ignore_index=True),check_dtype=False,atol=2e-12,rtol=0)
    for a,b in zip(report['training_scores'],scores):
        for key in a:
            np.testing.assert_allclose(a[key],b[key],atol=2e-12,rtol=0)
    best=sorted([x for x in scores if x['eligible']],key=lambda s:(-s['daily_delta'],-s['rows'],s['node']))[0]
    assert best['node']==report['chosen_leaf']['node']
    assert paths[best['node']]==report['conditions']
    condition=' AND '.join(f"{x['feature']}{x['op']}{x['threshold']}" for x in report['conditions'])
    expected=c.sql('SELECT date,code,half,board,decision_shares,formula_input_valid AND '+condition+' AS selected FROM features ORDER BY date,code').df()
    selected=pd.read_parquet(ROOT/'selection.parquet')
    pd.testing.assert_frame_equal(selected,expected,check_dtype=False,check_exact=True)
    result=dict(passed=True,selection_report_sha256=sha(ROOT/'selection_report.json'),
        source_training_only=True,all_leaf_paths_and_daily_scores_rebuilt=True,selected=int(selected.selected.sum()),
        new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'selection_verification.json',result)
    return result


def analyze():
    check=json.loads((ROOT/'selection_verification.json').read_text())
    assert check['passed'] and check['selection_report_sha256']==sha(ROOT/'selection_report.json')
    for name in ['full_labels.parquet','full_label_report.json','full_label_verification.json']:
        path=ROOT/name
        if not path.exists():
            path.symlink_to((SOURCE/name).resolve())
        assert sha(path)==sha(SOURCE/name)
    return common_analysis(ROOT,PROTOCOL)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['freeze','verify','analyze'])
    print(json.dumps(globals()[parser.parse_args().stage](),ensure_ascii=False,indent=2))
