"""Boost shallow rules while pruning branches supported by too few dates."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from . import tail_formula_additive as base
from . import tail_formula_context_2024 as linkage
from . import tail_formula_float as inputs
from . import tail_formula_recent as study
from . import tail_formula_relative as relative
from .corporate_cash import save_json,sha


def compact_tree(estimator,x,dates,minimum_days):
    original=estimator.tree_
    paths=estimator.decision_path(x).tocsc()
    supports=[len(np.unique(dates[paths.indices[paths.indptr[i]:paths.indptr[i+1]]]))
        for i in range(original.node_count)]
    output={k:[] for k in ['feature','threshold','children_left','children_right',
        'n_node_samples','weighted_n_node_samples','value','impurity','training_days']}
    collapsed=0

    def visit(node):
        nonlocal collapsed
        left,right=int(original.children_left[node]),int(original.children_right[node])
        leaf=left<0
        if not leaf and min(supports[left],supports[right])<minimum_days:
            leaf=True;collapsed+=1
        index=len(output['feature'])
        output['feature'].append(-2 if leaf else int(original.feature[node]))
        output['threshold'].append(-2. if leaf else float(original.threshold[node]))
        output['children_left'].append(-1);output['children_right'].append(-1)
        output['n_node_samples'].append(int(original.n_node_samples[node]))
        output['weighted_n_node_samples'].append(float(original.weighted_n_node_samples[node]))
        output['value'].append(float(original.value[node,0,0]))
        output['impurity'].append(float(original.impurity[node]))
        output['training_days'].append(supports[node])
        if not leaf:
            output['children_left'][index]=visit(left)
            output['children_right'][index]=visit(right)
        return index

    visit(0)
    return output,collapsed


def setup(fold):
    if fold=='combined':
        linkage.ROOT=Path('data/research/tail_formula_day_support_2024')
        linkage.H2=Path('data/research/tail_formula_day_support_recent')
        linkage.COMBINED=Path('data/research/tail_formula_day_support_2025')
        linkage.PROTOCOL=Path('config/tail_formula_day_support_combined_protocol.json')
        linkage.H2_SELECTION_SHA=None;linkage.H2_MODEL_SHA=None;linkage.H2_OUTCOMES_PREVIOUSLY_SEEN=False
    else:
        inputs.setup(fold)
        root=Path('data/research/tail_formula_day_support_'+fold)
        protocol=Path('config/tail_formula_day_support_'+fold+'_protocol.json')
        base.ROOT=root;base.PROTOCOL=protocol;relative.PROTOCOL=protocol
        study.ROOT=root;study.PROTOCOL=protocol


def make_estimator(random):
    return DecisionTreeRegressor(criterion='friedman_mse',max_depth=3,min_samples_leaf=300,random_state=random)


def verify_reference(fold):
    """Check the unpruned algorithm against the existing independent GB fit."""
    root=Path('data/research/tail_formula_float_'+fold)
    r=json.loads((root/'model_report.json').read_text())
    v=json.loads((root/'model_verification.json').read_text())
    assert v['passed'] and v['model_report_sha256']==sha(root/'model_report.json')
    t=relative.training('relative');x=base.encode(t);y=t.target.to_numpy()
    date_codes=np.unique(t.date.to_numpy(),return_inverse=True)[1]
    w=1/t.groupby('date').code.transform('size').to_numpy()
    score=np.full(len(t),np.average(y,weights=w));random=np.random.RandomState(20260927)
    np.testing.assert_allclose(score[0],r['bias'],rtol=0,atol=2e-12)
    checks=0
    for expected in r['trees'][:3]:
        estimator=make_estimator(random).fit(x,y-score,sample_weight=w)
        tree,_=compact_tree(estimator,x,date_codes,0)
        for key in ['feature','threshold','children_left','children_right','n_node_samples']:
            np.testing.assert_array_equal(tree[key],expected[key])
        for key in ['weighted_n_node_samples','value','impurity']:
            np.testing.assert_allclose(tree[key],expected[key],rtol=0,atol=2e-10)
        score+=.05*estimator.predict(x);checks+=len(tree['feature'])
    proof=dict(passed=True,protocol_sha256=sha(base.PROTOCOL),reference_model_sha256=sha(root/'model_report.json'),
        unpruned_initial_trees_reproduced=3,node_checks=checks,training_rows=len(t),
        no_new_evaluation_outcomes_read=True,new_2026_prices_read=False)
    base.ROOT.mkdir(parents=True,exist_ok=True)
    save_json(base.ROOT/'reference_algorithm_verification.json',proof)
    return proof


def model():
    root=base.ROOT
    if (root/'model_report.json').exists():
        raise ValueError('Do not replace the frozen day-supported model')
    reference=json.loads((root/'reference_algorithm_verification.json').read_text())
    assert reference['passed'] and reference['protocol_sha256']==sha(base.PROTOCOL)
    config=json.loads(base.PROTOCOL.read_text());minimum_days=config['minimum_leaf_training_days']
    assert minimum_days==20 and config['model_max_depth']==3
    t=relative.training('relative');x=base.encode(t);y=t.target.to_numpy()
    dates=np.unique(t.date.to_numpy(),return_inverse=True)[1]
    w=1/t.groupby('date').code.transform('size').to_numpy()
    bias=float(np.average(y,weights=w));score=np.full(len(t),bias)
    random=np.random.RandomState(20260927);trees=[];collapsed=[]
    for _ in range(64):
        estimator=make_estimator(random).fit(x,y-score,sample_weight=w)
        tree,n=compact_tree(estimator,x,dates,minimum_days)
        score+=.05*np.asarray(tree['value'])[base.leaf_indices(x,tree)]
        trees.append(tree);collapsed.append(n)
    start,end,_=relative.training_scope()
    r=dict(protocol_sha256=sha(base.PROTOCOL),feature_report_sha256=sha(base.FEATURES/'feature_report.json'),
        label_report_sha256=sha(base.SOURCE/'full_label_report.json'),rows=len(t),days=t.date.nunique(),
        last_observation=t.next_date.max(),parameters=dict(loss='squared_error',n_estimators=64,learning_rate=.05,
            max_depth=3,min_samples_leaf=300,subsample=1.,random_state=20260927,criterion='friedman_mse',
            minimum_leaf_training_days=minimum_days),feature_names=list(base.EXPRESSIONS),variant='relative',
        learning_rate=.05,bias=bias,trees=trees,training_start=start,training_end=end,
        algorithm='逐棵拟合加权残差；若某节点任一子节点不足20个训练日则合并为父叶；后续残差使用已剪枝预测重新计算。',
        collapsed_splits_per_tree=collapsed,reference_algorithm_verification_sha256=sha(root/'reference_algorithm_verification.json'),
        new_2025_score_groups_read=bool(t.date.ge('2025-01-01').any()),new_2025H2_score_groups_read=False,
        new_2026_prices_read=False,no_exit_rules=True)
    np.testing.assert_allclose(score,base.predict(x,r),rtol=0,atol=2e-12)
    r['thresholds']=[dict(id=i,training_quantile=q,threshold=float(np.quantile(score,q))) for i,q in enumerate(base.QUANTILES)]
    save_json(root/'model_report.json',r)
    return {k:v for k,v in r.items() if k!='trees'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('fold',choices=['2024','recent','combined'])
    p.add_argument('stage',choices=['verify_reference','model','verify_model','scores','freeze','verify','analyze','diagnose'])
    a=p.parse_args();setup(a.fold)
    if a.fold=='combined':
        assert a.stage in ['freeze','verify','analyze']
        r=(linkage.common_analysis(linkage.COMBINED,linkage.PROTOCOL) if a.stage=='analyze'
            else getattr(linkage,'combine' if a.stage=='freeze' else 'verify_combined')())
    elif a.stage=='verify_reference':
        r=verify_reference(a.fold)
    elif a.stage=='model':
        r=model()
    elif a.stage=='verify_model':
        r=relative.verify_model('relative')
    elif a.stage in ['freeze','verify']:
        r=getattr(study,a.stage)()
    else:
        r=getattr(base,a.stage)()
    print(json.dumps(r,ensure_ascii=False,indent=2))
