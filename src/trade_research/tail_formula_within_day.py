"""Boost stock differences after projecting out each training day's residual mean."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from . import tail_formula_additive as base
from . import tail_formula_context_2024 as linkage
from . import tail_formula_float as inputs
from . import tail_formula_recent as study
from . import tail_formula_relative as relative
from .corporate_cash import save_json, sha

PARAMETERS=dict(loss='within_day_squared_error',n_estimators=64,learning_rate=.05,max_depth=3,
    min_samples_leaf=300,subsample=1.,random_state=20260927,criterion='friedman_mse')


def setup(fold):
    if fold=='combined':
        linkage.ROOT=Path('data/research/tail_formula_within_day_2024')
        linkage.H2=Path('data/research/tail_formula_within_day_recent')
        linkage.COMBINED=Path('data/research/tail_formula_within_day_2025')
        linkage.PROTOCOL=Path('config/tail_formula_within_day_combined_protocol.json')
        linkage.H2_SELECTION_SHA=None;linkage.H2_MODEL_SHA=None;linkage.H2_OUTCOMES_PREVIOUSLY_SEEN=False
    else:
        inputs.setup(fold)
        root=Path('data/research/tail_formula_within_day_'+fold)
        protocol=Path('config/tail_formula_within_day_'+fold+'_protocol.json')
        base.ROOT=root;base.PROTOCOL=protocol;relative.PROTOCOL=protocol
        study.ROOT=root;study.PROTOCOL=protocol
        config=json.loads(protocol.read_text())
        assert config['feature_report_sha256']==sha(inputs.ROOT/'feature_report.json')
        assert config['label_report_sha256']==sha(base.SOURCE/'full_label_report.json')


def training():
    start,end,_=relative.training_scope()
    return base.training(start=start,end=end)


def centered(values,indices,counts):
    means=np.bincount(indices,weights=values,minlength=len(counts))/counts
    return values-means[indices],means


def model():
    root=base.ROOT
    if (root/'model_report.json').exists():
        raise ValueError('Do not refit a frozen within-day model')
    t=training();x=base.encode(t)
    dates,indices=np.unique(t.date.to_numpy(),return_inverse=True);counts=np.bincount(indices)
    y,_=centered(t.opportunity15.to_numpy(),indices,counts)
    w=1/counts[indices];score=np.zeros(len(t));trees=[];projections=[]
    rng=np.random.RandomState(20260927)
    for index in range(64):
        residual,means=centered(y-score,indices,counts)
        estimator=DecisionTreeRegressor(criterion='friedman_mse',max_depth=3,min_samples_leaf=300,
            random_state=rng).fit(x,residual,sample_weight=w)
        q=estimator.tree_
        tree=dict(feature=q.feature.tolist(),threshold=q.threshold.tolist(),children_left=q.children_left.tolist(),
            children_right=q.children_right.tolist(),n_node_samples=q.n_node_samples.tolist(),
            weighted_n_node_samples=q.weighted_n_node_samples.tolist(),value=q.value.reshape(-1).tolist(),impurity=q.impurity.tolist())
        trees.append(tree)
        step=np.asarray(tree['value'])[base.leaf_indices(x,tree)]
        np.testing.assert_allclose(step,estimator.predict(x),rtol=0,atol=2e-12)
        projections.append(dict(tree=index,removed_day_means=means.tolist(),
            maximum_centered_day_mean=float(np.max(np.abs(np.bincount(indices,weights=residual)/counts)))))
        score+=.05*step
    start,end,_=relative.training_scope()
    r=dict(protocol_sha256=sha(base.PROTOCOL),feature_report_sha256=sha(base.FEATURES/'feature_report.json'),
        label_report_sha256=sha(base.SOURCE/'full_label_report.json'),rows=len(t),days=len(dates),
        last_observation=t.next_date.max(),parameters=PARAMETERS,feature_names=list(base.EXPRESSIONS),
        variant='within_day',learning_rate=.05,bias=0.,trees=trees,training_start=start,training_end=end,
        training_dates=dates.tolist(),residual_projections=projections,score_is_not_probability=True,
        new_2025_score_groups_read=bool(t.date.ge('2025-01-01').any()),new_2025H2_score_groups_read=False,
        new_2026_prices_read=False,no_exit_rules=True)
    assert t.next_date.lt(end).all() and t.date.ge(start).all()
    np.testing.assert_allclose(score,base.predict(x,r),rtol=0,atol=2e-12)
    r['thresholds']=[dict(id=i,training_quantile=q,threshold=float(np.quantile(score,q))) for i,q in enumerate(base.QUANTILES)]
    root.mkdir(parents=True,exist_ok=True);save_json(root/'model_report.json',r)
    return {k:v for k,v in r.items() if k not in ['trees','training_dates','residual_projections']}


def verify_model():
    root=base.ROOT;r=json.loads((root/'model_report.json').read_text())
    for key,path in [('protocol_sha256',base.PROTOCOL),('feature_report_sha256',base.FEATURES/'feature_report.json'),
        ('label_report_sha256',base.SOURCE/'full_label_report.json')]:assert r[key]==sha(path)
    start,end,where=relative.training_scope()
    assert r['training_start']==start and r['training_end']==end and r['last_observation']<end
    assert r['parameters']==PARAMETERS and r['variant']=='within_day' and r['bias']==0 and r['learning_rate']==.05
    names=list(base.EXPRESSIONS);assert r['feature_names']==names and len(names)==48
    c=base.conn();c.read_parquet(str(base.FEATURES/'features.parquet')).create_view('features')
    encoded=','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::INT AS {n}' for n in names)
    d=c.sql(f'''WITH intersected AS(SELECT f.date,f.code,l.next_date,l.opportunity15,{encoded}
        FROM features f JOIN read_parquet('{base.SOURCE}/full_labels.parquet') l USING(date,code)
        WHERE {where} AND known15 AND formula_input_valid)
        SELECT *,opportunity15-avg(opportunity15) OVER(PARTITION BY date) AS target,
        1./count(*) OVER(PARTITION BY date) AS w FROM intersected ORDER BY date,code''').df()
    original=training()
    pd.testing.assert_frame_equal(d[['date','code','next_date']],original[['date','code','next_date']],check_exact=True)
    assert len(d)==r['rows'] and d.date.nunique()==r['days'] and d.next_date.max()==r['last_observation']
    assert sorted(d.date.unique())==r['training_dates']
    y=d.target.to_numpy();x=d[names].to_numpy(dtype='int32');w=d.w.to_numpy()
    np.testing.assert_array_equal(x,base.encode(original))
    expected_y=original.opportunity15-original.groupby('date').opportunity15.transform('mean')
    np.testing.assert_allclose(y,expected_y,rtol=0,atol=2e-12)
    score=np.zeros(len(d));checks=0;max_day_mean=0.
    assert len(r['trees'])==len(r['residual_projections'])==64
    for index,(tree,projection) in enumerate(zip(r['trees'],r['residual_projections'])):
        rd=pd.DataFrame(dict(row_id=np.arange(len(d)),date=d.date,residual=y-score));c.register('residuals',rd)
        means=c.sql('SELECT date,avg(residual) AS mean FROM residuals GROUP BY date ORDER BY date').df()
        rebuilt=c.sql('SELECT row_id,date,residual-avg(residual) OVER(PARTITION BY date) AS residual FROM residuals ORDER BY row_id').df()
        np.testing.assert_allclose(means['mean'],projection['removed_day_means'],rtol=0,atol=2e-10)
        assert projection['tree']==index
        drift=float(rebuilt.groupby('date').residual.mean().abs().max());max_day_mean=max(max_day_mean,drift)
        assert drift<2e-12 and projection['maximum_centered_day_mean']<2e-12
        residual=rebuilt.residual.to_numpy();masks={0:np.ones(len(d),dtype=bool)};levels={0:0};terminal=np.empty(len(d))
        assert len(tree['feature'])<=15
        for node,left in enumerate(tree['children_left']):
            mask=masks[node];weights=w[mask];assert int(mask.sum())==tree['n_node_samples'][node] and levels[node]<=3
            mean=np.average(residual[mask],weights=weights)
            variance=np.average((residual[mask]-mean)**2,weights=weights)
            np.testing.assert_allclose(weights.sum(),tree['weighted_n_node_samples'][node],rtol=0,atol=1e-8)
            np.testing.assert_allclose(mean,tree['value'][node],rtol=0,atol=2e-10)
            np.testing.assert_allclose(variance,tree['impurity'][node],rtol=0,atol=2e-10)
            if left<0:
                assert mask.sum()>=300;terminal[mask]=mean
            else:
                right=tree['children_right'][node];lower=x[:,tree['feature'][node]]<=tree['threshold'][node]
                masks[left],masks[right]=mask&lower,mask&~lower
                levels[left]=levels[right]=levels[node]+1
            checks+=1
        score+=.05*terminal
    c.close()
    for q,cut in zip(base.QUANTILES,r['thresholds']):
        assert q==cut['training_quantile']
        np.testing.assert_allclose(np.quantile(score,q),cut['threshold'],rtol=0,atol=2e-10)
    np.testing.assert_allclose(score,base.predict(x,r),rtol=0,atol=2e-10)
    assert r['new_2025_score_groups_read']==bool(d.date.ge('2025-01-01').any())
    assert not d.date.ge('2025-07-01').any()
    proof=dict(passed=True,model_report_sha256=sha(root/'model_report.json'),rows=len(d),node_checks=checks,
        all_intersection_targets_encodings_and_date_weights_rebuilt=True,
        all_64_daily_residual_projections_independently_rebuilt=True,maximum_daily_residual_mean=max_day_mean,
        all_node_residual_means_variances_and_quantiles_rebuilt=True,training_start=start,training_end=end,
        new_2025_score_groups_read=bool(d.date.ge('2025-01-01').any()),new_2025H2_score_groups_read=False,
        new_2026_prices_read=False,no_exit_rules=True)
    save_json(root/'model_verification.json',proof);return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('fold',choices=['2024','recent','combined'])
    p.add_argument('stage',choices=['model','verify_model','scores','freeze','verify','analyze'])
    a=p.parse_args();setup(a.fold)
    if a.fold=='combined':
        assert a.stage in ['freeze','verify','analyze']
        r=(linkage.common_analysis(linkage.COMBINED,linkage.PROTOCOL) if a.stage=='analyze'
            else getattr(linkage,'combine' if a.stage=='freeze' else 'verify_combined')())
    elif a.stage in ['model','verify_model']:
        r=globals()[a.stage]()
    elif a.stage in ['freeze','verify']:
        r=getattr(study,a.stage)()
    else:
        r=getattr(base,a.stage)()
    print(json.dumps(r,ensure_ascii=False,indent=2))
