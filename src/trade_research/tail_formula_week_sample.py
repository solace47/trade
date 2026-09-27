"""Shallow residual rules trained on a predeclared whole-week subsample."""
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
from .corporate_cash import save_json,sha


def setup(fold):
    if fold=='combined':
        linkage.ROOT=Path('data/research/tail_formula_week_sample_2024')
        linkage.H2=Path('data/research/tail_formula_week_sample_recent')
        linkage.COMBINED=Path('data/research/tail_formula_week_sample_2025')
        linkage.PROTOCOL=Path('config/tail_formula_week_sample_combined_protocol.json')
        linkage.H2_SELECTION_SHA=None;linkage.H2_MODEL_SHA=None;linkage.H2_OUTCOMES_PREVIOUSLY_SEEN=False
    else:
        inputs.setup(fold)
        root=Path('data/research/tail_formula_week_sample_'+fold)
        protocol=Path('config/tail_formula_week_sample_'+fold+'_protocol.json')
        base.ROOT=root;base.PROTOCOL=protocol;relative.PROTOCOL=protocol
        study.ROOT=root;study.PROTOCOL=protocol
        config=json.loads(protocol.read_text())
        assert config['feature_report_sha256']==sha(inputs.ROOT/'feature_report.json')


def model():
    root=base.ROOT
    if (root/'model_report.json').exists():
        raise ValueError('Do not replace the frozen week-sampled model')
    config=json.loads(base.PROTOCOL.read_text())
    fraction=config['training_week_fraction']; seed=config['random_seed']
    assert fraction==.75 and seed==20260927 and config['model_max_depth']==3
    t=relative.training('relative'); x=base.encode(t); y=t.target.to_numpy()
    times=pd.to_datetime(t.date)
    weeks=(times-pd.to_timedelta(times.dt.weekday,unit='D')).dt.strftime('%Y-%m-%d').to_numpy()
    unique=np.unique(weeks); size=int(np.ceil(fraction*len(unique)))
    w=1/t.groupby('date').code.transform('size').to_numpy()
    bias=float(np.average(y,weights=w)); score=np.full(len(t),bias)
    rng=np.random.default_rng(seed); trees=[]; schedule=[]
    for i in range(64):
        picked=sorted(rng.choice(unique,size=size,replace=False).tolist())
        mask=np.isin(weeks,picked)
        estimator=DecisionTreeRegressor(criterion='friedman_mse',max_depth=3,min_samples_leaf=300,
            random_state=seed+i).fit(x[mask],(y-score)[mask],sample_weight=w[mask])
        q=estimator.tree_
        tree=dict(feature=q.feature.tolist(),threshold=q.threshold.tolist(),children_left=q.children_left.tolist(),
            children_right=q.children_right.tolist(),n_node_samples=q.n_node_samples.tolist(),
            weighted_n_node_samples=q.weighted_n_node_samples.tolist(),value=q.value.reshape(-1).tolist(),
            impurity=q.impurity.tolist())
        trees.append(tree)
        step=np.asarray(tree['value'])[base.leaf_indices(x,tree)]
        np.testing.assert_allclose(step,estimator.predict(x),rtol=0,atol=2e-12)
        score+=.05*step
        schedule.append(dict(tree=i,week_starts=picked,rows=int(mask.sum()),days=int(t.loc[mask,'date'].nunique())))
    start,end,_=relative.training_scope()
    r=dict(protocol_sha256=sha(base.PROTOCOL),feature_report_sha256=sha(base.FEATURES/'feature_report.json'),
        label_report_sha256=sha(base.SOURCE/'full_label_report.json'),rows=len(t),days=t.date.nunique(),
        last_observation=t.next_date.max(),parameters=dict(loss='squared_error',n_estimators=64,learning_rate=.05,
            max_depth=3,min_samples_leaf=300,random_state=seed,criterion='friedman_mse',training_week_fraction=fraction),
        feature_names=list(base.EXPRESSIONS),variant='relative',learning_rate=.05,bias=bias,trees=trees,
        training_start=start,training_end=end,training_week_starts=unique.tolist(),weeks_per_tree=size,
        sampling_schedule=schedule,
        algorithm='按完整周无放回抽样拟合残差，保留所选周全部训练股票及日期等权，再更新全部训练行分数。',
        new_2025_score_groups_read=bool(t.date.ge('2025-01-01').any()),new_2025H2_score_groups_read=False,
        new_2026_prices_read=False,no_exit_rules=True)
    np.testing.assert_allclose(score,base.predict(x,r),rtol=0,atol=2e-12)
    r['thresholds']=[dict(id=i,training_quantile=q,threshold=float(np.quantile(score,q))) for i,q in enumerate(base.QUANTILES)]
    root.mkdir(parents=True,exist_ok=True);save_json(root/'model_report.json',r)
    return {k:v for k,v in r.items() if k not in ['trees','sampling_schedule','training_week_starts']}


def verify_model():
    root=base.ROOT;r=json.loads((root/'model_report.json').read_text())
    for key,path in [('protocol_sha256',base.PROTOCOL),('feature_report_sha256',base.FEATURES/'feature_report.json'),
        ('label_report_sha256',base.SOURCE/'full_label_report.json')]:
        assert r[key]==sha(path)
    config=json.loads(base.PROTOCOL.read_text());start,end,where=relative.training_scope()
    assert r['training_start']==start and r['training_end']==end and r['last_observation']<end
    assert r['variant']=='relative' and r['feature_names']==list(base.EXPRESSIONS) and len(base.EXPRESSIONS)==48
    assert config['training_week_fraction']==.75 and config['random_seed']==20260927
    assert r['parameters']==dict(loss='squared_error',n_estimators=64,learning_rate=.05,max_depth=3,
        min_samples_leaf=300,random_state=20260927,criterion='friedman_mse',training_week_fraction=.75)
    f=base.feature_inputs();c=base.conn();c.register('features',f)
    c.execute(f'''CREATE VIEW targets AS WITH l AS(SELECT date,code,opportunity15 AS utility
        FROM read_parquet('{base.SOURCE}/full_labels.parquet') WHERE {where} AND known15)
        SELECT date,code,utility-avg(utility) OVER(PARTITION BY date) AS target FROM l''')
    names=list(base.EXPRESSIONS)
    encoded=','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::INT AS {n}' for n in names)
    d=c.sql('''SELECT date,code,target,1./count(*) OVER(PARTITION BY date) AS w,
        strftime(date_trunc('week',date::DATE),'%Y-%m-%d') AS week_start,'''+encoded+'''
        FROM features JOIN targets USING(date,code) WHERE formula_input_valid ORDER BY date,code''').df()
    c.close();original=relative.training('relative')
    assert d[['date','code']].equals(original[['date','code']]) and len(d)==r['rows'] and d.date.nunique()==r['days']
    assert original.next_date.lt(end).all() and original.date.ge(start).all()
    np.testing.assert_allclose(d.target,original.target,rtol=0,atol=2e-12)
    x=d[names].to_numpy(dtype='int32');y=d.target.to_numpy();w=d.w.to_numpy()
    np.testing.assert_array_equal(x,base.encode(original))
    np.testing.assert_allclose(r['bias'],np.average(y,weights=w),rtol=0,atol=2e-12)
    weeks=sorted(d.week_start.unique());size=(3*len(weeks)+3)//4
    assert r['training_week_starts']==weeks and r['weeks_per_tree']==size
    assert size<len(weeks) and len(r['trees'])==64 and len(r['sampling_schedule'])==64 and r['learning_rate']==.05
    rng=np.random.default_rng(20260927);score=np.full(len(d),r['bias']);checks=0
    sample_days=[];sample_rows=[]
    for index,(tree,item) in enumerate(zip(r['trees'],r['sampling_schedule'])):
        expected=sorted(rng.choice(weeks,size=size,replace=False).tolist())
        assert item['tree']==index and item['week_starts']==expected and len(set(expected))==size
        selected=d.week_start.isin(expected).to_numpy()
        assert item['rows']==int(selected.sum()) and item['days']==d.loc[selected,'date'].nunique()
        # Every date and every stock of a sampled week must receive the same inclusion decision.
        assert pd.Series(selected).groupby(d.week_start).nunique().eq(1).all()
        assert len(tree['feature'])<=15
        residual=y-score;masks={0:np.ones(len(d),dtype=bool)};levels={0:0};terminal=np.empty(len(d))
        for node in range(len(tree['feature'])):
            mask=masks[node];fit_mask=mask&selected;weights=w[fit_mask]
            assert levels[node]<=3 and fit_mask.sum()==tree['n_node_samples'][node]
            mean=np.average(residual[fit_mask],weights=weights)
            variance=np.average((residual[fit_mask]-mean)**2,weights=weights)
            np.testing.assert_allclose(weights.sum(),tree['weighted_n_node_samples'][node],rtol=0,atol=1e-8)
            np.testing.assert_allclose(mean,tree['value'][node],rtol=0,atol=2e-10)
            np.testing.assert_allclose(variance,tree['impurity'][node],rtol=0,atol=2e-10)
            left,right=tree['children_left'][node],tree['children_right'][node]
            if left<0:
                assert right<0 and fit_mask.sum()>=300
                terminal[mask]=mean
            else:
                split=x[:,tree['feature'][node]]<=tree['threshold'][node]
                masks[left],masks[right]=mask&split,mask&~split
                levels[left]=levels[right]=levels[node]+1
            checks+=1
        score+=.05*terminal;sample_days.append(item['days']);sample_rows.append(item['rows'])
    np.testing.assert_allclose(score,base.predict(x,r),rtol=0,atol=2e-10)
    for q,item in zip(base.QUANTILES,r['thresholds']):
        assert item['training_quantile']==q
        np.testing.assert_allclose(np.quantile(score,q),item['threshold'],rtol=0,atol=2e-10)
    proof=dict(passed=True,model_report_sha256=sha(root/'model_report.json'),rows=len(d),node_checks=checks,
        all_targets_integer_inputs_day_weights_residual_means_and_variances_rebuilt=True,
        all_week_blocks_sampling_schedules_and_entire_week_inclusions_rebuilt=True,
        training_weeks=len(weeks),sampled_weeks_per_tree=size,minimum_sampled_days=min(sample_days),
        maximum_sampled_days=max(sample_days),minimum_sampled_rows=min(sample_rows),maximum_sampled_rows=max(sample_rows),
        training_start=start,training_end=end,new_2025H2_score_groups_read=False,new_2026_prices_read=False,no_exit_rules=True)
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
