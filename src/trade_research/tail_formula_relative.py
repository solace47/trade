"""Two fixed additive objectives: same-day opportunity and risk-adjusted opportunity."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from . import tail_formula_additive as base
from .corporate_cash import save_json,sha

PROTOCOL=Path('config/tail_formula_relative_protocol.json')


def setup(variant):
    base.ROOT=Path('data/research/tail_formula_'+variant)
    base.PROTOCOL=PROTOCOL


def training(variant):
    t=base.training()
    c=base.conn()
    # Price opportunities and adverse marks only; no future exit fields.
    labels=c.execute('''SELECT date,code,opportunity15,adverse_return15 FROM read_parquet(?)
        WHERE next_date<'2025-01-01' AND known15''',[str(base.SOURCE/'full_labels.parquet')]).df()
    assert labels.adverse_return15.notna().all()
    labels['utility']=labels.opportunity15-(labels.adverse_return15.le(-.03) if variant=='risk' else 0)
    labels['target']=labels.utility-labels.groupby('date').utility.transform('mean')
    t=t.merge(labels[['date','code','target']],on=['date','code'],validate='one_to_one')
    return t.sort_values(['date','code']).reset_index(drop=True)


def model(variant):
    root=base.ROOT
    if (root/'model_report.json').exists():
        raise ValueError('Do not refit the frozen relative model')
    root.mkdir(parents=True,exist_ok=True)
    train=training(variant)
    x=base.encode(train);y=train.target.to_numpy()
    w=1/train.groupby('date').code.transform('size').to_numpy()
    model=GradientBoostingRegressor(loss='squared_error',n_estimators=64,learning_rate=.05,max_depth=2,
        min_samples_leaf=300,subsample=1.,random_state=20260927)
    model.fit(x,y,sample_weight=w)
    trees=[]
    for estimator in model.estimators_.ravel():
        t=estimator.tree_
        trees.append(dict(feature=t.feature.tolist(),threshold=t.threshold.tolist(),children_left=t.children_left.tolist(),
            children_right=t.children_right.tolist(),n_node_samples=t.n_node_samples.tolist(),
            weighted_n_node_samples=t.weighted_n_node_samples.tolist(),value=t.value.reshape(-1).tolist(),impurity=t.impurity.tolist()))
    r=dict(protocol_sha256=sha(PROTOCOL),feature_report_sha256=sha(base.FEATURES/'feature_report.json'),
        label_report_sha256=sha(base.SOURCE/'full_label_report.json'),rows=len(train),days=train.date.nunique(),
        last_observation=train.next_date.max(),parameters=model.get_params(),feature_names=list(base.EXPRESSIONS),
        variant=variant,learning_rate=.05,bias=float(np.ravel(model.init_.constant_)[0]),trees=trees,
        new_2025_score_groups_read=False,new_2026_prices_read=False,no_exit_rules=True)
    manual=base.predict(x,r)
    np.testing.assert_allclose(manual,model.predict(x),rtol=0,atol=2e-12)
    r['thresholds']=[dict(id=i,training_quantile=q,threshold=float(np.quantile(manual,q))) for i,q in enumerate(base.QUANTILES)]
    save_json(root/'model_report.json',r)
    return {k:v for k,v in r.items() if k!='trees'}


def verify_model(variant):
    root=base.ROOT
    r=json.loads((root/'model_report.json').read_text())
    assert r['protocol_sha256']==sha(PROTOCOL) and r['variant']==variant
    assert r['feature_report_sha256']==sha(base.FEATURES/'feature_report.json')
    assert r['label_report_sha256']==sha(base.SOURCE/'full_label_report.json')
    f=base.feature_inputs();c=base.conn();c.register('features',f)
    utility='opportunity15'+('-CAST(adverse_return15<=-.03 AS INTEGER)' if variant=='risk' else '')
    c.execute(f'''CREATE VIEW targets AS WITH l AS(SELECT date,code,{utility} AS utility
        FROM read_parquet('{base.SOURCE}/full_labels.parquet') WHERE next_date<'2025-01-01' AND known15)
        SELECT date,code,utility-avg(utility) OVER(PARTITION BY date) AS target FROM l''')
    names=list(base.EXPRESSIONS)
    encoded=','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::INT AS {n}' for n in names)
    d=c.sql('''SELECT date,code,target,1./count(*) OVER(PARTITION BY date) AS w,'''+encoded+'''
        FROM features JOIN targets USING(date,code) WHERE formula_input_valid ORDER BY date,code''').df()
    original=training(variant)
    np.testing.assert_allclose(d.target,original.target,rtol=0,atol=2e-12)
    x=d[names].to_numpy(dtype='int32');y=d.target.to_numpy();w=d.w.to_numpy()
    np.testing.assert_array_equal(x,base.encode(original))
    np.testing.assert_allclose(r['bias'],np.average(y,weights=w),rtol=0,atol=2e-12)
    score=np.full(len(d),r['bias']);checks=0
    assert len(r['trees'])==64 and r['learning_rate']==.05
    for tree in r['trees']:
        residual=y-score;masks={0:np.ones(len(d),dtype=bool)};terminal=np.empty(len(d))
        for i in range(len(tree['feature'])):
            mask=masks[i];weights=w[mask]
            assert int(mask.sum())==tree['n_node_samples'][i]
            mean=np.average(residual[mask],weights=weights)
            variance=np.average((residual[mask]-mean)**2,weights=weights)
            np.testing.assert_allclose(weights.sum(),tree['weighted_n_node_samples'][i],rtol=0,atol=1e-8)
            np.testing.assert_allclose(mean,tree['value'][i],rtol=0,atol=2e-10)
            np.testing.assert_allclose(variance,tree['impurity'][i],rtol=0,atol=2e-10)
            left,right=tree['children_left'][i],tree['children_right'][i]
            if left<0:
                assert mask.sum()>=300
                terminal[mask]=mean
            else:
                lower=x[:,tree['feature'][i]]<=tree['threshold'][i]
                masks[left],masks[right]=mask&lower,mask&~lower
            checks+=1
        score+=.05*terminal
    for q,t in zip(base.QUANTILES,r['thresholds']):
        assert q==t['training_quantile']
        np.testing.assert_allclose(np.quantile(score,q),t['threshold'],rtol=0,atol=2e-10)
    proof=dict(passed=True,model_report_sha256=sha(root/'model_report.json'),variant=variant,rows=len(d),node_checks=checks,
        all_targets_integer_inputs_day_weights_residual_means_and_variances_rebuilt=True,
        new_2025_score_groups_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(root/'model_verification.json',proof)
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('variant',choices=['relative','risk'])
    p.add_argument('stage',choices=['model','verify_model','scores','calibrate','analyze'])
    args=p.parse_args();setup(args.variant)
    result=globals()[args.stage](args.variant) if args.stage in ['model','verify_model'] else getattr(base,args.stage)()
    print(json.dumps(result,ensure_ascii=False,indent=2))
