"""Paired squared and Huber losses on one fixed 09:59 reference-price target."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor

from . import tail_formula_additive as base
from . import tail_formula_before1000 as labels
from . import tail_formula_boundary_evaluation as evaluation
from . import tail_formula_context_2024 as linkage
from . import tail_formula_float as inputs
from . import tail_formula_prior_day as adapter
from . import tail_formula_recent as study
from .corporate_cash import save_json, sha

STEM='tail_formula_endpoint_robust'
ROOT=Path('data/research')/STEM
PROTOCOL=Path('config')/(STEM+'_v2_protocol.json')


def weighted_quantile(value,weight,q):
    value=np.asarray(value);weight=np.asarray(weight)
    assert np.isfinite(value).all() and np.isfinite(weight).all() and (weight>0).all() and 0<q<1
    order=np.argsort(value);cumulative=np.cumsum(weight[order])
    return float(value[order[min(np.searchsorted(cumulative,q*cumulative[-1]),len(order)-1)]])


def huber_leaf(residual,weight,delta):
    median=weighted_quantile(residual,weight,.5)
    return median+np.average(np.clip(residual-median,-delta,delta),weights=weight)


def sources():
    p=json.loads(PROTOCOL.read_text());assert sklearn.__version__=='1.7.2'
    assert p['alpha']==.9 and not p['new_2026_prices_allowed']
    assert p['supersedes_before_any_fit_sha256']==sha(Path('config')/(STEM+'_protocol.json'))
    for file,digest in p['source_hashes'].items():assert sha(Path(file))==digest
    manifest=json.loads((ROOT/'source/manifest.json').read_text())
    for item in manifest['sources']:
        rel=item['url'].split('/1.7.2/')[1]
        local=Path(sklearn.__file__).parent.parent/rel
        assert sha(local)==item['sha256']==sha(Path(item['path']))
    return p


def setup(arm,fold):
    sources();stem='tail_formula_endpoint_'+arm
    combined=Path('config')/(stem+'_combined_protocol.json');p=json.loads(combined.read_text())
    control=Path(p['control'])
    for stage in ['selection','analysis']:
        proof=json.loads((control/(stage+'_verification.json')).read_text())
        assert proof['passed'] and proof[stage+'_report_sha256']==p['control_'+stage+'_report_sha256']==sha(control/(stage+'_report.json'))
    adapter.STEM=stem;adapter.ROOT=inputs.ROOT;adapter.EXPRESSIONS=inputs.EXPRESSIONS;adapter.HEADER=inputs.HEADER
    adapter.COMBINED_PROTOCOL=combined;adapter.setup(fold);base.SOURCE=labels.ROOT
    for name in ['2024','recent']:
        q=json.loads((Path('config')/(stem+'_'+name+'_protocol.json')).read_text())
        assert q['master_protocol_sha256']==sha(PROTOCOL)
        assert q['feature_report_sha256']==sha(inputs.ROOT/'feature_report.json')
        assert q['label_report_sha256']==sha(labels.ROOT/'full_label_report.json')
        assert q['expected_features']==48 and q['loss_arm']==arm
        assert q['training_reference_required']=='mark_0959_return15'


def training():
    p=json.loads(base.PROTOCOL.read_text());start,end=p['training_start'],p['training_end']
    t=base.training(start=start,end=end)
    c=base.conn()
    l=c.execute('''SELECT date,code,mark_0959_return15,opportunity15 FROM read_parquet(?)
        WHERE date>=? AND next_date<? AND known15''',[str(labels.ROOT/'full_labels.parquet'),start,end]).df();c.close()
    finite=np.isfinite(l.mark_0959_return15)
    assert int((~finite).sum())==p['expected_reference_missing_known_rows']
    assert len(t)==p['original_training_rows']
    l=l.loc[finite].copy()
    l['target']=100*l.mark_0959_return15
    l['target']-=l.groupby('date').target.transform('mean')
    l['binary_target']=l.opportunity15-l.groupby('date').opportunity15.transform('mean')
    if p['loss_arm']=='binary':l['target']=l.binary_target
    out=t.merge(l[['date','code','target','binary_target']],on=['date','code'],validate='one_to_one')
    assert len(out)==p['expected_training_rows'] and out.date.nunique()==p['expected_training_days']==241
    return out.sort_values(['date','code']).reset_index(drop=True)


def independent_training():
    p=json.loads(base.PROTOCOL.read_text());start,end=p['training_start'],p['training_end']
    f=base.feature_inputs()[['date','code','formula_input_valid',*base.EXPRESSIONS]]
    c=base.conn();c.register('features',f)
    c.execute('''CREATE TEMP TABLE raw_labels AS SELECT * FROM read_parquet(?)
        WHERE date>=? AND next_date<? AND known15''',[str(labels.ROOT/'full_labels.parquet'),start,end])
    checks=c.sql('''WITH x AS(SELECT *,decision_shares*(entry_vwap+greatest(entry_vwap*.0015,.005)) AS bv,
        decision_shares*(price_0959-greatest(price_0959*.0015,.005)) AS sv FROM raw_labels),
        y AS(SELECT *,bv+greatest(5.,bv*.0003)+bv*.00001 AS buy,
        sv-greatest(5.,sv*.0003)-sv*.00051 AS sell FROM x)
        SELECT count(*) AS rows,count(*) FILTER(WHERE NOT isfinite(mark_0959_return15) OR mark_0959_return15 IS NULL) AS missing,
        max(abs(buy-buy_cash15)) AS buy_error,max(abs(sell/buy-1-mark_0959_return15)) AS mark_error FROM y''').df().iloc[0]
    assert checks.missing==p['expected_reference_missing_known_rows'] and checks.buy_error<1e-7 and checks.mark_error<2e-12
    endpoint='100*mark_0959_return15-avg(100*mark_0959_return15) OVER(PARTITION BY date)'
    binary='opportunity15-avg(opportunity15) OVER(PARTITION BY date)'
    target=binary if p['loss_arm']=='binary' else endpoint
    c.sql(f'''SELECT date,code,{target} AS target,{binary} AS binary_target
        FROM raw_labels WHERE isfinite(mark_0959_return15)''').create_view('targets')
    names=list(base.EXPRESSIONS)
    fields=','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::INT AS {n}' for n in names)
    d=c.sql('SELECT date,code,target,binary_target,1./count(*) OVER(PARTITION BY date) AS weight,'+fields+
        ' FROM features JOIN targets USING(date,code) WHERE formula_input_valid ORDER BY date,code').df();c.close()
    t=training();pd.testing.assert_frame_equal(d[['date','code']],t[['date','code']],check_exact=True)
    np.testing.assert_allclose(d.target,t.target,rtol=0,atol=2e-12)
    np.testing.assert_allclose(d.binary_target,t.binary_target,rtol=0,atol=2e-12)
    np.testing.assert_allclose(d.weight,1/t.groupby('date').code.transform('size'),rtol=0,atol=0)
    np.testing.assert_array_equal(d[names].to_numpy(dtype='int32'),base.encode(t))
    return d,t,dict(original_known_rows=int(checks['rows']),reference_missing_rows=int(checks.missing),
        original_training_rows=p['original_training_rows'],removed_training_rows=p['original_training_rows']-len(t),
        max_buy_cash_difference=float(checks.buy_error),max_reference_difference=float(checks.mark_error))


def verify_inputs(fold):
    d,t,checks=independent_training();ROOT.mkdir(parents=True,exist_ok=True)
    proof=dict(passed=True,master_protocol_sha256=sha(PROTOCOL),feature_report_sha256=sha(inputs.ROOT/'feature_report.json'),
        label_report_sha256=sha(labels.ROOT/'full_label_report.json'),rows=len(t),days=t.date.nunique(),
        first_signal=t.date.min(),last_observation=t.next_date.max(),checks=checks,
        all_reference_costs_targets_keys_date_weights_and_integer_inputs_rebuilt=True,
        same_finite_reference_training_intersection_for_all_three_arms=True,
        inference_pool_unchanged_and_future_reference_never_used_to_select=True,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/('training_'+fold+'_verification.json'),proof);return proof


def model(arm,fold):
    root=base.ROOT;assert not (root/'model_report.json').exists()
    proof=json.loads((ROOT/('training_'+fold+'_verification.json')).read_text())
    assert proof['passed'] and proof['master_protocol_sha256']==sha(PROTOCOL)
    train=training();x=base.encode(train);y=train.target.to_numpy();w=1/train.groupby('date').code.transform('size').to_numpy()
    p=json.loads(base.PROTOCOL.read_text());estimator=GradientBoostingRegressor(**p['parameters'])
    estimator.fit(x,y,sample_weight=w);trees=[]
    for fitted in estimator.estimators_.ravel():
        tree=fitted.tree_
        trees.append({key:getattr(tree,key).tolist() for key in ['feature','threshold','children_left','children_right','n_node_samples','weighted_n_node_samples','impurity']})
        trees[-1]['value']=tree.value.reshape(-1).tolist()
    report=dict(protocol_sha256=sha(base.PROTOCOL),master_protocol_sha256=sha(PROTOCOL),
        training_input_verification_sha256=sha(ROOT/('training_'+fold+'_verification.json')),
        feature_report_sha256=sha(inputs.ROOT/'feature_report.json'),label_report_sha256=sha(labels.ROOT/'full_label_report.json'),
        rows=len(train),days=train.date.nunique(),last_observation=train.next_date.max(),parameters=estimator.get_params(),
        feature_names=list(inputs.EXPRESSIONS),variant='endpoint_'+arm,learning_rate=.05,bias=float(np.ravel(estimator.init_.constant_)[0]),trees=trees,
        training_start=p['training_start'],training_end=p['training_end'],new_2025_score_groups_read=bool(train.date.ge('2025-01-01').any()),
        training_reference_required=p['training_reference_required'],
        new_2025H2_score_groups_read=False,new_2026_prices_read=False,no_exit_rules=True)
    score=base.predict(x,report);np.testing.assert_allclose(score,estimator.predict(x),rtol=0,atol=2e-12)
    report['thresholds']=[dict(id=i,training_quantile=q,threshold=float(np.quantile(score,q))) for i,q in enumerate(base.QUANTILES)]
    root.mkdir(parents=True,exist_ok=True);save_json(root/'model_report.json',report)
    return {k:v for k,v in report.items() if k!='trees'}


def verify_model(arm,fold):
    p=json.loads(base.PROTOCOL.read_text());r=json.loads((base.ROOT/'model_report.json').read_text())
    for key,path in [('protocol_sha256',base.PROTOCOL),('master_protocol_sha256',PROTOCOL),
        ('feature_report_sha256',inputs.ROOT/'feature_report.json'),('label_report_sha256',labels.ROOT/'full_label_report.json'),
        ('training_input_verification_sha256',ROOT/('training_'+fold+'_verification.json'))]:assert r[key]==sha(path)
    assert all(r['parameters'][key]==value for key,value in p['parameters'].items())
    assert r['variant']=='endpoint_'+arm and r['feature_names']==list(inputs.EXPRESSIONS)
    d,t,checks=independent_training();x=d[list(inputs.EXPRESSIONS)].to_numpy(dtype='int32');y=d.target.to_numpy();w=d.weight.to_numpy()
    bias=weighted_quantile(y,w,.5) if arm=='huber' else np.average(y,weights=w)
    np.testing.assert_allclose(r['bias'],bias,rtol=0,atol=2e-12)
    score=np.full(len(d),r['bias']);nodes=0;leaf_checks=0;deltas=[]
    assert len(r['trees'])==64 and r['learning_rate']==.05 and r['rows']==len(d) and r['days']==241
    for tree in r['trees']:
        residual=y-score;delta=weighted_quantile(np.abs(residual),w,.9) if arm=='huber' else None
        gradient=np.clip(residual,-delta,delta) if arm=='huber' else residual
        if delta is not None:deltas.append(delta)
        masks={0:np.ones(len(d),dtype=bool)};levels={0:0};terminal=np.empty(len(d))
        for i in range(len(tree['feature'])):
            assert levels[i]<=3;mask=masks[i];weights=w[mask];left,right=tree['children_left'][i],tree['children_right'][i]
            assert int(mask.sum())==tree['n_node_samples'][i]
            mean=np.average(gradient[mask],weights=weights);variance=np.average((gradient[mask]-mean)**2,weights=weights)
            np.testing.assert_allclose(weights.sum(),tree['weighted_n_node_samples'][i],rtol=0,atol=1e-8)
            np.testing.assert_allclose(variance,tree['impurity'][i],rtol=0,atol=2e-8)
            value=huber_leaf(residual[mask],weights,delta) if arm=='huber' and left<0 else mean
            np.testing.assert_allclose(value,tree['value'][i],rtol=0,atol=2e-10)
            if left<0:
                assert mask.sum()>=300;terminal[mask]=tree['value'][i];leaf_checks+=1
            else:
                lower=x[:,tree['feature'][i]]<=tree['threshold'][i]
                masks[left],masks[right]=mask&lower,mask&~lower;levels[left]=levels[right]=levels[i]+1
            nodes+=1
        score+=.05*terminal
    for q,cut in zip(base.QUANTILES,r['thresholds']):
        assert q==cut['training_quantile'];np.testing.assert_allclose(np.quantile(score,q),cut['threshold'],rtol=0,atol=2e-10)
    assert r['last_observation']==t.next_date.max()<p['training_end'] and t.date.min()>=p['training_start']
    proof=dict(passed=True,model_report_sha256=sha(base.ROOT/'model_report.json'),variant=r['variant'],rows=len(d),node_checks=nodes,
        leaf_checks=leaf_checks,huber_stage_deltas=deltas,reference_checks=checks,
        all_targets_inputs_weights_gradients_node_statistics_leaf_updates_and_quantiles_rebuilt=True,
        training_start=p['training_start'],training_end=p['training_end'],new_2026_prices_read=False,no_exit_rules=True)
    save_json(base.ROOT/'model_verification.json',proof);return proof


if __name__=='__main__':
    from .tail_formula_offset_logit48 import verify_scores
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['verify_inputs','model','verify_model','scores','verify_scores','freeze','verify','analyze'])
    parser.add_argument('--arm',choices=['squared','huber','binary'],default='squared')
    parser.add_argument('--fold',choices=['2024','recent','combined'],default='2024')
    args=parser.parse_args();setup(args.arm,args.fold)
    if args.stage=='analyze':
        for arm in ['squared','huber','binary']:
            for fold in ['2024','recent','2025']:
                root=Path('data/research')/('tail_formula_endpoint_'+arm+'_'+fold)
                proof=json.loads((root/'selection_verification.json').read_text())
                assert proof['passed'] and proof['selection_report_sha256']==sha(root/'selection_report.json')
        result=evaluation.analyze(linkage.COMBINED if args.fold=='combined' else base.ROOT,linkage.PROTOCOL if args.fold=='combined' else base.PROTOCOL)
    elif args.fold=='combined':
        assert args.stage in ['freeze','verify'];result=linkage.combine() if args.stage=='freeze' else linkage.verify_combined()
    elif args.stage=='verify_inputs':result=verify_inputs(args.fold)
    elif args.stage=='model':result=model(args.arm,args.fold)
    elif args.stage=='verify_model':result=verify_model(args.arm,args.fold)
    elif args.stage=='verify_scores':result=verify_scores()
    elif args.stage=='freeze':result=study.freeze()
    elif args.stage=='verify':
        report=json.loads((base.ROOT/'model_report.json').read_text())
        assert (base.ROOT/'frozen_numeric_core.tdx').read_text()==base.native_core(report,report['thresholds'][3]['threshold'],base.EXPRESSIONS,base.HEADER)
        result=study.verify()
    else:result=base.scores()
    print(json.dumps(result,ensure_ascii=False,indent=2))
