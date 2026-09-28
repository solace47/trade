"""Fixed one-year training with a 63-training-date half-life, without new inputs."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from . import tail_formula_additive as base
from . import tail_formula_before1000 as labels
from . import tail_formula_boundary_evaluation as evaluation
from . import tail_formula_context_2024 as linkage
from . import tail_formula_float as inputs
from . import tail_formula_prior_day as adapter
from . import tail_formula_recent as study
from . import tail_formula_relative as relative
from .corporate_cash import save_json, sha

STEM='tail_formula_recency_weight'
PROTOCOL=Path('config')/(STEM+'_protocol.json')


def date_weights(dates,half_life=63):
    """Normalize date totals first; split each total equally among its stocks."""
    assert half_life>0 and len(dates)>0 and dates.notna().all()
    unique=sorted(dates.unique())
    age=np.arange(len(unique)-1,-1,-1)
    totals=np.exp2(-age/half_life);totals*=len(unique)/totals.sum()
    table=pd.DataFrame(dict(date=unique,age=age,date_weight=totals))
    weight=dates.map(dict(zip(unique,totals)))/dates.groupby(dates).transform('size')
    return weight.to_numpy(),table


def setup(fold):
    p=json.loads(PROTOCOL.read_text())
    assert p['half_life_training_dates']==63 and not p['new_2026_prices_allowed']
    for file,digest in p['source_hashes'].items():assert sha(Path(file))==digest
    combined=Path('config')/(STEM+'_combined_protocol.json');q=json.loads(combined.read_text())
    assert q['master_protocol_sha256']==sha(PROTOCOL)
    control=Path(q['control'])
    for stage in ['selection','analysis']:
        proof=json.loads((control/(stage+'_verification.json')).read_text())
        assert proof['passed'] and proof[stage+'_report_sha256']==q['control_'+stage+'_report_sha256']==sha(control/(stage+'_report.json'))
    adapter.STEM=STEM;adapter.ROOT=inputs.ROOT;adapter.EXPRESSIONS=inputs.EXPRESSIONS;adapter.HEADER=inputs.HEADER
    adapter.COMBINED_PROTOCOL=combined;adapter.setup(fold);base.SOURCE=labels.ROOT
    for name in ['2024','recent']:
        q=json.loads((Path('config')/(STEM+'_'+name+'_protocol.json')).read_text())
        assert q['master_protocol_sha256']==sha(PROTOCOL) and q['half_life_training_dates']==63
        assert q['feature_report_sha256']==sha(inputs.ROOT/'feature_report.json')
        assert q['label_report_sha256']==sha(labels.ROOT/'full_label_report.json')
        assert q['parameters']==p['parameters'] and q['expected_features']==48


def model():
    root=base.ROOT;assert not (root/'model_report.json').exists()
    train=relative.training('relative');p=json.loads(base.PROTOCOL.read_text())
    assert len(train)==p['expected_training_rows'] and train.date.nunique()==241
    x=base.encode(train);y=train.target.to_numpy();w,dates=date_weights(train.date)
    estimator=GradientBoostingRegressor(**p['parameters']);estimator.fit(x,y,sample_weight=w)
    trees=[]
    for fitted in estimator.estimators_.ravel():
        t=fitted.tree_
        tree={key:getattr(t,key).tolist() for key in ['feature','threshold','children_left','children_right',
            'n_node_samples','weighted_n_node_samples','impurity']}
        tree['value']=t.value.reshape(-1).tolist();trees.append(tree)
    report=dict(protocol_sha256=sha(base.PROTOCOL),master_protocol_sha256=sha(PROTOCOL),
        feature_report_sha256=sha(inputs.ROOT/'feature_report.json'),label_report_sha256=sha(labels.ROOT/'full_label_report.json'),
        rows=len(train),days=train.date.nunique(),last_observation=train.next_date.max(),parameters=estimator.get_params(),
        feature_names=list(inputs.EXPRESSIONS),variant='relative_recency63',learning_rate=.05,
        bias=float(np.ravel(estimator.init_.constant_)[0]),trees=trees,training_date_weights=dates.to_dict('records'),
        date_weight_effective_count=float(dates.date_weight.sum()**2/(dates.date_weight**2).sum()),
        training_start=p['training_start'],training_end=p['training_end'],
        new_2025_score_groups_read=bool(train.date.ge('2025-01-01').any()),new_2025H2_score_groups_read=False,
        new_2026_prices_read=False,no_exit_rules=True)
    score=base.predict(x,report);np.testing.assert_allclose(score,estimator.predict(x),rtol=0,atol=2e-12)
    report['thresholds']=[dict(id=i,training_quantile=q,threshold=float(np.quantile(score,q))) for i,q in enumerate(base.QUANTILES)]
    root.mkdir(parents=True,exist_ok=True);save_json(root/'model_report.json',report)
    return {k:v for k,v in report.items() if k not in ['trees','training_date_weights']}


def verify_model():
    root=base.ROOT;p=json.loads(base.PROTOCOL.read_text());r=json.loads((root/'model_report.json').read_text())
    for field,path in [('protocol_sha256',base.PROTOCOL),('master_protocol_sha256',PROTOCOL),
        ('feature_report_sha256',inputs.ROOT/'feature_report.json'),('label_report_sha256',labels.ROOT/'full_label_report.json')]:
        assert r[field]==sha(path)
    assert r['variant']=='relative_recency63' and r['feature_names']==list(inputs.EXPRESSIONS)
    assert all(r['parameters'][k]==v for k,v in p['parameters'].items())
    start,end,where=relative.training_scope();c=base.conn();names=list(inputs.EXPRESSIONS)
    c.register('features',base.feature_inputs()[['date','code','formula_input_valid',*names]])
    c.execute(f'''CREATE VIEW targets AS SELECT date,code,opportunity15-avg(opportunity15) OVER(PARTITION BY date) AS target
        FROM read_parquet('{labels.ROOT}/full_labels.parquet') WHERE {where} AND known15''')
    c.sql('SELECT * FROM features JOIN targets USING(date,code) WHERE formula_input_valid').create_view('valid')
    c.sql('''WITH dates AS(SELECT DISTINCT date FROM valid),ranks AS(
        SELECT date,(count(*) OVER()-dense_rank() OVER(ORDER BY date))::INT AS age FROM dates),
        powers AS(SELECT *,power(2.,-age/63.) AS raw_weight FROM ranks)
        SELECT date,age,raw_weight/avg(raw_weight) OVER() AS date_weight FROM powers''').create_view('date_weights')
    dates=c.sql('SELECT * FROM date_weights ORDER BY date').df()
    fields=','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::INT AS {n}' for n in names)
    d=c.sql('SELECT date,code,target,date_weight/count(*) OVER(PARTITION BY date) AS w,'+fields+
        ' FROM valid JOIN date_weights USING(date) ORDER BY date,code').df();c.close()
    t=relative.training('relative');original_w,original_dates=date_weights(t.date)
    pd.testing.assert_frame_equal(d[['date','code']],t[['date','code']],check_exact=True)
    pd.testing.assert_frame_equal(dates,original_dates,check_dtype=False,rtol=0,atol=2e-12)
    pd.testing.assert_frame_equal(dates,pd.DataFrame(r['training_date_weights']),check_dtype=False,rtol=0,atol=2e-12)
    np.testing.assert_allclose(d.target,t.target,rtol=0,atol=2e-12)
    np.testing.assert_allclose(d.w,original_w,rtol=0,atol=2e-15)
    np.testing.assert_array_equal(d[names].to_numpy(dtype='int32'),base.encode(t))
    assert len(d)==r['rows']==p['expected_training_rows'] and len(dates)==r['days']==241
    np.testing.assert_allclose(r['date_weight_effective_count'],dates.date_weight.sum()**2/(dates.date_weight**2).sum(),rtol=0,atol=2e-10)
    x=d[names].to_numpy(dtype='int32');y=d.target.to_numpy();w=d.w.to_numpy()
    np.testing.assert_allclose(r['bias'],np.average(y,weights=w),rtol=0,atol=2e-12)
    score=np.full(len(d),r['bias']);nodes=0;leaves=0
    assert len(r['trees'])==64 and r['learning_rate']==.05
    for tree in r['trees']:
        residual=y-score;masks={0:np.ones(len(d),dtype=bool)};levels={0:0};terminal=np.empty(len(d))
        for i in range(len(tree['feature'])):
            assert levels[i]<=3;mask=masks[i];weights=w[mask];left,right=tree['children_left'][i],tree['children_right'][i]
            assert int(mask.sum())==tree['n_node_samples'][i]
            mean=np.average(residual[mask],weights=weights);variance=np.average((residual[mask]-mean)**2,weights=weights)
            np.testing.assert_allclose(weights.sum(),tree['weighted_n_node_samples'][i],rtol=0,atol=1e-8)
            np.testing.assert_allclose(mean,tree['value'][i],rtol=0,atol=2e-10)
            np.testing.assert_allclose(variance,tree['impurity'][i],rtol=0,atol=2e-8)
            if left<0:
                assert mask.sum()>=300;terminal[mask]=tree['value'][i];leaves+=1
            else:
                lower=x[:,tree['feature'][i]]<=tree['threshold'][i]
                masks[left],masks[right]=mask&lower,mask&~lower;levels[left]=levels[right]=levels[i]+1
            nodes+=1
        score+=.05*terminal
    for q,cut in zip(base.QUANTILES,r['thresholds']):
        assert q==cut['training_quantile'];np.testing.assert_allclose(np.quantile(score,q),cut['threshold'],rtol=0,atol=2e-10)
    assert r['last_observation']==t.next_date.max()<end and t.date.min()>=start
    proof=dict(passed=True,model_report_sha256=sha(root/'model_report.json'),rows=len(d),days=len(dates),node_checks=nodes,
        leaf_checks=leaves,all_training_keys_targets_weights_integer_inputs_node_statistics_and_quantiles_rebuilt=True,
        original_training_intersection_unchanged=True,training_start=start,training_end=end,new_2026_prices_read=False,no_exit_rules=True)
    save_json(root/'model_verification.json',proof);return proof


if __name__=='__main__':
    from .tail_formula_offset_logit48 import verify_scores
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['model','verify_model','scores','verify_scores','freeze','verify','analyze'])
    parser.add_argument('--fold',choices=['2024','recent','combined'],default='2024')
    args=parser.parse_args();setup(args.fold)
    if args.stage=='analyze':
        for fold in ['2024','recent','2025']:
            root=Path('data/research')/(STEM+'_'+fold);proof=json.loads((root/'selection_verification.json').read_text())
            assert proof['passed'] and proof['selection_report_sha256']==sha(root/'selection_report.json')
        result=evaluation.analyze(linkage.COMBINED if args.fold=='combined' else base.ROOT,
                                  linkage.PROTOCOL if args.fold=='combined' else base.PROTOCOL)
    elif args.fold=='combined':
        assert args.stage in ['freeze','verify'];result=linkage.combine() if args.stage=='freeze' else linkage.verify_combined()
    elif args.stage=='model':result=model()
    elif args.stage=='verify_model':result=verify_model()
    elif args.stage=='verify_scores':result=verify_scores()
    elif args.stage=='freeze':result=study.freeze()
    elif args.stage=='verify':
        report=json.loads((base.ROOT/'model_report.json').read_text())
        assert (base.ROOT/'frozen_numeric_core.tdx').read_text()==base.native_core(report,report['thresholds'][3]['threshold'],base.EXPRESSIONS,base.HEADER)
        result=study.verify()
    else:result=base.scores()
    print(json.dumps(result,ensure_ascii=False,indent=2))
