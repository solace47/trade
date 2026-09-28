"""Fixed additive objectives with optional same-day centering."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from . import tail_formula_additive as base
from .corporate_cash import save_json,sha

PROTOCOL=Path('config/tail_formula_relative_protocol.json')


def training_scope():
    from datetime import date
    r=json.loads(PROTOCOL.read_text())
    start,end=r.get('training_start'),r.get('training_end','2025-01-01')
    assert date.fromisoformat(end).isoformat()==end
    assert start is None or date.fromisoformat(start).isoformat()==start
    where=f"next_date<'{end}'"+(f" AND date>='{start}'" if start else '')
    return start,end,where


def setup(variant):
    base.ROOT=Path('data/research/tail_formula_'+variant)
    base.PROTOCOL=PROTOCOL


def duration_source():
    from . import tail_formula_duration_labels as durations
    config=json.loads(PROTOCOL.read_text())
    assert sha(durations.PROTOCOL)==config['duration_labels_protocol_sha256']
    report=durations.checked_report()
    assert report['full_label_report_sha256']==sha(base.SOURCE/'full_label_report.json')
    assert report['full_label_verification_sha256']==sha(base.SOURCE/'full_label_verification.json')
    return durations.ROOT


def training_size_buckets(frame, groups):
    """Rank the visible universe before any next-morning label intersection."""
    assert groups in [1,5]
    assert not frame.duplicated(['date','code']).any() and np.isfinite(frame.S01).all()
    ranks=frame.groupby('date').S01.rank(method='average')
    counts=frame.groupby('date').code.transform('size')
    out=frame[['date','code']].copy()
    out['size_bucket']=np.floor(groups*(ranks-.5)/counts).astype('int8')
    assert out.size_bucket.between(0,groups-1).all()
    return out


def training(variant):
    assert variant in ['relative','risk','absolute','margin','rank','duration','downside','quality','space','group_relative']
    start,end,where=training_scope()
    t=base.training(start=start,end=end)
    c=base.conn()
    if variant=='group_relative':
        groups=json.loads(PROTOCOL.read_text())['training_size_groups']
        f=base.feature_inputs()[['date','code','S01','formula_input_valid']]
        visible=f.loc[f.formula_input_valid & f.date.lt(end) & (True if start is None else f.date.ge(start))]
        buckets=training_size_buckets(visible,groups)
        t=t.merge(buckets,on=['date','code'],validate='one_to_one')
        assert len(t)>0
        t['target']=t.opportunity15-t.groupby(['date','size_bucket']).opportunity15.transform('mean')
        c.close()
        return t.sort_values(['date','code']).reset_index(drop=True)
    if variant=='duration':
        source=duration_source()
        labels=c.execute(f'''SELECT date,code,profitable_fraction15 AS utility
            FROM read_parquet(?) WHERE {where}''',[str(source/'duration_labels.parquet')]).df()
        assert np.isfinite(labels.utility).all() and labels.utility.between(0,1).all()
        labels['target']=labels.utility-labels.groupby('date').utility.transform('mean')
        out=t.merge(labels[['date','code','target']],on=['date','code'],validate='one_to_one')
        assert len(out)==len(t)
        return out.sort_values(['date','code']).reset_index(drop=True)
    # Price opportunities and adverse marks only; no future exit fields.
    extra=',sustained_return15' if variant in ['margin','rank','quality','space'] else ''
    labels=c.execute(f'''SELECT date,code,opportunity15,adverse_return15{extra} FROM read_parquet(?)
        WHERE {where} AND known15''',[str(base.SOURCE/'full_labels.parquet')]).df()
    # A fully observed morning with no positive-volume minute has no profit
    # opportunity, while its reference-price marks remain unavailable. Only
    # objectives using downside need a finite adverse mark; do not fabricate
    # one or drop the known zero opportunity from the same-day base mean.
    assert np.isfinite(labels.opportunity15).all() and labels.opportunity15.isin([0,1]).all()
    if variant in ['risk','downside','quality']:
        assert np.isfinite(labels.adverse_return15).all()
    if variant=='rank':
        # Rank only the exact training intersection, not rows excluded by input quality.
        t=t.merge(labels[['date','code','sustained_return15']],on=['date','code'],validate='one_to_one')
        assert np.isfinite(t.sustained_return15).all()
        grouped=t.groupby('date').sustained_return15
        t['target']=(grouped.rank(method='average')-.5)/grouped.transform('size')-.5
        np.testing.assert_allclose(t.groupby('date').target.mean(),0,rtol=0,atol=2e-12)
        return t.sort_values(['date','code']).reset_index(drop=True)
    if variant=='margin':
        assert np.isfinite(labels.sustained_return15).all()
        labels['utility']=labels.sustained_return15.clip(-.01,.01)/.01
    elif variant=='downside':
        labels['utility']=labels.adverse_return15.le(-.03).astype(float)
    elif variant=='quality':
        assert np.isfinite(labels.sustained_return15).all()
        labels['utility']=(labels.sustained_return15.ge(.01)&labels.adverse_return15.gt(-.03)).astype(float)
    elif variant=='space':
        assert np.isfinite(labels.sustained_return15).all()
        labels['utility']=labels.sustained_return15.ge(.01).astype(float)
    else:
        labels['utility']=labels.opportunity15-(labels.adverse_return15.le(-.03) if variant=='risk' else 0)
    labels['target']=labels.utility if variant in ['absolute','downside'] else labels.utility-labels.groupby('date').utility.transform('mean')
    t=t.merge(labels[['date','code','target']],on=['date','code'],validate='one_to_one')
    return t.sort_values(['date','code']).reset_index(drop=True)


def model(variant):
    root=base.ROOT
    if (root/'model_report.json').exists():
        raise ValueError('Do not refit the frozen relative model')
    root.mkdir(parents=True,exist_ok=True)
    train=training(variant)
    start,end,_=training_scope()
    x=base.encode(train);y=train.target.to_numpy()
    w=1/train.groupby('date').code.transform('size').to_numpy()
    depth=json.loads(PROTOCOL.read_text()).get('model_max_depth',2)
    assert depth in [2,3]
    model=GradientBoostingRegressor(loss='squared_error',n_estimators=64,learning_rate=.05,max_depth=depth,
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
        training_start=start,training_end=end,new_2025_score_groups_read=bool(train.date.ge('2025-01-01').any()),
        new_2025H2_score_groups_read=bool(train.date.ge('2025-07-01').any()),new_2026_prices_read=False,no_exit_rules=True)
    if variant=='duration':
        r['duration_label_report_sha256']=sha(duration_source()/'duration_report.json')
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
    start,end,where=training_scope()
    assert r.get('training_start')==start and r.get('training_end','2025-01-01')==end
    config=json.loads(PROTOCOL.read_text())
    depth=config.get('model_max_depth',2)
    minimum_days=config.get('minimum_leaf_training_days',0)
    assert r['parameters']['max_depth']==depth
    # Pandas column lookup is case-sensitive; DuckDB identifiers are not.
    # Project exact model names before registration so a retained source
    # field such as ip01 cannot shadow the model input IP01.
    columns=['date','code','formula_input_valid',*base.EXPRESSIONS]
    assert len({name.casefold() for name in columns})==len(columns)
    f=base.feature_inputs()[columns];c=base.conn();c.register('features',f)
    utility='opportunity15'+('-CAST(adverse_return15<=-.03 AS INTEGER)' if variant=='risk' else '')
    if variant=='margin':
        utility='greatest(-1.,least(1.,sustained_return15/.01))'
    elif variant=='downside':
        utility='CAST(adverse_return15<=-.03 AS INTEGER)'
    elif variant=='quality':
        utility='CAST(sustained_return15>=.01 AND adverse_return15>-.03 AS INTEGER)'
    elif variant=='space':
        utility='CAST(sustained_return15>=.01 AS INTEGER)'
    target='utility' if variant in ['absolute','downside'] else 'utility-avg(utility) OVER(PARTITION BY date)'
    if variant=='group_relative':
        groups=config['training_size_groups'];assert groups in [1,5]
        visible_where=f"formula_input_valid AND date<'{end}'"+(f" AND date>='{start}'" if start else '')
        # Compute membership using all valid visible rows, then join only
        # labels known strictly before the training boundary.
        c.execute(f'''CREATE VIEW targets AS WITH visible AS(
            SELECT date,code,rank() OVER(PARTITION BY date ORDER BY S01) AS first_rank,
                count(*) OVER(PARTITION BY date,S01) AS tied,
                count(*) OVER(PARTITION BY date) AS n
            FROM features WHERE {visible_where}), members AS(
            SELECT date,code,floor({groups}*(first_rank+(tied-1)/2.-.5)/n)::INT AS size_bucket FROM visible),
            known AS(SELECT date,code,size_bucket,opportunity15 AS utility
                FROM members JOIN read_parquet('{base.SOURCE}/full_labels.parquet') USING(date,code)
                WHERE {where} AND known15)
            SELECT date,code,size_bucket,utility-avg(utility) OVER(PARTITION BY date,size_bucket) AS target FROM known''')
    elif variant=='duration':
        source=duration_source()
        assert r['duration_label_report_sha256']==sha(source/'duration_report.json')
        c.execute(f'''CREATE VIEW targets AS WITH l AS(
            SELECT date,code,bit_count(profitable_mask15)/30. AS utility
            FROM read_parquet('{source}/duration_labels.parquet') WHERE {where})
            SELECT date,code,utility-avg(utility) OVER(PARTITION BY date) AS target FROM l''')
    elif variant=='rank':
        c.execute(f'''CREATE VIEW targets AS WITH l AS (
            SELECT l.date,l.code,l.sustained_return15 AS outcome
            FROM read_parquet('{base.SOURCE}/full_labels.parquet') l
            JOIN features f USING(date,code)
            WHERE {where} AND known15 AND formula_input_valid), ranked AS (
            SELECT *,rank() OVER(PARTITION BY date ORDER BY outcome) AS first_rank,
                count(*) OVER(PARTITION BY date,outcome) AS tied,
                count(*) OVER(PARTITION BY date) AS n FROM l)
            SELECT date,code,(first_rank+(tied-1)/2.-.5)/n-.5 AS target FROM ranked''')
    else:
        c.execute(f'''CREATE VIEW targets AS WITH l AS(SELECT date,code,{utility} AS utility
            FROM read_parquet('{base.SOURCE}/full_labels.parquet') WHERE {where} AND known15)
            SELECT date,code,{target} AS target FROM l''')
    names=list(base.EXPRESSIONS)
    encoded=','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::INT AS {n}' for n in names)
    d=c.sql('''SELECT date,code,target,1./count(*) OVER(PARTITION BY date) AS w,'''+encoded+'''
        FROM features JOIN targets USING(date,code) WHERE formula_input_valid ORDER BY date,code''').df()
    original=training(variant)
    assert original.next_date.lt(end).all() and (start is None or original.date.ge(start).all())
    assert r['new_2025H2_score_groups_read']==bool(original.date.ge('2025-07-01').any())
    np.testing.assert_allclose(d.target,original.target,rtol=0,atol=2e-12)
    assert d[['date','code']].equals(original[['date','code']])
    if variant=='group_relative':
        membership=c.sql('SELECT date,code,size_bucket FROM targets ORDER BY date,code').df()
        pd.testing.assert_frame_equal(membership,original[['date','code','size_bucket']],check_exact=True,check_dtype=False)
        group_means=original.groupby(['date','size_bucket']).target.mean()
        assert group_means.abs().max()<2e-12
    if variant=='rank':
        assert np.isfinite(d.target).all() and d.target.between(-.5,.5,inclusive='neither').all()
        np.testing.assert_allclose(d.groupby('date').target.mean(),0,rtol=0,atol=2e-12)
    x=d[names].to_numpy(dtype='int32');y=d.target.to_numpy();w=d.w.to_numpy()
    np.testing.assert_array_equal(x,base.encode(original))
    np.testing.assert_allclose(r['bias'],np.average(y,weights=w),rtol=0,atol=2e-12)
    score=np.full(len(d),r['bias']);checks=0;minimum_observed_days=len(d)
    dates=np.unique(d.date.to_numpy(),return_inverse=True)[1] if minimum_days else None
    assert len(r['trees'])==64 and r['learning_rate']==.05
    for tree in r['trees']:
        assert len(tree['feature'])<=2**(depth+1)-1
        residual=y-score;masks={0:np.ones(len(d),dtype=bool)};levels={0:0};terminal=np.empty(len(d))
        for i in range(len(tree['feature'])):
            assert levels[i]<=depth
            mask=masks[i];weights=w[mask]
            if minimum_days:
                observed_days=len(np.unique(dates[mask]))
                assert tree['training_days'][i]==observed_days
            assert int(mask.sum())==tree['n_node_samples'][i]
            mean=np.average(residual[mask],weights=weights)
            variance=np.average((residual[mask]-mean)**2,weights=weights)
            np.testing.assert_allclose(weights.sum(),tree['weighted_n_node_samples'][i],rtol=0,atol=1e-8)
            np.testing.assert_allclose(mean,tree['value'][i],rtol=0,atol=2e-10)
            np.testing.assert_allclose(variance,tree['impurity'][i],rtol=0,atol=2e-10)
            left,right=tree['children_left'][i],tree['children_right'][i]
            if left<0:
                assert mask.sum()>=300
                if minimum_days:
                    assert observed_days>=minimum_days
                    minimum_observed_days=min(minimum_observed_days,observed_days)
                terminal[mask]=mean
            else:
                lower=x[:,tree['feature'][i]]<=tree['threshold'][i]
                masks[left],masks[right]=mask&lower,mask&~lower
                levels[left]=levels[right]=levels[i]+1
            checks+=1
        score+=.05*terminal
    for q,t in zip(base.QUANTILES,r['thresholds']):
        assert q==t['training_quantile']
        np.testing.assert_allclose(np.quantile(score,q),t['threshold'],rtol=0,atol=2e-10)
    proof=dict(passed=True,model_report_sha256=sha(root/'model_report.json'),variant=variant,rows=len(d),node_checks=checks,
        all_targets_integer_inputs_day_weights_residual_means_and_variances_rebuilt=True,
        training_start=start,training_end=end,new_2025_score_groups_read=bool(original.date.ge('2025-01-01').any()),
        new_2025H2_score_groups_read=bool(original.date.ge('2025-07-01').any()),new_2026_prices_read=False,no_exit_rules=True)
    if minimum_days:
        proof.update(minimum_leaf_training_days_required=minimum_days,
            minimum_leaf_training_days_observed=minimum_observed_days,all_node_date_support_rebuilt=True)
    if variant=='rank':
        proof.update(all_tied_ranks_and_training_intersection_rebuilt=True,
            maximum_absolute_daily_target_mean=float(d.groupby('date').target.mean().abs().max()))
    if variant=='duration':
        proof.update(duration_label_report_sha256=sha(duration_source()/'duration_report.json'),
            all_duration_targets_rebuilt_from_masks=True)
    if variant=='group_relative':
        proof.update(training_size_groups=config['training_size_groups'],
            all_visible_memberships_before_label_intersection_rebuilt=True,
            observed_date_groups=len(group_means),maximum_absolute_group_target_mean=float(group_means.abs().max()))
    save_json(root/'model_verification.json',proof)
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('variant',choices=['relative','risk'])
    p.add_argument('stage',choices=['model','verify_model','scores','calibrate','analyze'])
    args=p.parse_args();setup(args.variant)
    result=globals()[args.stage](args.variant) if args.stage in ['model','verify_model'] else getattr(base,args.stage)()
    print(json.dumps(result,ensure_ascii=False,indent=2))
