"""Independently rebuild top-ten tie discounts, dynamic pairs and leaf updates."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from trade_research import tail_formula_top_pairs_raw as study
from trade_research import tail_formula_additive as base
from trade_research import tail_formula_relative as relative
from trade_research.corporate_cash import save_json,sha


def verify(fold):
    study.setup(fold);root=base.ROOT;r=json.loads((root/'model_report.json').read_text())
    p=json.loads(base.PROTOCOL.read_text());start,end,where=relative.training_scope()
    for key,file in [('protocol_sha256',base.PROTOCOL),('feature_report_sha256',base.FEATURES/'feature_report.json'),
        ('label_report_sha256',base.SOURCE/'full_label_report.json'),('training_pairs_sha256',root/'training_pairs.parquet'),('training_days_sha256',root/'training_days.parquet')]:
        assert r[key]==sha(file)
    assert r['parameters']==p['parameters']==study.PARAMETERS
    assert r['bias']==0 and r['learning_rate']==.05 and r['variant']=='same_date_binary_top10_pairs'
    assert r['training_start']==start and r['training_end']==end and r['last_observation']<end
    names=list(base.EXPRESSIONS);assert names==r['feature_names'] and len(names)==48
    f=base.feature_inputs()[['date','code','formula_input_valid',*names]]
    c=base.conn();c.register('features',f)
    c.sql(f"SELECT date,code,next_date,opportunity15 FROM read_parquet('{base.SOURCE}/full_labels.parquet') WHERE {where} AND known15").create_view('labels')
    encode=','.join(f'floor(least(greatest(100*"{name}"+10000+.000001,0),999999))::INT AS "{name}"' for name in names)
    d=c.sql('SELECT l.date,l.code,l.next_date,l.opportunity15,1./count(*) OVER(PARTITION BY l.date) AS w,'+encode+
        ' FROM labels l JOIN features f USING(date,code) WHERE f.formula_input_valid ORDER BY l.date,l.code').df();c.close()
    t=base.training(start=start,end=end)
    pd.testing.assert_frame_equal(d[['date','code','next_date','opportunity15']],t[['date','code','next_date','opportunity15']],check_exact=True)
    assert (len(d),d.date.nunique())==(r['rows'],r['days'])==(p['expected_rows'],241)
    assert d.next_date.max()==r['last_observation'] and d.date.lt('2025-07-01').all()
    x=d[names].to_numpy('int32');np.testing.assert_array_equal(x,base.encode(t));w=d.w.to_numpy()
    # Independent cyclic-list construction, rather than producer np.resize.
    rng=np.random.default_rng(20260927);blocks=[];days=[]
    for day,ids in d.groupby('date',sort=True).groups.items():
        a=np.asarray([i for i in ids if d.opportunity15.iloc[i]==1],dtype='int64')
        b=np.asarray([i for i in ids if d.opportunity15.iloc[i]==0],dtype='int64')
        n=max(len(a),len(b));active=bool(len(a) and len(b))
        days.append(dict(date=day,positive=len(a),negative=len(b),pairs=8*n if active else 0,informative=active))
        if not active:
            continue
        for turn in range(8):
            first=rng.permutation(a).tolist();second=rng.permutation(b).tolist()
            blocks.append(pd.DataFrame(dict(date=day,turn=turn,positive=(first*math.ceil(n/len(first)))[:n],
                negative=(second*math.ceil(n/len(second)))[:n],weight=1/(8*n))))
    pairs=pd.concat(blocks,ignore_index=True)
    pd.testing.assert_frame_equal(pd.read_parquet(root/'training_pairs.parquet'),pairs,check_exact=True)
    pd.testing.assert_frame_equal(pd.read_parquet(root/'training_days.parquet'),pd.DataFrame(days),check_exact=True)
    assert len(pairs)==r['pairs'] and sum(q['informative'] for q in days)==r['informative_days']
    assert [q['date'] for q in days if not q['informative']]==r['constant_label_days']
    np.testing.assert_allclose(pairs.groupby('date').weight.sum(),1,rtol=0,atol=2e-12)
    pos=pairs.positive.to_numpy('int64');neg=pairs.negative.to_numpy('int64');pw=pairs.weight.to_numpy()
    np.testing.assert_array_equal(d.date.to_numpy()[pos],d.date.to_numpy()[neg])
    assert d.opportunity15.to_numpy()[pos].all() and not d.opportunity15.to_numpy()[neg].any()
    coverage=np.bincount(np.concatenate([pos,neg]),minlength=len(d))
    assert (coverage[d.date.isin([q['date'] for q in days if q['informative']])]>=8).all()
    assert (coverage[d.date.isin(r['constant_label_days'])]==0).all()
    score=np.zeros(len(d));checks=0;max_day_gradient=0.
    ranking_connection=base.conn()
    day_ids=pd.factorize(d.date,sort=True)[0]
    max_weight_error=0.
    max_variance_absolute_error=0.;max_variance_relative_error=0.
    pair_day=pairs.date
    assert len(r['trees'])==len(r['trace'])==64
    for iteration,(tree,trace) in enumerate(zip(r['trees'],r['trace'])):
        if iteration == 0:
            original_model=json.loads((Path('data/research')/('tail_formula_pairwise_'+fold)/'model_report.json').read_text())
            assert tree == original_model['trees'][0], 'The initial uniform-weight tree must match exactly'
        rank_input=d[['date','code']].copy()
        rank_input['i']=np.arange(len(d));rank_input['score']=score
        ranking_connection.register('rank_input',rank_input)
        geometry=ranking_connection.sql("""WITH ranked AS(
            SELECT *,row_number() OVER(PARTITION BY date ORDER BY score DESC,code) AS position,
                rank() OVER(PARTITION BY date ORDER BY score DESC) AS first_position,
                count(*) OVER(PARTITION BY date,score) AS tie_size FROM rank_input),
            discounted AS(SELECT *,CASE WHEN position<=10 THEN 1./log2(position+1) ELSE 0. END AS discount FROM ranked)
            SELECT i,avg(discount) OVER(PARTITION BY date,score) AS mean_discount,
                CASE WHEN tie_size>1 THEN
                    2.*sum((tie_size-1-2*(position-first_position))*discount) OVER(PARTITION BY date,score)
                    /(tie_size*(tie_size-1)) ELSE 0. END AS within_discount
            FROM discounted ORDER BY i""").df()
        ranking_connection.unregister('rank_input')
        a=geometry.mean_discount.to_numpy();b=geometry.within_discount.to_numpy()
        delta=np.where(score[pos]==score[neg],b[pos],np.abs(a[pos]-a[neg]))
        denominator=pd.Series(delta).groupby(pair_day).transform('sum').to_numpy()
        assert (denominator>0).all()
        pw=delta/denominator
        produced=study.rank_weights(score,day_ids,pos,neg)
        max_weight_error=max(max_weight_error,float(np.max(np.abs(produced-pw))))
        np.testing.assert_allclose(pw,produced,rtol=0,atol=2e-14)
        np.testing.assert_allclose(pd.Series(pw).groupby(pair_day).sum(),1,rtol=0,atol=2e-12)
        if iteration==0:
            np.testing.assert_allclose(pw,pairs.weight,rtol=0,atol=2e-14)
        assert np.count_nonzero(pw)==trace['nonzero_weight_pairs']
        np.testing.assert_allclose(pw.max(),trace['max_pair_weight'],rtol=0,atol=2e-14)
        margin=score[pos]-score[neg]
        rho=1/(1+np.exp(margin));force=pw*rho;curv=pw*rho*(1-rho)
        gradient=np.zeros(len(d));np.add.at(gradient,pos,force);np.add.at(gradient,neg,-force)
        target=gradient/w
        max_day_gradient=max(max_day_gradient,float(pd.Series(gradient).groupby(d.date).sum().abs().max()))
        loss=float(np.sum(pw*np.log1p(np.exp(-margin))))
        assert trace['iteration']==iteration
        np.testing.assert_allclose(trace['loss_before'],loss,rtol=0,atol=2e-9)
        np.testing.assert_allclose(trace['gradient_sum'],gradient.sum(),rtol=0,atol=2e-10)
        masks={0:np.ones(len(d),dtype=bool)};levels={0:0};leaves=np.empty(len(d),dtype='int32')
        assert len(tree['feature'])<=15
        for node,left in enumerate(tree['children_left']):
            mask=masks[node];ww=w[mask];mean=np.average(target[mask],weights=ww)
            variance=np.average((target[mask]-mean)**2,weights=ww)
            assert mask.sum()==tree['n_node_samples'][node] and levels[node]<=3
            np.testing.assert_allclose(ww.sum(),tree['weighted_n_node_samples'][node],rtol=0,atol=2e-8)
            np.testing.assert_allclose(mean,tree['gradient_value'][node],rtol=0,atol=2e-9)
            variance_error=abs(variance-tree['impurity'][node])
            max_variance_absolute_error=max(max_variance_absolute_error,float(variance_error))
            max_variance_relative_error=max(max_variance_relative_error,float(variance_error/max(1.,abs(variance))))
            np.testing.assert_allclose(variance,tree['impurity'][node],rtol=2e-10,atol=2e-9)
            right=tree['children_right'][node]
            if left<0:
                assert right<0 and mask.sum()>=300;leaves[mask]=node
            else:
                assert 0<=tree['feature'][node]<48
                split=x[:,tree['feature'][node]]<=tree['threshold'][node]
                masks[left],masks[right]=mask&split,mask&~split;levels[left]=levels[right]=levels[node]+1
            checks+=1
        np.testing.assert_array_equal(leaves,base.leaf_indices(x,tree))
        a,b=leaves[pos],leaves[neg];cross=a!=b;n=len(tree['feature'])
        numerator=np.bincount(a,weights=force,minlength=n)-np.bincount(b,weights=force,minlength=n)
        curvature=np.bincount(a[cross],weights=curv[cross],minlength=n)+np.bincount(b[cross],weights=curv[cross],minlength=n)
        clipped=0
        for node in np.flatnonzero(np.asarray(tree['children_left'])<0):
            wanted=float(np.clip(numerator[node]/curvature[node],-2,2)) if curvature[node] else 0.
            np.testing.assert_allclose(tree['leaf_numerator'][str(node)],numerator[node],rtol=0,atol=2e-9)
            np.testing.assert_allclose(tree['leaf_curvature'][str(node)],curvature[node],rtol=0,atol=2e-9)
            np.testing.assert_allclose(tree['value'][node],wanted,rtol=0,atol=2e-9)
            clipped+=int(abs(tree['value'][node])==2)
        assert clipped==trace['clipped_leaves']
        score+=.05*np.asarray(tree['value'])[leaves]
        np.testing.assert_allclose(trace['loss_after'],np.sum(pw*np.log1p(np.exp(score[neg]-score[pos]))),rtol=0,atol=2e-9)
    ranking_connection.close()
    assert max_day_gradient<2e-12
    np.testing.assert_allclose(score,base.predict(x,r),rtol=0,atol=2e-12)
    for item,q in zip(r['thresholds'],base.QUANTILES):
        assert item['training_quantile']==q
        np.testing.assert_allclose(item['threshold'],np.quantile(score,q),rtol=0,atol=2e-12)
    proof=dict(passed=True,model_report_sha256=sha(root/'model_report.json'),rows=len(d),node_checks=checks,
        training_pairs=len(pairs),informative_days=r['informative_days'],max_absolute_day_gradient=max_day_gradient,
        all_training_keys_integer_inputs_date_weights_and_fixed_pairs_rebuilt=True,
        all_64_rank_tie_geometries_and_pair_weights_independently_rebuilt=True,
        first_all_tied_round_recovers_uniform_weights=True,max_pair_weight_difference=max_weight_error,
        first_tree_matches_original_uniform_pair_model_exactly=True,
        node_variance_absolute_tolerance=2e-9,node_variance_relative_tolerance=2e-10,
        max_node_variance_absolute_error=max_variance_absolute_error,
        max_node_variance_relative_error=max_variance_relative_error,
        all_pair_gradients_and_same_leaf_cancellation_rebuilt=True,all_tree_nodes_and_leaf_directional_curvatures_rebuilt=True,
        all_export_scores_and_training_thresholds_rebuilt=True,constant_response_days_preserved=True,
        new_2025H2_score_groups_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(root/'model_verification.json',proof);return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fold',choices=['2024','recent'],required=True)
    print(json.dumps(verify(p.parse_args().fold),ensure_ascii=False,indent=2))
