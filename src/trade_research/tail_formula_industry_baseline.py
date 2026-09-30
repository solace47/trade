"""Learn stock-specific opportunities after historical industry and day centering."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_morning_range as prior
from .corporate_cash import save_json,sha

STEM='tail_formula_industry_baseline'
ROOT=Path('data/research')/STEM
INPUTS=ROOT/'inputs'
PROTOCOL=Path('config')/(STEM+'_input_protocol.json')
INTENT=Path('config')/(STEM+'_intent.json')
META=prior.META
HEADER=prior.HEADER
CORE_GATE='AMREADY AND RTREADY'
ARMS={'control':prior.EXPRESSIONS,'industry':prior.EXPRESSIONS}
TRAINING_EVENT='industry_and_day_centered_full_known_pool'
MINIMUM_GROUP=11


def checked():
    prior.checked();p=json.loads(PROTOCOL.read_text())
    assert p['intent_sha256']==sha(INTENT) and p['arms']==ARMS and p['native_header']==HEADER
    assert p['minimum_industry_known_group']==MINIMUM_GROUP and p['training_event']==TRAINING_EVENT
    assert p['expected_keys']==1258085 and p['expected_valid']==1117397
    assert p['no_new_raw_extraction'] and not p['new_2026_prices_allowed']
    for file,digest in p['source_hashes'].items():assert sha(Path(file))==digest,file
    complete=json.loads(Path(p['conditional_completion']).read_text())
    gate=json.loads(Path(p['conditional_gate']).read_text())
    assert complete['passed'] and gate['passed'] and not gate['supports_2024_extension']
    lookup=json.loads((ROOT/'classification_metadata_lookup_verification.json').read_text())
    assert lookup['passed'] and lookup['full_memberships_sha256']==sha(ROOT/'historical_memberships.parquet')
    for file,digest in lookup['source_hashes'].items():assert sha(Path(file))==digest,file
    return p


def center(frame):
    """Center the full known label pool before intersecting visible input quality."""
    assert not frame.duplicated(['date','code']).any()
    assert frame.opportunity15.isin([0,1]).all()
    f=frame.copy();daily=f.groupby('date').opportunity15
    sector=f.groupby(['date','industry']).opportunity15
    f['day_mean']=daily.transform('mean')
    f['sector_count']=sector.transform('size').fillna(0).astype('int64')
    f['sector_mean']=sector.transform('mean')
    f['industry_baseline_used']=f.industry.notna() & f.sector_count.ge(MINIMUM_GROUP)
    f['baseline']=f.sector_mean.where(f.industry_baseline_used,f.day_mean)
    f['residual']=f.opportunity15-f.baseline
    f['residual_day_mean']=f.groupby('date').residual.transform('mean')
    f['target']=f.residual-f.residual_day_mean
    assert np.isfinite(f.target).all()
    np.testing.assert_allclose(f.groupby('date').target.mean(),0,rtol=0,atol=2e-12)
    return f.sort_values(['date','code']).reset_index(drop=True)


def features():
    checked();INPUTS.mkdir(parents=True,exist_ok=True)
    assert not (INPUTS/'feature_report.json').exists()
    old=json.loads((prior.INPUTS/'feature_report.json').read_text())
    for name in ['features.parquet','full_labels.parquet','full_label_report.json','full_label_verification.json']:
        (INPUTS/name).symlink_to((prior.INPUTS/name).resolve())
    assert sha(INPUTS/'features.parquet')==old['features_sha256']
    for kind in ['feature','native_input']:
        proof=json.loads((prior.INPUTS/(kind+'_verification.json')).read_text())
        assert proof['passed'] and proof['feature_report_sha256']==sha(prior.INPUTS/'feature_report.json')
    meta=pd.read_parquet(INPUTS/'features.parquet',columns=META)
    assert len(meta)==1258085 and meta.formula_input_valid.sum()==1117397
    r=dict(protocol_sha256=sha(PROTOCOL),features_sha256=sha(INPUTS/'features.parquet'),rows=len(meta),
        valid=int(meta.formula_input_valid.sum()),expressions=ARMS['industry'],native_header=HEADER,
        immutable_original_50_complete_feature_file_reused=True,
        parent_feature_report_sha256=sha(prior.INPUTS/'feature_report.json'),
        no_new_input_arithmetic=True,new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(INPUTS/'feature_report.json',r)
    common=dict(passed=True,feature_report_sha256=sha(INPUTS/'feature_report.json'),
        original_50_complete_file_sha256_exactly_equal=True,no_new_input_arithmetic=True,
        new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(INPUTS/'feature_verification.json',dict(**common,effective_input_intersection_unchanged=True,
        domain_reference=str(prior.INPUTS),parent_verification_sha256=sha(prior.INPUTS/'feature_verification.json')))
    save_json(INPUTS/'native_input_verification.json',dict(**common,
        parent_native_verification_sha256=sha(prior.INPUTS/'native_input_verification.json'),
        software_compilation_verified=False,native_source_parity_verified=False))
    return dict(rows=len(meta),valid=int(meta.formula_input_valid.sum()))


def target_paths(fold):
    root=ROOT/'training_targets'/fold
    return root,root/'targets.parquet',root/'target_report.json',root/'target_verification.json'


def targets():
    p=checked();counts=[]
    for fold,spec in p['folds'].items():
        root,file,report,_=target_paths(fold);assert not report.exists();root.mkdir(parents=True,exist_ok=True)
        labels=pd.read_parquet(INPUTS/'full_labels.parquet',columns=['date','code','next_date','opportunity15'],
            filters=[('date','>=',spec['training_start']),('next_date','<',spec['training_end']),('known15','==',True)])
        members=pd.read_parquet(ROOT/'historical_memberships.parquet',columns=['date','code','industry'])
        f=labels.merge(members,on=['date','code'],how='left',validate='one_to_one')
        assert len(f)==len(labels) and f.next_date.lt(spec['training_end']).all()
        f=center(f);f.to_parquet(file,index=False,compression='zstd')
        r=dict(protocol_sha256=sha(PROTOCOL),fold=fold,training_start=spec['training_start'],training_end=spec['training_end'],
            targets_sha256=sha(file),full_known_pool_rows=len(f),days=f.date.nunique(),last_observation=f.next_date.max(),
            industry_baseline_rows=int(f.industry_baseline_used.sum()),unknown_industry_rows=int(f.industry.isna().sum()),
            small_industry_rows=int((f.industry.notna() & ~f.industry_baseline_used).sum()),
            minimum_industry_known_group=MINIMUM_GROUP,training_event=TRAINING_EVENT,
            full_known_baselines_before_input_intersection=True,classification_metadata_sha256=sha(ROOT/'classification_metadata_lookup_verification.json'),
            full_label_report_sha256=sha(INPUTS/'full_label_report.json'),new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
        save_json(report,r);counts.append({k:r[k] for k in ['fold','full_known_pool_rows','industry_baseline_rows','unknown_industry_rows','small_industry_rows']})
    return counts


def independent_targets(fold):
    p=checked();spec=p['folds'][fold];c=base.conn()
    c.read_parquet(str(INPUTS/'full_labels.parquet')).create_view('labels')
    c.read_parquet('data/research/industry_intervals.parquet').create_view('industry_history')
    d=c.sql(f'''WITH known AS(SELECT date,code,next_date,opportunity15 FROM labels
        WHERE known15 AND date>='{spec['training_start']}' AND next_date<'{spec['training_end']}'),
        classified AS(SELECT l.*,h.industry FROM known l LEFT JOIN industry_history h
            ON h.code=l.code AND l.date>=h.effective_date AND(h.next_effective_date IS NULL OR l.date<h.next_effective_date)
            AND h."asof"<l.date AND h.updateDate<l.date
            AND date_diff('day',CAST(h."asof" AS DATE),CAST(l.date AS DATE))<=370),
        stats AS(SELECT *,avg(opportunity15) OVER(PARTITION BY date) AS day_mean,
            CASE WHEN industry IS NULL THEN 0 ELSE count(*) OVER(PARTITION BY date,industry) END AS sector_count,
            CASE WHEN industry IS NULL THEN NULL ELSE avg(opportunity15) OVER(PARTITION BY date,industry) END AS sector_mean
            FROM classified), baselines AS(SELECT *,industry IS NOT NULL AND sector_count>={MINIMUM_GROUP} AS industry_baseline_used,
            CASE WHEN industry IS NOT NULL AND sector_count>={MINIMUM_GROUP} THEN sector_mean ELSE day_mean END AS baseline
            FROM stats), residuals AS(SELECT *,opportunity15-baseline AS residual FROM baselines),
        daily AS(SELECT *,avg(residual) OVER(PARTITION BY date) AS residual_day_mean FROM residuals)
        SELECT *,residual-residual_day_mean AS target FROM daily ORDER BY date,code''').df();c.close()
    assert not d.duplicated(['date','code']).any() and d.next_date.lt(spec['training_end']).all()
    return d


def verify_targets():
    checked();proofs=[]
    meta=pd.read_parquet(INPUTS/'features.parquet',columns=META)
    for fold in json.loads(PROTOCOL.read_text())['folds']:
        _,file,report,verification=target_paths(fold);r=json.loads(report.read_text())
        assert r['protocol_sha256']==sha(PROTOCOL) and r['targets_sha256']==sha(file)
        actual=pd.read_parquet(file);expected=independent_targets(fold)
        keys=['date','code','next_date','opportunity15','industry','sector_count','industry_baseline_used']
        pd.testing.assert_frame_equal(actual[keys],expected[keys],check_exact=True,check_dtype=False)
        for name in ['day_mean','sector_mean','baseline','residual','residual_day_mean','target']:
            np.testing.assert_allclose(actual[name],expected[name],rtol=0,atol=2e-12,equal_nan=True)
        np.testing.assert_allclose(expected.groupby('date').target.mean(),0,rtol=0,atol=2e-12)
        projected=meta.loc[meta.formula_input_valid,['date','code']].merge(expected,on=['date','code'],validate='one_to_one')
        v=dict(passed=True,target_report_sha256=sha(report),fold=fold,training_rows=len(projected),training_days=projected.date.nunique(),
            all_historical_memberships_label_maturity_industry_and_daily_means_fallbacks_and_targets_sql_rebuilt=True,
            baselines_before_input_intersection=True,all_full_known_daily_target_means_zero=True,
            target_max_difference=float((actual.target-expected.target).abs().max()),
            new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
        save_json(verification,v);proofs.append(v)
    return proofs


def training():
    p=checked();q=json.loads(base.PROTOCOL.read_text());fold=q['fold'];spec=p['folds'][fold]
    assert q['arm']=='industry' and q['training_event']==TRAINING_EVENT
    assert q['training_start']==spec['training_start'] and q['training_end']==spec['training_end']
    _,file,report,verification=target_paths(fold)
    r=json.loads(report.read_text());v=json.loads(verification.read_text())
    assert v['passed'] and v['target_report_sha256']==sha(report) and r['targets_sha256']==sha(file)
    for item in [report,verification]:assert q['input_receipts'][str(item)]==sha(item)
    old=base.training(start=spec['training_start'],end=spec['training_end'])
    out=old.merge(pd.read_parquet(file,columns=['date','code','target']),on=['date','code'],validate='one_to_one')
    pd.testing.assert_frame_equal(old[['date','code']],out[['date','code']],check_exact=True)
    assert len(out)==q['expected_training_rows']==v['training_rows']
    assert out.date.nunique()==q['expected_training_days']==v['training_days']
    assert out.next_date.max()==q['expected_last_observation']<q['evaluation_start']
    assert np.isfinite(out.target).all()
    return out


def model():
    from sklearn.ensemble import GradientBoostingRegressor
    root=base.ROOT;assert not (root/'model_report.json').exists();root.mkdir(parents=True,exist_ok=True)
    q=json.loads(base.PROTOCOL.read_text());train=training()
    x=base.encode(train);y=train.target.to_numpy();w=1/train.groupby('date').code.transform('size').to_numpy()
    estimator=GradientBoostingRegressor(**q['parameters']);estimator.fit(x,y,sample_weight=w)
    trees=[]
    for fitted in estimator.estimators_.ravel():
        t=fitted.tree_
        trees.append(dict(feature=t.feature.tolist(),threshold=t.threshold.tolist(),children_left=t.children_left.tolist(),
            children_right=t.children_right.tolist(),n_node_samples=t.n_node_samples.tolist(),
            weighted_n_node_samples=t.weighted_n_node_samples.tolist(),value=t.value.reshape(-1).tolist(),impurity=t.impurity.tolist()))
    _,_,target_report,target_verification=target_paths(q['fold'])
    r=dict(protocol_sha256=sha(base.PROTOCOL),feature_report_sha256=sha(INPUTS/'feature_report.json'),
        label_report_sha256=sha(INPUTS/'full_label_report.json'),rows=len(train),days=train.date.nunique(),
        last_observation=train.next_date.max(),parameters=estimator.get_params(),feature_names=list(base.EXPRESSIONS),
        variant='relative',training_event=TRAINING_EVENT,learning_rate=.05,
        bias=float(np.ravel(estimator.init_.constant_)[0]),trees=trees,
        target_report_sha256=sha(target_report),target_verification_sha256=sha(target_verification),
        training_start=q['training_start'],training_end=q['training_end'],
        new_2025_score_groups_read=bool(train.date.ge('2025-01-01').any()),
        new_2025H2_score_groups_read=bool(train.date.ge('2025-07-01').any()),new_2026_prices_read=False,no_exit_rules=True)
    manual=base.predict(x,r);np.testing.assert_allclose(manual,estimator.predict(x),rtol=0,atol=2e-12)
    r['thresholds']=[dict(id=i,training_quantile=t,threshold=float(np.quantile(manual,t))) for i,t in enumerate(base.QUANTILES)]
    save_json(root/'model_report.json',r);return {k:v for k,v in r.items() if k!='trees'}


def verify_model():
    q=json.loads(base.PROTOCOL.read_text());r=json.loads((base.ROOT/'model_report.json').read_text())
    assert r['protocol_sha256']==sha(base.PROTOCOL) and r['training_event']==TRAINING_EVENT and r['variant']=='relative'
    assert r['feature_report_sha256']==sha(INPUTS/'feature_report.json') and r['label_report_sha256']==sha(INPUTS/'full_label_report.json')
    assert r['feature_names']==list(base.EXPRESSIONS) and all(r['parameters'][k]==v for k,v in q['parameters'].items())
    _,file,report,verification=target_paths(q['fold'])
    assert r['target_report_sha256']==sha(report) and r['target_verification_sha256']==sha(verification)
    assert json.loads(verification.read_text())['passed'] and json.loads(report.read_text())['targets_sha256']==sha(file)
    f=base.feature_inputs()[['date','code','formula_input_valid',*base.EXPRESSIONS]]
    c=base.conn();c.register('features',f)
    encoded=','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::INT AS {n}' for n in base.EXPRESSIONS)
    d=c.sql(f'''SELECT date,code,target,next_date,1./count(*) OVER(PARTITION BY date) AS w,{encoded}
        FROM features JOIN read_parquet('{file}') USING(date,code) WHERE formula_input_valid ORDER BY date,code''').df();c.close()
    original=training()
    pd.testing.assert_frame_equal(d[['date','code','next_date']],original[['date','code','next_date']],check_exact=True)
    np.testing.assert_allclose(d.target,original.target,rtol=0,atol=2e-12)
    assert len(d)==r['rows']==q['expected_training_rows'] and d.date.nunique()==r['days']==q['expected_training_days']
    assert d.next_date.max()==r['last_observation']==q['expected_last_observation']<q['evaluation_start']
    x=d[list(base.EXPRESSIONS)].to_numpy(dtype='int32');y=d.target.to_numpy();w=d.w.to_numpy()
    np.testing.assert_array_equal(x,base.encode(original))
    np.testing.assert_array_equal(w,1/original.groupby('date').code.transform('size').to_numpy())
    np.testing.assert_allclose(r['bias'],np.average(y,weights=w),rtol=0,atol=2e-12)
    assert len(r['trees'])==64 and r['learning_rate']==.05
    score=np.full(len(d),r['bias']);checks=0
    for tree in r['trees']:
        assert len(tree['feature'])<=15
        residual=y-score;masks={0:np.ones(len(d),dtype=bool)};levels={0:0};terminal=np.empty(len(d))
        for i in range(len(tree['feature'])):
            assert levels[i]<=3
            mask=masks[i];weights=w[mask]
            assert int(mask.sum())==tree['n_node_samples'][i]
            mean=np.average(residual[mask],weights=weights);variance=np.average((residual[mask]-mean)**2,weights=weights)
            np.testing.assert_allclose(weights.sum(),tree['weighted_n_node_samples'][i],rtol=0,atol=1e-8)
            np.testing.assert_allclose(mean,tree['value'][i],rtol=0,atol=2e-10)
            np.testing.assert_allclose(variance,tree['impurity'][i],rtol=0,atol=2e-10)
            left,right=tree['children_left'][i],tree['children_right'][i]
            if left<0:
                assert mask.sum()>=300;terminal[mask]=mean
            else:
                lower=x[:,tree['feature'][i]]<=tree['threshold'][i]
                masks[left],masks[right]=mask&lower,mask&~lower
                levels[left]=levels[right]=levels[i]+1
            checks+=1
        score+=.05*terminal
    for quantile,cut in zip(base.QUANTILES,r['thresholds']):
        assert quantile==cut['training_quantile']
        np.testing.assert_allclose(np.quantile(score,quantile),cut['threshold'],rtol=0,atol=2e-10)
    proof=dict(passed=True,model_report_sha256=sha(base.ROOT/'model_report.json'),variant='relative',training_event=TRAINING_EVENT,
        rows=len(d),node_checks=checks,target_verification_sha256=sha(verification),
        all_targets_integer_inputs_day_weights_residual_means_and_variances_rebuilt=True,
        training_start=q['training_start'],training_end=q['training_end'],
        new_2025_score_groups_read=bool(original.date.ge('2025-01-01').any()),
        new_2025H2_score_groups_read=bool(original.date.ge('2025-07-01').any()),new_2026_prices_read=False,no_exit_rules=True)
    save_json(base.ROOT/'model_verification.json',proof);return proof


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['features','targets','verify_targets'])
    args=parser.parse_args();print(json.dumps(globals()[args.stage](),ensure_ascii=False,indent=2))
