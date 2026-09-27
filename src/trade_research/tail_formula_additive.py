"""A fixed sum of shallow rules, exportable as native IF arithmetic."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from .corporate_cash import save_json,sha
from .tail_formula_joint import ROOT as FEATURES,EXPRESSIONS,HEADER
from .tail_formula_intraday import SOURCE,conn
from .tail_formula_selective import statistics
from .tail_formula_1000_daily import analyze as common_analysis

ROOT=Path('data/research/tail_formula_additive')
PROTOCOL=Path('config/tail_formula_additive_protocol.json')
QUANTILES=[.97,.98,.99,.995,.998,.999,.9995]


def feature_inputs():
    r=json.loads((FEATURES/'feature_report.json').read_text())
    v=json.loads((FEATURES/'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256']==sha(FEATURES/'feature_report.json')
    assert r['features_sha256']==sha(FEATURES/'features.parquet')
    return pd.read_parquet(FEATURES/'features.parquet')


def labels(where):
    r=json.loads((SOURCE/'full_label_report.json').read_text())
    v=json.loads((SOURCE/'full_label_verification.json').read_text())
    assert v['passed'] and v['label_report_sha256']==sha(SOURCE/'full_label_report.json')
    assert r['labels_sha256']==sha(SOURCE/'full_labels.parquet')
    c=conn()
    return c.execute(f'''SELECT date,code,next_date,known15,opportunity15,known_no_trade
        FROM read_parquet(?) WHERE {where}''',[str(SOURCE/'full_labels.parquet')]).df()


def training(f=None):
    f=feature_inputs() if f is None else f
    l=labels("next_date<'2025-01-01'")
    assert l.next_date.lt('2025-01-01').all()
    t=f.loc[f.formula_input_valid].merge(l.loc[l.known15],on=['date','code'],validate='one_to_one')
    return t.sort_values(['date','code']).reset_index(drop=True)


def encode(f):
    x=f[list(EXPRESSIONS)].to_numpy(dtype=float)
    assert np.isfinite(x).all()
    return np.floor(np.clip(100*x+10000+.000001,0,999999)).astype('int32')


def leaf_indices(x,t):
    node=np.zeros(len(x),dtype='int32')
    left=np.asarray(t['children_left'])
    right=np.asarray(t['children_right'])
    feature=np.asarray(t['feature'])
    threshold=np.asarray(t['threshold'])
    for _ in range(2):
        active=left[node]>=0
        n=node[active]
        branch=x[active,feature[n]]<=threshold[n]
        node[active]=np.where(branch,left[n],right[n])
    assert (left[node]<0).all()
    return node


def predict(x,r):
    score=np.full(len(x),r['bias'],dtype=float)
    for t in r['trees']:
        score+=r['learning_rate']*np.asarray(t['value'])[leaf_indices(x,t)]
    return score


def model():
    if (ROOT/'model_report.json').exists():
        raise ValueError('Do not refit the fixed additive model')
    ROOT.mkdir(parents=True,exist_ok=True)
    train=training()
    x=encode(train)
    y=train.opportunity15.to_numpy()
    w=1/train.groupby('date').code.transform('size').to_numpy()
    model=GradientBoostingClassifier(loss='log_loss',n_estimators=64,learning_rate=.05,max_depth=2,
        min_samples_leaf=300,subsample=1.,random_state=20260927)
    model.fit(x,y,sample_weight=w)
    trees=[]
    for estimator in model.estimators_.ravel():
        t=estimator.tree_
        trees.append(dict(feature=t.feature.tolist(),threshold=t.threshold.tolist(),
            children_left=t.children_left.tolist(),children_right=t.children_right.tolist(),
            n_node_samples=t.n_node_samples.tolist(),weighted_n_node_samples=t.weighted_n_node_samples.tolist(),
            value=t.value.reshape(-1).tolist(),impurity=t.impurity.tolist()))
    prior=model.init_.class_prior_
    r=dict(protocol_sha256=sha(PROTOCOL),feature_report_sha256=sha(FEATURES/'feature_report.json'),
        label_report_sha256=sha(SOURCE/'full_label_report.json'),rows=len(train),days=train.date.nunique(),
        last_observation=train.next_date.max(),parameters=model.get_params(),feature_names=list(EXPRESSIONS),
        learning_rate=.05,bias=float(np.log(prior[1]/prior[0])),class_prior=prior.tolist(),trees=trees,
        new_2025_score_groups_read=False,new_2026_prices_read=False,no_exit_rules=True)
    manual=predict(x,r)
    np.testing.assert_allclose(manual,model.decision_function(x),rtol=0,atol=2e-12)
    r['thresholds']=[dict(id=i,training_quantile=q,threshold=float(np.quantile(manual,q))) for i,q in enumerate(QUANTILES)]
    save_json(ROOT/'model_report.json',r)
    return {k:v for k,v in r.items() if k not in ['trees']}


def verify_model():
    r=json.loads((ROOT/'model_report.json').read_text())
    assert r['protocol_sha256']==sha(PROTOCOL)
    assert r['feature_report_sha256']==sha(FEATURES/'feature_report.json')
    assert r['label_report_sha256']==sha(SOURCE/'full_label_report.json')
    assert len(r['trees'])==64 and r['learning_rate']==.05
    train=training()
    c=conn()
    c.register('training',train)
    e=c.sql('SELECT date,code,opportunity15 AS y,1./count(*) OVER(PARTITION BY date) AS w FROM training ORDER BY date,code').df()
    names=list(EXPRESSIONS)
    expressions=','.join(f'floor(least(greatest(100*{name}+10000+.000001,0),999999))::INT AS {name}' for name in names)
    x=c.sql('SELECT '+expressions+' FROM training ORDER BY date,code').df().to_numpy(dtype='int32')
    np.testing.assert_array_equal(x,encode(train))
    positive=float(np.average(e.y,weights=e.w))
    np.testing.assert_allclose(r['bias'],np.log(positive/(1-positive)),atol=2e-12,rtol=0)
    score=np.full(len(train),r['bias'])
    node_checks=0
    for tree in r['trees']:
        p=1/(1+np.exp(-score))
        residual=e.y.to_numpy()-p
        masks={0:np.ones(len(train),dtype=bool)}
        terminal=np.empty(len(train))
        for i in range(len(tree['feature'])):
            mask=masks[i]
            assert int(mask.sum())==tree['n_node_samples'][i]
            weights=e.w.to_numpy()[mask]
            mean=float(np.average(residual[mask],weights=weights))
            variance=float(np.average((residual[mask]-mean)**2,weights=weights))
            np.testing.assert_allclose(weights.sum(),tree['weighted_n_node_samples'][i],atol=1e-8,rtol=0)
            np.testing.assert_allclose(variance,tree['impurity'][i],atol=2e-10,rtol=0)
            left,right=tree['children_left'][i],tree['children_right'][i]
            if left<0:
                assert mask.sum()>=300
                value=np.sum(weights*residual[mask])/np.sum(weights*p[mask]*(1-p[mask]))
                terminal[mask]=value
            else:
                value=mean
                lower=x[:,tree['feature'][i]]<=tree['threshold'][i]
                masks[left],masks[right]=mask&lower,mask&~lower
            np.testing.assert_allclose(value,tree['value'][i],atol=3e-10,rtol=0)
            node_checks+=1
        score+=r['learning_rate']*terminal
    for q,item in zip(QUANTILES,r['thresholds']):
        assert q==item['training_quantile']
        np.testing.assert_allclose(np.quantile(score,q),item['threshold'],atol=3e-10,rtol=0)
    proof=dict(passed=True,model_report_sha256=sha(ROOT/'model_report.json'),rows=len(train),node_checks=node_checks,
        sql_integer_inputs_and_day_weights_rebuilt=True,all_residual_variances_and_newton_leaf_values_rebuilt=True,
        new_2025_score_groups_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'model_verification.json',proof)
    return proof


def scores():
    if (ROOT/'score_report.json').exists():
        raise ValueError('Do not replace fixed model scores')
    proof=json.loads((ROOT/'model_verification.json').read_text())
    assert proof['passed'] and proof['model_report_sha256']==sha(ROOT/'model_report.json')
    r=json.loads((ROOT/'model_report.json').read_text())
    f=feature_inputs()
    out=f[['date','code','half','board','decision_shares','formula_input_valid']].copy()
    out['score']=np.nan
    valid=f.formula_input_valid
    out.loc[valid,'score']=predict(encode(f.loc[valid]),r)
    out.to_parquet(ROOT/'scores.parquet',index=False,compression='zstd')
    report=dict(protocol_sha256=sha(PROTOCOL),model_report_sha256=sha(ROOT/'model_report.json'),
        feature_report_sha256=sha(FEATURES/'feature_report.json'),scores_sha256=sha(ROOT/'scores.parquet'),
        rows=len(out),valid=int(valid.sum()),new_2025_score_groups_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'score_report.json',report)
    return report


def native_tree(t,index=0):
    if t['children_left'][index]<0:
        return format(.05*t['value'][index],'.17g')
    threshold=int(np.floor(t['threshold'][index]))
    name=f"X{t['feature'][index]+1:02d}"
    return f"IF({name}<={threshold},{native_tree(t,t['children_left'][index])},{native_tree(t,t['children_right'][index])})"


def native_core(model,threshold,expressions=EXPRESSIONS,header=HEADER):
    core=header+'\n'.join(f'{k}:={v};' for k,v in expressions.items())+'\n'
    core+='\n'.join(f'X{i:02d}:=INTPART(MIN(MAX(100*{name}+10000+0.000001,0),999999));' for i,name in enumerate(expressions,1))+'\n'
    core+='\n'.join(f'T{i:02d}:={native_tree(t)};' for i,t in enumerate(model['trees'],1))+'\n'
    core+='SC:='+format(model['bias'],'.17g')+'+'+'+'.join(f'T{i:02d}' for i in range(1,65))+';\n'
    return core+'CORE:SC>'+format(threshold,'.17g')+';\n'


def calibrate(max_p95=50):
    if (ROOT/'selection_report.json').exists():
        raise ValueError('Do not replace the H1-selected threshold')
    check=json.loads((ROOT/'score_verification.json').read_text())
    assert check['passed'] and check['score_report_sha256']==sha(ROOT/'score_report.json')
    model=json.loads((ROOT/'model_report.json').read_text())
    f=pd.read_parquet(ROOT/'scores.parquet')
    l=labels("date>='2025-01-01' AND next_date<'2025-07-01'")
    base=l.groupby('date').opportunity15.mean()
    r=f.merge(l,on=['date','code'],validate='one_to_one')
    days=[];scores=[]
    for cut in model['thresholds']:
        q=r.loc[r.formula_input_valid&r.score.gt(cut['threshold'])].copy()
        q['unknown']=~q.known15&~q.known_no_trade
        d=q.groupby('date').agg(rows=('code','size'),known=('known15','sum'),success=('opportunity15','sum'),
            unknown=('unknown','sum'),no_trade=('known_no_trade','sum')).reset_index()
        d['rate']=d.success/d.known.replace(0,np.nan);d['lower']=d.success/d.rows
        d['base_rate']=d.date.map(base);d['delta']=d.rate-d.base_rate;d['cut_id']=cut['id']
        days.append(d)
        s=dict(**cut,**statistics(d))
        s['admitted']=bool(s['days']>=30 and s['known']>=100 and s['mean_selected_per_day']<=20
            and (max_p95 is None or s['p95_selected']<=max_p95))
        scores.append(s)
    admitted=[s for s in scores if s['admitted']]
    best=sorted(admitted,key=lambda s:(-s['lower'],-s['delta'],-s['known'],s['threshold']))[0] if admitted else None
    pd.concat(days,ignore_index=True).to_parquet(ROOT/'calibration_days.parquet',index=False,compression='zstd')
    out=f[['date','code','half','board','decision_shares']].copy()
    out['selected']=f.formula_input_valid&f.score.gt(best['threshold']) if best else False
    out.to_parquet(ROOT/'selection.parquet',index=False,compression='zstd')
    report=dict(protocol_sha256=sha(PROTOCOL),model_report_sha256=sha(ROOT/'model_report.json'),
        score_report_sha256=sha(ROOT/'score_report.json'),calibration_label_report_sha256=sha(SOURCE/'full_label_report.json'),
        calibration_days_sha256=sha(ROOT/'calibration_days.parquet'),selection_sha256=sha(ROOT/'selection.parquet'),
        thresholds=scores,chosen_threshold=best,admitted_thresholds=len(admitted),calibration_max_p95=max_p95,
        selected=int(out.selected.sum()),by_half=out.groupby('half').selected.agg(['size','sum']).reset_index().to_dict('records'),
        admission_is_diagnostic_only=True,calibration_period='2025H1',new_2025H2_score_groups_read=False,
        new_2026_prices_read=False,no_exit_rules=True,multi_day_holding_study=False,software_compilation_verified=False)
    if best:
        core=native_core(model,best['threshold'],EXPRESSIONS,HEADER)
        (ROOT/'frozen_numeric_core.tdx').write_text(core)
        report['core_sha256']=sha(ROOT/'frozen_numeric_core.tdx')
    save_json(ROOT/'selection_report.json',report)
    return {k:v for k,v in report.items() if k!='thresholds'}


def analyze():
    r=json.loads((ROOT/'selection_report.json').read_text())
    if r['chosen_threshold'] is None:
        raise ValueError('No threshold met count/coverage conditions; continue research without publishing this model')
    return common_analysis(ROOT,PROTOCOL)


def diagnose():
    """Describe concentration after evaluation without retuning the model."""
    s=pd.read_parquet(ROOT/'selection.parquet')
    f=feature_inputs()[['date','code','A01','A04','A19','D02']]
    m=json.loads((ROOT/'model_report.json').read_text())
    distributions=[]
    for half in ['2024H1','2024H2','2025H1','2025H2']:
        q=s.loc[s.half.eq(half)&s.selected].merge(f,on=['date','code'],validate='one_to_one')
        counts=q.groupby('date').size().sort_values(ascending=False)
        distributions.append(dict(half=half,rows=len(q),top5_date_fraction=float(counts.head(5).sum()/len(q)) if len(q) else None,
            top5_dates=counts.head(5).to_dict(),medians=q[['A01','A04','A19','D02']].median().to_dict()))
    gains={name:0. for name in m['feature_names']}
    for t in m['trees']:
        for i,left in enumerate(t['children_left']):
            if left<0:
                continue
            right=t['children_right'][i];imp=t['impurity'];w=t['weighted_n_node_samples']
            gains[m['feature_names'][t['feature'][i]]]+=w[i]*imp[i]-w[left]*imp[left]-w[right]*imp[right]
    total=sum(gains.values())
    gains={key:value/total for key,value in sorted(gains.items(),key=lambda kv:-kv[1])}
    report=dict(selection_report_sha256=sha(ROOT/'selection_report.json'),model_report_sha256=sha(ROOT/'model_report.json'),
        source_feature_report_sha256=sha(FEATURES/'feature_report.json'),training_split_importance=gains,
        selected_distributions=distributions,posthoc_diagnostic_only=True)
    save_json(ROOT/'diagnostic_report.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['model','verify_model','scores','calibrate','analyze','diagnose'])
    print(json.dumps(globals()[p.parse_args().stage](),ensure_ascii=False,indent=2))
