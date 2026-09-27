"""A fixed date-equal ridge baseline for the same 48 next-morning inputs."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from . import tail_formula_additive as base
from . import tail_formula_context_2024 as linkage
from . import tail_formula_float as original
from . import tail_formula_relative as relative
from .corporate_cash import save_json, sha


def setup(fold):
    if fold=='combined':
        linkage.ROOT=Path('data/research/tail_formula_ridge_2024')
        linkage.H2=Path('data/research/tail_formula_ridge_recent')
        linkage.COMBINED=Path('data/research/tail_formula_ridge_2025')
        linkage.PROTOCOL=Path('config/tail_formula_ridge_combined_protocol.json')
        linkage.H2_SELECTION_SHA=None;linkage.H2_MODEL_SHA=None;linkage.H2_OUTCOMES_PREVIOUSLY_SEEN=False
    else:
        original.setup(fold)
        base.ROOT=Path('data/research/tail_formula_ridge_'+fold)
        base.PROTOCOL=Path('config/tail_formula_ridge_'+fold+'_protocol.json')
        relative.PROTOCOL=base.PROTOCOL


def config():
    p=json.loads(base.PROTOCOL.read_text())
    assert p['feature_report_sha256']==sha(base.FEATURES/'feature_report.json')
    assert p['label_report_sha256']==sha(base.SOURCE/'full_label_report.json')
    assert p['ridge_alpha']==1 and p['evaluation_start']==p['training_end']
    return p


def predict(x,m):
    result=np.full(len(x),m['bias'],dtype=float)
    for i,(coef,mean,scale) in enumerate(zip(m['coefficients'],m['means'],m['scales'])):
        # Preserve the native expression's explicit left-to-right arithmetic.
        result+=coef*(x[:,i]-mean)/scale
    return result


def model():
    root=base.ROOT;p=config()
    if (root/'model_report.json').exists():
        raise ValueError('Do not refit the frozen ridge score')
    root.mkdir(parents=True,exist_ok=True)
    t=relative.training('relative');x=base.encode(t).astype(float);y=t.target.to_numpy()
    w=1/t.groupby('date').code.transform('size').to_numpy()/t.date.nunique();w/=w.sum()
    means=np.average(x,axis=0,weights=w)
    centered=x-means
    scales=np.sqrt(np.average(centered*centered,axis=0,weights=w))
    constant=scales.le(1e-12) if isinstance(scales,pd.Series) else scales<=1e-12
    scales[constant]=1
    z=centered/scales;bias=float(np.average(y,weights=w))
    gram=z.T@(z*w[:,None]);rhs=z.T@(w*(y-bias))
    coefficients=np.linalg.solve(gram+np.eye(x.shape[1]),rhs)
    r=dict(protocol_sha256=sha(base.PROTOCOL),feature_report_sha256=sha(base.FEATURES/'feature_report.json'),
        label_report_sha256=sha(base.SOURCE/'full_label_report.json'),model_family='date_equal_ridge',
        rows=len(t),days=t.date.nunique(),training_start=p['training_start'],training_end=p['training_end'],
        last_observation=t.next_date.max(),feature_names=list(base.EXPRESSIONS),ridge_alpha=1,
        weights_sum=float(w.sum()),bias=bias,means=means.tolist(),scales=scales.tolist(),coefficients=coefficients.tolist(),
        constant_columns=[n for n,flag in zip(base.EXPRESSIONS,constant) if flag],
        new_2025_score_groups_read=bool(t.date.ge('2025-01-01').any()),new_2025H2_score_groups_read=False,
        new_2026_prices_read=False,no_exit_rules=True,score_is_not_probability=True)
    score=predict(x,r)
    r['thresholds']=[dict(id=0,training_quantile=.995,threshold=float(np.quantile(score,.995)))]
    assert t.next_date.lt(p['training_end']).all() and t.date.ge(p['training_start']).all()
    save_json(root/'model_report.json',r)
    return {k:v for k,v in r.items() if k not in ['means','scales','coefficients']}


def checked_model():
    p=config();r=json.loads((base.ROOT/'model_report.json').read_text())
    for key,path in [('protocol_sha256',base.PROTOCOL),('feature_report_sha256',base.FEATURES/'feature_report.json'),
        ('label_report_sha256',base.SOURCE/'full_label_report.json')]:
        assert r[key]==sha(path)
    assert r['model_family']=='date_equal_ridge' and r['ridge_alpha']==p['ridge_alpha']==1
    assert r['training_start']==p['training_start'] and r['training_end']==p['training_end']
    assert r['feature_names']==list(base.EXPRESSIONS) and len(r['coefficients'])==len(r['means'])==len(r['scales'])==48
    assert np.isfinite([r['coefficients'],r['means'],r['scales']]).all() and min(r['scales'])>0
    assert r['last_observation']<p['evaluation_start']
    return p,r


def training_sql(c,p):
    names=list(base.EXPRESSIONS)
    encoded=','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::INT AS {n}' for n in names)
    c.execute(f'''CREATE VIEW training AS WITH l AS(SELECT date,code,opportunity15 FROM read_parquet('{base.SOURCE}/full_labels.parquet')
        WHERE known15 AND date>='{p['training_start']}' AND next_date<'{p['training_end']}'),
        target AS(SELECT date,code,opportunity15-avg(opportunity15) OVER(PARTITION BY date) AS target FROM l),
        raw_weights AS(SELECT f.date,f.code,target,1./count(*) OVER(PARTITION BY f.date) AS raw_w,{encoded}
            FROM read_parquet('{base.FEATURES}/features.parquet') f JOIN target USING(date,code) WHERE formula_input_valid)
        SELECT * EXCLUDE(raw_w),raw_w/sum(raw_w) OVER() AS w FROM raw_weights''')


def verify_model():
    p,m=checked_model();root=base.ROOT;c=base.conn();training_sql(c,p)
    d=c.sql('SELECT * FROM training ORDER BY date,code').df();names=list(base.EXPRESSIONS)
    original_train=relative.training('relative')
    pd.testing.assert_frame_equal(d[['date','code']],original_train[['date','code']],check_exact=True)
    np.testing.assert_allclose(d.target,original_train.target,rtol=0,atol=2e-12)
    np.testing.assert_array_equal(d[names].to_numpy(),base.encode(original_train))
    means=np.array(c.sql('SELECT '+','.join(f'sum(w*{n})/sum(w)' for n in names)+' FROM training').fetchone())
    scales=np.sqrt(np.array(c.sql('SELECT '+','.join(f'sum(w*({n}-{mu:.17e})*({n}-{mu:.17e}))/sum(w)'
        for n,mu in zip(names,means))+' FROM training').fetchone()))
    constants=scales<=1e-12;scales[constants]=1
    np.testing.assert_allclose(means,m['means'],rtol=0,atol=2e-7)
    np.testing.assert_allclose(scales,m['scales'],rtol=0,atol=2e-7)
    assert m['constant_columns']==[n for n,v in zip(names,constants) if v]
    z=(d[names].to_numpy(dtype=float)-means)/scales
    reference=Ridge(alpha=1.,fit_intercept=True,solver='cholesky').fit(z,d.target,sample_weight=d.w)
    np.testing.assert_allclose(reference.coef_,m['coefficients'],rtol=0,atol=2e-10)
    np.testing.assert_allclose(reference.intercept_,m['bias'],rtol=0,atol=2e-10)
    actual=predict(d[names].to_numpy(dtype=float),m)
    expected=reference.predict(z)
    np.testing.assert_allclose(actual,expected,rtol=0,atol=2e-10)
    np.testing.assert_allclose(np.quantile(expected,.995),m['thresholds'][0]['threshold'],rtol=0,atol=2e-10)
    assert m['rows']==len(d) and m['days']==d.date.nunique()
    assert original_train.next_date.max()==m['last_observation']
    assert abs(d.w.sum()-1)<2e-12 and abs(m['weights_sum']-1)<2e-12
    c.close()
    proof=dict(passed=True,model_report_sha256=sha(root/'model_report.json'),rows=len(d),days=d.date.nunique(),
        all_targets_integer_inputs_date_weights_training_means_and_scales_rebuilt=True,
        all_coefficients_verified_with_independent_cholesky_estimator=True,
        max_training_score_difference=float(np.max(np.abs(actual-expected))),
        training_start=p['training_start'],training_end=p['training_end'],new_2026_prices_read=False,no_exit_rules=True)
    save_json(root/'model_verification.json',proof);return proof


def scores():
    root=base.ROOT;p,m=checked_model()
    if (root/'score_report.json').exists():
        raise ValueError('Do not replace frozen ridge scores')
    proof=json.loads((root/'model_verification.json').read_text())
    assert proof['passed'] and proof['model_report_sha256']==sha(root/'model_report.json')
    f=base.feature_inputs();out=f[['date','code','half','board','decision_shares','formula_input_valid']].copy()
    out['score']=np.nan;valid=f.formula_input_valid
    out.loc[valid,'score']=predict(base.encode(f.loc[valid]).astype(float),m)
    out.to_parquet(root/'scores.parquet',index=False,compression='zstd')
    r=dict(protocol_sha256=sha(base.PROTOCOL),model_report_sha256=sha(root/'model_report.json'),
        feature_report_sha256=sha(base.FEATURES/'feature_report.json'),scores_sha256=sha(root/'scores.parquet'),
        rows=len(out),valid=int(valid.sum()),no_outcome_based_scoring=True,new_2026_prices_read=False,no_exit_rules=True)
    save_json(root/'score_report.json',r);return r


def verify_scores():
    root=base.ROOT;p,m=checked_model();r=json.loads((root/'score_report.json').read_text())
    for key,path in [('protocol_sha256',base.PROTOCOL),('model_report_sha256',root/'model_report.json'),
        ('feature_report_sha256',base.FEATURES/'feature_report.json'),('scores_sha256',root/'scores.parquet')]:
        assert r[key]==sha(path)
    c=base.conn();names=list(base.EXPRESSIONS)
    c.read_parquet(str(base.FEATURES/'features.parquet')).create_view('features')
    encoded=','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::DOUBLE AS X{i:02d}' for i,n in enumerate(names,1))
    c.sql('SELECT date,code,'+encoded+' FROM features WHERE formula_input_valid').create_view('encoded')
    terms=[f'({coef:.17e}*(X{i:02d}-{mean:.17e})/{scale:.17e})'
           for i,(coef,mean,scale) in enumerate(zip(m['coefficients'],m['means'],m['scales']),1)]
    expression=format(m['bias'],'.17e')+'+'+'+'.join(terms)
    c.sql('SELECT date,code,'+expression+' AS score FROM encoded').create_view('rebuilt')
    expected=c.sql('''SELECT f.date,f.code,f.half,f.board,f.decision_shares,f.formula_input_valid,r.score
        FROM features f LEFT JOIN rebuilt r USING(date,code) ORDER BY date,code''').df()
    actual=pd.read_parquet(root/'scores.parquet')
    pd.testing.assert_frame_equal(actual.drop(columns='score'),expected.drop(columns='score'),check_exact=True)
    np.testing.assert_allclose(actual.score,expected.score,rtol=0,atol=2e-12,equal_nan=True)
    assert actual.score.notna().equals(actual.formula_input_valid)
    cut=m['thresholds'][0]['threshold'];np.testing.assert_array_equal(actual.score.gt(cut),expected.score.gt(cut))
    training_sql(c,p)
    train=c.sql('SELECT score FROM rebuilt JOIN training USING(date,code) ORDER BY date,code').df()
    np.testing.assert_allclose(np.quantile(train.score,.995),cut,rtol=0,atol=2e-12)
    c.close()
    proof=dict(passed=True,score_report_sha256=sha(root/'score_report.json'),rows=len(actual),valid=int(actual.formula_input_valid.sum()),
        all_scores_encodings_threshold_flags_and_training_quantile_rebuilt=True,
        max_score_difference=float((actual.score-expected.score).abs().max()),new_2026_prices_read=False,no_exit_rules=True)
    save_json(root/'score_verification.json',proof);return proof


def native_core(m):
    core=base.HEADER+'\n'.join(f'{k}:={v};' for k,v in base.EXPRESSIONS.items())+'\n'
    core+='\n'.join(f'X{i:02d}:=INTPART(MIN(MAX(100*{n}+10000+0.000001,0),999999));' for i,n in enumerate(base.EXPRESSIONS,1))+'\n'
    for i,(coef,mean,scale) in enumerate(zip(m['coefficients'],m['means'],m['scales']),1):
        core+=f'L{i:02d}:={coef:.17g}*(X{i:02d}-{mean:.17g})/{scale:.17g};\n'
    core+='SC:='+format(m['bias'],'.17g')+'+'+'+'.join(f'L{i:02d}' for i in range(1,49))+';\n'
    return core+'CORE:SC>'+format(m['thresholds'][0]['threshold'],'.17g')+';\n'


def freeze():
    root=base.ROOT;p,m=checked_model()
    if (root/'selection_report.json').exists():
        raise ValueError('Do not replace the frozen ridge selection')
    s=json.loads((root/'score_report.json').read_text());proof=json.loads((root/'score_verification.json').read_text())
    assert proof['passed'] and proof['score_report_sha256']==sha(root/'score_report.json')
    assert s['scores_sha256']==sha(root/'scores.parquet') and s['model_report_sha256']==sha(root/'model_report.json')
    f=pd.read_parquet(root/'scores.parquet');out=f[['date','code','half','board','decision_shares']].copy()
    out['selected']=(f.date.ge(p['evaluation_start'])&f.date.lt(p['evaluation_end'])&f.formula_input_valid&f.score.gt(m['thresholds'][0]['threshold']))
    out.to_parquet(root/'selection.parquet',index=False,compression='zstd')
    (root/'frozen_numeric_core.tdx').write_text(native_core(m))
    r=dict(protocol_sha256=sha(base.PROTOCOL),model_report_sha256=sha(root/'model_report.json'),
        score_report_sha256=sha(root/'score_report.json'),selection_sha256=sha(root/'selection.parquet'),
        core_sha256=sha(root/'frozen_numeric_core.tdx'),chosen_threshold=m['thresholds'][0],selected=int(out.selected.sum()),
        by_half=out.groupby('half').selected.agg(['size','sum']).reset_index().to_dict('records'),
        evaluation_start=p['evaluation_start'],evaluation_end=p['evaluation_end'],new_group_outcomes_read=False,
        year_2025_is_exploratory=True,new_2026_prices_read=False,no_exit_rules=True,software_compilation_verified=False)
    save_json(root/'selection_report.json',r);return r


def verify():
    import re
    root=base.ROOT;p,m=checked_model();r=json.loads((root/'selection_report.json').read_text())
    for key,path in [('protocol_sha256',base.PROTOCOL),('model_report_sha256',root/'model_report.json'),
        ('score_report_sha256',root/'score_report.json'),('selection_sha256',root/'selection.parquet'),('core_sha256',root/'frozen_numeric_core.tdx')]:
        assert r[key]==sha(path)
    cut=m['thresholds'][0];assert r['chosen_threshold']==cut and cut['training_quantile']==.995
    c=base.conn();c.read_parquet(str(root/'scores.parquet')).create_view('scores')
    expected=c.sql(f'''SELECT date,code,half,board,decision_shares,
        date>='{p['evaluation_start']}' AND date<'{p['evaluation_end']}' AND formula_input_valid
        AND score>{cut['threshold']:.17e} AS selected FROM scores ORDER BY date,code''').df();c.close()
    pd.testing.assert_frame_equal(pd.read_parquet(root/'selection.parquet'),expected,check_exact=True)
    assert int(expected.selected.sum())==r['selected']
    core=(root/'frozen_numeric_core.tdx').read_text();assert core==native_core(m)
    terms=re.findall(r'^L(\d{2}):=([^;]+);$',core,re.M)
    assert [i for i,_ in terms]==[f'{i:02d}' for i in range(1,49)]
    for index,expr in terms:
        coeff,other=expr.split('*(X'+index+'-',1);mean,scale=other.split(')/')
        i=int(index)-1
        assert [float(coeff),float(mean),float(scale)]==[m['coefficients'][i],m['means'][i],m['scales'][i]]
    assert expected.loc[expected.selected,'date'].ge(m['training_end']).all()
    proof=dict(passed=True,selection_report_sha256=sha(root/'selection_report.json'),rows=len(expected),
        all_selection_flags_rebuilt=True,all_native_coefficients_and_sum_order_preserved=True,
        no_training_period_selection=True,no_new_group_outcomes_read=True,new_2026_prices_read=False,no_exit_rules=True)
    save_json(root/'selection_verification.json',proof);return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['model','verify_model','scores','verify_scores','freeze','verify','analyze'])
    p.add_argument('--fold',choices=['2024','recent','combined'],default='2024');a=p.parse_args();setup(a.fold)
    if a.fold=='combined':
        assert a.stage in ['freeze','verify','analyze']
        r=(linkage.common_analysis(linkage.COMBINED,linkage.PROTOCOL) if a.stage=='analyze'
            else getattr(linkage,'combine' if a.stage=='freeze' else 'verify_combined')())
    elif a.stage=='analyze':
        r=linkage.common_analysis(base.ROOT,base.PROTOCOL)
    else:
        r=globals()[a.stage]()
    print(json.dumps(r,ensure_ascii=False,indent=2))
