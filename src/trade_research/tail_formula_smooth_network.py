"""One fixed eight-unit smooth model and a matched-penalty affine control."""
import argparse
import json
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.neural_network import MLPRegressor
from threadpoolctl import threadpool_limits

from . import tail_formula_additive as base
from . import tail_formula_context_2024 as linkage
from . import tail_formula_float as original
from . import tail_formula_polynomial_ridge as reuse
from . import tail_formula_relative as relative
from .corporate_cash import save_json, sha

STEM = 'tail_formula_smooth_network'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
ARM = None
ALPHA = .0001


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    for path, digest in {**p['references'], **p['local_primary_source_hashes']}.items():
        assert sha(Path(path)) == digest
    assert sklearn.__version__ == p['sklearn_version'] == '1.7.2'
    assert p['arms'] == ['linear', 'smooth'] and p['ridge_alpha'] == ALPHA
    r = json.loads((original.ROOT / 'feature_report.json').read_text())
    v = json.loads((original.ROOT / 'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(original.ROOT / 'feature_report.json')
    assert r['features_sha256'] == sha(original.ROOT / 'features.parquet')
    return p


def setup(arm, fold):
    global ARM
    assert arm in ['linear', 'smooth'] and fold in ['2024', 'recent', 'combined']
    ARM = arm
    base.FEATURES = original.ROOT; base.EXPRESSIONS = original.EXPRESSIONS; base.HEADER = original.HEADER
    base.SOURCE = Path('data/research/tail_formula_before1000')
    stem = STEM + '_' + arm
    base.ROOT = Path('data/research') / (stem + '_' + ('2025' if fold == 'combined' else fold))
    base.PROTOCOL = Path('config') / (stem + '_' + fold + '_protocol.json'); relative.PROTOCOL = base.PROTOCOL
    # These four existing artifact routines are model-family independent. Every
    # stage runs in its own process; their mathematical callbacks are explicit.
    reuse.checked_model = checked_model; reuse.predict = predict
    reuse.native_core = native_core; reuse.verify_native = verify_native
    reuse.ROOT = ROOT; reuse.MASTER = PROTOCOL
    if fold == 'combined':
        linkage.ROOT = Path('data/research') / (stem + '_2024')
        linkage.H2 = Path('data/research') / (stem + '_recent')
        linkage.COMBINED = base.ROOT; linkage.PROTOCOL = base.PROTOCOL
        linkage.H2_SELECTION_SHA = None; linkage.H2_MODEL_SHA = None; linkage.H2_OUTCOMES_PREVIOUSLY_SEEN = False


def config():
    master = checked_sources(); p = json.loads(base.PROTOCOL.read_text())
    assert p['inputs_protocol_sha256'] == sha(PROTOCOL) and p['arm'] == ARM
    assert p['feature_report_sha256'] == sha(original.ROOT / 'feature_report.json')
    assert p['label_report_sha256'] == sha(base.SOURCE / 'full_label_report.json')
    assert p['ridge_alpha'] == ALPHA and p['parameters'] == master['parameters']
    assert p['expected_features'] == 48 and p['training_quantile'] == .995
    assert p['training_end'] == p['evaluation_start'] and p['window_end'] == '09:59'
    return p


def standardized(x, m):
    return np.clip((x-np.asarray(m['input_means'])) / np.asarray(m['input_scales']), -5, 5)


def loss_gradient(z, y, weights, hidden, hidden_bias, output, bias, alpha=ALPHA):
    """Independent full-batch objective, not the optimizer's private backprop."""
    h = np.tanh(z @ hidden + hidden_bias)
    residual = h @ output + bias-y
    loss = .5*np.dot(weights, residual*residual) + .5*alpha*(np.square(hidden).sum()+np.square(output).sum())
    delta = weights*residual
    hidden_delta = delta[:, None]*output*(1-h*h)
    gradient = [z.T @ hidden_delta+alpha*hidden, hidden_delta.sum(axis=0),
                h.T @ delta+alpha*output, np.array([delta.sum()])]
    return float(loss), gradient


def predict(x, m):
    # Match the explicitly exported addition order. Backend tanh is checked
    # separately before any economic evaluation.
    out = np.empty(len(x))
    for start in range(0, len(x), 8192):
        z = standardized(x[start:start+8192], m)
        score = np.full(len(z), m['bias'])
        if m['arm'] == 'linear':
            for j in range(48): score += m['coefficients'][j]*z[:, j]
        else:
            for j in range(8):
                a = np.full(len(z), m['hidden_bias'][j])
                for i in range(48): a += m['hidden_weights'][i][j]*z[:, i]
                h = 2/(1+np.exp(-2*np.clip(a, -20, 20)))-1
                score += m['coefficients'][j]*h
        out[start:start+len(z)] = score
    return out


def model():
    p = config(); master = checked_sources(); root = base.ROOT
    assert not (root / 'model_report.json').exists()
    t = relative.training('relative')
    spec = next(v for v in master['training'] if v['start'] == p['training_start'])
    assert len(t) == spec['rows'] and t.date.nunique() == spec['days'] == 241
    x = base.encode(t).astype(float); y = t.target.to_numpy()
    w = 1/t.groupby('date').code.transform('size').to_numpy()/241; w /= w.sum()
    means = np.average(x, axis=0, weights=w)
    scales = np.sqrt(np.average((x-means)**2, axis=0, weights=w)); constant = scales <= 1e-12; scales[constant] = 1
    m = dict(arm=ARM, input_means=means.tolist(), input_scales=scales.tolist(),
        input_constant_indices=np.flatnonzero(constant).tolist(), standardized_clip=5, scale_floor=1e-12,
        ridge_alpha=ALPHA, parameters=p['parameters'], weights_sum=float(w.sum()))
    z = standardized(x, m)
    with threadpool_limits(limits=2):
        if ARM == 'linear':
            mu = w @ z; ym = float(w @ y); centered = z-mu
            coef = np.linalg.solve(centered.T @ (w[:, None]*centered)+ALPHA*np.eye(48), centered.T @ (w*(y-ym)))
            bias = float(ym-mu @ coef); residual = z @ coef+bias-y
            m.update(coefficients=coef.tolist(), bias=bias, optimization_warnings=[], iteration_budget_reached=False,
                optimizer_loss=float(.5*np.dot(w, residual*residual)+.5*ALPHA*np.square(coef).sum()))
        else:
            parameters = dict(p['parameters']); parameters['hidden_layer_sizes'] = tuple(parameters['hidden_layer_sizes'])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                estimator = MLPRegressor(**parameters).fit(z, y, sample_weight=w)
            m.update(hidden_weights=estimator.coefs_[0].tolist(), hidden_bias=estimator.intercepts_[0].tolist(),
                coefficients=estimator.coefs_[1][:, 0].tolist(), bias=float(estimator.intercepts_[1][0]),
                optimizer_loss=float(estimator.loss_), iterations=int(estimator.n_iter_),
                optimization_warnings=[dict(category=type(v.message).__name__, message=str(v.message)) for v in caught],
                iteration_budget_reached=bool(estimator.n_iter_ >= p['parameters']['max_iter']))
    m.update(protocol_sha256=sha(base.PROTOCOL), inputs_protocol_sha256=sha(PROTOCOL),
        feature_report_sha256=sha(original.ROOT / 'feature_report.json'), label_report_sha256=sha(base.SOURCE / 'full_label_report.json'),
        model_family='date_equal_fixed_smooth_comparison', feature_names=list(original.EXPRESSIONS),
        rows=len(t), days=241, training_start=p['training_start'], training_end=p['training_end'], last_observation=t.next_date.max(),
        score_is_not_probability=True, new_2026_prices_read=False, no_exit_rules=True)
    scores = predict(x, m)
    m['thresholds'] = [dict(id=0, training_quantile=.995, threshold=float(np.quantile(scores, .995)))]
    root.mkdir(parents=True, exist_ok=True); save_json(root / 'model_report.json', m)
    return dict(arm=ARM, rows=len(t), days=241, optimizer_loss=m['optimizer_loss'], iterations=m.get('iterations'),
        iteration_budget_reached=m['iteration_budget_reached'], optimization_warnings=m['optimization_warnings'])


def checked_model():
    p = config(); m = json.loads((base.ROOT / 'model_report.json').read_text())
    for key, path in [('protocol_sha256', base.PROTOCOL), ('inputs_protocol_sha256', PROTOCOL),
                      ('feature_report_sha256', original.ROOT / 'feature_report.json'), ('label_report_sha256', base.SOURCE / 'full_label_report.json')]:
        assert m[key] == sha(path)
    assert m['arm'] == ARM and m['model_family'] == 'date_equal_fixed_smooth_comparison'
    assert m['feature_names'] == list(original.EXPRESSIONS) and m['ridge_alpha'] == ALPHA
    assert m['standardized_clip'] == 5 and m['scale_floor'] == 1e-12 and m['parameters'] == p['parameters']
    assert min(m['input_scales']) > 0 and m['last_observation'] < p['training_end']
    for key in ['input_means', 'input_scales', 'coefficients', 'bias']:
        assert np.isfinite(m[key]).all()
    assert len(m['coefficients']) == (48 if ARM == 'linear' else 8)
    if ARM == 'smooth':
        assert np.asarray(m['hidden_weights']).shape == (48, 8) and np.asarray(m['hidden_bias']).shape == (8,)
        assert np.isfinite(m['hidden_weights']).all() and np.isfinite(m['hidden_bias']).all()
    return p, m


def verify_model():
    p, m = checked_model(); c = base.conn(); reuse.register_features(c); reuse.training_sql(c, p)
    d = c.sql('SELECT * FROM training ORDER BY date,code').df(); t = relative.training('relative')
    pd.testing.assert_frame_equal(d[['date', 'code']], t[['date', 'code']], check_exact=True)
    np.testing.assert_allclose(d.target, t.target, rtol=0, atol=2e-12)
    np.testing.assert_array_equal(d[list(original.EXPRESSIONS)].to_numpy(), base.encode(t))
    w = 1/t.groupby('date').code.transform('size').to_numpy()/241; w /= w.sum()
    np.testing.assert_allclose(d.w, w, rtol=0, atol=2e-15)
    names = list(original.EXPRESSIONS)
    means = np.array(c.sql('SELECT '+','.join(f'sum(w*{n})/sum(w)' for n in names)+' FROM training').fetchone())
    variance = c.sql('SELECT '+','.join(f'sum(w*({n}-({v:.17e}))*({n}-({v:.17e})))/sum(w)' for n,v in zip(names,means))+' FROM training').fetchone()
    scales = np.sqrt(variance); constant = scales <= 1e-12; scales[constant] = 1
    np.testing.assert_allclose(means, m['input_means'], rtol=0, atol=2e-7)
    np.testing.assert_allclose(scales, m['input_scales'], rtol=0, atol=2e-7)
    assert np.flatnonzero(constant).tolist() == m['input_constant_indices']
    expressions = [f'least(greatest(({n}-({mu:.17e}))/({s:.17e}),-5e0),5e0) AS Z{i}'
        for i,(n,mu,s) in enumerate(zip(names,m['input_means'],m['input_scales']))]
    z = c.sql('SELECT '+','.join(expressions)+' FROM training ORDER BY date,code').df().to_numpy(); c.close()
    y = d.target.to_numpy(); w = d.w.to_numpy(); w /= w.sum(); coef = np.asarray(m['coefficients'])
    with threadpool_limits(limits=2):
        if ARM == 'linear':
            reference = m['bias'] + np.einsum('ij,j->i', z, coef)
            residual = reference-y
            loss = .5*np.dot(w,residual*residual)+.5*ALPHA*np.square(coef).sum()
            gradient = [np.einsum('i,ij->j',w*residual,z)+ALPHA*coef, np.array([np.sum(w*residual)])]
            assert max(float(np.max(np.abs(v))) for v in gradient) < 2e-10
        else:
            hidden = np.asarray(m['hidden_weights']); hb = np.asarray(m['hidden_bias'])
            h = np.tanh(np.einsum('ij,jk->ik', z, hidden)+hb)
            reference = m['bias'] + np.einsum('ij,j->i',h,coef); residual = reference-y
            hidden_delta = (w*residual)[:,None]*coef*(1-h*h)
            gradient = [np.einsum('ij,ik->jk',z,hidden_delta)+ALPHA*hidden, hidden_delta.sum(axis=0),
                np.einsum('i,ij->j',w*residual,h)+ALPHA*coef, np.array([np.sum(w*residual)])]
            loss = .5*np.dot(w,residual*residual)+.5*ALPHA*(np.square(hidden).sum()+np.square(coef).sum())
            original_loss, original_gradient = loss_gradient(z,y,w,hidden,hb,coef,m['bias'])
            np.testing.assert_allclose(loss,original_loss,rtol=0,atol=2e-12)
            for a,b in zip(gradient,original_gradient): np.testing.assert_allclose(a,b,rtol=0,atol=2e-10)
        np.testing.assert_allclose(loss,m['optimizer_loss'],rtol=0,atol=2e-10)
        actual = predict(d[names].to_numpy(),m)
        np.testing.assert_allclose(actual,reference,rtol=0,atol=2e-10)
    cut = m['thresholds'][0]['threshold']
    np.testing.assert_allclose(np.quantile(reference,.995),cut,rtol=0,atol=2e-10)
    np.testing.assert_array_equal(actual>cut, reference>cut)
    assert len(d)==m['rows'] and d.date.nunique()==m['days']==241
    assert t.next_date.max()==m['last_observation'] and abs(m['weights_sum']-1)<2e-12
    proof = dict(passed=True,model_report_sha256=sha(base.ROOT/'model_report.json'),rows=len(d),days=241,
        parameter_checks=49 if ARM=='linear' else 401,all_targets_encodings_date_weights_sql_verified=True,
        all_input_statistics_and_training_objective_verified=True,all_training_gradients_independently_rebuilt=True,
        max_gradient=float(max(np.max(np.abs(v)) for v in gradient)),
        iteration_budget_reached=m['iteration_budget_reached'],optimizer_warnings=m['optimization_warnings'],
        max_training_score_difference=float(np.max(np.abs(actual-reference))),new_2026_prices_read=False,no_exit_rules=True)
    save_json(base.ROOT/'model_verification.json',proof);return proof


def native_definitions(m):
    definitions = []
    for i,name in enumerate(m['feature_names'],1):
        definitions.append((f'X{i:02d}',f'INTPART(MIN(MAX(100*{name}+10000+0.000001,0),999999))'))
    for i,(mu,s) in enumerate(zip(m['input_means'],m['input_scales']),1):
        definitions.append((f'PZ{i:02d}',f'MIN(MAX((X{i:02d}-({mu:.17e}))/({s:.17e}),-5),5)'))
    if m['arm']=='smooth':
        for j in range(8):
            terms=[f'({m["hidden_bias"][j]:.17e})']+[f'({m["hidden_weights"][i][j]:.17e})*PZ{i+1:02d}' for i in range(48)]
            definitions.append((f'NH{j+1:02d}','+'.join(terms)))
            definitions.append((f'NV{j+1:02d}',f'2/(1+EXP(-2*MIN(MAX(NH{j+1:02d},-20),20)))-1'))
        terms=[f'({v:.17e})*NV{j+1:02d}' for j,v in enumerate(m['coefficients'])]
    else:terms=[f'({v:.17e})*PZ{j+1:02d}' for j,v in enumerate(m['coefficients'])]
    definitions.append(('SC',f'({m["bias"]:.17e})+'+'+'.join(terms)))
    return definitions


def native_core(m):
    return original.HEADER+'\n'.join(f'{n}:={v};' for n,v in original.EXPRESSIONS.items())+'\n'+\
        '\n'.join(f'{n}:={v};' for n,v in native_definitions(m))+f'\nCORE:SC>{m["thresholds"][0]["threshold"]:.17e};\n'


def verify_native(core,m):
    pairs=re.findall(r'^([A-Z][A-Z0-9]*):=([^;]+);$',core,re.M)
    assert len(pairs)==len({n.casefold() for n,_ in pairs})
    d=dict(pairs)
    for n,v in native_definitions(m):assert d[n]==v
    assert core.endswith(f'CORE:SC>{m["thresholds"][0]["threshold"]:.17e};\n')
    return d


def verify_scores():
    p,m=checked_model();root=base.ROOT;r=json.loads((root/'score_report.json').read_text())
    for key,path in [('protocol_sha256',base.PROTOCOL),('model_report_sha256',root/'model_report.json'),
        ('feature_report_sha256',original.ROOT/'feature_report.json'),('scores_sha256',root/'scores.parquet'),('numeric_score_sha256',root/'numeric_score.tdx')]:assert r[key]==sha(path)
    core=(root/'numeric_score.tdx').read_text();assert core==native_core(m);d=verify_native(core,m)
    sql=lambda v:re.sub(r'\bINTPART\(','floor(',re.sub(r'\bMIN\(','least(',re.sub(r'\bMAX\(','greatest(',v)))
    c=base.conn();reuse.register_features(c)
    c.sql('SELECT date,code,'+','.join(sql(d[f'X{i:02d}'])+f'::DOUBLE AS X{i:02d}' for i in range(1,49))+' FROM features WHERE formula_input_valid').create_view('encoded')
    c.sql('SELECT date,code,'+','.join(sql(d[f'PZ{i:02d}'])+f' AS PZ{i:02d}' for i in range(1,49))+' FROM encoded').create_view('transformed')
    source='transformed'
    if ARM=='smooth':
        c.sql('SELECT date,code,'+','.join(sql(d[f'NH{i:02d}'])+f' AS NH{i:02d}' for i in range(1,9))+' FROM transformed').create_view('hidden')
        c.sql('SELECT date,code,'+','.join(sql(d[f'NV{i:02d}'])+f' AS NV{i:02d}' for i in range(1,9))+' FROM hidden').create_view('nonlinear');source='nonlinear'
    c.sql('SELECT date,code,'+sql(d['SC'])+' AS score FROM '+source).create_view('rebuilt')
    expected=c.sql('SELECT f.date,f.code,f.half,f.board,f.decision_shares,f.formula_input_valid,r.score FROM features f LEFT JOIN rebuilt r USING(date,code) ORDER BY date,code').df()
    actual=pd.read_parquet(root/'scores.parquet')
    pd.testing.assert_frame_equal(actual.drop(columns='score'),expected.drop(columns='score'),check_exact=True)
    np.testing.assert_allclose(actual.score,expected.score,rtol=0,atol=2e-10,equal_nan=True)
    assert actual.score.notna().equals(actual.formula_input_valid)
    cut=m['thresholds'][0]['threshold'];np.testing.assert_array_equal(actual.score.gt(cut),expected.score.gt(cut))
    reuse.training_sql(c,p);c.register('replayed',expected[['date','code','score']])
    train=c.sql('SELECT score FROM replayed JOIN training USING(date,code) ORDER BY date,code').df()
    np.testing.assert_allclose(np.quantile(train.score,.995),cut,rtol=0,atol=2e-10);c.close()
    proof=dict(passed=True,score_report_sha256=sha(root/'score_report.json'),rows=len(actual),valid=int(actual.formula_input_valid.sum()),
        all_exported_native_arithmetic_sql_replayed=True,all_threshold_flags_and_training_quantile_verified=True,
        max_score_difference=float((actual.score-expected.score).abs().max()),software_compilation_verified=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(root/'score_verification.json',proof);return proof


def main(arm=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['model','verify_model','scores','verify_scores','freeze','verify','analyze'])
    p.add_argument('--arm',choices=['linear','smooth'],required=arm is None,default=arm)
    p.add_argument('--fold',choices=['2024','recent','combined'],default='2024');a=p.parse_args();setup(a.arm,a.fold)
    if a.fold=='combined' and a.stage!='analyze':
        assert a.stage in ['freeze','verify'];result=getattr(linkage,'combine' if a.stage=='freeze' else 'verify_combined')()
    elif a.stage in ['scores','freeze','verify','analyze']:result=getattr(reuse,a.stage)()
    else:result=globals()[a.stage]()
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
