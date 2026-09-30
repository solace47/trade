"""Two new fixed relative models, plus exact reuse of the two old controls."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_relative as relative
from . import tail_formula_morning_trend as study
from .corporate_cash import save_json, sha
from .tail_formula_offset_logit48 import verify_scores

PROTOCOL = Path('config/tail_formula_morning_trend_model_protocol.json')
OLD = {fold:study.prior.ROOT/'morning'/fold for fold in ['2025h1','2025h2']}


def checked():
    study.checked(); p = json.loads(PROTOCOL.read_text())
    assert p['input_protocol_sha256']==sha(study.PROTOCOL)
    assert p['old_controls']=={f:str(r) for f,r in OLD.items()}
    assert p['arms']==study.ARMS and p['threshold']==.995
    for file,digest in p['source_hashes'].items():
        assert sha(Path(file))==digest,file
    fr = json.loads((study.INPUTS/'feature_report.json').read_text())
    for kind in ['feature','native_input']:
        v = json.loads((study.INPUTS/(kind+'_verification.json')).read_text())
        assert v['passed'] and v['feature_report_sha256']==sha(study.INPUTS/'feature_report.json')
    assert fr['features_sha256']==sha(study.INPUTS/'features.parquet')
    assert json.loads((study.INPUTS/'feature_verification.json').read_text())['effective_input_intersection_unchanged']
    return p


def fold_protocol(arm,fold):
    return Path('config')/(study.STEM+'_'+arm+'_'+fold+'_protocol.json')


def protocols():
    p = checked(); base.FEATURES=base.SOURCE=study.INPUTS; base.EXPRESSIONS=study.ARMS['control']
    counts = {}; records = []
    receipts = {str(study.INPUTS/f):sha(study.INPUTS/f) for f in
        ['feature_report.json','feature_verification.json','native_input_verification.json',
         'full_label_report.json','full_label_verification.json']}
    for fold,spec in p['folds'].items():
        t = base.training(start=spec['training_start'],end=spec['training_end'])
        assert t.next_date.max()<spec['evaluation_start']
        counts[fold] = dict(rows=len(t),days=t.date.nunique(),last_observation=t.next_date.max())
    for arm,expr in study.ARMS.items():
        for fold,spec in p['folds'].items():
            a = counts[fold]
            q = dict(master_protocol_sha256=sha(PROTOCOL),arm=arm,fold=fold,**spec,
                expected_training_rows=a['rows'],expected_training_days=a['days'],
                expected_last_observation=a['last_observation'],expected_features=len(expr),
                feature_names=list(expr),parameters=p['parameters'],model_max_depth=3,threshold=.995,
                target='relative',input_receipts=receipts,no_training_period_selection=True,
                new_2026_prices_allowed=False,no_exit_rules=True)
            file = fold_protocol(arm,fold); assert not file.exists(); save_json(file,q)
            result = subprocess.run(['.venv/bin/python','scripts/find_existing_tail_formula_models.py',
                '--protocol',str(file),'--variant','relative'],capture_output=True,text=True,check=True)
            records.append(dict(arm=arm,fold=fold,protocol_sha256=sha(file),lookup=json.loads(result.stdout)))
    assert not any(r['lookup']['matches'] for r in records if r['arm']=='trend'), 'Audit and reuse matching morning-trend models'
    out = dict(passed=True,master_protocol_sha256=sha(PROTOCOL),counts=counts,records=records,
        exactly_two_new_models_allowed=True,two_old_controls_to_reuse_after_exact_audit=True,
        all_four_protocols_before_fitting=True,no_new_group_outcomes_read=True,new_2026_prices_read=False,no_exit_rules=True)
    save_json(study.ROOT/'prefit_lookup_verification.json',out)
    return dict(prefit_sha256=sha(study.ROOT/'prefit_lookup_verification.json'),counts=counts)


def setup(arm,fold):
    p = checked(); base.ROOT=study.ROOT/arm/fold; base.FEATURES=base.SOURCE=study.INPUTS
    base.EXPRESSIONS=study.ARMS[arm]; base.HEADER=study.HEADER
    base.PROTOCOL=relative.PROTOCOL=fold_protocol(arm,fold)
    q = json.loads(base.PROTOCOL.read_text()); pre = json.loads((study.ROOT/'prefit_lookup_verification.json').read_text())
    assert q['master_protocol_sha256']==sha(PROTOCOL)==pre['master_protocol_sha256'] and pre['passed']
    assert q['arm']==arm and q['fold']==fold and q['feature_names']==list(base.EXPRESSIONS)
    assert q['parameters']==p['parameters'] and all(q[k]==v for k,v in p['folds'][fold].items())
    assert any(r['arm']==arm and r['fold']==fold and r['protocol_sha256']==sha(base.PROTOCOL) for r in pre['records'])
    for file,digest in q['input_receipts'].items():assert sha(Path(file))==digest
    return q


def reuse(fold):
    q = setup('control',fold); root = base.ROOT; parent = OLD[fold]
    assert not (root/'model_report.json').exists()
    cols = [*study.META,*study.ARMS['control']]
    new = pd.read_parquet(study.INPUTS/'features.parquet',columns=cols)
    old = pd.read_parquet(study.prior.INPUTS/'features.parquet',columns=cols,filters=[('date','>=','2024-01-01'),('date','<','2026-01-01')]).reset_index(drop=True)
    pd.testing.assert_frame_equal(new,old,check_exact=True)
    pm = json.loads((parent/'model_report.json').read_text()); sr = json.loads((parent/'score_report.json').read_text())
    mv = json.loads((parent/'model_verification.json').read_text()); sv = json.loads((parent/'score_verification.json').read_text())
    assert mv['passed'] and mv['model_report_sha256']==sha(parent/'model_report.json')
    assert sv['passed'] and sv['score_report_sha256']==sha(parent/'score_report.json')
    assert pm['feature_report_sha256']==sr['feature_report_sha256']==sha(study.prior.INPUTS/'feature_report.json')
    assert sr['model_report_sha256']==sha(parent/'model_report.json') and sr['scores_sha256']==sha(parent/'scores.parquet')
    assert pm['label_report_sha256']==sha(study.INPUTS/'full_label_report.json')
    assert pm['feature_names']==q['feature_names'] and pm['variant']=='relative'
    assert all(pm['parameters'][k]==v for k,v in q['parameters'].items())
    fresh = relative.training('relative'); c = base.conn(); c.register('parent',old)
    fields = ','.join('f.'+n for n in study.ARMS['control'])
    prior = c.sql(f'''WITH labels AS(SELECT date,code,next_date,opportunity15,
        opportunity15-avg(opportunity15) OVER(PARTITION BY date) AS target
        FROM read_parquet('{study.prior.INPUTS}/full_labels.parquet')
        WHERE known15 AND date>='{q['training_start']}' AND next_date<'{q['training_end']}')
        SELECT date,code,next_date,opportunity15,target,1./count(*) OVER(PARTITION BY date) AS w,{fields}
        FROM parent f JOIN labels USING(date,code) WHERE formula_input_valid ORDER BY date,code''').df(); c.close()
    fields = ['date','code','next_date','opportunity15',*study.ARMS['control']]
    pd.testing.assert_frame_equal(fresh[fields],prior[fields],check_exact=True,check_dtype=False)
    np.testing.assert_allclose(fresh.target,prior.target,rtol=0,atol=2e-12)
    np.testing.assert_array_equal(1/fresh.groupby('date').code.transform('size'),prior.w)
    assert len(prior)==q['expected_training_rows']==pm['rows']
    assert prior.date.nunique()==q['expected_training_days']==pm['days']
    assert prior.next_date.max()==q['expected_last_observation']==pm['last_observation']<q['evaluation_start']
    assert pm['training_start']==q['training_start'] and pm['training_end']==q['training_end']
    scores = pd.read_parquet(parent/'scores.parquet',filters=[('date','>=','2024-01-01'),('date','<','2026-01-01')]).reset_index(drop=True); pd.testing.assert_frame_equal(scores[study.META],new[study.META],check_exact=True)
    root.mkdir(parents=True,exist_ok=True)
    m = dict(pm); m.update(protocol_sha256=sha(base.PROTOCOL),feature_report_sha256=sha(study.INPUTS/'feature_report.json'),
        label_report_sha256=sha(study.INPUTS/'full_label_report.json'),no_model_fit_performed=True,
        reused_model_report_sha256=sha(parent/'model_report.json'),reused_source_root=str(parent))
    save_json(root/'model_report.json',m)
    for field in ['parameters','trees','bias','feature_names','thresholds','rows','days','last_observation']:
        assert m[field]==pm[field]
    scores.to_parquet(root/'scores.parquet',index=False,compression='zstd')
    projected = pd.read_parquet(root/'scores.parquet'); pd.testing.assert_frame_equal(projected,scores,check_exact=True)
    r = dict(sr); r.update(protocol_sha256=sha(base.PROTOCOL),model_report_sha256=sha(root/'model_report.json'),
        feature_report_sha256=sha(study.INPUTS/'feature_report.json'),no_prediction_performed=True,
        rows=len(scores),valid=int(scores.formula_input_valid.sum()),scores_sha256=sha(root/'scores.parquet'),
        original_all_year_score_report_sha256=sha(parent/'score_report.json'),only_2024_2025_complete_score_projection=True,
        reused_score_report_sha256=sha(parent/'score_report.json'))
    save_json(root/'score_report.json',r)
    save_json(root/'model_verification.json',dict(passed=True,model_report_sha256=sha(root/'model_report.json'),
        reused_model_verification_sha256=sha(parent/'model_verification.json'),rows=len(prior),days=pm['days'],
        node_checks=mv['node_checks'],exact_input_target_weight_and_parameter_equivalence_before_proof_reuse=True,
        all_original_arithmetic_checks_reused=True,no_new_arithmetic_or_fit=True,new_2026_prices_read=False,no_exit_rules=True))
    save_json(root/'score_verification.json',dict(passed=True,score_report_sha256=sha(root/'score_report.json'),rows=len(scores),
        valid=int(scores.formula_input_valid.sum()),reused_score_verification_sha256=sha(parent/'score_verification.json'),
        original_threshold_flag_checks=sv['threshold_flag_checks'],threshold_flag_checks=0,
        all_complete_score_metadata_values_quantiles_and_flags_reused_exactly=True,no_prediction_performed=True,
        new_2026_prices_read=False,no_exit_rules=True))
    proof = dict(passed=True,master_protocol_sha256=sha(PROTOCOL),fold=fold,parent=str(parent),
        source_hashes={str(parent/f):sha(parent/f) for f in ['model_report.json','model_verification.json','scores.parquet','score_report.json','score_verification.json']},
        rows=len(prior),days=pm['days'],complete_feature_frame_and_training_keys_targets_weights_exactly_matched=True,
        all_equations_bias_parameters_thresholds_and_full_scores_preserved=True,
        original_source_score_rows=1258085,projected_score_rows=len(scores),no_new_2023_inputs_or_predictions=True,
        no_refit_or_prediction=True,new_2026_prices_read=False,no_exit_rules=True)
    save_json(root/'control_reuse_verification.json',proof); return dict(fold=fold,rows=len(prior),days=pm['days'],reused=True)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['protocols','reuse','model','verify_model','scores','verify_scores'])
    p.add_argument('--fold',choices=['2025h1','2025h2']); a = p.parse_args()
    if a.stage=='protocols':result = protocols()
    elif a.stage=='reuse':result = reuse(a.fold)
    else:
        assert a.fold; q = setup('trend',a.fold)
        if a.stage=='model':result = relative.model('relative')
        elif a.stage=='verify_model':result = relative.verify_model('relative')
        elif a.stage=='scores':result = base.scores()
        else:result = verify_scores(expected_expressions=base.EXPRESSIONS)
    print(json.dumps(result,ensure_ascii=False,indent=2))
