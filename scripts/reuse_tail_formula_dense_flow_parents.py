"""Reuse four fixed parent models and project their scores without prediction."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trade_research import tail_formula_additive as base
from trade_research import tail_formula_dense_flow as study
from trade_research import tail_formula_dense_flow_model as model
from trade_research import tail_formula_relative as relative
from trade_research.corporate_cash import save_json, sha

PROTOCOL = Path('config/tail_formula_dense_flow_reuse_protocol.json')
OLD = {arm: {fold: study.price.ROOT / ('path' if arm=='price' else arm) / fold
             for fold in ['2025h1','2025h2']} for arm in ['control','price']}


def checked():
    p = json.loads(PROTOCOL.read_text()); study.checked()
    assert p['master_protocol_sha256'] == sha(study.PROTOCOL)
    assert p['old_models'] == {a:{f:str(r) for f,r in folds.items()} for a,folds in OLD.items()}
    for file,digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    return p


def prepare():
    study.checked(); result = model.protocols()
    pre = json.loads((study.ROOT / 'prefit_lookup_verification.json').read_text())
    assert pre['passed'] and pre['all_six_protocols_before_fitting']
    assert not any(x['lookup']['matches'] for x in pre['records'] if x['arm']=='flow')
    return dict(**result, only_two_flow_models_to_fit=True, four_cached_parents_require_exact_reuse=True)


def reuse(arm, fold):
    checked(); p = model.setup(arm,fold); root = base.ROOT; parent = OLD[arm][fold]
    assert not (root / 'model_report.json').exists() and not (root / 'scores.parquet').exists()
    fv = json.loads((study.INPUTS / 'feature_verification.json').read_text())
    assert fv['passed'] and fv['effective_input_intersection_unchanged']
    columns = [*study.META, *study.ARMS[arm]]
    fresh_inputs = pd.read_parquet(study.INPUTS / 'features.parquet', columns=columns)
    parent_inputs = pd.read_parquet(study.price.INPUTS / 'features.parquet', columns=columns,
                                    filters=[('date','>=','2024-01-01')])
    pd.testing.assert_frame_equal(fresh_inputs,parent_inputs,check_exact=True)
    pm = json.loads((parent / 'model_report.json').read_text())
    mv = json.loads((parent / 'model_verification.json').read_text())
    sr = json.loads((parent / 'score_report.json').read_text())
    sv = json.loads((parent / 'score_verification.json').read_text())
    assert mv['passed'] and mv['model_report_sha256'] == sha(parent / 'model_report.json')
    assert sv['passed'] and sv['score_report_sha256'] == sha(parent / 'score_report.json')
    assert sr['model_report_sha256'] == sha(parent / 'model_report.json')
    assert sr['scores_sha256'] == sha(parent / 'scores.parquet')
    assert pm['feature_report_sha256'] == sr['feature_report_sha256'] == sha(study.price.INPUTS / 'feature_report.json')
    assert pm['label_report_sha256'] == sha(study.INPUTS / 'full_label_report.json')
    assert pm['feature_names'] == p['feature_names'] and pm['variant'] == 'relative'
    assert pm['training_start'] == p['training_start'] and pm['training_end'] == p['training_end']
    assert all(pm['parameters'][k] == v for k,v in p['parameters'].items())
    # Independently reconstruct the old training target from all known labels,
    # then intersect with old inputs. Keep day weights after that intersection.
    fresh = relative.training('relative'); c = base.conn()
    c.register('parent_features',parent_inputs)
    fields = ','.join('f.'+n for n in study.ARMS[arm])
    original = c.sql(f'''WITH labels AS(SELECT date,code,next_date,opportunity15,
        opportunity15-avg(opportunity15) OVER(PARTITION BY date) AS target
        FROM read_parquet('{study.price.INPUTS}/full_labels.parquet')
        WHERE known15 AND date>='{p['training_start']}' AND next_date<'{p['training_end']}')
        SELECT date,code,next_date,opportunity15,target,1./count(*) OVER(PARTITION BY date) AS w,{fields}
        FROM parent_features f JOIN labels USING(date,code) WHERE formula_input_valid ORDER BY date,code''').df(); c.close()
    names = ['date','code','next_date','opportunity15',*study.ARMS[arm]]
    pd.testing.assert_frame_equal(fresh[names],original[names],check_exact=True,check_dtype=False)
    np.testing.assert_allclose(fresh.target,original.target,rtol=0,atol=2e-12)
    np.testing.assert_array_equal(1/fresh.groupby('date').code.transform('size'),original.w)
    assert len(original) == p['expected_training_rows'] == pm['rows']
    assert original.date.nunique() == p['expected_training_days'] == pm['days']
    assert original.next_date.max() == p['expected_last_observation'] == pm['last_observation'] < p['evaluation_start']
    report = pm.copy()
    report.update(protocol_sha256=sha(base.PROTOCOL),feature_report_sha256=sha(study.INPUTS / 'feature_report.json'),
        label_report_sha256=sha(study.INPUTS / 'full_label_report.json'),reused_source_root=str(parent),
        reused_model_report_sha256=sha(parent / 'model_report.json'),reuse_protocol_sha256=sha(PROTOCOL),no_model_fit_performed=True)
    root.mkdir(parents=True,exist_ok=True); save_json(root / 'model_report.json',report)
    for field in ['feature_names','parameters','trees','bias','learning_rate','thresholds','rows','days','last_observation','training_start','training_end']:
        assert report[field] == pm[field]
    scores = pd.read_parquet(parent / 'scores.parquet',filters=[('date','>=','2024-01-01')])
    pd.testing.assert_frame_equal(scores[study.META],fresh_inputs[study.META],check_exact=True)
    assert scores.loc[scores.formula_input_valid,'score'].notna().all()
    scores.to_parquet(root / 'scores.parquet',index=False,compression='zstd')
    pd.testing.assert_frame_equal(pd.read_parquet(root / 'scores.parquet'),scores,check_exact=True)
    sr.update(protocol_sha256=sha(base.PROTOCOL),model_report_sha256=sha(root / 'model_report.json'),
        feature_report_sha256=sha(study.INPUTS / 'feature_report.json'),scores_sha256=sha(root / 'scores.parquet'),
        rows=len(scores),valid=int(scores.formula_input_valid.sum()),reused_score_report_sha256=sha(parent / 'score_report.json'),
        no_prediction_performed=True,only_date_projection_of_verified_parent_scores=True)
    save_json(root / 'score_report.json',sr)
    mv.update(model_report_sha256=sha(root / 'model_report.json'),reused_model_verification_sha256=sha(parent / 'model_verification.json'),
        all_targets_integer_inputs_day_weights_residual_means_and_variances_rebuilt=False,
        original_arithmetic_verification_reused_after_all_training_inputs_labels_targets_weights_and_models_equal=True)
    save_json(root / 'model_verification.json',mv)
    old_rows = sv['rows']; old_checks = sv['threshold_flag_checks']
    sv.update(score_report_sha256=sha(root / 'score_report.json'),rows=len(scores),valid=int(scores.formula_input_valid.sum()),
        original_threshold_flag_checks=old_checks,threshold_flag_checks=0,original_full_score_rows=old_rows,
        reused_score_verification_sha256=sha(parent / 'score_verification.json'),
        all_integer_encodings_tree_scores_and_training_quantiles_rebuilt=False,
        original_arithmetic_verification_reused_after_all_inputs_models_and_weights_equal=True,
        all_projected_score_metadata_values_and_flags_identical=True,no_prediction_performed=True)
    save_json(root / 'score_verification.json',sv)
    receipt = dict(passed=True,arm=arm,fold=fold,reuse_protocol_sha256=sha(PROTOCOL),source_root=str(parent),
        original_model_report_sha256=sha(parent / 'model_report.json'),original_score_report_sha256=sha(parent / 'score_report.json'),
        original_scores_sha256=sha(parent / 'scores.parquet'),model_report_sha256=sha(root / 'model_report.json'),
        score_report_sha256=sha(root / 'score_report.json'),projected_rows=len(scores),training_rows=len(original),
        all_raw_training_keys_inputs_labels_targets_and_day_weights_equal=True,
        all_parent_equations_parameters_thresholds_scores_and_projected_metadata_identical=True,
        no_model_fit_performed=True,no_prediction_performed=True,new_2026_prices_read=False,no_exit_rules=True)
    save_json(root / 'parent_reuse_verification.json',receipt)
    return {k:v for k,v in receipt.items() if not k.endswith('sha256')}


def complete():
    checked(); receipts = {}; records = []
    for arm,folds in OLD.items():
        for fold,parent in folds.items():
            root = study.ROOT / arm / fold
            proof = json.loads((root / 'parent_reuse_verification.json').read_text())
            assert proof['passed'] and proof['reuse_protocol_sha256'] == sha(PROTOCOL)
            for kind in ['model','score']:
                v = json.loads((root / (kind+'_verification.json')).read_text())
                assert v['passed'] and v[kind+'_report_sha256'] == sha(root / (kind+'_report.json')) == proof[kind+'_report_sha256']
            for file in ['parent_reuse_verification.json','model_report.json','model_verification.json','score_report.json','score_verification.json','scores.parquet']:
                receipts[str(root / file)] = sha(root / file)
            records.append(dict(arm=arm,fold=fold,parent=str(parent)))
    out = dict(passed=True,reuse_protocol_sha256=sha(PROTOCOL),source_hashes=receipts,records=records,
        all_four_parent_models_and_projected_scores_reused=True,only_two_flow_models_are_new_fits=True,
        no_parent_prediction_or_model_fitting=True,new_2026_prices_read=False,no_exit_rules=True)
    save_json(study.ROOT / 'parent_reuse_verification.json',out)
    return dict(parent_reuse_sha256=sha(study.ROOT / 'parent_reuse_verification.json'))


def checked_complete():
    checked(); r = json.loads((study.ROOT / 'parent_reuse_verification.json').read_text())
    assert r['passed'] and r['reuse_protocol_sha256'] == sha(PROTOCOL)
    for file,digest in r['source_hashes'].items():
        assert sha(Path(file)) == digest
    return r


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('stage',choices=['prepare','reuse','complete'])
    parser.add_argument('--arm',choices=list(OLD)); parser.add_argument('--fold',choices=['2025h1','2025h2'])
    a = parser.parse_args()
    result = reuse(a.arm,a.fold) if a.stage=='reuse' else globals()[a.stage]()
    print(json.dumps(result,ensure_ascii=False,indent=2))
