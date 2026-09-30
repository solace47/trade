"""Reuse already verified full 2023–2025 control scores, with no prediction.

The previous daily-regression controls already extended canonical models
to exactly this complete key set. New minute inputs do not affect them if
all 48 values, metadata and validity flags are equal. Reuse the equation
proofs too, while clearly recording that their arithmetic is not rerun.
"""
import json
from pathlib import Path

import pandas as pd

import reuse_tail_formula_tail_regression_control as historical
from trade_research import tail_formula_additive as base
from trade_research import tail_formula_tail_regression as study
from trade_research import tail_formula_tail_regression_model as model
from trade_research.corporate_cash import save_json, sha

PREVIOUS = Path('data/research/tail_formula_daily_regression')


def run():
    historical.common.checked()
    columns = [*study.META,*study.ARMS['control']]
    new = pd.read_parquet(study.INPUTS/'features.parquet',columns=columns)
    old = pd.read_parquet(PREVIOUS/'inputs/features.parquet',columns=columns)
    pd.testing.assert_frame_equal(new,old,check_exact=True)
    fv = json.loads((PREVIOUS/'inputs/feature_verification.json').read_text())
    assert fv['passed'] and fv['feature_report_sha256'] == sha(PREVIOUS/'inputs/feature_report.json')
    fr = json.loads((PREVIOUS/'inputs/feature_report.json').read_text())
    assert fr['features_sha256'] == sha(PREVIOUS/'inputs/features.parquet')
    receipts = []
    fields = ['feature_names','parameters','trees','bias','learning_rate','thresholds','rows','days',
        'last_observation','training_start','training_end','variant']
    for fold in historical.OLD:
        model.setup('control',fold);root = base.ROOT;prior = PREVIOUS/'control'/fold
        assert not (root/'scores.parquet').exists() and not (root/'model_verification.json').exists()
        m = json.loads((root/'model_report.json').read_text())
        pm = json.loads((prior/'model_report.json').read_text())
        assert m['no_model_fit_performed'] and pm['no_model_fit_performed']
        assert {k:m[k] for k in fields} == {k:pm[k] for k in fields}
        mv = json.loads((prior/'model_verification.json').read_text())
        sv = json.loads((prior/'score_verification.json').read_text())
        sr = json.loads((prior/'score_report.json').read_text())
        assert mv['passed'] and mv['model_report_sha256'] == sha(prior/'model_report.json')
        assert sv['passed'] and sv['score_report_sha256'] == sha(prior/'score_report.json')
        assert sr['model_report_sha256'] == sha(prior/'model_report.json')
        assert sr['scores_sha256'] == sha(prior/'scores.parquet')
        scores = pd.read_parquet(prior/'scores.parquet',columns=study.META)
        pd.testing.assert_frame_equal(scores,new[study.META],check_exact=True)
        (root/'scores.parquet').symlink_to((prior/'scores.parquet').resolve())
        sr.update(protocol_sha256=sha(base.PROTOCOL),model_report_sha256=sha(root/'model_report.json'),
            feature_report_sha256=sha(study.INPUTS/'feature_report.json'),
            reused_score_report_sha256=sha(prior/'score_report.json'),no_prediction_performed=True)
        save_json(root/'score_report.json',sr)
        mv.update(model_report_sha256=sha(root/'model_report.json'),reused_model_verification_sha256=sha(prior/'model_verification.json'),
            all_targets_integer_inputs_day_weights_residual_means_and_variances_rebuilt=False,
            original_arithmetic_verification_reused_after_full_inputs_models_and_weights_equal=True)
        save_json(root/'model_verification.json',mv)
        sv.update(score_report_sha256=sha(root/'score_report.json'),reused_score_verification_sha256=sha(prior/'score_verification.json'),
            all_integer_encodings_tree_scores_and_training_quantiles_rebuilt=False,
            original_arithmetic_verification_reused_after_full_inputs_models_and_weights_equal=True,
            all_1815129_metadata_and_input_validity_flags_equal=True,no_prediction_performed=True)
        save_json(root/'score_verification.json',sv)
        receipt = historical.common.scores(fold)
        receipt.update(entire_score_table_reused_source=str(prior),entire_score_table_source_sha256=sha(prior/'scores.parquet'),
            no_prediction_performed=True,all_1815129_keys_and_48_values_and_validity_identical=True)
        save_json(root/'score_reuse_verification.json',receipt)
        receipts.append(dict(fold=fold,source=str(prior),rows=len(scores),
            score_reuse_verification_sha256=sha(root/'score_reuse_verification.json'),
            model_verification_sha256=sha(root/'model_verification.json'),score_verification_sha256=sha(root/'score_verification.json')))
    proof = dict(passed=True,feature_report_sha256=sha(study.INPUTS/'feature_report.json'),
        previous_feature_report_sha256=sha(PREVIOUS/'inputs/feature_report.json'),records=receipts,
        all_1815129_full_inputs_metadata_and_validity_equal=True,
        all_four_control_equations_q995_scores_and_arithmetic_proofs_reused=True,
        no_prediction_performed=True,no_model_fit_performed=True,new_2026_prices_read=False,no_exit_rules=True)
    save_json(study.ROOT/'full_control_score_reuse_verification.json',proof)
    result = historical.common.complete()
    # Bind the transitive proof in the already checked shared-control receipt.
    path = study.ROOT/'control_reuse_verification.json';r = json.loads(path.read_text())
    r['source_hashes'][str(study.ROOT/'full_control_score_reuse_verification.json')] = sha(study.ROOT/'full_control_score_reuse_verification.json')
    r['all_control_score_reprediction_avoided'] = True;save_json(path,r)
    result['control_reuse_sha256'] = sha(path)
    result['all_four_cached_scores_reused'] = True
    return result


if __name__ == '__main__':
    print(json.dumps(run(),ensure_ascii=False,indent=2))
