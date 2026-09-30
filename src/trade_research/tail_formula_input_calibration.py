"""Fix a score cutoff from post-training visible inputs, without their outcomes."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_morning_range as prior
from .corporate_cash import save_json, sha

STEM = 'tail_formula_input_calibration'
ROOT = Path('data/research') / STEM
INPUTS = prior.INPUTS
PROTOCOL = Path('config') / (STEM + '_input_protocol.json')
INTENT = Path('config') / (STEM + '_intent.json')
META = prior.META
HEADER = prior.HEADER
CORE_GATE = 'AMREADY AND RTREADY'
ARMS = {'control': prior.EXPRESSIONS, 'calibrated': prior.EXPRESSIONS}
QUANTILE = .995


def checked():
    prior.checked()
    p = json.loads(PROTOCOL.read_text())
    assert p['intent_sha256'] == sha(INTENT)
    assert p['arms'] == ARMS and p['native_header'] == HEADER
    assert p['quantile'] == QUANTILE and p['expected_keys'] == 1258085
    assert p['expected_valid'] == 1117397 and not p['new_2026_prices_allowed']
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest, file
    gate = json.loads(Path(p['conditional_gate']).read_text())
    assert gate['passed'] and not gate['supports_2024_extension']
    lookup = json.loads((ROOT / 'metadata_lookup_verification.json').read_text())
    assert lookup['passed'] and lookup['intent_sha256'] == sha(INTENT)
    for file, digest in lookup['source_hashes'].items():
        assert sha(Path(file)) == digest, file
    for kind in ['feature', 'native_input']:
        v = json.loads((INPUTS / (kind + '_verification.json')).read_text())
        assert v['passed'] and v['feature_report_sha256'] == sha(INPUTS / 'feature_report.json')
    return p


def visible_score_cut(frame, start, end):
    """The calibration population depends only on dates and valid visible inputs."""
    f = frame.loc[frame.date.ge(start) & frame.date.lt(end),
                  ['date', 'code', 'formula_input_valid', 'score']].copy()
    assert not f.duplicated(['date', 'code']).any()
    valid = f.loc[f.formula_input_valid]
    assert len(valid) > 0 and np.isfinite(valid.score).all()
    return float(np.quantile(valid.score.to_numpy(), QUANTILE, method='linear')), f


def calibration_paths(fold):
    root = ROOT / 'models' / fold
    return root, root / 'calibration_report.json', root / 'calibration_verification.json'


def calibrate(fold):
    p = checked()
    spec = p['folds'][fold]
    root, report, _ = calibration_paths(fold)
    assert not report.exists()
    model = json.loads((root / 'model_report.json').read_text())
    proof = json.loads((root / 'score_verification.json').read_text())
    assert model['training_end'] == spec['training_end'] == spec['calibration_start']
    assert model['last_observation'] < spec['calibration_start']
    assert spec['calibration_end'] == spec['evaluation_start']
    assert proof['passed'] and proof['score_report_sha256'] == sha(root / 'score_report.json')
    frame = pd.read_parquet(root / 'scores.parquet',
        columns=['date', 'code', 'formula_input_valid', 'score'],
        filters=[('date', '>=', spec['calibration_start']), ('date', '<', spec['calibration_end'])])
    cut, frame = visible_score_cut(frame, spec['calibration_start'], spec['calibration_end'])
    counts = next(x for x in json.loads((ROOT / 'metadata_lookup_verification.json').read_text())['records'] if x['fold'] == fold)
    assert len(frame) == counts['calibration_keys']
    assert frame.formula_input_valid.sum() == counts['calibration_valid']
    assert frame.date.nunique() == counts['calibration_days']
    r = dict(protocol_sha256=sha(PROTOCOL), fold=fold,
        model_report_sha256=sha(root / 'model_report.json'),
        score_report_sha256=sha(root / 'score_report.json'), scores_sha256=sha(root / 'scores.parquet'),
        calibration_start=spec['calibration_start'], calibration_end=spec['calibration_end'],
        rows=len(frame), valid=int(frame.formula_input_valid.sum()), days=frame.date.nunique(),
        quantile=QUANTILE, quantile_method='linear_unweighted_visible_rows', threshold=cut,
        training_control_threshold=model['thresholds'][3]['threshold'],
        no_calibration_label_values_read=True, no_outcome_based_threshold_choice=True,
        no_new_group_evaluation=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(report, r)
    return r


def verify_calibration(fold):
    p = checked()
    spec = p['folds'][fold]
    root, report, verification = calibration_paths(fold)
    r = json.loads(report.read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL)
    assert r['scores_sha256'] == sha(root / 'scores.parquet')
    c = base.conn()
    where = f"date>='{spec['calibration_start']}' AND date<'{spec['calibration_end']}'"
    expected = c.sql(f"SELECT date,code,formula_input_valid FROM read_parquet('{INPUTS}/features.parquet') WHERE {where} ORDER BY date,code").df()
    actual = c.sql(f"SELECT date,code,formula_input_valid FROM read_parquet('{root}/scores.parquet') WHERE {where} ORDER BY date,code").df()
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    values = c.sql(f"SELECT count(*) AS valid,quantile_cont(score,{QUANTILE}) AS threshold,count(*) FILTER(WHERE score IS NULL OR NOT isfinite(score)) AS bad FROM read_parquet('{root}/scores.parquet') WHERE {where} AND formula_input_valid").df().iloc[0]
    c.close()
    assert values.valid == r['valid'] and values.bad == 0
    np.testing.assert_allclose(values.threshold, r['threshold'], rtol=0, atol=2e-12)
    v = dict(passed=True, calibration_report_sha256=sha(report), fold=fold,
        all_complete_calibration_keys_and_visible_validity_sql_verified=True,
        fixed_linear_quantile_independently_sql_rebuilt=True,
        no_known_label_or_future_execution_filter=True, no_calibration_label_values_read=True,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(verification, v)
    return v
