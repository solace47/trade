"""Calibrate absolute scores on visible inputs of stocks excluded from each fit."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_input_calibration as calibration
from . import tail_formula_morning_range as prior
from .corporate_cash import save_json, sha

STEM = 'tail_formula_stock_holdout'
ROOT = Path('data/research') / STEM
INPUTS = prior.INPUTS
PROTOCOL = Path('config') / (STEM + '_input_protocol.json')
INTENT = Path('config') / (STEM + '_intent.json')
META = prior.META
HEADER = prior.HEADER
CORE_GATE = 'AMREADY AND RTREADY'
ARMS = {'control': prior.EXPRESSIONS, 'cross': prior.EXPRESSIONS}
LABEL_FIELDS = ['date', 'code', 'next_date', 'known15', 'opportunity15', 'known_no_trade']


def group_ids(codes):
    assert codes.str.fullmatch(r'(sh|sz)\.\d{6}').all()
    return codes.str[-1].astype(int).ge(5).astype('int32')


def model_root(fold, group):
    assert fold in ['2025h1', '2025h2'] and group in [0, 1]
    return ROOT / 'models' / fold / ('group' + str(group))


def checked():
    prior.checked()
    p = json.loads(PROTOCOL.read_text())
    assert p['intent_sha256'] == sha(INTENT) and p['arms'] == ARMS
    assert p['folds'] == json.loads(INTENT.read_text())['folds']
    assert p['expected_keys'] == 1258085 and p['expected_valid'] == 1117397
    assert p['quantile'] == .995 and not p['new_2026_prices_allowed']
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest, file
    gate = json.loads(Path(p['conditional_gate']).read_text())
    assert gate['passed'] and not gate['supports_2024_extension']
    metadata = json.loads((ROOT / 'metadata_lookup_verification.json').read_text())
    assert metadata['passed'] and metadata['intent_sha256'] == sha(INTENT)
    assert metadata['full_native_six_digit_last_digit_routing_sql_verified']
    for file, digest in metadata['source_hashes'].items():
        assert sha(Path(file)) == digest, file
    return p


def project_labels(fold, group):
    """Copy only this model's label scope, allowing unchanged model verifiers."""
    p = checked(); spec = p['folds'][fold]
    root = model_root(fold, group) / 'labels'; root.mkdir(parents=True, exist_ok=True)
    path = root / 'full_labels.parquet'
    assert not path.exists()
    digits = tuple(str(x) for x in (range(5) if group == 0 else range(5, 10)))
    c = base.conn()
    actual = c.execute('SELECT ' + ','.join(LABEL_FIELDS) + ''' FROM read_parquet(?)
        WHERE date>=? AND next_date<? AND right(code,1) IN (?,?,?,?,?) ORDER BY date,code''',
        [str(INPUTS / 'full_labels.parquet'), spec['training_start'], spec['training_end'], *digits]).df()
    actual.to_parquet(path, index=False, compression='zstd')
    # Different code predicate and direct parent query, before accepting values.
    expected = c.execute('SELECT ' + ','.join(LABEL_FIELDS) + ''' FROM read_parquet(?)
        WHERE date>=? AND next_date<? AND (cast(substr(code,9,1) AS INT)>=5)::INT=? ORDER BY date,code''',
        [str(INPUTS / 'full_labels.parquet'), spec['training_start'], spec['training_end'], group]).df()
    pd.testing.assert_frame_equal(pd.read_parquet(path), expected, check_exact=True)
    assert group_ids(actual.code).eq(group).all()
    assert actual.date.ge(spec['training_start']).all() and actual.next_date.lt(spec['training_end']).all()
    c.register('projected', actual)
    counts = c.execute('''SELECT count(*) AS rows,count(DISTINCT date) AS days,max(next_date) AS last_observation
        FROM projected l JOIN read_parquet(?) f USING(date,code) WHERE l.known15 AND f.formula_input_valid''',
        [str(INPUTS / 'features.parquet')]).df().iloc[0]
    c.close()
    metadata = next(r for r in json.loads((ROOT / 'metadata_lookup_verification.json').read_text())['records']
                    if r['fold'] == fold and r['group'] == group)
    assert counts.rows == metadata['training_rows'] and counts.days == metadata['training_days']
    assert counts.last_observation == metadata['last_observation']
    report = dict(protocol_sha256=sha(PROTOCOL), labels_sha256=sha(path), rows=len(actual),
        fields=LABEL_FIELDS, fold=fold, training_group=group,
        training_start=spec['training_start'], training_end=spec['training_end'],
        original_label_report_sha256=sha(INPUTS / 'full_label_report.json'),
        original_labels_sha256=sha(INPUTS / 'full_labels.parquet'),
        opposite_group_label_values_not_in_training_projection=True,
        no_evaluation_period_labels_in_projection=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(root / 'full_label_report.json', report)
    proof = dict(passed=True, label_report_sha256=sha(root / 'full_label_report.json'),
        all_projected_fields_exactly_equal_to_independent_parent_sql=True,
        original_valid_training_intersection_and_counts_verified=True,
        opposite_group_labels_excluded_from_model_target=True,
        no_new_group_evaluation=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(root / 'full_label_verification.json', proof)
    return dict(fold=fold, group=group, projected_rows=len(actual), training_rows=int(counts.rows))


def visible_group_cut(frame, start, end, training_group):
    assert training_group in [0, 1]
    opposite = frame.loc[group_ids(frame.code).eq(1 - training_group)]
    return calibration.visible_score_cut(opposite, start, end)


def calibrate(fold, group):
    p = checked(); spec = p['folds'][fold]; root = model_root(fold, group)
    assert not (root / 'calibration_report.json').exists()
    m = json.loads((root / 'model_report.json').read_text())
    v = json.loads((root / 'score_verification.json').read_text())
    assert v['passed'] and v['score_report_sha256'] == sha(root / 'score_report.json')
    assert m['last_observation'] < spec['training_end'] == spec['evaluation_start']
    frame = pd.read_parquet(root / 'scores.parquet',
        filters=[('date', '>=', spec['training_start']), ('date', '<', spec['training_end'])])
    cut, frame = visible_group_cut(frame, spec['training_start'], spec['training_end'], group)
    counts = next(r for r in json.loads((ROOT / 'metadata_lookup_verification.json').read_text())['records']
                  if r['fold'] == fold and r['group'] == group)
    assert len(frame) == counts['calibration_keys'] and frame.formula_input_valid.sum() == counts['calibration_valid']
    assert frame.date.nunique() == counts['calibration_days']
    report = dict(protocol_sha256=sha(PROTOCOL), fold=fold, training_group=group, calibration_group=1-group,
        model_report_sha256=sha(root / 'model_report.json'), score_report_sha256=sha(root / 'score_report.json'),
        scores_sha256=sha(root / 'scores.parquet'), calibration_start=spec['training_start'],
        calibration_end=spec['training_end'], rows=len(frame), valid=int(frame.formula_input_valid.sum()),
        days=frame.date.nunique(), quantile=.995, quantile_method='linear_unweighted_opposite_stock_visible_rows',
        threshold=cut, no_calibration_label_values_read=True, no_outcome_based_threshold_choice=True,
        same_dates_stock_holdout_is_not_chronological_holdout=True,
        no_new_group_evaluation=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(root / 'calibration_report.json', report)
    return report


def verify_calibration(fold, group):
    p = checked(); spec = p['folds'][fold]; root = model_root(fold, group)
    r = json.loads((root / 'calibration_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['scores_sha256'] == sha(root / 'scores.parquet')
    c = base.conn()
    predicate = 'date>=? AND date<? AND (cast(substr(code,9,1) AS INT)>=5)::INT=?'
    args = [spec['training_start'], spec['training_end'], 1-group]
    expected = c.execute('SELECT date,code,formula_input_valid FROM read_parquet(?) WHERE ' + predicate + ' ORDER BY date,code',
        [str(INPUTS / 'features.parquet'), *args]).df()
    actual = c.execute('SELECT date,code,formula_input_valid FROM read_parquet(?) WHERE ' + predicate + ' ORDER BY date,code',
        [str(root / 'scores.parquet'), *args]).df()
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    values = c.execute('''SELECT count(*) AS valid,quantile_cont(score,.995) AS threshold,
        count(*) FILTER(WHERE score IS NULL OR NOT isfinite(score)) AS bad FROM read_parquet(?) WHERE '''
        + predicate + ' AND formula_input_valid', [str(root / 'scores.parquet'), *args]).df().iloc[0]
    c.close()
    assert len(actual) == r['rows'] and actual.date.nunique() == r['days']
    assert values.valid == r['valid'] and values.bad == 0
    np.testing.assert_allclose(values.threshold, r['threshold'], rtol=0, atol=2e-12)
    proof = dict(passed=True, calibration_report_sha256=sha(root / 'calibration_report.json'),
        all_opposite_group_keys_validity_and_fixed_linear_quantile_independently_sql_rebuilt=True,
        no_known_label_or_future_execution_filter=True, no_calibration_label_values_read=True,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(root / 'calibration_verification.json', proof)
    return proof


def route_flags(frame, score0, score1, cuts, cross):
    g = group_ids(frame.code).to_numpy()
    use0 = g == (1 if cross else 0)
    selected = np.where(use0, np.asarray(score0) > cuts[0], np.asarray(score1) > cuts[1])
    return frame.formula_input_valid.to_numpy() & selected
