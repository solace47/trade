"""Train a confirmed-success certificate, preserving unknown event intervals."""
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_morning_range as prior
from . import tail_formula_volume_memory as original_source
from .corporate_cash import save_json, sha

STEM = 'tail_formula_confirmed_success'
ROOT = Path('data/research') / STEM
INPUTS = ROOT / 'inputs'
PROTOCOL = Path('config') / (STEM + '_protocol.json')
LABELS = prior.INPUTS
META, EXPRESSIONS, HEADER = prior.META, prior.EXPRESSIONS, prior.HEADER
OBJECTIVE = 'mature_confirmed_success_certificate_relative'


def event_interval(known, no_trade, opportunity):
    """Unknown outcomes remain [0,1]; zero lower evidence is not an outcome."""
    known, no_trade = np.asarray(known, bool), np.asarray(no_trade, bool)
    opportunity = np.asarray(opportunity, float)
    assert known.shape == no_trade.shape == opportunity.shape
    assert not (known & no_trade).any()
    assert np.isfinite(opportunity[known]).all() and np.isin(opportunity[known], [0, 1]).all()
    lower = np.where(known, opportunity, 0.)
    upper = np.where(known, opportunity, np.where(no_trade, 0., 1.))
    return lower, upper


def checked():
    p = json.loads(PROTOCOL.read_text())
    assert p['objective_identifier'] == OBJECTIVE and p['feature_names'] == list(EXPRESSIONS)
    assert p['maximum_new_fits'] == 4 and p['threshold'] == .995 and not p['new_2026_prices_allowed']
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest, file
    v = json.loads((ROOT / 'audit_verification.json').read_text())
    assert v['passed'] and v['supports_complete_model_protocol']
    assert not v['requires_explicit_delayed_and_missing_date_policy']
    committed = subprocess.run(['git', 'show', 'HEAD:docs/selection-formula.md'], text=True,
                               capture_output=True, check=True).stdout
    assert sha(PROTOCOL) in committed, 'Complete protocol must precede numeric target projection'
    return p


def target_data(spec):
    c = base.conn()
    l = c.sql(f'''SELECT date,code,next_date,known15,known_no_trade,opportunity15
        FROM read_parquet('{LABELS}/full_labels.parquet')
        WHERE date>='{spec['training_start']}' AND date<'{spec['training_end']}'
          AND next_date>date AND next_date<'{spec['training_end']}' ORDER BY date,code''').df()
    c.close()
    assert not l.duplicated(['date', 'code']).any()
    l['lower_certificate'], l['upper_event'] = event_interval(l.known15, l.known_no_trade, l.opportunity15)
    # Use the complete mature label pool before intersection with visible input quality.
    l['date_lower_mean'] = l.groupby('date').lower_certificate.transform('mean')
    l['target'] = l.lower_certificate - l.date_lower_mean
    return l


def training(spec, independent=False):
    f = base.feature_inputs()
    if not independent:
        l = target_data(spec)
        t = f.loc[f.formula_input_valid].merge(l, on=['date', 'code'], validate='one_to_one')
        t['w'] = 1 / t.groupby('date').code.transform('size')
        return t.sort_values(['date', 'code']).reset_index(drop=True)
    c = base.conn(); c.register('features', f)
    encoded = ','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::INT AS {n}' for n in EXPRESSIONS)
    d = c.sql(f'''WITH mature AS (
        SELECT date,code,next_date,known15,known_no_trade,opportunity15,
          CASE WHEN known15 THEN opportunity15 ELSE 0. END AS lower_certificate,
          CASE WHEN known15 THEN opportunity15 WHEN known_no_trade THEN 0. ELSE 1. END AS upper_event
        FROM read_parquet('{LABELS}/full_labels.parquet')
        WHERE date>='{spec['training_start']}' AND date<'{spec['training_end']}'
          AND next_date>date AND next_date<'{spec['training_end']}'), targets AS (
        SELECT *,avg(lower_certificate) OVER(PARTITION BY date) AS date_lower_mean FROM mature)
        SELECT f.date,f.code,next_date,known15,known_no_trade,opportunity15,lower_certificate,upper_event,
          date_lower_mean,lower_certificate-date_lower_mean AS target,
          1./count(*) OVER(PARTITION BY f.date) AS w,{encoded}
        FROM features f JOIN targets USING(date,code) WHERE formula_input_valid ORDER BY date,code''').df()
    c.close()
    return d


def verify_training(fold, spec, save=True):
    t, d = training(spec), training(spec, True)
    common = ['date', 'code', 'next_date', 'known15', 'known_no_trade', 'opportunity15',
              'lower_certificate', 'upper_event', 'date_lower_mean', 'target', 'w']
    pd.testing.assert_frame_equal(t[common], d[common], check_exact=True, check_dtype=False)
    np.testing.assert_array_equal(base.encode(t), d[list(EXPRESSIONS)].to_numpy('int32'))
    audited = pd.read_parquet(ROOT / (fold + '_mature_metadata.parquet'))
    pd.testing.assert_frame_equal(t[[*META, 'next_date', 'known15', 'known_no_trade']],
                                 audited[[*META, 'next_date', 'known15', 'known_no_trade']], check_exact=True)
    assert len(t) == spec['expected_training_rows'] and t.date.nunique() == spec['expected_training_days']
    assert t.next_date.max() == spec['expected_last_observation'] < spec['evaluation_start']
    unknown = ~t.known15 & ~t.known_no_trade
    assert len(t.loc[unknown]) == spec['expected_unknown_rows']
    assert t.loc[unknown, 'lower_certificate'].eq(0).all() and t.loc[unknown, 'upper_event'].eq(1).all()
    np.testing.assert_allclose(t.groupby('date').w.sum(), 1., rtol=0, atol=2e-12)
    proof = dict(passed=True, objective_identifier=OBJECTIVE, rows=len(t), days=t.date.nunique(),
                 known_rows=int(t.known15.sum()), confirmed_no_trade_rows=int(t.known_no_trade.sum()),
                 unknown_rows=int(unknown.sum()), last_observation=t.next_date.max(),
                 all_targets_intervals_integer_inputs_date_weights_independent_sql_equal=True,
                 complete_original_mature_metadata_exactly_equal=True,
                 unknown_actual_outcomes_and_returns_not_imputed=True,
                 full_date_baseline_before_valid_input_intersection=True,
                 original_label_report_sha256=sha(LABELS / 'full_label_report.json'),
                 new_2026_prices_read=False, no_exit_rules=True)
    if save:
        path = ROOT / 'models' / fold / 'training_verification.json'
        assert not path.exists(); save_json(path, proof)
    return proof


def prepare():
    p = checked(); assert not (INPUTS / 'feature_report.json').exists()
    INPUTS.mkdir(parents=True, exist_ok=True)
    f = original_source.original()
    f.to_parquet(INPUTS / 'features.parquet', index=False, compression='zstd')
    # Rebuild the full frame from the immutable sources with an independent SQL union.
    c = base.conn(); columns = ','.join([*META, *EXPRESSIONS])
    a, b = p['feature_files']
    independent = c.sql(f'''SELECT {columns} FROM read_parquet('{a}') WHERE date<'2024-01-01'
        UNION ALL SELECT {columns} FROM read_parquet('{b}') ORDER BY date,code''').df(); c.close()
    pd.testing.assert_frame_equal(f, independent, check_exact=True)
    assert len(f) == 1815129 and int(f.formula_input_valid.sum()) == 1602413
    report = dict(protocol_sha256=sha(PROTOCOL), features_sha256=sha(INPUTS / 'features.parquet'),
                  rows=len(f), valid=int(f.formula_input_valid.sum()), source_hashes=p['source_hashes'],
                  original_50_values_metadata_and_validity_unchanged=True,
                  no_new_price_inputs=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(INPUTS / 'feature_report.json', report)
    save_json(INPUTS / 'feature_verification.json', dict(passed=True,
        feature_report_sha256=sha(INPUTS / 'feature_report.json'),
        all_original_keys_50_values_and_validity_independent_sql_exact=True,
        shared_2024_sources_exact_before_union=True, new_2026_prices_read=False))
    base.FEATURES, base.EXPRESSIONS = INPUTS, EXPRESSIONS
    records = []
    for fold, spec in p['folds'].items():
        records.append(dict(fold=fold, **verify_training(fold, spec)))
    save_json(ROOT / 'target_verification.json', dict(passed=True, protocol_sha256=sha(PROTOCOL),
        feature_verification_sha256=sha(INPUTS / 'feature_verification.json'), records=records,
        original_economic_labels_unchanged=True, zero_is_lower_evidence_not_unknown_actual_event=True,
        no_new_model_fit_or_new_selection_evaluation=True, new_2026_prices_read=False, no_exit_rules=True))
    return dict(feature_report_sha256=sha(INPUTS / 'feature_report.json'),
                feature_verification_sha256=sha(INPUTS / 'feature_verification.json'),
                target_verification_sha256=sha(ROOT / 'target_verification.json'), records=records)
