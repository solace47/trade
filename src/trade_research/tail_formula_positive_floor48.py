"""Use a natural cost-zero morning price floor as a training-only event.

The original economic outcomes and the linear fitting/scoring math are reused.
Fresh processes keep the adapter's module configuration local to each stage.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_probability_linear48 as engine
from . import tail_formula_additive as base
from .corporate_cash import save_json, sha

STEM = 'tail_formula_positive_floor48'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
SOURCE = engine.INPUTS
INPUTS = ROOT / 'inputs'
ARMS, META, HEADER, PARAMETERS = engine.ARMS, engine.META, engine.HEADER, engine.PARAMETERS
TARGET = 'original_three_active_positive_minutes_and_observed_active_minute_low_cost_floor_above_zero'
_CHECKED_MODEL = engine.checked_model
native_core = engine.native_core


def master():
    p = json.loads(PROTOCOL.read_text())
    assert p['training_event'] == TARGET and p['floor_threshold'] == 0
    assert p['parameters'] == PARAMETERS and p['arms'] == ARMS and p['threshold'] == .995
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    return p


def utility(f):
    # An unknown floor cannot turn an original positive opportunity into a
    # known utility. A known original zero makes this AND event false already.
    assert np.isfinite(f.loc[f.known15 & f.opportunity15.eq(1), 'adverse_return15']).all()
    return (f.opportunity15.eq(1) & f.adverse_return15.gt(0)).astype(float).where(f.known15)


def prepare():
    master(); assert not (INPUTS / 'full_label_report.json').exists()
    INPUTS.mkdir(parents=True, exist_ok=True)
    for name in ['features.parquet', 'feature_report.json', 'feature_verification.json', 'native_input_verification.json']:
        (INPUTS / name).symlink_to((SOURCE / name).resolve())
    f = pd.read_parquet(SOURCE / 'full_labels.parquet'); out = f.copy()
    out['opportunity15'] = utility(f)
    out.to_parquet(INPUTS / 'full_labels.parquet', index=False, compression='zstd')
    save_json(INPUTS / 'full_label_report.json', dict(protocol_sha256=sha(PROTOCOL),
        labels_sha256=sha(INPUTS / 'full_labels.parquet'), rows=len(out), training_event=TARGET,
        source_label_report_sha256=sha(SOURCE / 'full_label_report.json'),
        source_label_verification_sha256=sha(SOURCE / 'full_label_verification.json'),
        source_labels_sha256=sha(SOURCE / 'full_labels.parquet'),
        opportunity15_is_training_utility_alias_only=True,
        original_economic_outcomes_and_all_known_unknown_flags_unchanged=True,
        no_new_raw_price_extraction=True, new_2026_prices_read=False, no_exit_rules=True))
    return dict(rows=len(out), label_report_sha256=sha(INPUTS / 'full_label_report.json'))


def verify():
    master(); r = json.loads((INPUTS / 'full_label_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['training_event'] == TARGET
    assert r['labels_sha256'] == sha(INPUTS / 'full_labels.parquet')
    f = pd.read_parquet(SOURCE / 'full_labels.parquet'); got = pd.read_parquet(INPUTS / 'full_labels.parquet')
    other = list(f.columns.drop('opportunity15'))
    pd.testing.assert_frame_equal(got[other], f[other], check_exact=True)
    pd.testing.assert_series_equal(got.opportunity15, utility(f), check_names=False, check_exact=True)
    c = base.conn()
    expected = c.sql(f'''SELECT * REPLACE(CASE WHEN known15 THEN
        CAST(opportunity15=1 AND coalesce(adverse_return15>0,false) AS DOUBLE)
        ELSE NULL END AS opportunity15) FROM read_parquet('{SOURCE}/full_labels.parquet')
        ORDER BY date,code''').df(); c.close()
    pd.testing.assert_frame_equal(got, expected, check_exact=True)
    assert len(got) == r['rows'] == 1815129
    assert got.opportunity15.notna().equals(f.known15)
    assert got.loc[f.known15, 'opportunity15'].le(f.loc[f.known15, 'opportunity15']).all()
    for name in ['features.parquet', 'feature_report.json', 'feature_verification.json', 'native_input_verification.json']:
        assert sha(INPUTS / name) == sha(SOURCE / name)
    proof = dict(passed=True, label_report_sha256=sha(INPUTS / 'full_label_report.json'), rows=len(got),
        all_original_keys_metadata_known_unknown_and_adverse_values_exact=True,
        full_training_utility_alias_independently_sql_verified=True,
        no_original_positive_case_has_an_undefined_floor=True,
        all_old_features_and_native_input_proofs_reused=True,
        economic_evaluation_will_use_original_before1000_labels=True,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(INPUTS / 'full_label_verification.json', proof)
    return proof


def configure():
    engine.STEM, engine.ROOT, engine.PROTOCOL, engine.INPUTS = STEM, ROOT, PROTOCOL, INPUTS
    engine.checked_model = checked_model


def checked():
    master(); configure(); p = engine.checked()
    r = json.loads((INPUTS / 'full_label_report.json').read_text())
    assert r['training_event'] == TARGET and r['source_labels_sha256'] == sha(SOURCE / 'full_labels.parquet')
    return p


def setup(arm, fold):
    checked(); return engine.setup(arm, fold)


def model():
    result = engine.model()
    file = base.ROOT / 'model_report.json'; m = json.loads(file.read_text())
    m['training_event'] = TARGET
    m['training_utility_verification_sha256'] = sha(INPUTS / 'full_label_verification.json')
    save_json(file, m)
    return result


def checked_model():
    p, m = _CHECKED_MODEL()
    assert m['training_event'] == TARGET
    assert m['training_utility_verification_sha256'] == sha(INPUTS / 'full_label_verification.json')
    return p, m


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'verify', 'model', 'verify_model', 'scores', 'verify_scores'])
    p.add_argument('--arm', choices=list(ARMS))
    p.add_argument('--fold', choices=['2024h1', '2024h2', '2025h1', '2025h2'])
    a = p.parse_args()
    if a.stage in ['prepare', 'verify']:
        result = globals()[a.stage]()
    else:
        assert a.arm and a.fold; setup(a.arm, a.fold)
        result = model() if a.stage == 'model' else getattr(engine, a.stage)()
    print(json.dumps(result, ensure_ascii=False, indent=2))
