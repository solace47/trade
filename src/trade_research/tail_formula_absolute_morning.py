"""Exact original morning inputs for the absolute binary-opportunity study."""
import json
from pathlib import Path

from . import tail_formula_morning_range as prior
from .corporate_cash import sha

STEM = 'tail_formula_absolute_morning'
ROOT = Path('data/research') / STEM
INPUTS = prior.INPUTS
PROTOCOL = Path('config') / (STEM + '_input_protocol.json')
INTENT = Path('config') / (STEM + '_intent.json')
META = prior.META
ARMS = {'control': prior.EXPRESSIONS, 'absolute': prior.EXPRESSIONS}
HEADER = prior.HEADER
CORE_GATE = 'AMREADY'


def checked():
    prior.checked()
    p = json.loads(PROTOCOL.read_text())
    assert p['intent_sha256'] == sha(INTENT)
    assert p['arms'] == ARMS and p['native_header'] == HEADER
    assert p['shared_input_root'] == str(INPUTS)
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest, file
    complete = json.loads(Path(p['conditional_completion']).read_text())
    gate = json.loads(Path(p['conditional_gate']).read_text())
    assert complete['passed'] and gate['passed'] and not gate['supports_2024_extension']
    fr = json.loads((INPUTS / 'feature_report.json').read_text())
    assert fr['features_sha256'] == sha(INPUTS / 'features.parquet')
    fv = json.loads((INPUTS / 'feature_verification.json').read_text())
    nv = json.loads((INPUTS / 'native_input_verification.json').read_text())
    assert fv['passed'] and nv['passed']
    assert fv['rows'] == p['expected_keys'] == 1258085
    assert fv['valid'] == p['expected_valid'] == 1117397
    assert fv['effective_input_intersection_unchanged']
    assert fv['feature_report_sha256'] == nv['feature_report_sha256'] == sha(INPUTS / 'feature_report.json')
    assert p['no_new_raw_extraction'] and not p['new_2026_prices_allowed']
    return p
