"""Reuse four stock-group models; require both fixed score gates for selection."""
import json
from pathlib import Path

from . import tail_formula_stock_relative as parent
from .corporate_cash import sha

STEM = 'tail_formula_stock_agreement'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_input_protocol.json')
INTENT = Path('config') / (STEM + '_intent.json')
prior = parent.prior
INPUTS = parent.INPUTS
META = parent.META
HEADER = parent.HEADER
CORE_GATE = parent.CORE_GATE
ARMS = {'control': prior.EXPRESSIONS, 'agreement': prior.EXPRESSIONS}


def checked():
    prior.checked()
    p = json.loads(PROTOCOL.read_text())
    assert p['intent_sha256'] == sha(INTENT) and p['arms'] == ARMS
    assert p['expected_keys'] == 1258085 and p['expected_valid'] == 1117397
    assert p['new_model_fits_allowed'] == p['new_score_predictions_allowed'] == 0
    assert not p['new_2026_prices_allowed']
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest, file
    gate = json.loads((parent.ROOT / 'stock_relative_gate.json').read_text())
    complete = json.loads((parent.ROOT / 'complete_results_manifest.json').read_text())
    joint = json.loads((parent.ROOT / 'joint_selection_freeze.json').read_text())
    assert gate['passed'] and not gate['supports_2024_extension'] and complete['passed'] and joint['passed']
    assert complete['joint_sha256'] == sha(parent.ROOT / 'joint_selection_freeze.json')
    assert len(joint['models']) == 4 and len(complete['comparisons']) == 6
    for file, digest in joint['source_hashes'].items():
        assert sha(Path(file)) == digest, file
    assert next(s for s in joint['selections'] if s['group'] == 'cross2025')['selected'] > 0
    return p
