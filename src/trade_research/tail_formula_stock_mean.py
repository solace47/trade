"""Equal-weight scores from the frozen disjoint-stock relative models."""
import json
from pathlib import Path

from . import tail_formula_stock_agreement as previous
from .corporate_cash import sha

STEM = 'tail_formula_stock_mean'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_input_protocol.json')
INTENT = Path('config') / (STEM + '_intent.json')
parent = previous.parent
prior = previous.prior
INPUTS = previous.INPUTS
META = previous.META
HEADER = previous.HEADER
CORE_GATE = previous.CORE_GATE
ARMS = {'control': prior.EXPRESSIONS, 'mean': prior.EXPRESSIONS}


def checked():
    previous.checked()
    p = json.loads(PROTOCOL.read_text())
    assert p['intent_sha256'] == sha(INTENT) and p['arms'] == ARMS
    assert p['expected_keys'] == 1258085 and p['expected_valid'] == 1117397
    assert p['weights'] == [.5, .5] and p['quantile'] == .995
    assert p['new_model_fits_allowed'] == p['new_model_predictions_allowed'] == 0
    assert not p['new_2026_prices_allowed']
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest, file
    gate = json.loads((previous.ROOT / 'stock_agreement_gate.json').read_text())
    complete = json.loads((previous.ROOT / 'complete_results_manifest.json').read_text())
    assert gate['passed'] and not gate['supports_2024_extension'] and complete['passed']
    assert len(complete['comparisons']) == 6
    return p
