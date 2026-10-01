"""Same audited inputs, matched absolute opportunity targets, no new raw scans."""
import json
from pathlib import Path
import subprocess

from .tail_formula_emotion_inputs import (
    META, CONTROL, EXPRESSIONS, HEADER, HELPER, DAILY_HELPER, NATIVE_GATE,
)
from .research_io import check_runtime, check_sources, sha

STEM = 'tail_formula_emotion_absolute'
ROOT = Path('data/research') / STEM
INPUTS = Path('data/research/tail_formula_emotion_transition/inputs')
PROTOCOL = Path('config') / (STEM + '_input_protocol.json')
TARGET = 'absolute'


def checked():
    check_runtime()
    p = json.loads(PROTOCOL.read_text())
    assert subprocess.check_output(['git', 'show', f'HEAD:{PROTOCOL}']) == PROTOCOL.read_bytes()
    assert p['target'] == TARGET and p['maximum_new_fits'] == 8
    assert p['arms'] == {'control': CONTROL, 'memory': EXPRESSIONS}
    assert p['native_header'] == HEADER and p['native_helper'] == HELPER
    assert p['daily_helper'] == DAILY_HELPER
    assert p['expected_keys'] == 1815129 and p['expected_original_valid'] == 1602413
    assert p['newly_invalid_allowed'] == 0 and not p['new_2026_prices_allowed']
    check_sources(p['source_hashes'])
    for file, digest in p['reused_inputs'].items():
        assert sha(Path(file)) == digest, file
    before = Path(p['preceding_result_root'])
    complete = json.loads((before / 'complete_results_manifest.json').read_text())
    gate = json.loads((before / 'memory_gate.json').read_text())
    assert complete['passed'] and not gate['supports_further_validation']
    assert not all(gate['criteria'].values())
    for name in ['feature', 'native_input', 'full_label']:
        proof = json.loads((INPUTS / (name + '_verification.json')).read_text())
        report = 'full_label_report.json' if name == 'full_label' else 'feature_report.json'
        field = 'label_report_sha256' if name == 'full_label' else 'feature_report_sha256'
        assert proof['passed'] and proof[field] == sha(INPUTS / report)
    return p
