"""Reuse the fixed 48-input study method with two verified past-morning inputs."""
import argparse
import json
from pathlib import Path

from . import tail_formula_additive as base
from . import tail_formula_context_2024 as linkage
from . import tail_formula_morning_history as inputs
from . import tail_formula_prior_day as adapter
from . import tail_formula_recent as study
from . import tail_formula_relative as relative
from .corporate_cash import sha

STEM = 'tail_formula_morning_history'


def configure():
    adapter.STEM = STEM; adapter.ROOT = inputs.ROOT; adapter.PROTOCOL = inputs.PROTOCOL
    adapter.COMBINED_PROTOCOL = Path('config') / (STEM + '_combined_protocol.json')
    adapter.CONTROL = Path('data/research') / (STEM + '_control')
    adapter.EXPRESSIONS = inputs.EXPRESSIONS; adapter.HEADER = inputs.HEADER
    native = json.loads((inputs.ROOT / 'native_input_verification.json').read_text())
    assert native['passed'] and native['feature_report_sha256'] == sha(inputs.ROOT / 'feature_report.json')
    assert native['feature_verification_sha256'] == sha(inputs.ROOT / 'feature_verification.json')
    for fold in ['2024', 'recent', 'combined']:
        p = json.loads((Path('config') / (STEM + '_' + fold + '_protocol.json')).read_text())
        assert p['inputs_protocol_sha256'] == sha(inputs.PROTOCOL)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('fold', choices=['2024', 'recent', 'combined', 'control'])
    p.add_argument('stage', choices=['model', 'verify_model', 'scores', 'freeze', 'verify', 'analyze'])
    a = p.parse_args(); configure()
    if a.fold == 'control':
        if a.stage == 'analyze':
            result = linkage.common_analysis(adapter.CONTROL, adapter.COMBINED_PROTOCOL)
        else:
            assert a.stage in ['freeze', 'verify']; result = adapter.control(a.stage + '_control')
    else:
        adapter.setup(a.fold)
        if a.stage == 'analyze':
            assert (Path('data/research') / (STEM + '_2025') / 'selection_verification.json').exists()
            assert (adapter.CONTROL / 'selection_verification.json').exists()
        if a.fold == 'combined':
            assert a.stage in ['freeze', 'verify', 'analyze']
            result = (linkage.common_analysis(linkage.COMBINED, linkage.PROTOCOL) if a.stage == 'analyze'
                      else getattr(linkage, 'combine' if a.stage == 'freeze' else 'verify_combined')())
        elif a.stage in ['model', 'verify_model']:
            result = getattr(relative, a.stage)('relative')
        elif a.stage in ['freeze', 'verify']:
            result = getattr(study, a.stage)()
        else:
            result = getattr(base, a.stage)()
    print(json.dumps(result, ensure_ascii=False, indent=2))
