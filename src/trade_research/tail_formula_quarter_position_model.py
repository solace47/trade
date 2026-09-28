"""Prior-quarter inputs with unchanged coverage and a reused 48-input control."""
import argparse
import json
from pathlib import Path

from . import tail_formula_additive as base
from . import tail_formula_before1000 as labels
from . import tail_formula_boundary_evaluation as evaluation
from . import tail_formula_context_2024 as linkage
from . import tail_formula_prior_day as adapter
from . import tail_formula_quarter_position as inputs
from . import tail_formula_recent as study
from . import tail_formula_relative as relative
from .corporate_cash import sha

CONTROL = Path('data/research/tail_formula_before1000_model_2025')
COMBINED_PROTOCOL = Path('config') / (inputs.STEM + '_combined_protocol.json')


def setup(fold):
    r = json.loads((inputs.ROOT / 'feature_report.json').read_text())
    v = json.loads((inputs.ROOT / 'feature_verification.json').read_text())
    n = json.loads((inputs.ROOT / 'native_input_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(inputs.ROOT / 'feature_report.json')
    assert v['effective_input_intersection_unchanged'] and r['newly_invalid'] == 0
    assert n['passed'] and n['feature_report_sha256'] == sha(inputs.ROOT / 'feature_report.json')
    assert n['feature_verification_sha256'] == sha(inputs.ROOT / 'feature_verification.json')
    assert n['native_protocol_sha256'] == sha(inputs.NATIVE_PROTOCOL)
    control = json.loads(COMBINED_PROTOCOL.read_text())
    assert control['control_selection_report_sha256'] == sha(CONTROL / 'selection_report.json')
    assert control['control_analysis_report_sha256'] == sha(CONTROL / 'analysis_report.json')
    for stage in ['selection', 'analysis']:
        proof = json.loads((CONTROL / (stage + '_verification.json')).read_text())
        assert proof['passed'] and proof[stage + '_report_sha256'] == sha(CONTROL / (stage + '_report.json'))
    adapter.STEM = inputs.STEM; adapter.ROOT = inputs.ROOT
    adapter.EXPRESSIONS = inputs.EXPRESSIONS; adapter.HEADER = inputs.HEADER
    adapter.COMBINED_PROTOCOL = COMBINED_PROTOCOL; adapter.setup(fold)
    base.SOURCE = labels.ROOT; base.native_core = inputs.native_core
    for name in ['2024', 'recent']:
        p = json.loads((Path('config') / (inputs.STEM + '_' + name + '_protocol.json')).read_text())
        assert p['inputs_protocol_sha256'] == sha(inputs.PROTOCOL)
        assert p['feature_report_sha256'] == sha(inputs.ROOT / 'feature_report.json')
        assert p['label_report_sha256'] == sha(labels.ROOT / 'full_label_report.json')
        assert p['expected_features'] == 50
        assert p['native_protocol_sha256'] == sha(inputs.NATIVE_PROTOCOL)
        assert p['native_input_verification_sha256'] == sha(inputs.ROOT / 'native_input_verification.json')


def verify_model():
    p = json.loads(base.PROTOCOL.read_text()); r = json.loads((base.ROOT / 'model_report.json').read_text())
    assert r['feature_names'] == list(inputs.EXPRESSIONS) and len(r['feature_names']) == 50
    assert r['days'] == p['expected_training_days'] == 241
    assert all(r['parameters'][k] == v for k,v in p['parameters'].items())
    return relative.verify_model('relative')


if __name__ == '__main__':
    from .tail_formula_offset_logit48 import verify_scores
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['model', 'verify_model', 'scores', 'verify_scores', 'freeze', 'verify', 'analyze'])
    parser.add_argument('--fold', choices=['2024', 'recent', 'combined'], default='2024')
    args = parser.parse_args(); setup(args.fold)
    if args.stage == 'analyze':
        for fold in ['2024', 'recent', '2025']:
            root = Path('data/research') / (inputs.STEM + '_' + fold)
            proof = json.loads((root / 'selection_verification.json').read_text())
            assert proof['passed'] and proof['selection_report_sha256'] == sha(root / 'selection_report.json')
        result = evaluation.analyze(linkage.COMBINED if args.fold == 'combined' else base.ROOT,
                                    linkage.PROTOCOL if args.fold == 'combined' else base.PROTOCOL)
    elif args.fold == 'combined':
        assert args.stage in ['freeze', 'verify']
        result = linkage.combine() if args.stage == 'freeze' else linkage.verify_combined()
    elif args.stage == 'model':
        result = relative.model('relative')
    elif args.stage == 'verify_model':
        result = verify_model()
    elif args.stage == 'verify_scores':
        result = verify_scores()
    elif args.stage == 'freeze':
        result = study.freeze()
    elif args.stage == 'verify':
        model = json.loads((base.ROOT / 'model_report.json').read_text())
        core = base.native_core(model, model['thresholds'][3]['threshold'], base.EXPRESSIONS, base.HEADER)
        assert (base.ROOT / 'frozen_numeric_core.tdx').read_text() == core
        result = study.verify()
    else:
        result = base.scores()
    print(json.dumps(result, ensure_ascii=False, indent=2))
