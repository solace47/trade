"""Fixed two-fold fits with an exact-quality original-input control if needed."""
import argparse
import json
from pathlib import Path

from . import tail_formula_additive as base
from . import tail_formula_before1000 as labels
from . import tail_formula_boundary_evaluation as evaluation
from . import tail_formula_context_2024 as linkage
from . import tail_formula_prior_day as adapter
from . import tail_formula_recent as study
from . import tail_formula_relative as relative
from . import tail_formula_prior_close as inputs
from .corporate_cash import sha


def setup(arm, fold):
    p, _ = inputs.raw.checked()
    original = json.loads((inputs.ROOT / 'feature_report.json').read_text())
    root = inputs.ROOT if arm == 'joint' else inputs.CONTROL_INPUTS
    if arm == 'control':
        assert original['newly_invalid'] > 0
    r = json.loads((root / 'feature_report.json').read_text())
    v = json.loads((root / 'feature_verification.json').read_text())
    n = json.loads((inputs.ROOT / 'native_input_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(root / 'feature_report.json')
    assert r['features_sha256'] == sha(root / 'features.parquet') == original['features_sha256']
    assert n['passed'] and n['feature_report_sha256'] == sha(inputs.ROOT / 'feature_report.json')
    expressions = inputs.EXPRESSIONS if arm == 'joint' else inputs.previous.EXPRESSIONS
    assert r['expressions'] == expressions and r['native_header'] == inputs.HEADER
    stem = inputs.STEM + '_' + arm
    adapter.STEM, adapter.ROOT = stem, root
    adapter.EXPRESSIONS, adapter.HEADER = expressions, inputs.HEADER
    adapter.COMBINED_PROTOCOL = Path('config') / (stem + '_combined_protocol.json')
    base.native_core = inputs.native_core
    adapter.setup(fold)
    base.SOURCE = labels.ROOT
    for name in ['2024', 'recent']:
        q = json.loads((Path('config') / (stem + '_' + name + '_protocol.json')).read_text())
        assert q['inputs_protocol_sha256'] == sha(inputs.PROTOCOL) and q['parameters'] == p['parameters']
        assert q['feature_report_sha256'] == sha(root / 'feature_report.json')
        assert q['native_input_verification_sha256'] == sha(inputs.ROOT / 'native_input_verification.json')
        assert q['expected_features'] == len(expressions)
        assert q['expected_rows'] == original['training_counts'][name]['rows']
        assert q['expected_training_days'] == original['training_counts'][name]['days']


def main():
    from .tail_formula_offset_logit48 import verify_scores
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['model', 'verify_model', 'scores', 'verify_scores', 'freeze', 'verify', 'analyze'])
    p.add_argument('--arm', choices=['joint', 'control'], required=True)
    p.add_argument('--fold', choices=['2024', 'recent', 'combined'], default='2024')
    a = p.parse_args()
    setup(a.arm, a.fold)
    if a.stage == 'analyze':
        assert a.fold == 'combined', 'Only annual aggregation; both halves are already included'
        for arm in ['joint'] + (['control'] if (inputs.CONTROL_INPUTS / 'feature_report.json').exists() else []):
            for fold in ['2024', 'recent', '2025']:
                root = Path('data/research') / (inputs.STEM + '_' + arm + '_' + fold)
                v = json.loads((root / 'selection_verification.json').read_text())
                assert v['passed'] and v['selection_report_sha256'] == sha(root / 'selection_report.json')
        result = evaluation.analyze(linkage.COMBINED, linkage.PROTOCOL)
    elif a.fold == 'combined':
        assert a.stage in ['freeze', 'verify']
        result = linkage.combine() if a.stage == 'freeze' else linkage.verify_combined()
    elif a.stage == 'model':
        result = relative.model('relative')
    elif a.stage == 'verify_model':
        r = json.loads((base.ROOT / 'model_report.json').read_text())
        p = json.loads(base.PROTOCOL.read_text())
        assert r['rows'] == p['expected_rows'] and r['days'] == p['expected_training_days']
        assert r['feature_names'] == list(base.EXPRESSIONS)
        assert all(r['parameters'][k] == v for k, v in p['parameters'].items())
        result = relative.verify_model('relative')
    elif a.stage == 'verify_scores':
        result = verify_scores()
    elif a.stage == 'freeze':
        result = study.freeze()
    elif a.stage == 'verify':
        r = json.loads((base.ROOT / 'model_report.json').read_text())
        assert (base.ROOT / 'frozen_numeric_core.tdx').read_text() == inputs.native_core(r, r['thresholds'][3]['threshold'], base.EXPRESSIONS, base.HEADER)
        result = study.verify()
    else:
        result = base.scores()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
