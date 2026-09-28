"""Learn early morning opportunity; evaluate the unchanged full morning window."""
import argparse
import json
from pathlib import Path

from . import tail_formula_additive as base
from . import tail_formula_before1000 as evaluation_labels
from . import tail_formula_boundary_evaluation as evaluation
from . import tail_formula_context_2024 as linkage
from . import tail_formula_early_opportunity_labels as targets
from . import tail_formula_float as inputs
from . import tail_formula_prior_day as adapter
from . import tail_formula_recent as study
from . import tail_formula_relative as relative
from .corporate_cash import sha

STEM = 'tail_formula_early_opportunity'


def setup(fold):
    adapter.STEM = STEM; adapter.ROOT = inputs.ROOT
    adapter.EXPRESSIONS = inputs.EXPRESSIONS; adapter.HEADER = inputs.HEADER
    adapter.COMBINED_PROTOCOL = Path('config') / (STEM + '_combined_protocol.json')
    adapter.setup(fold)
    base.SOURCE = targets.ROOT
    proof = json.loads((targets.ROOT / 'full_label_verification.json').read_text())
    assert proof['passed'] and proof['scope'] == 'training_target_only' and proof['not_for_evaluation']
    for name in ['2024', 'recent']:
        p = json.loads((Path('config') / (STEM + '_' + name + '_protocol.json')).read_text())
        assert p['training_target_protocol_sha256'] == sha(targets.PROTOCOL)
        assert p['label_report_sha256'] == sha(targets.ROOT / 'full_label_report.json')
        assert p['evaluation_label_report_sha256'] == sha(evaluation_labels.ROOT / 'full_label_report.json')
        assert p['feature_report_sha256'] == sha(inputs.ROOT / 'feature_report.json')
        assert p['expected_features'] == len(inputs.EXPRESSIONS) == 48


def verify_model():
    p = json.loads(base.PROTOCOL.read_text()); r = json.loads((base.ROOT / 'model_report.json').read_text())
    assert r['days'] == p['expected_training_days'] == 241
    assert r['feature_names'] == list(inputs.EXPRESSIONS)
    assert all(r['parameters'][k] == v for k, v in p['parameters'].items())
    return relative.verify_model('relative')


if __name__ == '__main__':
    from .tail_formula_offset_logit48 import verify_scores
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['model', 'verify_model', 'scores', 'verify_scores', 'freeze', 'verify', 'analyze'])
    p.add_argument('--fold', choices=['2024', 'recent', 'combined'], default='2024')
    a = p.parse_args(); setup(a.fold)
    if a.stage == 'analyze':
        assert all((Path('data/research') / (STEM + '_' + f) / 'selection_verification.json').exists()
                   for f in ['2024', 'recent', '2025'])
        root = linkage.COMBINED if a.fold == 'combined' else base.ROOT
        protocol = linkage.PROTOCOL if a.fold == 'combined' else base.PROTOCOL
        result = evaluation.analyze(root, protocol)
    elif a.fold == 'combined':
        assert a.stage in ['freeze', 'verify']
        result = linkage.combine() if a.stage == 'freeze' else linkage.verify_combined()
    elif a.stage == 'model':
        result = relative.model('relative')
    elif a.stage == 'verify_model':
        result = verify_model()
    elif a.stage == 'verify_scores':
        result = verify_scores()
    elif a.stage in ['freeze', 'verify']:
        result = getattr(study, a.stage)()
    else:
        result = getattr(base, a.stage)()
    print(json.dumps(result, ensure_ascii=False, indent=2))
