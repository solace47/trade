"""Compare daily and daily-size training baselines using the same 48 inputs."""
import argparse
import json
from pathlib import Path

from . import tail_formula_additive as base
from . import tail_formula_before1000 as labels
from . import tail_formula_before1000_model as original
from . import tail_formula_boundary_evaluation as evaluation
from . import tail_formula_context_2024 as linkage
from . import tail_formula_recent as study
from . import tail_formula_relative as relative
from .corporate_cash import sha

STEM = 'tail_formula_size_baseline'
PROTOCOL = Path('config') / (STEM + '_protocol.json')


def setup(variant, fold):
    p = json.loads(PROTOCOL.read_text())
    assert variant in ['daily', 'size'] and p['size_groups'] == {'daily': 1, 'size': 5}
    original.STEM = STEM + '_' + variant
    original.setup(fold)
    for f in ['2024', 'recent']:
        q = json.loads((Path('config') / (original.STEM + '_' + f + '_protocol.json')).read_text())
        assert q['baseline_protocol_sha256'] == sha(PROTOCOL)
        assert q['training_size_groups'] == p['size_groups'][variant]


def verify_model():
    p = json.loads(base.PROTOCOL.read_text())
    r = json.loads((base.ROOT / 'model_report.json').read_text())
    assert r['days'] == p['expected_training_days'] == 241
    assert r['rows'] == p['expected_rows']
    assert r['feature_names'] == list(base.EXPRESSIONS) and len(r['feature_names']) == 48
    assert all(r['parameters'][k] == v for k, v in p['parameters'].items())
    return relative.verify_model('group_relative')


if __name__ == '__main__':
    from .tail_formula_offset_logit48 import verify_scores
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['model', 'verify_model', 'scores', 'verify_scores', 'freeze', 'verify', 'analyze'])
    p.add_argument('--variant', choices=['daily', 'size'], required=True)
    p.add_argument('--fold', choices=['2024', 'recent', 'combined'], default='2024')
    a = p.parse_args(); setup(a.variant, a.fold)
    if a.stage == 'analyze':
        assert all((Path('data/research') / (STEM + '_' + v + '_' + f) / 'selection_verification.json').exists()
                   for v in ['daily', 'size'] for f in ['2024', 'recent', '2025'])
        root = linkage.COMBINED if a.fold == 'combined' else base.ROOT
        protocol = linkage.PROTOCOL if a.fold == 'combined' else base.PROTOCOL
        result = evaluation.analyze(root, protocol)
    elif a.fold == 'combined':
        assert a.stage in ['freeze', 'verify']
        result = linkage.combine() if a.stage == 'freeze' else linkage.verify_combined()
    elif a.stage == 'model':
        result = relative.model('group_relative')
    elif a.stage == 'verify_model':
        result = verify_model()
    elif a.stage == 'verify_scores':
        result = verify_scores()
    elif a.stage in ['freeze', 'verify']:
        result = getattr(study, a.stage)()
    else:
        result = base.scores()
    print(json.dumps(result, ensure_ascii=False, indent=2))
