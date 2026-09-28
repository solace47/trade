"""Evaluate fixed prior-day price-path inputs with the conservative labels."""
import argparse
import json
from pathlib import Path

from . import tail_formula_limit_breadth_model as comparison
from . import tail_formula_daily_efficiency as inputs
from . import tail_formula_additive as base
from . import tail_formula_boundary_evaluation as evaluation
from . import tail_formula_context_2024 as linkage
from . import tail_formula_recent as study
from . import tail_formula_relative as relative


def configure(fold):
    comparison.inputs = inputs
    comparison.CONTROL = Path('data/research') / (inputs.STEM + '_control')
    comparison.configure(fold)


if __name__ == '__main__':
    from .tail_formula_offset_logit48 import verify_scores
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['model', 'verify_model', 'scores', 'verify_scores', 'freeze', 'verify', 'analyze'])
    p.add_argument('--fold', choices=['2024', 'recent', 'combined', 'control'], default='2024')
    a = p.parse_args(); configure(a.fold)
    if a.stage == 'analyze':
        assert all((Path('data/research') / (inputs.STEM + '_' + f) / 'selection_verification.json').exists()
                   for f in ['2024', 'recent', '2025', 'control'])
        root = comparison.CONTROL if a.fold == 'control' else linkage.COMBINED if a.fold == 'combined' else base.ROOT
        protocol = inputs.COMBINED_PROTOCOL if a.fold in ['combined', 'control'] else base.PROTOCOL
        result = evaluation.analyze(root, protocol)
    elif a.fold == 'control':
        assert a.stage in ['freeze', 'verify']; result = comparison.control(a.stage)
    elif a.fold == 'combined':
        assert a.stage in ['freeze', 'verify']
        result = linkage.combine() if a.stage == 'freeze' else linkage.verify_combined()
    elif a.stage == 'model':
        result = relative.model('relative')
    elif a.stage == 'verify_model':
        result = comparison.verify_model()
    elif a.stage == 'verify_scores':
        result = verify_scores()
    elif a.stage in ['freeze', 'verify']:
        result = getattr(study, a.stage)()
    else:
        result = base.scores()
    print(json.dumps(result, ensure_ascii=False, indent=2))
