"""Fit the predeclared opportunity-order target and retain full-window evaluation."""
import argparse
import json
from pathlib import Path

from . import tail_formula_additive as base
from . import tail_formula_boundary_evaluation as evaluation
from . import tail_formula_context_2024 as linkage
from . import tail_formula_early_opportunity as adapter
from . import tail_formula_path_order_labels as targets
from . import tail_formula_recent as study
from . import tail_formula_relative as relative

STEM='tail_formula_path_order'


def setup(fold):
    adapter.STEM=STEM;adapter.targets=targets;adapter.setup(fold)
    for name in ['2024','recent','combined']:
        p=json.loads((Path('config')/(STEM+'_'+name+'_protocol.json')).read_text())
        assert p['training_target_window_end']=='09:59'


if __name__=='__main__':
    from .tail_formula_offset_logit48 import verify_scores
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['model','verify_model','scores','verify_scores','freeze','verify','analyze'])
    p.add_argument('--fold',choices=['2024','recent','combined'],default='2024');a=p.parse_args();setup(a.fold)
    if a.stage=='analyze':
        assert all((Path('data/research')/(STEM+'_'+f)/'selection_verification.json').exists() for f in ['2024','recent','2025'])
        root=linkage.COMBINED if a.fold=='combined' else base.ROOT
        result=evaluation.analyze(root,linkage.PROTOCOL if a.fold=='combined' else base.PROTOCOL)
    elif a.fold=='combined':
        assert a.stage in ['freeze','verify'];result=linkage.combine() if a.stage=='freeze' else linkage.verify_combined()
    elif a.stage=='model':
        result=relative.model('relative')
    elif a.stage=='verify_model':
        result=adapter.verify_model()
    elif a.stage=='verify_scores':
        result=verify_scores()
    elif a.stage in ['freeze','verify']:
        result=getattr(study,a.stage)()
    else:
        result=base.scores()
    print(json.dumps(result,ensure_ascii=False,indent=2))
