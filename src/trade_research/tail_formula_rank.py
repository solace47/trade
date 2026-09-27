"""Within-day opportunity ranks on the fixed native 48-input formula family."""
import argparse
import json
from pathlib import Path

from . import tail_formula_additive as base
from . import tail_formula_context_2024 as linkage
from . import tail_formula_float as inputs
from . import tail_formula_recent as study
from . import tail_formula_relative as relative


def setup(fold):
    if fold=='combined':
        linkage.ROOT=Path('data/research/tail_formula_rank_2024')
        linkage.H2=Path('data/research/tail_formula_rank_recent')
        linkage.COMBINED=Path('data/research/tail_formula_rank_2025')
        linkage.PROTOCOL=Path('config/tail_formula_rank_combined_protocol.json')
        linkage.H2_SELECTION_SHA=None;linkage.H2_MODEL_SHA=None;linkage.H2_OUTCOMES_PREVIOUSLY_SEEN=False
    else:
        inputs.setup(fold)
        root=Path('data/research/tail_formula_rank_'+fold)
        protocol=Path('config/tail_formula_rank_'+fold+'_protocol.json')
        base.ROOT=root;base.PROTOCOL=protocol;relative.PROTOCOL=protocol
        study.ROOT=root;study.PROTOCOL=protocol


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('fold',choices=['2024','recent','combined'])
    p.add_argument('stage',choices=['model','verify_model','scores','freeze','verify','analyze','diagnose'])
    a=p.parse_args();setup(a.fold)
    if a.fold=='combined':
        assert a.stage in ['freeze','verify','analyze']
        r=(linkage.common_analysis(linkage.COMBINED,linkage.PROTOCOL) if a.stage=='analyze'
            else getattr(linkage,'combine' if a.stage=='freeze' else 'verify_combined')())
    elif a.stage in ['model','verify_model']:
        r=getattr(relative,a.stage)('rank')
    elif a.stage in ['freeze','verify']:
        r=getattr(study,a.stage)()
    else:
        r=getattr(base,a.stage)()
    print(json.dumps(r,ensure_ascii=False,indent=2))
