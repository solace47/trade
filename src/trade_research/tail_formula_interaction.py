"""A single depth-three control with unchanged recent training and score cutoff."""
import argparse
import json
from pathlib import Path

from . import tail_formula_recent as study
from . import tail_formula_relative as relative
from . import tail_formula_additive as base

ROOT=Path('data/research/tail_formula_interaction')
PROTOCOL=Path('config/tail_formula_interaction_protocol.json')


def setup():
    study.ROOT=ROOT;study.PROTOCOL=PROTOCOL
    study.setup()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['model','verify_model','scores','freeze','verify','analyze','diagnose'])
    a=p.parse_args();setup()
    if a.stage in ['model','verify_model']:
        r=getattr(relative,a.stage)('relative')
    elif a.stage in ['freeze','verify']:
        r=getattr(study,a.stage)()
    else:
        r=getattr(base,a.stage)()
    print(json.dumps(r,ensure_ascii=False,indent=2))
