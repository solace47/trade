"""Earlier time fold of the fixed intraday-market relative formula method."""
import argparse
import json
from pathlib import Path

from . import tail_formula_additive as base
from . import tail_formula_context as context
from . import tail_formula_recent as study
from . import tail_formula_relative as relative

ROOT=Path('data/research/tail_formula_context_2024')
PROTOCOL=Path('config/tail_formula_context_2024_protocol.json')


def setup():
    context.setup('relative')
    base.ROOT=ROOT;base.PROTOCOL=PROTOCOL;relative.PROTOCOL=PROTOCOL
    study.ROOT=ROOT;study.PROTOCOL=PROTOCOL


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
