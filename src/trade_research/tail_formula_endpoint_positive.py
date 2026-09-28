"""A binary event: the fixed 09:59 reference mark strictly covers its costs."""
from pathlib import Path

from . import tail_formula_endpoint_robust as engine


if __name__=='__main__':
    engine.ROOT=Path('data/research/tail_formula_endpoint_positive')
    engine.PROTOCOL=Path('config/tail_formula_endpoint_positive_protocol.json')
    engine.MODEL_STEM='tail_formula_endpoint_positive'
    engine.VARIANT='endpoint_positive'
    engine.main()
