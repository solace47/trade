"""The paired 09:59 price targets without subtracting the same-date mean."""
from pathlib import Path

from . import tail_formula_endpoint_robust as engine


if __name__=='__main__':
    engine.ROOT=Path('data/research/tail_formula_endpoint_absolute')
    engine.PROTOCOL=Path('config/tail_formula_endpoint_absolute_protocol.json')
    engine.MODEL_STEM='tail_formula_endpoint_absolute'
    engine.VARIANT='endpoint_absolute'
    engine.main()
