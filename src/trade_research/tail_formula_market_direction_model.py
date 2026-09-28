"""Fit the fixed direction-conditioned stock/index excess move extension."""
from . import tail_formula_market_direction as inputs
from . import tail_formula_range_change_model as engine

engine.inputs = inputs

if __name__ == '__main__':
    engine.main()
