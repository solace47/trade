"""Fit the fixed same-day stock/index association extension."""
from . import tail_formula_intraday_beta as inputs
from . import tail_formula_range_change_model as engine

engine.inputs = inputs

if __name__ == '__main__':
    engine.main()
