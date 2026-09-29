"""Fit the fixed brokerage versus bank market-context extension."""
from . import tail_formula_financial_context as inputs
from . import tail_formula_range_change_model as engine

engine.inputs = inputs

if __name__ == '__main__':
    engine.main()
