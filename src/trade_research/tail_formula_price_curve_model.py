"""Test the frozen four coefficients of the complete visible price path."""
from . import tail_formula_price_curve as inputs
from . import tail_formula_range_change_model as engine

engine.inputs = inputs

if __name__ == '__main__':
    engine.main()
