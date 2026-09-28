"""Frozen amount-concentration inputs in the original two-fold model design."""
from . import tail_formula_amount_concentration as inputs
from . import tail_formula_range_change_model as engine

engine.inputs = inputs
engine.base.native_core = inputs.native_core

if __name__ == '__main__':
    engine.main()
