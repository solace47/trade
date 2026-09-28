"""Use the frozen prior-day close-location input with the original model design."""
from . import tail_formula_daily_pressure as inputs
from . import tail_formula_range_change_model as engine

engine.inputs = inputs
engine.base.native_core = inputs.native_core

if __name__ == '__main__':
    engine.main()
