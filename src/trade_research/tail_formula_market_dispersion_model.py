"""Direct dispersion levels with the unchanged-input 48-feature control."""
from . import tail_formula_market_dispersion as inputs
from . import tail_formula_range_change_model as engine

engine.inputs = inputs
engine.base.native_core = inputs.native_core

if __name__ == '__main__':
    engine.main()
