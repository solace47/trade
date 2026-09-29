"""Keep the original learner while exposing stock changes in percentage units."""
from . import tail_formula_unscaled_prices as inputs
from . import tail_formula_range_change_model as engine

if __name__ == '__main__':
    engine.inputs = inputs
    engine.main()
