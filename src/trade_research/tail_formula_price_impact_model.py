"""Two fixed 49-input folds for absolute price change per amount."""
from . import tail_formula_price_impact as inputs
from . import tail_formula_range_change_model as engine

original_setup = engine.setup


def setup(fold):
    original_setup(fold)
    engine.base.native_core = inputs.native_core


if __name__ == '__main__':
    engine.inputs = inputs
    engine.setup = setup
    engine.main()
