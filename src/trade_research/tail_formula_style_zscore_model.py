"""Two fixed 48-input folds with four contemporaneously normalized style inputs."""
from . import tail_formula_style_zscore as inputs
from . import tail_formula_range_change_model as engine

original_setup = engine.setup


def setup(fold):
    original_setup(fold)
    engine.base.native_core = inputs.native_core


if __name__ == '__main__':
    engine.inputs = inputs
    engine.setup = setup
    engine.main()
