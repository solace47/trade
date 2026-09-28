"""Use the fixed native-input engine for longest late-window price runs."""
from . import tail_formula_range_change_model as engine
from . import tail_formula_max_run as inputs


if __name__ == '__main__':
    engine.inputs = inputs
    engine.main()
