"""Use the fixed native-input engine for ordered past-window excursions."""
from . import tail_formula_range_change_model as engine
from . import tail_formula_path_excursion as inputs


if __name__ == '__main__':
    engine.inputs = inputs
    engine.main()
