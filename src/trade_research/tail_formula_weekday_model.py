"""Use the unchanged native-input study engine for five weekday indicators."""
from . import tail_formula_range_change_model as engine
from . import tail_formula_weekday as inputs


if __name__ == '__main__':
    engine.inputs = inputs
    engine.main()
