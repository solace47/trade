"""Original two chronological folds with only the fixed bipower-gap input."""
from . import tail_formula_bipower_gap as inputs
from . import tail_formula_range_change_model as engine

if __name__ == '__main__':
    engine.inputs = inputs
    engine.main()
