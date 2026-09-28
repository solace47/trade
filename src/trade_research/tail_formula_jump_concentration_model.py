"""Two frozen time folds with the squared-return concentration input."""
from . import tail_formula_range_change_model as engine
from . import tail_formula_jump_concentration as inputs

if __name__ == '__main__':
    engine.inputs = inputs
    engine.main()
