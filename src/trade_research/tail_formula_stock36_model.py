"""Two fixed 36-input folds without direct or relative index inputs."""
from . import tail_formula_stock36 as inputs
from . import tail_formula_range_change_model as engine

if __name__ == '__main__':
    engine.inputs = inputs
    engine.main()
