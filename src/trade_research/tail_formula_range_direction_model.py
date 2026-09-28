"""Fixed original-capacity folds for daily range migration inputs."""
from . import tail_formula_range_direction as inputs
from . import tail_formula_range_change_model as engine

if __name__ == "__main__":
    engine.inputs = inputs
    engine.main()
