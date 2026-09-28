"""Two fixed time folds using the independently checked lagged volume inputs."""
from . import tail_formula_range_change_model as engine
from . import tail_formula_volume_lead as inputs

if __name__ == '__main__':
    engine.inputs = inputs
    engine.main()
