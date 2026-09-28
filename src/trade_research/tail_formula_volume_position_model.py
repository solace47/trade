"""Evaluate the two fixed late-volume price-position inputs."""
from . import tail_formula_volume_position as inputs
from . import tail_formula_range_change_model as engine

engine.inputs = inputs

if __name__ == '__main__':
    engine.main()
