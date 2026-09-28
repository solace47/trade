"""Frozen cross-sectional rotation inputs with unchanged two-fold training."""
from . import tail_formula_cross_rotation as inputs
from . import tail_formula_range_change_model as engine

engine.inputs = inputs
engine.base.native_core = inputs.native_core

if __name__ == '__main__':
    engine.main()
