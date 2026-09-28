"""Test the two frozen visible cost and capacity combinations."""
from . import tail_formula_entry_friction as inputs
from . import tail_formula_range_change_model as engine

engine.inputs = inputs

if __name__ == '__main__':
    engine.main()
