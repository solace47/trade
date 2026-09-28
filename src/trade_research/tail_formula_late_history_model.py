"""Frozen prior late-session mean on the existing 48-input model design."""
from . import tail_formula_late_history as inputs
from . import tail_formula_range_change_model as engine

engine.inputs = inputs

if __name__ == '__main__':
    engine.main()
