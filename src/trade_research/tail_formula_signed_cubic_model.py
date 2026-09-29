"""Fixed chronological models adding only the signed cubic input."""
from . import tail_formula_signed_cubic as inputs
from . import tail_formula_range_change_model as engine

if __name__ == '__main__':
    engine.inputs = inputs
    engine.main()
