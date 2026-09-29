"""Fixed two chronological models with the single historical-event distance."""
from . import tail_formula_event_anchor as inputs
from . import tail_formula_range_change_model as engine

if __name__ == '__main__':
    engine.inputs = inputs
    engine.main()
