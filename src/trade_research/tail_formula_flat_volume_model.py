"""Fixed unchanged-close volume share on the original relative-opportunity model."""
from . import tail_formula_flat_volume as inputs
from . import tail_formula_range_change_model as engine

engine.inputs=inputs

if __name__=='__main__':
    engine.main()
