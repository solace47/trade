"""Fixed original-capacity folds for the two overnight jump second moments."""
from . import tail_formula_gap_semivariance as inputs
from . import tail_formula_range_change_model as engine

if __name__=='__main__':
    engine.inputs=inputs
    engine.main()
