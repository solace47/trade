"""Two frozen time folds with completed daily close-change extremes."""
from . import tail_formula_range_change_model as engine
from . import tail_formula_daily_extremes as inputs

if __name__ == '__main__':
    engine.inputs = inputs
    engine.main()
