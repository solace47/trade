"""Frozen historical return-volume response on the shared two-fold design."""
from . import tail_formula_daily_response as inputs
from . import tail_formula_range_change_model as engine

engine.inputs=inputs
engine.base.native_core=inputs.native_core

if __name__=='__main__':
    engine.main()
