"""Frozen time-varying VWAP path on the existing relative-opportunity model."""
from . import tail_formula_vwap_path as inputs
from . import tail_formula_range_change_model as engine

engine.inputs=inputs
engine.base.native_core=inputs.native_core

if __name__=='__main__':
    engine.main()
