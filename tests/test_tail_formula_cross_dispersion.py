import numpy as np

from trade_research.tail_formula_cross_dispersion import standardized


def test_flat_section_and_fixed_basis_point_floor():
    np.testing.assert_allclose(standardized(np.array([1.,1.01]),np.array([1.,1.]),np.array([0.,.001])),[0,1],rtol=0,atol=1e-12)


def test_same_relative_strength_survives_common_shift_and_scale():
    values=np.array([-.5,1.,2.5]);mean=values.mean();std=values.std(ddof=0)
    expected=standardized(values,mean,std)
    np.testing.assert_allclose(standardized(3*values+5,3*mean+5,3*std),expected,rtol=0,atol=1e-12)
