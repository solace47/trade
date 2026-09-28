import numpy as np
from trade_research.tail_formula_gap_semivariance import components


def test_two_sides_retain_zero_mean_jump_size():
    previous=np.full((2,20),10.)
    opened=previous.copy();opened[1,:10]=9.;opened[1,10:]=11.
    result=components(opened,previous,np.array([2.,2.]))
    np.testing.assert_array_equal(result[0],[0.,0.])
    np.testing.assert_allclose(result[1],[np.sqrt(.005)*50]*2,rtol=0,atol=1e-13)


def test_one_sided_jumps_and_share_price_rounding():
    previous=np.full((1,20),10.)
    opened=np.full((1,20),10.1)
    result=components(opened,previous,np.array([1.]))
    np.testing.assert_allclose(result,[[0.,1.]],rtol=0,atol=1e-13)
    np.testing.assert_array_equal(result,components(opened+1e-6,previous-1e-6,np.array([1.])))
