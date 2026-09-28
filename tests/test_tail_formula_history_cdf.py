import numpy as np

from trade_research.tail_formula_history_cdf import midcdf


def test_ties_receive_half_weight_and_endpoints_are_bounded():
    prices=np.tile([100,200,300],(5,1));volume=np.tile([1,6,3],(5,1))
    np.testing.assert_allclose(midcdf([50,100,200,300,400],prices,volume),[0,5,40,85,100],rtol=0,atol=1e-12)


def test_equal_mean_and_variance_need_not_determine_cumulative_mass():
    prices=np.array([[100,400,400,700],[100,300,600,600]])
    assert prices[0].mean()==prices[1].mean() and prices[0].var()==prices[1].var()
    np.testing.assert_allclose(midcdf([500,500],prices,np.ones((2,4))),[75,50],rtol=0,atol=1e-12)
