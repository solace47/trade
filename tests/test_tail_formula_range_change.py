import numpy as np

from trade_research.tail_formula_range_change import prior_ranges


def test_current_shock_does_not_change_prior_signal():
    high=np.full(30,11.);low=np.full(30,9.);close=np.full(30,10.)
    before=prior_ranges(high,low,close)
    high[25:]=100;low[25:]=90;close[25:]=95
    after=prior_ranges(high,low,close)
    for left,right in zip(before,after):
        np.testing.assert_allclose(left[:26],right[:26],rtol=0,atol=0,equal_nan=True)
    assert after[0][26]>before[0][26]


def test_gap_is_counted_and_ratio_is_invariant_to_common_price_scale():
    high=np.r_[np.full(20,11.),np.full(10,21.)];low=high-2;close=high-1
    short,long=prior_ranges(high,low,close)
    assert short[21]==3.8 and long[21]==2.45
    short2,long2=prior_ranges(3*high,3*low,3*close)
    np.testing.assert_allclose(short/long,short2/long2,rtol=0,atol=1e-12,equal_nan=True)
