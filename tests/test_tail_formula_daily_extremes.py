import numpy as np

from trade_research.tail_formula_daily_extremes import extremes, native_value


def test_preserve_signed_extremes_flat_and_unknown():
    prices = np.array([np.arange(21,0,-1),np.arange(1,22),np.ones(21),np.ones(21),np.ones(21)],float)
    prices[3,9] = np.nan
    prices[4,20] = 0
    got = extremes(prices,np.ones(5))
    np.testing.assert_allclose(got[:3],[[100,5],[-100/21,-50],[0,0]],atol=2e-13)
    assert np.isnan(got[3:]).all()
    assert np.isnan(extremes(np.ones((2,21)),[0,np.nan])).all()


def test_formula_covers_both_endpoints_and_atr_units():
    for index in [0,10,19]:
        chronological = np.ones(21)*10
        chronological[index+1:] = 11
        prices = chronological[::-1]
        expected = [5,0]
        np.testing.assert_allclose(extremes([prices],[2])[0],expected,atol=2e-13)
        np.testing.assert_allclose(native_value(prices,2),expected,atol=2e-13)
        np.testing.assert_allclose(native_value(prices*100,2),expected,atol=2e-13)
        np.testing.assert_allclose(native_value(prices[::-1],2),[0,-100/22],atol=2e-13)
