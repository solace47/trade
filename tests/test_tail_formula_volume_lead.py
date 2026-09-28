import numpy as np

from trade_research.tail_formula_volume_lead import moments, response, native_values


def calculated(prices, volume):
    m = moments(np.array([prices]), np.array([volume]))
    return np.array([response(m[n+'_volume'],m[n+'_return'],m[n+'_product'])[0]
                     for n in ['vl_lead','vl_follow']])


def test_impulse_before_and_after_return_have_distinct_signs():
    prices = np.r_[np.repeat(10.,15),np.repeat(11.,15)]
    vol = np.ones(29)
    vol[13] = 100
    first = calculated(prices,vol)
    vol = np.ones(29)
    vol[15] = 100
    last = calculated(prices,vol)
    assert first[0] > 0 and first[1] < 0 and last[0] < 0 and last[1] > 0


def test_zero_windows_units_and_generated_native_boundaries():
    prices = np.array([10+(i%7)*.03 for i in range(30)])
    for vol in [np.zeros(29),np.ones(29),np.r_[np.zeros(28),100.],np.r_[100.,np.zeros(28)],np.arange(29.)]:
        values = calculated(prices,vol)
        np.testing.assert_allclose(values, calculated(prices,vol*100),atol=2e-13)
        native = native_values(prices,np.r_[9999.,vol/100],1)
        np.testing.assert_allclose(values,native,atol=2e-12)
        np.testing.assert_array_equal(native,native_values(prices,np.r_[.001,vol/100],1,.01))
    np.testing.assert_array_equal(calculated(np.ones(30)*10,np.arange(29.)),[0,0])
