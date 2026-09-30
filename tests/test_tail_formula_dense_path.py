"""Dense coordinates retain minute order and honor the same native cutoff."""
import numpy as np
from trade_research.tail_formula_dense_path import measure,native_value


def test_all_prices_recovered_before_common_encoding():
    prices = (1000+np.arange(30)**2)/100
    atr = 2.3;coords = measure(prices.reshape(1,-1),np.array([atr]))[0]
    recovered = prices[-1]*(1+coords*atr/100)
    np.testing.assert_allclose(recovered,prices[-2::-1],rtol=0,atol=1e-12)
    swapped = prices.copy();swapped[5:10] = swapped[5:10][::-1]
    assert not np.array_equal(coords,measure(swapped.reshape(1,-1),np.array([atr]))[0])


def test_native_dense_prices_ignore_outside_and_future_quotes():
    prices = (1200+np.arange(30)**2)/100
    got = native_value(prices,2.3)
    np.testing.assert_array_equal(got,native_value(prices,2.3,.01))
    np.testing.assert_allclose(got,measure(prices.reshape(1,-1),np.array([2.3]))[0],rtol=0,atol=2e-11)
