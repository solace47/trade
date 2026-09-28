import numpy as np

from trade_research.tail_formula_price_curve import (
    WEIGHTS, MAXIMUM, SQUARES, project, measure, ordinary_power_fit, native_value)


def test_known_polynomial_is_recovered_independently_and_intercept_cancels():
    np.testing.assert_array_equal(WEIGHTS @ WEIGHTS.T, np.diag(SQUARES))
    np.testing.assert_array_equal(WEIGHTS.sum(axis=1), np.zeros(4))
    coefficients = np.array([[.4, -.2, .7, -.1], [0., 2., 0., 0.]])
    y = coefficients @ (WEIGHTS/MAXIMUM[:, None]) + 7
    np.testing.assert_allclose(project(y), coefficients, rtol=0, atol=1e-13)
    np.testing.assert_allclose(ordinary_power_fit(y), coefficients, rtol=0, atol=1e-13)
    np.testing.assert_allclose(project(y[:, ::-1]), coefficients*[-1, 1, -1, 1], rtol=0, atol=1e-13)


def test_identical_endpoints_do_not_erase_curvature_and_flat_is_observed_zero():
    curved = np.round(10+.1*((np.arange(30)-14.5)/14.5)**2, 2)
    flat = np.full(30, curved[0])
    assert curved[0] == curved[-1] == flat[0] == flat[-1]
    valid, values = measure(np.array([curved, flat]), np.array([2., 2.]))
    assert valid.all() and values[0, 1] > 0
    np.testing.assert_allclose(values[1], 0, rtol=0, atol=0)
    np.testing.assert_allclose(values[0, [0, 2]], 0, rtol=0, atol=1e-13)


def test_invalid_prices_are_not_imputed_and_actual_native_future_terms_are_absent():
    prices = np.array([np.full(30, 10.) for _ in range(3)])
    prices[0, 4] = np.nan
    prices[1, 8] = 0
    prices[2, 9] = 10.001
    valid, values = measure(prices, np.full(3, 2.))
    assert not valid.any() and np.isnan(values).all()
    prices = np.round(10+.001*np.arange(30)**2, 2)
    valid, values = measure(prices[None, :], np.array([2.]))
    native = native_value(prices, 2.)
    changed = native_value(prices, 2., .01)
    np.testing.assert_allclose(list(native.values()), values[0], rtol=0, atol=1e-12)
    assert native == changed
