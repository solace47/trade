import numpy as np

from trade_research.tail_formula_prior_trend import native_value, prior_state


def test_exact_tie_and_one_cent_on_each_side():
    prices = np.full((3, 20), 10.); prices[1, 0] += .01; prices[2, 0] -= .01
    value, difference, valid = prior_state(prices)
    np.testing.assert_array_equal(value, [0, 1, 0])
    np.testing.assert_array_equal(difference, [0, 19, -19])
    assert valid.all()
    for row, expected in zip(prices, value):
        assert native_value(np.r_[row, .01]) == expected == native_value(np.r_[row, 10000.], .01)


def test_oldest_used_close_changes_state_and_missing_stays_missing():
    prices = np.full((3, 20), 10.); prices[0, -1] = 9.; prices[1, -1] = 11.; prices[2, -1] = np.nan
    value, difference, valid = prior_state(prices)
    np.testing.assert_allclose(value, [1, 0, np.nan], equal_nan=True)
    np.testing.assert_array_equal(valid, [True, True, False])
    assert native_value(np.r_[prices[0], 1.]) == 1
    assert native_value(np.r_[prices[1], 1.]) == 0
