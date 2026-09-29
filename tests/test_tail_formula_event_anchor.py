import numpy as np

from trade_research.tail_formula_event_anchor import anchor, native_value


def test_rational_ties_use_recent_day_and_price_is_the_event_close():
    prices = np.array([[1.21, 1.10, 1.00] + [1.00]*18])
    value, price, lag, valid = anchor(prices, np.array([1.25]), np.array([2.0]))
    assert valid[0] and price[0] == 121 and lag[0] == 1
    actual, selected = native_value(prices[0], 1.25, 2.0)
    assert selected == 121
    np.testing.assert_allclose(actual, value[0], rtol=0, atol=1e-12)
    assert native_value(prices[0], .5, 2.0)[1] == 121


def test_oldest_event_flat_history_and_negative_changes_keep_defined_anchors():
    prices = np.ones((3, 21))*10
    prices[0, -1] = 9
    prices[2, :-1] = np.linspace(5, 9.9, 20)
    _, chosen, lag, valid = anchor(prices, np.ones(3)*10, np.ones(3))
    assert valid.all() and lag.tolist() == [20, 1, 20]
    for i in range(3):
        assert native_value(prices[i], 10, 1)[1] == chosen[i]
