import numpy as np
import pytest

from trade_research.tail_formula_price_impact import measure


def test_units_active_denominator_and_flat_prices():
    prices = np.full((1, 30), 101.); prices[0, 0] = 100.
    volume = np.full((1, 29), 1000.); amount = np.full((1, 29), 1e6)
    result = measure(prices, volume, amount)
    assert result['valid'][0]
    np.testing.assert_allclose(result['IL01'], [100/29])
    volume[0, -1] = amount[0, -1] = 0
    result = measure(prices, volume, amount)
    np.testing.assert_allclose(result['IL01'], [100/28])
    flat = measure(np.full((1, 30), 100.), volume, amount)
    assert flat['valid'][0] and flat['IL01'][0] == 0


def test_unknown_and_inconsistent_activity_do_not_become_zero_signal():
    prices = np.full((4, 30), 100.); volume = np.full((4, 29), 1000.); amount = np.full((4, 29), 1e6)
    volume[0] = amount[0] = 0
    amount[1, 0] = 0
    volume[2, 0] = 0
    amount[3, 0] = .001
    result = measure(prices, volume, amount)
    assert not result['valid'].any() and np.isnan(result['IL01']).all()
    with pytest.raises(AssertionError):
        measure(prices[:, 1:], volume, amount)
