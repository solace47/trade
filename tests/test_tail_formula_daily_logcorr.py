import numpy as np

from trade_research.tail_formula_daily_logcorr import correlation, native_value


def test_order_alignment_and_flat_sequences():
    prices = np.arange(1, 21, dtype=float)
    volumes = np.arange(20, dtype=float)
    positive = correlation(prices[None], volumes[None])[0]
    assert positive > 90
    np.testing.assert_allclose(correlation(prices[None], volumes[::-1][None]), -positive)
    np.testing.assert_allclose(correlation(np.ones((1, 20)), volumes[None]), 0)
    np.testing.assert_allclose(correlation(prices[None], np.ones((1, 20))), 0)
    assert abs(correlation(prices[None], volumes[np.roll(np.arange(20), 10)][None])[0]) < positive


def test_native_share_units_and_missing_values():
    prices = np.arange(1, 21, dtype=float)
    volumes = np.arange(20, dtype=float)
    expected = correlation(prices[None], volumes[None])[0]
    np.testing.assert_allclose(native_value(prices, volumes/100), expected, atol=1e-10)
    assert abs(native_value(prices, volumes)-expected) > 1
    for bad in [np.nan, -1, .5]:
        changed = volumes.copy(); changed[0] = bad
        assert np.isnan(correlation(prices[None], changed[None])[0])
