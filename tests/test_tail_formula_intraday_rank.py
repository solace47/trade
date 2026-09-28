import numpy as np

from trade_research.tail_formula_intraday_rank import rank_sequence


def test_equal_ratios_at_different_prices_and_half_ties():
    p = np.arange(1, 23) * 100
    q = p * 2
    actual = rank_sequence(q, p, np.ones(22, bool))
    assert np.isnan(actual[:20]).all()
    np.testing.assert_array_equal(actual[20:], [50, 50])
    q[20] += 1
    assert rank_sequence(q, p, np.ones(22, bool))[20] == 100


def test_future_prices_do_not_change_current_and_missing_is_not_skipped():
    p = np.full(43, 1000)
    q = np.arange(43) + 1000
    good = np.ones(43, bool)
    original = rank_sequence(q, p, good)
    q[21:] = 500
    changed = rank_sequence(q, p, good)
    np.testing.assert_array_equal(original[:21], changed[:21])
    good[20] = False
    missing = rank_sequence(q, p, good)
    assert np.isnan(missing[20:41]).all()
    assert np.isfinite(missing[41:]).all()


def test_cross_products_keep_nearby_ratios_distinct():
    p = np.full(21, 2**26 - 1)
    q = p.copy()
    q[-1] -= 1
    assert rank_sequence(q, p, np.ones(21, bool))[-1] == 0
