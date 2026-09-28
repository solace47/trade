import numpy as np

from trade_research.tail_formula_late_history import prior_tail_mean


def test_current_and_future_close_cannot_enter_current_signal():
    q = np.full(42, 10.)
    cl = np.arange(42) / 100 + 10
    known = np.ones(42, bool)
    actual = prior_tail_mean(q, cl, known)
    expected = sum((cl[i] - 10) / 10 for i in range(20)) / 20
    np.testing.assert_allclose(actual[20], expected, rtol=0, atol=1e-15)
    cl[20:] = 50
    changed = prior_tail_mean(q, cl, known)
    np.testing.assert_array_equal(actual[:21], changed[:21])
    assert changed[21] != actual[21]


def test_missing_stock_day_is_not_skipped_or_filled():
    q = np.full(42, 10.)
    cl = np.full(42, 10.1)
    known = np.ones(42, bool)
    known[20] = False
    values = prior_tail_mean(q, cl, known)
    np.testing.assert_allclose(values[20], .01, rtol=0, atol=1e-15)
    assert np.isnan(values[21:41]).all()
    np.testing.assert_allclose(values[41], .01, rtol=0, atol=1e-15)


def test_integer_cent_recovery_and_flat_history():
    q = np.full(21, 10.00000001)
    cl = np.full(21, 10.00999999)
    v = prior_tail_mean(q, cl, np.ones(21, bool))
    np.testing.assert_allclose(v[20], .001, rtol=0, atol=1e-15)
    assert prior_tail_mean(q, q, np.ones(21, bool))[20] == 0
