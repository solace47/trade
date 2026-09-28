import numpy as np

from trade_research.tail_formula_quarter_position import position


def test_flat_history_has_cent_denominator_and_breakouts_are_not_clamped():
    first, second = position(np.array([10., 10.01, 15.]), np.array([2., 2., 2.]),
                             np.array([60000., 60000., 60000.]),
                             np.array([1000., 1000., 900.]), np.array([1000., 1000., 1100.]))
    np.testing.assert_allclose(first, [0., .05, 25.], atol=1e-12)
    np.testing.assert_allclose(second, [0., 100., 300.], atol=1e-10)
