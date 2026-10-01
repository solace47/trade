import numpy as np

from trade_research.tail_formula_morning_extrema_order import measure, literal


def test_latest_ties_and_zero_volume_extremes_in_actual_native_header():
    high, low, volume = np.full((2, 121), 110.), np.full((2, 121), 100.), np.ones((2, 121))
    high[0, [10, 90]] = 120; low[0, [20, 80]] = 90
    high[0, 120] = 999; low[0, 120] = 1; volume[0, 120] = 0
    high[1, [10, 90]] = 120; low[1, [20, 110]] = 90
    expected = np.array([[100 * 10 / 120, 100 * 40 / 120], [100 * -20 / 120, 100 * 10 / 120]])
    np.testing.assert_allclose(measure(high, low, volume), expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(literal(high, low, volume), expected, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(literal(np.full((1, 121), 100.), np.full((1, 121), 100.), np.ones((1, 121))), [[0, 0]])
