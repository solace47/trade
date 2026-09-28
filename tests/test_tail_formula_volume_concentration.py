import numpy as np

from trade_research.tail_formula_volume_concentration import concentration


def test_concentration_bounds_and_unknowns():
    values = concentration([[5, 5, 5, 5], [0, 0, 0, 20], [0, 0, 0, 0], [5, np.nan, 5, 5], [-1, 2, 3, 4]])
    np.testing.assert_allclose(values[:2], [0, 100], atol=1e-12)
    assert np.isnan(values[2:]).all()


def test_same_total_order_and_share_lot_invariance():
    volume = np.array([[100, 200, 300, 400], [250, 250, 250, 250], [0, 0, 0, 1000]], dtype=float)
    actual = concentration(volume)
    assert actual[1] < actual[0] < actual[2]
    np.testing.assert_allclose(actual, concentration(volume/100), rtol=0, atol=1e-12)
    np.testing.assert_allclose(actual, concentration(volume[:, ::-1]), rtol=0, atol=1e-12)
