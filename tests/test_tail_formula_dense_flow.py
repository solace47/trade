"""Ordered fractions preserve zeros, reject unknowns, and ignore future bars."""
import numpy as np
from trade_research.tail_formula_dense_flow import fractions, native_value


def test_order_zero_volume_and_share_lot_invariance():
    volumes = np.arange(29, dtype=float)**2
    got = fractions(volumes.reshape(1,-1))[0]
    assert got[-1] == 0
    np.testing.assert_allclose(got.sum(), 100, rtol=0, atol=2e-12)
    np.testing.assert_allclose(got, native_value(volumes), rtol=0, atol=2e-11)
    np.testing.assert_allclose(got, native_value(volumes/100), rtol=0, atol=2e-11)
    np.testing.assert_array_equal(native_value(volumes), native_value(volumes,.01))
    swapped = volumes.copy(); swapped[10:15] = swapped[10:15][::-1]
    assert not np.array_equal(got, fractions(swapped.reshape(1,-1))[0])


def test_unknown_negative_and_zero_total_are_not_imputed():
    volumes = np.zeros((3,29)); volumes[1,0] = np.nan; volumes[2,0] = -1
    assert np.isnan(fractions(volumes)).all()
