import numpy as np

from trade_research.tail_formula_signed_cubic import native_value, signed_cubic


def test_direction_distinguishes_equal_total_variation_and_even_concentration():
    a = np.zeros((1, 29)); a[0, :3] = [-2, 1, 1]
    b = -a
    assert a.sum() == b.sum() == 0
    assert np.sum(a**2) == np.sum(b**2) and np.sum(a**4) == np.sum(b**4)
    av, aq, ac = signed_cubic(a); bv, bq, bc = signed_cubic(b)
    np.testing.assert_allclose(av, [-100/np.sqrt(6)])
    np.testing.assert_array_equal(av, -bv)
    np.testing.assert_array_equal(aq, bq)
    np.testing.assert_array_equal(ac, -bc)
    np.testing.assert_allclose(signed_cubic(5*a)[0], av)
    np.testing.assert_allclose(signed_cubic(a[:, ::-1])[0], av)


def test_flat_and_one_move_boundaries():
    returns = np.zeros((3, 29)); returns[1, 0] = 1; returns[2, -1] = -1
    np.testing.assert_array_equal(signed_cubic(returns)[0], [0, 100, -100])
    for row, expected in zip(returns, [0, 100, -100]):
        price = 10*np.exp(np.r_[0, np.cumsum(row/100)])
        np.testing.assert_allclose(native_value(price), expected)
        assert native_value(price, .01) == native_value(price, 1000000.)


def test_native_expression_on_mixed_direction_window():
    returns = np.tile([.1, -.04, .02], 10)[:29]
    price = 8*np.exp(np.r_[0, np.cumsum(returns/100)])
    expected = signed_cubic(returns[None, :])[0][0]
    np.testing.assert_allclose(native_value(price), expected, atol=1e-10, rtol=0)
