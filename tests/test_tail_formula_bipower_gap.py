import numpy as np

from trade_research.tail_formula_bipower_gap import COEFFICIENT, bipower_gap, native_value


def test_flat_and_same_energy_different_adjacency():
    r = np.zeros((4, 29))
    r[1, [0, 1]] = [1, -1]
    r[2, [0, 28]] = [1, -1]
    r[3] = 1
    gap, squared, adjacent = bipower_gap(r)
    np.testing.assert_allclose(gap, [0, 100*(1-COEFFICIENT/2), 100, 0], rtol=0, atol=1e-12)
    assert squared[1] == squared[2] == 2 and adjacent[1] == 1 and adjacent[2] == 0
    np.testing.assert_allclose(bipower_gap(-r*2)[0], gap, rtol=0, atol=1e-12)


def test_native_29_returns_include_both_edges_but_not_outside_prices():
    for positions in [[0], [28], [0, 1], [27, 28], [0, 28], []]:
        r = np.zeros(29); r[positions] = .1
        prices = 10*np.exp(np.r_[0, np.cumsum(r)]/100)
        value = bipower_gap((100*np.log(prices[1:]/prices[:-1]))[None, :])[0][0]
        np.testing.assert_allclose(native_value(prices), value, rtol=0, atol=1e-10)
        assert native_value(prices, .01) == native_value(prices, 1000000)
