"""Economic boundary semantics, including exact ties and sparse distributions."""
from fractions import Fraction

import numpy as np
import pytest

from trade_research.tail_formula_market_bins import BOUNDARIES, bin_index, helper_atoms, native_values


def test_half_open_bins_at_every_price_boundary():
    prices = np.array([9799, 9800, 9801, 9899, 9900, 9949, 9950, 9999, 10000, 10049, 10050, 10099, 10100, 10199, 10200, 10201])
    expected = [sum(Fraction(int(p)-10000,10000)*10000 >= b for b in BOUNDARIES) for p in prices]
    np.testing.assert_array_equal(bin_index(np.full(len(prices),10000), prices), expected)


def test_crowding_and_midpoint_use_the_whole_bin_and_include_own_stock():
    # All 3,000 stocks lie in [0, +0.5%): every stock gets midpoint 50 and mass 100.
    all_flat = [3000,0,0,0,0,3000,3000,3000]
    np.testing.assert_array_equal(native_values(10000,10000,all_flat), [50,100])
    # The uppermost open bin contains one stock and all others are below it.
    one_extreme = [3000,100,200,300,400,600,800,2999]
    np.testing.assert_allclose(native_values(10000,10200,one_extreme), [100*2999.5/3000,100/3000])
    assert np.isnan(native_values(10000,10000,[1999,0,0,0,0,1999,1999,1999])).all()


def test_invalid_members_do_not_enter_any_count():
    assert helper_atoms(0,0,False) == [0]*8
    with pytest.raises(AssertionError):
        bin_index(np.array([0]),np.array([100]))
    with pytest.raises(AssertionError):
        native_values(10000,10000,[3000,0,500,499,1000,2000,2500,3000])
