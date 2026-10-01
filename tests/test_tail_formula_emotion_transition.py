import numpy as np
from trade_research.tail_formula_emotion_transition import transition_atoms


def test_equal_marginal_breadth_can_have_different_transitions():
    stable = transition_atoms([9,11], [9,11], [10,10])
    rotation = transition_atoms([9,11], [11,9], [10,10])
    np.testing.assert_array_equal(stable.sum(axis=0), [0,0])
    np.testing.assert_array_equal(rotation.sum(axis=0), [1,1])


def test_flat_prices_and_source_unknown_are_distinct():
    a = transition_atoms([10,9,11,np.nan], [11,10,10,11], [10,10,10,10])
    np.testing.assert_array_equal(a[:3], np.zeros((3,2)))
    assert np.isnan(a[3]).all()


def test_cent_boundary_and_out_of_tick_prices():
    a = transition_atoms([9.99,10.01,9.991], [10.01,9.99,10.01], [10,10,10])
    np.testing.assert_array_equal(a[:2], [[1,0],[0,1]])
    assert np.isnan(a[2]).all()
