import numpy as np
from prepare_tail_formula_direction_target import direction


def test_equal_barriers_and_persistence_use_first_completed_event():
    prices=np.array([[100,101,101,101,99,99,99],[99,99,99,100,101,101,101],[100,100,100,100,100,100,100]],dtype=float)
    up,down,utility=direction(prices,np.ones_like(prices,dtype=bool),np.array([100.,100.,100.]))
    np.testing.assert_array_equal(up,[3,6,-1]);np.testing.assert_array_equal(down,[6,2,-1]);np.testing.assert_array_equal(utility,[1,-1,0])


def test_zero_volume_breaks_a_three_bar_direction_event():
    p=np.array([[101.,101.,101.,100.,99.,99.,99.]])
    a=np.array([[True,False,True,True,True,True,True]])
    up,down,utility=direction(p,a,np.array([100.]))
    np.testing.assert_array_equal(up,[-1]);np.testing.assert_array_equal(down,[6]);np.testing.assert_array_equal(utility,[-1])
