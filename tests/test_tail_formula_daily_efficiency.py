import numpy as np
from trade_research.tail_formula_daily_efficiency import efficiency


def test_direction_bounds_flat_and_unknown_are_distinct():
    prices = [[6,5,4,3,2,1], [1,2,3,4,5,6], [2,2,2,2,2,2], [6,5,4,3,2,np.nan]]
    np.testing.assert_allclose(efficiency(prices,5), [100,-100,0,np.nan], equal_nan=True)


def test_equal_net_movement_distinguishes_retraced_path():
    steady = [6,5,4,3,2,1]
    retraced = [6,2,5,2,4,1]
    a,b = efficiency([steady,retraced],5)
    assert a == 100 and np.isclose(b, 100*5/15) and b < a
