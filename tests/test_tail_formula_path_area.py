import numpy as np
import pytest

from trade_research.tail_formula_path_area_labels import price_area


def test_negative_path_depth_and_inactive_slots_keep_fixed_denominator():
    marks = np.full((3,29), .01)
    active = np.ones((3,29),dtype=bool)
    marks[0,-1] = -.02
    active[1,1:] = False; marks[1,1:] = np.nan
    active[2,:] = False; marks[2,:] = np.nan
    np.testing.assert_allclose(price_area(marks,active),[(28*.01-.02)/29,.01/29,0],rtol=0,atol=1e-15)


def test_missing_active_mark_cannot_become_zero_contribution():
    marks=np.full((1,29),.01);marks[0,0]=np.nan
    with pytest.raises(AssertionError):
        price_area(marks,np.ones((1,29),dtype=bool))
