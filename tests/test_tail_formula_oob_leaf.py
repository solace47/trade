import copy

import numpy as np
import pytest

from trade_research.tail_formula_oob_leaf import revalue


def tree():
    return dict(feature=[0, 0, -2, -2, -2], threshold=[5, 2, -2, -2, -2],
        children_left=[1, 2, -1, -1, -1], children_right=[4, 3, -1, -1, -1], value=[99]*5)


def test_nearest_supported_parent_is_used_without_skipping_it():
    x = np.array([[1], [3], [4], [8], [9]])
    y = np.array([1., 2., 4., 10., 20.])
    got = revalue(tree(), x, y, np.ones(5), np.arange(5), np.ones(5, bool), 2, 2)
    assert got['value_source_node'][2] == 1
    assert got['value_source_node'][3] == 3
    assert got['value_source_node'][4] == 4
    np.testing.assert_allclose(got['raw_value'][2], 7/3)


def test_inbag_targets_and_old_values_cannot_contaminate_estimation():
    x = np.array([[1], [3], [4], [8], [9]])
    y = np.array([1., 2., 4., 10., 20.])
    oob = np.array([True, True, True, False, False])
    got = revalue(tree(), x, y, np.ones(5), np.arange(5), oob, 2, 2)
    assert got['value_source_node'][4] == 0
    changed = copy.deepcopy(tree())
    changed['value'] = [-123456]*5
    y[~oob] = 1e12
    again = revalue(changed, x, y, np.ones(5), np.arange(5), oob, 2, 2)
    assert got == again


def test_unsupported_root_fails_instead_of_borrowing_or_zero_filling():
    with pytest.raises(ValueError, match='supported out-of-bag root'):
        revalue(tree(), np.array([[1], [8]]), np.array([1., 2.]), np.ones(2), np.array([0, 0]),
            np.ones(2, bool), 2, 2)
