import copy

import numpy as np

from trade_research.tail_formula_leaf_transfer import refit_component, flatten
from trade_research.tail_formula_additive import predict


def source():
    return dict(bias=999., learning_rate=.05, trees=[dict(feature=[0, -2, -2],
        threshold=[0., -2., -2.], children_left=[1, -1, -1], children_right=[2, -1, -1],
        value=[20., 30., 40.])]*2)


def test_opposite_values_ignore_source_estimates_and_preserve_structure():
    s = source(); before = copy.deepcopy(s)
    x = np.array([[-1], [-1], [1], [1]]); y = np.array([0., 0., 1., 1.]); w = np.array([1., 2., 3., 4.])
    got = refit_component(s, x, y, w)
    changed = copy.deepcopy(s); changed['bias'] = -999.
    for tree in changed['trees']: tree['value'] = [-50., -60., -70.]
    assert got == refit_component(changed, x, y, w) and s == before
    assert all(q['after'] <= q['before'] for q in got['estimation_losses'])
    np.testing.assert_allclose(got['trees'][0]['value'][1:], [-.7, .3])
    bias, trees = flatten([got, before])
    np.testing.assert_allclose(predict(x, dict(bias=bias, trees=trees, learning_rate=.05)),
        (predict(x, got)+predict(x, before))/2, rtol=0, atol=2e-12)


def test_unsupported_leaf_gets_zero_without_borrowing_source_value():
    got = refit_component(source(), np.array([[-1], [-1]]), np.array([0., 1.]), np.ones(2))
    assert all(t['estimation_rows'][2] == t['estimation_weights'][2] == t['value'][2] == 0 for t in got['trees'])
    assert all(q['after'] <= q['before'] for q in got['estimation_losses'])
