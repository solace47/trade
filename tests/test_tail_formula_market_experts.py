import numpy as np
import pandas as pd

from trade_research import tail_formula_additive as base
from trade_research import tail_formula_market_experts as study


def test_exact_gate_boundary_and_inactive_expert_zero():
    tree = dict(feature=[0, -2, -2], threshold=[4.5, -2., -2.],
        children_left=[1, -1, -1], children_right=[2, -1, -1], value=[0., -2., 3.])
    x = np.zeros((6, 48), dtype='int32')
    x[:, study.GATE_INDEX] = [9999, 10000, 10001, 10001, 10000, 999999]
    x[:, 0] = [4, 5, 4, 5, 4, 5]
    lower = dict(state='lower', bias=.17, learning_rate=.05, trees=[tree])
    upper = dict(state='upper', bias=-.23, learning_rate=.05,
        trees=[dict(tree, value=[0., 7., -11.])])
    bias, trees = study.flatten([lower, upper], 'split')
    got = base.predict(x, dict(bias=bias, learning_rate=.05, trees=trees))
    np.testing.assert_allclose(got, [.07, .32, .12, -.78, .07, -.78], rtol=0, atol=1e-15)
    for expert, mask in study.state_masks(x, 'split'):
        component = lower if expert == 'lower' else upper
        np.testing.assert_allclose(got[mask], base.predict(x[mask], component), rtol=0, atol=1e-15)


def test_partition_retains_global_day_mass_without_reweighting():
    dates = pd.Series(['a', 'a', 'a', 'b', 'b'])
    w = 1 / dates.groupby(dates).transform('size').to_numpy()
    x = np.zeros((5, 48), dtype='int32')
    x[:, study.GATE_INDEX] = [10001, 10000, 10000, 10001, 10000]
    states = dict(study.state_masks(x, 'split'))
    # A date split between experts keeps its original unit weight in total.
    np.testing.assert_allclose([w[states['upper']].sum(), w[states['lower']].sum()], [5/6, 7/6])
    assert np.all(states['lower'] ^ states['upper'])
    assert study.state_masks(x, 'pooled')[0][1].all()
