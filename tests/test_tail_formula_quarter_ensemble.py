import numpy as np
import pandas as pd

from trade_research import tail_formula_additive as base
from trade_research.tail_formula_quarter_ensemble import component_masks, flatten, quarters


def test_observation_quarter_and_three_component_membership():
    next_dates = pd.Series(['2024-03-29', '2024-04-01', '2024-07-01', '2024-10-08'])
    assert quarters(next_dates).tolist() == ['2024Q1', '2024Q2', '2024Q3', '2024Q4']
    masks = component_masks(next_dates, ['2024Q1', '2024Q2', '2024Q3', '2024Q4'])
    for i, (_, mask) in enumerate(masks):
        assert not mask[i] and mask.sum() == 3
    # A signal on March 29 observed April 1 belongs to the second label quarter.
    assert not masks[1][1][1]


def test_flattened_score_is_equal_average_and_does_not_change_components():
    x = np.array([[0], [1], [2]], dtype='int32')
    components = []
    for i in range(4):
        tree = dict(feature=[0, -2, -2], threshold=[.5, -2., -2.], children_left=[1, -1, -1],
                    children_right=[2, -1, -1], value=[0., float(i), float(-2*i)])
        components.append(dict(bias=float(i)/10, learning_rate=.05, trees=[tree]))
    bias, trees = flatten(components)
    result = base.predict(x, dict(bias=bias, learning_rate=.05, trees=trees))
    expected = np.mean([base.predict(x, c) for c in components], axis=0)
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-14)
    assert components[-1]['trees'][0]['value'] == [0., 3., -6.]
