import numpy as np
import pytest

from trade_research import tail_formula_exchange_experts as study


def test_routing_preserves_inactive_zero_negative_bias_and_feature_boundary(monkeypatch):
    monkeypatch.setattr(study.mechanics, 'GATE_INDEX', 48)
    monkeypatch.setattr(study.mechanics, 'GATE', 10050)
    tree = dict(feature=[0, -2, -2], threshold=[2.5, -2, -2], children_left=[1, -1, -1],
                children_right=[2, -1, -1], value=[0, -4, 8])
    components = [dict(state=state, bias=bias, learning_rate=.05, trees=[tree])
                  for state, bias in [('lower', -.3), ('upper', .2)]]
    x = np.zeros((4, 49), dtype='int32'); x[:, 0] = [2, 3, 2, 3]; x[:, 48] = [10000, 10000, 10100, 10100]
    masks = study.state_masks(x)
    np.testing.assert_array_equal(masks[0][1], [True, True, False, False])
    np.testing.assert_array_equal(masks[1][1], [False, False, True, True])
    bias, trees = study.mechanics.flatten(components, 'split')
    got = study.base.predict(x, dict(bias=bias, trees=trees, learning_rate=.05))
    np.testing.assert_allclose(got, [-.5, .1, 0., .6], rtol=0, atol=1e-15)
    x[0, 48] = 10050
    with pytest.raises(AssertionError):
        study.state_masks(x)
