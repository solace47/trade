import numpy as np
import pandas as pd

from trade_research.tail_formula_offset_logit48 import date_baselines, newton_update, probabilities


def test_baseline_uses_full_known_pool_before_input_intersection():
    labels = pd.DataFrame(dict(date=['a'] * 4 + ['b'], known15=[True, True, True, False, True],
        opportunity15=[1., 1., 0., np.nan, 0.], formula_input_valid=[True, False, True, True, True]))
    daily = date_baselines(labels).set_index('date')
    assert daily.loc['a', 'rows'] == 3 and daily.loc['a', 'rate'] == 2 / 3
    assert daily.loc['b', 'rows'] == 1 and daily.loc['b', 'rate'] == 0


def test_zero_score_preserves_daily_baseline_and_odds_multiply():
    np.testing.assert_allclose(probabilities([.1, .5, .9], [0., 0., 0.]), [.1, .5, .9], atol=1e-15, rtol=0)
    # Doubling odds takes probability 1/3 to 1/2; it does not add a fixed risk.
    np.testing.assert_allclose(probabilities([1 / 3, .5], [np.log(2), np.log(2)]), [.5, 2 / 3], atol=1e-15, rtol=0)


def test_constant_response_days_remain_exact_and_have_no_newton_effect():
    y = np.array([0., 0., 1., 1.]); w = np.ones(4)
    p = probabilities(y, [-6.4, 6.4, -6.4, 6.4])
    np.testing.assert_array_equal(y, p)
    assert newton_update(y - p, p, w) == 0.


def test_newton_uses_curvature_and_prespecified_symmetric_cap():
    p = np.array([.2, .2]); w = np.array([1., 3.]); residual = np.array([.8, -.2])
    assert abs(newton_update(residual, p, w) - .3125) < 1e-14
    assert newton_update(np.array([.99]), np.array([.01]), np.ones(1)) == 2.
    assert newton_update(np.array([-.99]), np.array([.99]), np.ones(1)) == -2.
