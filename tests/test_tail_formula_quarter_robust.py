import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from trade_research.tail_formula_additive import predict
from trade_research.tail_formula_quarter_robust import fit, group_risks, tilted_distribution, tree_dict


def test_zero_tilt_reproduces_all_standard_boosted_trees():
    rng = np.random.RandomState(12)
    x = rng.randint(0, 50, size=(96, 4)).astype('int32')
    y = (x[:, 0] > 20).astype(float) + rng.normal(0, .1, len(x))
    groups = np.repeat(np.arange(4), 24)
    weights = np.tile(np.repeat([.5, .25, .125], 8), 4)
    pars = dict(n_estimators=8, max_depth=3, min_samples_leaf=4, learning_rate=.05,
        criterion='friedman_mse', random_state=20260927, temperature_fraction=.1)
    result = fit(x, y, weights, groups, np.full(4, .25), pars, tilted=False)
    old = GradientBoostingRegressor(loss='squared_error', **{k: v for k, v in pars.items() if k != 'temperature_fraction'}).fit(x, y, sample_weight=weights)
    np.testing.assert_allclose(predict(x, result), old.predict(x), rtol=0, atol=2e-12)
    for got, estimator in zip(result['trees'], old.estimators_.ravel()):
        expected = tree_dict(estimator)
        for key in ['feature', 'threshold', 'children_left', 'children_right', 'n_node_samples']:
            np.testing.assert_array_equal(got[key], expected[key])
        for key in ['value', 'impurity', 'weighted_n_node_samples']:
            np.testing.assert_allclose(got[key], expected[key], rtol=0, atol=2e-12)


def test_tilt_responds_to_relative_improvement_not_initial_hardness():
    initial = np.array([.1, .4, .2, .3])
    priors = np.array([.2, .3, .3, .2])
    probabilities, factors, _ = tilted_distribution(initial, initial, priors, .02)
    np.testing.assert_array_equal(probabilities, priors)
    np.testing.assert_array_equal(factors, np.ones(4))
    improved = initial - np.array([.03, 0, .01, .02])
    p, factors, _ = tilted_distribution(improved, initial, priors, .02)
    assert factors[1] > factors[2] > factors[3] > factors[0]
    np.testing.assert_allclose(p.sum(), 1)


def test_smooth_group_objective_gradient_matches_stage_weights():
    y = np.array([0., .5, 1., -.2, .2, .4, -.1, .3])
    groups = np.repeat(np.arange(4), 2)
    weights = np.array([.5, .5, .25, .75, .8, .2, .4, .6])
    priors = np.full(4, .25)
    initial = group_risks(y, np.zeros(8), weights, groups, 4)
    score = np.linspace(-.03, .1, 8)
    risk = group_risks(y, score, weights, groups, 4)
    probabilities, _, _ = tilted_distribution(risk, initial, priors, .02)
    analytic = 2*(score-y)*weights*probabilities[groups]
    def objective(z):
        return tilted_distribution(group_risks(y, z, weights, groups, 4), initial, priors, .02)[2]
    step = 1e-6
    finite = []
    for i in range(8):
        z = np.eye(8)[i]*step
        finite.append((objective(score+z)-objective(score-z))/(2*step))
    np.testing.assert_allclose(analytic, finite, rtol=0, atol=2e-10)
