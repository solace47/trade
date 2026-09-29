import numpy as np

from trade_research.tail_formula_smooth_network import loss_gradient, native_core, predict, verify_native


def test_weighted_regularized_gradient_matches_finite_differences():
    rng = np.random.default_rng(71)
    x = rng.normal(size=(17, 4)); y = rng.normal(size=17)
    w = np.arange(1, 18, dtype=float); w /= w.sum()
    values = [rng.normal(size=(4, 3)), rng.normal(size=3), rng.normal(size=3), np.array([.2])]
    evaluate = lambda v: loss_gradient(x, y, w, *v[:3], float(v[3][0]), alpha=.017)
    _, analytic = evaluate(values)
    for part, value in enumerate(values):
        for index in np.ndindex(value.shape):
            plus = [v.copy() for v in values]; minus = [v.copy() for v in values]
            plus[part][index] += 1e-6; minus[part][index] -= 1e-6
            difference = (evaluate(plus)[0]-evaluate(minus)[0])/2e-6
            np.testing.assert_allclose(difference, analytic[part][index], rtol=0, atol=2e-9)


def test_native_bounded_exponential_matches_tanh_and_stays_finite():
    from trade_research.tail_formula_float import EXPRESSIONS
    rng = np.random.default_rng(78)
    m = dict(arm='smooth', feature_names=list(EXPRESSIONS), input_means=[0.]*48, input_scales=[1.]*48,
        hidden_weights=rng.normal(size=(48, 8)).tolist(), hidden_bias=[.1]*8,
        coefficients=rng.normal(size=8).tolist(), bias=.2, thresholds=[dict(threshold=.1)])
    x = rng.normal(size=(31, 48))*5
    x[0] = 1e100; x[1] = -1e100
    expected = .2 + np.tanh(np.clip(x, -5, 5) @ np.asarray(m['hidden_weights'])+.1) @ np.asarray(m['coefficients'])
    actual = predict(x, m)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-13)
    verify_native(native_core(m), m)
