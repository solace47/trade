import copy

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from trade_research import tail_formula_polynomial_ridge as study


def test_complete_quadratic_dictionary_includes_squares_and_each_pair_once():
    terms = study.dictionary(48, 'quadratic')
    assert terms[:48] == [[i] for i in range(48)]
    assert len(terms) == 1224 and len({tuple(t) for t in terms}) == 1224
    assert terms[48:51] == [[0, 0], [0, 1], [0, 2]]
    assert terms[-1] == [47, 47]
    assert sum(len(t) == 2 and t[0] == t[1] for t in terms) == 48


def test_quadratic_recovers_interaction_with_no_marginal_linear_signal(monkeypatch):
    x = np.ones((16, 48)) * 10000
    x[:, :2] += np.tile([[-1, -1], [-1, 1], [1, -1], [1, 1]], (4, 1))
    y = (x[:, 0] - 10000) * (x[:, 1] - 10000)
    with threadpool_limits(limits=2):
        linear = study.fit_numeric(x, y, np.ones(len(x)), 'linear', block=3)
        quadratic = study.fit_numeric(x, y, np.ones(len(x)), 'quadratic', block=3)
    np.testing.assert_allclose(study.predict(x, linear), 0, atol=1e-12)
    np.testing.assert_allclose(study.predict(x, quadratic), .5 * y, atol=1e-12)
    assert quadratic['input_constant_indices'] == list(range(2, 48))
    assert min(quadratic['term_scales']) > 0
    quadratic['feature_names'] = list(study.original.EXPRESSIONS)
    quadratic['thresholds'] = [dict(id=0, training_quantile=.995, threshold=.4)]
    monkeypatch.setattr(study.base, 'HEADER', study.original.HEADER)
    monkeypatch.setattr(study.base, 'EXPRESSIONS', study.original.EXPRESSIONS)
    core = study.native_core(quadratic)
    study.verify_native(core, quadratic)
    changed = copy.deepcopy(quadratic)
    changed['coefficients'][49] += .01
    with pytest.raises(AssertionError):
        study.verify_native(core, changed)


def test_weighted_fit_matches_independent_estimator_and_evaluation_is_clipped():
    rng = np.random.default_rng(38)
    x = rng.integers(0, 20000, (137, 4)).astype(float)
    x[:, 3] = 9200
    y = rng.normal(size=len(x))
    w = rng.uniform(.1, 2, size=len(x))
    w /= w.sum()
    m = study.fit_numeric(x, y, w, 'quadratic', block=17)
    mean = np.average(x, weights=w, axis=0)
    scale = np.sqrt(np.average((x - mean) ** 2, weights=w, axis=0))
    scale[scale <= 1e-12] = 1
    z = np.maximum(-5, np.minimum(5, (x - mean) / scale))
    b = np.column_stack([*z.T, *(z[:, i] * z[:, j] for i in range(4) for j in range(i, 4))])
    bm = np.average(b, weights=w, axis=0)
    bs = np.sqrt(np.average((b - bm) ** 2, weights=w, axis=0))
    bs[bs <= 1e-12] = 1
    normalized = (b - bm) / bs
    reference = Ridge(alpha=1, fit_intercept=True, solver='cholesky').fit(normalized, y, sample_weight=w)
    np.testing.assert_allclose(m['coefficients'], reference.coef_, rtol=0, atol=2e-12)
    np.testing.assert_allclose(study.predict(x, m), reference.predict(normalized), rtol=0, atol=2e-12)
    future = x[:2].copy()
    future[:, 0] = [1e12, -1e12]
    clipped = study.standardized(future, m)
    np.testing.assert_array_equal(clipped[:, 0], [5, -5])
    assert m['input_means'] == mean.tolist()
