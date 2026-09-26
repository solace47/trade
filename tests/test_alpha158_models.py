import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from trade_research.alpha158_models import fit_robust, transform


def test_training_normalization_and_ridge_agree_with_separate_estimator():
    frame = pd.DataFrame({"a": [1., 2., 3., 4., 100., np.nan],
                          "b": [8., 2., 5., 0., 1., 4.]})
    target = pd.Series([.03, .02, -.01, -.6, .7, -.02])
    model = fit_robust(frame, target)
    medians = frame.median()
    scales = ((frame - medians).abs().median() + 1e-12) * 1.4826
    expected_x = ((frame - medians) / scales).clip(-3, 3).fillna(0)
    reference = Ridge(alpha=.05 * len(frame), fit_intercept=True, solver="svd")
    reference.fit(expected_x, target.clip(-.15, .15))
    np.testing.assert_allclose(model["median"], medians, atol=0, rtol=0)
    np.testing.assert_allclose(model["scale"], scales, atol=0, rtol=0)
    np.testing.assert_allclose(model["coefficients"], reference.coef_, atol=1e-12)
    assert model["intercept"] == pytest.approx(reference.intercept_, abs=1e-12)
    future = pd.DataFrame({"b": [2., -1e8], "a": [np.nan, 1e8]})
    expected = ((future[frame.columns] - medians) / scales).clip(-3, 3).fillna(0)
    np.testing.assert_allclose(transform(future, model), expected, atol=0, rtol=0)


def test_all_missing_training_column_stays_zero_when_future_is_observed():
    frame = pd.DataFrame({"missing": [np.nan] * 4, "constant": [2.] * 4})
    model = fit_robust(frame, pd.Series([.01, .02, .03, .04]))
    assert model["all_missing_columns"] == ["missing"]
    assert model["zero_mad_columns"] == ["missing", "constant"]
    actual = transform(pd.DataFrame({"missing": [99., -99.], "constant": [2., 3.]}), model)
    np.testing.assert_array_equal(actual, [[0., 0.], [0., 3.]])
    np.testing.assert_array_equal(model["coefficients"], [0., 0.])


def test_misaligned_or_nonfinite_training_values_are_rejected():
    frame = pd.DataFrame({"a": [1., 2.]})
    with pytest.raises(ValueError, match="align"):
        fit_robust(frame, pd.Series([.1, .2], index=[1, 0]))
    with pytest.raises(ValueError, match="align"):
        fit_robust(frame, pd.Series([.1, np.nan]))
    with pytest.raises(ValueError, match="Infinite"):
        fit_robust(frame.assign(a=np.inf), pd.Series([.1, .2]))
