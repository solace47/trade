import numpy as np
import pytest

from trade_research.tail_formula_quantile48 import weighted_quartile


@pytest.mark.parametrize(
    'values,weights,expected',
    [
        ([-.02, -.01, 0., .01], [1., 1., 1., 1.], -.02),
        ([-.02, -.01, 0., .01], [.1, .1, .7, .1], 0.),
        ([-.02, -.02, .01, .01], [1., 1., 1., 1.], -.02),
        ([-.1, .01, .02, .03, .04], [1., .25, .25, .25, .25], -.1),
    ],
)
def test_inverse_cdf_quartile_and_date_weights(values, weights, expected):
    y, w = np.array(values), np.array(weights)
    assert weighted_quartile(y, w) == expected
    assert weighted_quartile(y[::-1], w[::-1]) == expected
    assert weighted_quartile(y, w * 8) == expected


def test_lower_quartile_minimizes_weighted_pinball_loss():
    y = np.array([-.08, -.03, -.003, .002, .01, .12])
    w = np.array([.1, .17, .2, .3, .18, .05])
    quartile = weighted_quartile(y, w)
    assert quartile == -.03

    def loss(prediction):
        error = y - prediction
        return np.sum(w * np.where(error >= 0, .25 * error, -.75 * error))

    alternatives = np.r_[y, (y[:-1] + y[1:]) / 2, y.min() - .1, y.max() + .1]
    assert all(loss(quartile) <= loss(value) + 1e-15 for value in alternatives)
