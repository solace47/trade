"""A risk coefficient constraint keeps the forecast on its training scale."""

import numpy as np
import pandas as pd
import pytest

from trade_research.absolute_ridge import FEATURES, score
from trade_research.absolute_ridge_late_risk import constrain_late_risk


def test_positive_late_risk_weight_is_removed_without_mean_shift() -> None:
    frame = pd.DataFrame({name: [0.0, 1.0, 2.0] for name in FEATURES})
    median = pd.Series(0.0, index=FEATURES)
    scale = pd.Series(1.0, index=FEATURES)
    coefficients = np.zeros(len(FEATURES))
    coefficients[FEATURES.index("abs_return_last30")] = .02
    coefficients[FEATURES.index("position_1450")] = .01
    model = {"median": median, "scale": scale,
             "coefficients": coefficients, "intercept": 0.0}
    constrained = constrain_late_risk(model, frame, .03)

    assert constrained["coefficients"][FEATURES.index(
        "abs_return_last30")] == 0.0
    assert model["coefficients"][FEATURES.index(
        "abs_return_last30")] == .02
    assert score(frame, constrained).score.mean() == pytest.approx(.03)


def test_negative_late_risk_weight_is_preserved() -> None:
    frame = pd.DataFrame({name: [0.0, 1.0] for name in FEATURES})
    coefficients = np.zeros(len(FEATURES))
    coefficients[FEATURES.index("abs_return_last30")] = -.01
    model = {"median": pd.Series(0.0, index=FEATURES),
             "scale": pd.Series(1.0, index=FEATURES),
             "coefficients": coefficients, "intercept": 0.0}
    constrained = constrain_late_risk(model, frame, -.005)
    assert np.array_equal(constrained["coefficients"], coefficients)
    assert constrained["intercept"] == pytest.approx(0.0)
