import numpy as np
import pandas as pd
import pytest

from trade_research.round_number_geometry import controls, fit_day


def inputs(n=300):
    rng = np.random.default_rng(83)
    return pd.DataFrame({"date": "2024-05-06", "half": "2024H1",
        "code": [("sh." if i % 3 else "sz.") + str(600000 + i) for i in range(n)],
        "side": np.where(np.arange(n) % 2, "above", "below"),
        "distance_cents": rng.integers(1, 20, n), "day_return": rng.uniform(-.03, .03, n),
        "return_last29": rng.uniform(-.01, .01, n), "prior20_return": rng.uniform(-.20, .20, n),
        "price_1449": rng.uniform(6, 50, n), "amount_1449": rng.uniform(1e8, 1e9, n)})


def test_input_only_residual_weights_reproduce_conditional_coefficient():
    frame = inputs()
    audit, weighted = fit_day(frame)
    assert audit["passed"]
    matrix, _, _ = controls(frame)
    treatment = weighted.treatment.to_numpy()
    # An artificial response checks the algebra; no market outcome is used.
    response = matrix @ np.linspace(-.01, .01, matrix.shape[1]) + treatment * .02
    full_coefficient = np.linalg.lstsq(np.column_stack([matrix, treatment]), response, rcond=1e-10)[0][-1]
    assert weighted.contrast_weight @ response == pytest.approx(full_coefficient, abs=1e-12)
    assert full_coefficient == pytest.approx(.02, abs=1e-12)
    assert audit["orthogonality_error"] <= 1e-10


def test_only_constant_controls_are_removed():
    frame = inputs()
    frame["day_return"] = .01
    _, names, dropped = controls(frame)
    assert "day_return" in dropped and "day_return_squared" in dropped
    assert "return_last29" in names and "absolute_distance_2" in names


def test_single_side_and_small_cross_sections_are_retained_as_failures():
    frame = inputs()
    frame["side"] = "above"
    audit, rows = fit_day(frame)
    assert not audit["passed"] and not audit["checks"]["both_arms"]
    assert not audit["checks"]["residual_variance"]
    assert rows.contrast_weight.isna().all()
    audit, _ = fit_day(inputs(80))
    assert not audit["passed"] and not audit["checks"]["stock_days"]


def test_duplicate_stock_or_multiple_dates_cannot_create_more_information():
    frame = inputs()
    with pytest.raises(ValueError, match="unique"):
        fit_day(pd.concat([frame, frame.iloc[:1]], ignore_index=True))
    frame.loc[0, "date"] = "2024-05-07"
    with pytest.raises(ValueError, match="decision-date"):
        fit_day(frame)
