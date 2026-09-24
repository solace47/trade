"""Check the two-benchmark rolling regression calculation."""

import numpy as np
import pandas as pd

from trade_research.price_nonsynch import r_squared_from_moments


def test_r_squared_from_moments_matches_direct_regression() -> None:
    rng = np.random.default_rng(81)
    x1 = rng.normal(size=120)
    x2 = rng.normal(size=120)
    y = 1.5 * x1 - .7 * x2 + rng.normal(scale=.4, size=120)
    moments = pd.DataFrame([{
        "n": 120, "span": 119,
        "sum_y": y.sum(), "sum_x1": x1.sum(), "sum_x2": x2.sum(),
        "sum_y_sq": np.square(y).sum(),
        "sum_x1_sq": np.square(x1).sum(),
        "sum_x2_sq": np.square(x2).sum(),
        "sum_x1_x2": (x1 * x2).sum(),
        "sum_x1_y": (x1 * y).sum(),
        "sum_x2_y": (x2 * y).sum(),
    }])
    fitted = np.column_stack((np.ones(120), x1, x2)) @ np.linalg.lstsq(
        np.column_stack((np.ones(120), x1, x2)), y, rcond=None)[0]
    direct = 1 - np.square(y - fitted).sum() / np.square(y - y.mean()).sum()
    assert np.isclose(r_squared_from_moments(moments)[0], direct, atol=1e-12)
    moments.loc[0, "span"] = 140
    assert np.isnan(r_squared_from_moments(moments)[0])
