import duckdb
import numpy as np
import pandas as pd
import pytest

from trade_research.execution_drift import FEATURES, _candidates, _slope_and_interval


def test_candidate_never_uses_later_snapshot_price() -> None:
    prefix = pd.DataFrame({
        "date": ["2023-12-29", "2024-01-02", "2025-01-02", "2026-01-02"],
        "code": ["sh.600000"] * 4,
        "price_1420": [10.0] * 4, "price_1435": [10.0] * 4,
        "price_1449": [10.02] * 4, "amount_1449": [200_000_000] * 4,
        "high_1449": [10.1] * 4, "low_1449": [9.9] * 4,
        "return_last29": [.002] * 4, "return_last14": [.002] * 4,
        "quote_outside_traded_range": [False] * 4,
    })
    snapshots = pd.DataFrame({
        "date": prefix.date, "code": prefix.code,
        "preclose": [10.0] * 4, "isST": [0] * 4,
        "tradestatus": [1] * 4, "listing_age_sessions": [100] * 4,
        "reference_gap": [False] * 4,
        "return20_prior_adjusted": [0.0] * 4,
        "price_1450": [10.0] * 4, "amount_1450": [300_000_000] * 4,
    })
    connection = duckdb.connect()
    try:
        connection.register("prefix", prefix)
        connection.register("snapshots", snapshots)
        first = _candidates(connection)
        snapshots.price_1450 = 100.0
        snapshots.amount_1450 = 999_999_999.0
        connection.unregister("snapshots")
        connection.register("snapshots", snapshots)
        second = _candidates(connection)
    finally:
        connection.close()
    assert first.date.tolist() == ["2024-01-02", "2025-01-02"]
    pd.testing.assert_frame_equal(first, second)


def test_regression_recovers_within_day_price_drift_slope() -> None:
    rng = np.random.default_rng(4)
    rows = []
    for date in pd.date_range("2024-01-02", periods=98, freq="D"):
        for stock in range(30):
            features = rng.normal(size=len(FEATURES))
            rows.append({"date": date.strftime("%Y-%m-%d"), "board": "main",
                         "code": f"sh.60{stock:04d}",
                         **dict(zip(FEATURES, features, strict=True)),
                         "drift_bps": -8 * features[0] + 3 * features[1]
                         + rng.normal(scale=.01) + date.day % 5})
    result = _slope_and_interval(pd.DataFrame(rows), draws=100)
    assert result["tail_bps_per_pp"] == pytest.approx(-8, abs=.01)
    assert result["week_bootstrap_95"][1] < 0
