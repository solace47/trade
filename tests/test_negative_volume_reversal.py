from __future__ import annotations

import pandas as pd

from trade_research.negative_volume_reversal import _inputs


def test_history_excludes_current_and_future_minute_volume(tmp_path):
    prefix = tmp_path / "prefix" / "2024"
    snapshots = tmp_path / "snapshots"
    prefix.mkdir(parents=True)
    snapshots.mkdir()
    dates = pd.date_range("2024-01-01", periods=22)
    base = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"), "code": "sh.600000",
        "price_1449": 9.9, "price_1420": 9.9,
        "volume_1449": [100] * 20 + [200, 1000],
        "amount_1449": 150_000_000,
        "return_last29": 0.0,
        "quote_outside_traded_range": False,
    })
    base.to_parquet(prefix / "part_000.parquet")
    pd.DataFrame({
        "date": base.date, "code": base.code, "preclose": 10.0,
        "tradestatus": 1, "isST": 0, "listing_age_sessions": 100,
        "reference_gap": False, "return20_prior_adjusted": 0.0,
    }).to_parquet(snapshots / "2024.parquet")
    first = _inputs(tmp_path / "prefix", snapshots)
    assert first.date.tolist() == ["2024-01-21", "2024-01-22"]
    assert first.loc[0, "volume_median20"] == 100
    assert first.loc[0, "volume_ratio"] == 2
    base.loc[21, "volume_1449"] = 5000
    base.to_parquet(prefix / "part_000.parquet")
    second = _inputs(tmp_path / "prefix", snapshots)
    assert second.loc[0, "volume_ratio"] == first.loc[0, "volume_ratio"]
