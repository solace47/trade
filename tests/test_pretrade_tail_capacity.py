"""Keep the execution screen fully computable before order submission."""

import pandas as pd
import pytest

from trade_research.pretrade_tail_capacity import (
    build_last5_year, predict_capacity,
)


def test_capacity_prediction_uses_only_prior_volume_and_round_lots() -> None:
    price = pd.Series([10.0, 10.0, 1200.0])
    recent = pd.Series([400_000, 300_000, 1_000_000])
    result = predict_capacity(price, recent)
    assert result.planned_shares.tolist() == [10_000, 10_000, 0]
    assert result.predicted_feasible.tolist() == [True, False, False]
    five = predict_capacity(pd.Series([10.0, 10.0]),
                            pd.Series([130_000, 100_000]), 5)
    assert five.predicted_feasible.tolist() == [True, False]
    with pytest.raises(ValueError, match="Invalid"):
        predict_capacity(pd.Series([0.0]), pd.Series([100_000]))


def test_last_five_builder_requires_all_completed_labels(tmp_path) -> None:
    root = tmp_path / "minutes"
    for exchange in ("SH", "SZ"):
        (root / exchange).mkdir(parents=True)
    records = [
        {"timestamp": pd.Timestamp(f"2024-07-01 14:{minute:02d}"),
         "exchange": "SH", "symbol": "600001", "volume": 100}
        for minute in range(46, 52)
    ]
    records.extend([
        {"timestamp": pd.Timestamp(f"2024-07-02 14:{minute:02d}"),
         "exchange": "SH", "symbol": "600001", "volume": 100}
        for minute in range(46, 50)
    ])
    pd.DataFrame(records).to_parquet(root / "SH" / "stock.parquet")
    pd.DataFrame([{"timestamp": pd.Timestamp("2023-07-01 14:50"),
                   "exchange": "SZ", "symbol": "000001", "volume": 100}]
                 ).to_parquet(root / "SZ" / "old.parquet")
    result = pd.read_parquet(build_last5_year(2024, root, tmp_path / "output"))
    assert result[["date", "volume_last5", "bar_count"]].to_records(
        index=False).tolist() == [("2024-07-01", 500, 5)]
