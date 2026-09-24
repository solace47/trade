from pathlib import Path

import pandas as pd
import pytest

from trade_research.hf_audit import EXPECTED_LABELS
from trade_research.intraday_features import build_year


def test_intraday_features_stop_at_completed_1450_bar(tmp_path: Path) -> None:
    labels = EXPECTED_LABELS[:231]
    assert labels[-1] == "1450"
    minutes = pd.DataFrame({
        "symbol": "600000", "exchange": "SH",
        "timestamp": pd.to_datetime([f"2024-01-02 {label[:2]}:{label[2:]}"
                                     for label in labels] + ["2024-01-02 14:51"]),
        "close": [10.0 if label < "1421" else 11.0 for label in labels] + [99.0],
        "volume": [100] * len(labels) + [100_000],
        "turnover": [1_000.0 if label < "1421" else 1_100.0
                     for label in labels] + [9_900_000.0],
    })
    source = tmp_path / "minute.parquet"
    destination = tmp_path / "features.parquet"
    minutes.to_parquet(source, index=False)
    build_year([str(source)], 2024, destination, 1)
    features = pd.read_parquet(destination)
    assert len(features) == 1
    row = features.iloc[0]
    assert row["price_1450"] == 11.0
    assert row["return_last30"] == pytest.approx(0.1)
    assert row["volume_share_last30"] == pytest.approx(30 / 231)
    assert row["premium_to_last30_vwap"] == pytest.approx(0)
