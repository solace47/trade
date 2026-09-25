import math
from pathlib import Path

import pandas as pd
import pytest

from trade_research.late_variance import build_year


def test_variance_uses_only_completed_last_thirty_minutes(tmp_path: Path) -> None:
    labels = pd.date_range("2024-01-02 14:20", periods=32, freq="min")
    closes = [10.0] + [10.1] * 30 + [99.0]
    source = tmp_path / "minutes.parquet"
    output = tmp_path / "variance.parquet"
    pd.DataFrame({
        "symbol": "600000", "exchange": "SH", "timestamp": labels,
        "close": closes,
    }).to_parquet(source, index=False)

    build_year([str(source)], 2024, output, 1)
    frame = pd.read_parquet(output)
    assert len(frame) == 1
    assert frame.iloc[0].price_1450 == pytest.approx(10.1)
    assert frame.iloc[0].realized_variance_last30 == pytest.approx(
        math.log(10.1 / 10.0) ** 2
    )
    assert frame.iloc[0].downside_share_last30 == 0


def test_signed_variance_partitions_completed_returns(tmp_path: Path) -> None:
    labels = pd.date_range("2024-01-02 14:20", periods=32, freq="min")
    closes = [10.0, 10.1, 9.9, 10.0] + [10.0] * 27 + [99.0]
    source = tmp_path / "minutes.parquet"
    output = tmp_path / "variance.parquet"
    pd.DataFrame({
        "symbol": "600000", "exchange": "SH", "timestamp": labels,
        "close": closes,
    }).to_parquet(source, index=False)

    build_year([str(source)], 2024, output, 1)
    row = pd.read_parquet(output).iloc[0]
    downside = math.log(9.9 / 10.1) ** 2
    upside = math.log(10.1 / 10.0) ** 2 + math.log(10.0 / 9.9) ** 2
    assert row.price_1450 == pytest.approx(10.0)
    assert row.downside_variance_last30 == pytest.approx(downside)
    assert row.upside_variance_last30 == pytest.approx(upside)
    assert row.realized_variance_last30 == pytest.approx(downside + upside)
    assert row.downside_share_last30 == pytest.approx(downside / (downside + upside))
    assert row.down_moves_last30 == 1
    assert row.up_moves_last30 == 2


def test_variance_rejects_pre_2024_window(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="2024-2025"):
        build_year(["unused.parquet"], 2023, tmp_path / "variance.parquet")
