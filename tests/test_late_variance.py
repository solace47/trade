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


def test_variance_rejects_pre_2024_window(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="2024-2025"):
        build_year(["unused.parquet"], 2023, tmp_path / "variance.parquet")
