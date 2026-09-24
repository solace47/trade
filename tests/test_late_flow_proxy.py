from pathlib import Path

import pandas as pd
import pytest

from trade_research.late_flow_proxy import build_year


def test_proxy_uses_completed_minutes_and_keeps_flat_turnover(tmp_path: Path) -> None:
    timestamps = pd.date_range("2024-01-02 14:20", periods=32, freq="min")
    closes = [10.0]
    closes.extend(10 + i / 100 for i in range(1, 11))
    closes.extend(10.1 - i / 100 for i in range(1, 11))
    closes.extend([10.0] * 10)
    closes.append(99.0)  # 14:51 must not leak into a 14:50 decision.
    turnover = [0.0] + [100.0] * 10 + [200.0] * 10 + [300.0] * 10
    turnover.append(1_000_000.0)
    source = tmp_path / "minutes.parquet"
    output = tmp_path / "proxy.parquet"
    pd.DataFrame({
        "symbol": "600000", "exchange": "SH", "timestamp": timestamps,
        "close": closes, "turnover": turnover,
    }).to_parquet(source, index=False)

    build_year([str(source)], 2024, output, 1)
    frame = pd.read_parquet(output)
    assert len(frame) == 1
    assert frame.iloc[0].price_1450 == pytest.approx(10.0)
    assert frame.iloc[0].signed_turnover_proxy_last30 == pytest.approx(-1 / 6)
    assert frame.iloc[0].moving_turnover_share_last30 == pytest.approx(.5)


def test_proxy_rejects_pre_2024_outcome_window(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="2024-2025"):
        build_year(["unused.parquet"], 2023, tmp_path / "proxy.parquet")
