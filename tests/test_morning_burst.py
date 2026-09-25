from __future__ import annotations

import pandas as pd
import pytest

from trade_research.morning_burst import build_batch


def test_morning_burst_ignores_later_bars_and_rejects_missing_minutes(tmp_path):
    rows = []
    for date, missing in (("2024-05-06", False), ("2024-05-07", True)):
        labels = pd.date_range(f"{date} 09:30", f"{date} 11:30", freq="min")
        for stamp in labels:
            if missing and stamp.strftime("%H%M") == "1000":
                continue
            price = 10.2 if stamp.strftime("%H%M") >= "1000" else 10.0
            rows.append({"symbol": "600000", "exchange": "SH",
                         "timestamp": stamp, "close": price,
                         "high": price, "low": price, "volume": 1000,
                         "turnover": price * 1000})
        rows.append({"symbol": "600000", "exchange": "SH",
                     "timestamp": pd.Timestamp(f"{date} 14:55"),
                     "close": 100.0, "high": 100.0, "low": 100.0,
                     "volume": 1000, "turnover": 100000.0})
    source = tmp_path / "source.parquet"
    result = tmp_path / "result.parquet"
    pd.DataFrame(rows).to_parquet(source)

    build_batch([str(source)], result, threads=1)

    frame = pd.read_parquet(result)
    assert frame.date.tolist() == ["2024-05-06"]
    assert frame.code.tolist() == ["sh.600000"]
    assert frame.max5.iloc[0] == pytest.approx(.02)
    assert frame.bar_count.iloc[0] == 121
