"""Source audit handles zero-volume quotes and the official close correctly."""

import pandas as pd

from trade_research.hf_audit import EXPECTED_LABELS, audit_symbol


def test_zero_volume_open_quote_and_official_close(tmp_path):
    date = "2025-09-01"
    bars = pd.DataFrame([{
        "symbol": "600000", "exchange": "SH",
        "timestamp": pd.Timestamp(f"{date} {label[:2]}:{label[2:]}"),
        "open": 10.0, "high": 11.0, "low": 10.0, "close": 10.0,
        "volume": 100, "turnover": 1000.0,
    } for label in EXPECTED_LABELS])
    bars.loc[0, ["open", "high", "low", "close", "volume", "turnover"]] = [
        99, 99, 99, 99, 0, 0
    ]
    bars.loc[len(bars) - 1, ["open", "high", "low", "close", "volume", "turnover"]] = [
        10, 10.5, 10, 10.5, 0, 0
    ]
    minute_file = tmp_path / "minute.parquet"
    daily_file = tmp_path / "daily.parquet"
    bars.to_parquet(minute_file)
    pd.DataFrame([{
        "date": date, "code": "sh.600000", "tradestatus": 1,
        "open": 10.0, "high": 11.0, "low": 10.0, "close": 10.5,
        "volume": 23900, "amount": 239000.0,
    }]).to_parquet(daily_file)
    result, issues = audit_symbol("sh.600000", minute_file, daily_file, date, date)
    assert result["complete_days"] == 1
    assert result["zero_volume_auction_days"] == 1
    assert result["ohlc_mismatch_days"] == 0
    assert issues == []
