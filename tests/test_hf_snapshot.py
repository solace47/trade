"""The one-minute signal must be independent of post-14:50 data."""

import pandas as pd
from pandas.testing import assert_frame_equal

from trade_research.hf_audit import EXPECTED_LABELS
from trade_research.hf_snapshot import snapshot_for_symbol


def _history():
    dates = pd.bdate_range("2025-04-01", periods=75).strftime("%Y-%m-%d")
    daily = pd.DataFrame({
        "date": dates, "code": "sh.600000",
        "close": [10 + n * 0.01 for n in range(len(dates))],
        "open": [10 + n * 0.01 for n in range(len(dates))],
        "preclose": [10 + (n - 1) * 0.01 for n in range(len(dates))],
        "volume": 1_000_000, "tradestatus": 1, "isST": 0,
    })
    date = dates[-1]
    minute = pd.DataFrame([{
        "date": date, "timestamp": pd.Timestamp(f"{date} {label[:2]}:{label[2:]}"),
        "label": label, "open": 10.5 + n * 0.001,
        "high": 10.6 + n * 0.001, "low": 10.4 + n * 0.001,
        "close": 10.5 + n * 0.001, "volume": 10_000,
        "turnover": 105_000,
    } for n, label in enumerate(EXPECTED_LABELS)])
    return minute, daily


def test_post_cutoff_records_do_not_affect_signal():
    minute, daily = _history()
    original = snapshot_for_symbol(minute, daily, "sh.600000")
    assert len(original) == 1
    assert original.iloc[0]["signal_time"].strftime("%H:%M") == "14:50"
    later = minute["label"] > "1450"
    minute.loc[later, ["open", "high", "low", "close", "volume", "turnover"]] = [
        100, 100, 100, 100, 1_000_000_000, 100_000_000_000
    ]
    current = daily["date"] == daily["date"].iloc[-1]
    daily.loc[current, ["close", "volume"]] = [100, 1_000_000_000]
    assert_frame_equal(original, snapshot_for_symbol(minute, daily, "sh.600000"))


def test_missing_precutoff_minute_has_no_signal():
    minute, daily = _history()
    minute = minute.loc[minute["label"] != "1449"]
    assert snapshot_for_symbol(minute, daily, "sh.600000").empty


def test_zero_volume_auction_quote_does_not_set_traded_high():
    minute, daily = _history()
    minute.loc[minute["label"] == "0930", ["open", "high", "low", "close", "volume"]] = [
        99, 99, 99, 99, 0
    ]
    result = snapshot_for_symbol(minute, daily, "sh.600000").iloc[0]
    assert result["high_1450"] < 99
    assert result["open_1450"] == daily.iloc[-1]["open"]
