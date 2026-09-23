"""Tests that protect the 14:50 information boundary."""

import pandas as pd
from pandas.testing import assert_frame_equal

from trade_research.audit import EXPECTED_LABELS
from trade_research.snapshot import prior_daily_features, snapshot_for_symbol


def _example_history():
    dates = pd.bdate_range("2025-04-01", periods=70).strftime("%Y-%m-%d")
    daily = pd.DataFrame({
        "date": dates,
        "code": "sh.600000",
        "close": [10 + n * 0.01 for n in range(len(dates))],
        "high": [10 + n * 0.01 for n in range(len(dates))],
        "preclose": [10 + (n - 1) * 0.01 for n in range(len(dates))],
        "volume": 1_000_000,
        "amount": 10_000_000,
        "tradestatus": 1,
        "isST": 0,
    })
    signal_date = dates[-1]
    compact_date = signal_date.replace("-", "")
    minute = pd.DataFrame([{
        "date": signal_date,
        "time": compact_date + label + "00000",
        "code": "sh.600000",
        "open": 10.5 + number * 0.001,
        "high": 10.6 + number * 0.001,
        "low": 10.4 + number * 0.001,
        "close": 10.5 + number * 0.001,
        "volume": 10_000,
        "amount": 105_000,
    } for number, label in enumerate(EXPECTED_LABELS)])
    return minute, daily


def test_later_bars_and_final_daily_values_do_not_change_signal():
    minute, daily = _example_history()
    original = snapshot_for_symbol(minute, daily)
    assert len(original) == 1
    assert original.iloc[0]["signal_time"].endswith("145000000")
    later = minute["time"].str[8:12] > "1450"
    minute.loc[later, ["open", "high", "low", "close", "volume", "amount"]] = [
        100, 100, 100, 100, 1_000_000_000, 100_000_000_000
    ]
    current = daily["date"] == daily["date"].iloc[-1]
    daily.loc[current, ["close", "high", "volume", "amount"]] = [
        100, 100, 1_000_000_000, 100_000_000_000
    ]
    changed = snapshot_for_symbol(minute, daily)
    assert_frame_equal(original, changed)


def test_adjusted_prior_mean_uses_current_preopen_reference_factor():
    daily = pd.DataFrame({
        "date": pd.bdate_range("2025-01-01", periods=25).strftime("%Y-%m-%d"),
        "code": "sh.600000", "close": [10.0] * 24 + [5.0],
        "preclose": [10.0] * 24 + [5.0],
        "volume": 1_000_000, "tradestatus": 1, "isST": 0,
    })
    features = prior_daily_features(daily).iloc[-1]
    assert features["ma20_prior"] == 10.0
    assert features["ma20_prior_adjusted"] == 5.0
