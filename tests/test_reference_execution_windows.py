"""Morning and tail executions must never share a day's price or capacity."""
import pandas as pd
import pytest

from trade_research.hf_outcomes import Assumptions, EXIT_WINDOWS, outcomes_for_symbol
from trade_research.reference_gain_eval import account_with_windows, quality_windows


DAYS = ["2025-09-01", "2025-09-02", "2025-09-03"]


def inputs():
    signals = pd.DataFrame([{"date": DAYS[0], "code": "sh.600000", "isST": 0,
        "reference_gap": False, "price_1449": 10., "arm": "high", "pair_id": "a"}])
    records = []
    for day in DAYS:
        for window, price, volume in (("close", 10., 250_000), ("morning", 10.4, 50_000)):
            for label in EXIT_WINDOWS[window]:
                records.append({"date": day, "code": "sh.600000", "label": label,
                    "timestamp": pd.Timestamp(day + " " + label[:2] + ":" + label[2:]),
                    "open": price, "high": price, "low": price, "close": price,
                    "volume": volume, "turnover": price * volume})
    minute = pd.DataFrame(records).sort_values("timestamp")
    daily = pd.DataFrame([{"date": day, "tradestatus": 1, "isST": 0,
                          "preclose": 10., "close": 10.} for day in DAYS])
    return signals, minute, daily


def account(signals, minute, daily):
    rows = outcomes_for_symbol(signals, minute, daily, DAYS,
        Assumptions(target_notional=20_000), horizons=(1,),
        exit_labels=EXIT_WINDOWS["morning"], sizing_price_column="price_1449")
    rows["target_notional"] = 20_000
    rows["quality_clean_exit"] = rows.exit_status.eq("filled")
    windows = quality_windows(minute, DAYS, EXIT_WINDOWS["morning"])
    return account_with_windows(rows, signals, windows, DAYS).iloc[0], windows


def test_exit_window_joins_keep_morning_price_and_tail_entry_distinct():
    signals, minute, daily = inputs()
    result, windows = account(signals, minute, daily)
    assert result.entry_price == pytest.approx(10. * 1.0005)
    assert result.exit_price == pytest.approx(10.4 * .9995)
    assert result.entry_raw_vwap == result.entry_window_low == 10.
    assert result.exit_raw_vwap == result.exit_window_high == 10.4
    assert result.execution_source_valid and not result.unknown_after_buy
    assert len(windows) == 2 * len(DAYS)


def test_valid_tail_does_not_hide_invalid_morning_source():
    signals, minute, daily = inputs()
    bad = minute.date.eq(DAYS[1]) & minute.label.eq("0935")
    minute.loc[bad, "turnover"] *= 1.01
    result, _ = account(signals, minute, daily)
    assert result.entry_window_status == "valid"
    assert result.exit_window_status != "valid"
    assert not result.execution_source_valid and result.unknown_after_buy
    assert pd.isna(result.known_return5)


def test_morning_capacity_failure_retries_morning_without_using_tail_volume():
    signals, minute, daily = inputs()
    limited = minute.date.eq(DAYS[1]) & minute.label.isin(EXIT_WINDOWS["morning"])
    minute.loc[limited, "volume"] = 100
    minute.loc[limited, "turnover"] = 1040
    result, _ = account(signals, minute, daily)
    assert result.exit_date == DAYS[2]
    assert result.exit_delay_sessions == 1
    assert result.exit_price == pytest.approx(10.4 * .9995)


def test_legacy_tail_diagnostics_ignore_new_morning_bars():
    _, minute, _ = inputs()
    tail = minute.loc[minute.label.isin(EXIT_WINDOWS["close"])]
    pd.testing.assert_frame_equal(quality_windows(minute, DAYS), quality_windows(tail, DAYS))
    assert "window_role" not in quality_windows(minute, DAYS)
