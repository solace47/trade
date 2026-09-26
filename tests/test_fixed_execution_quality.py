import pandas as pd
import pytest

from trade_research.fixed_execution_quality import attach_quality, window_quality


def test_reproducible_vwap_does_not_mean_consistent_source_prices():
    raw = pd.DataFrame({"timestamp": pd.date_range("2024-07-01 14:52", periods=4, freq="min"),
        "open": 10., "high": 10.01, "low": 10., "close": 10.01,
        "volume": 10000, "turnover": [101000., 100000., 100000., 100000.]})
    result = window_quality("sh.600001", "2024-07-01", raw)
    assert result["window_status"] == "vwap_outside_bar_range"
    assert result["raw_vwap"] == 10.025
    assert result["aggregate_vwap_outside_window_tolerance"]


def rows():
    return pd.DataFrame({"date": "2024-07-01", "code": ["sh.600001", "sh.600002", "sh.600003"],
        "horizon": 1, "target_notional": 20000, "entry_status": ["filled", "filled", "volume_cap"],
        "unknown_after_buy": [False, True, False], "accounting_exit_date": ["2024-07-02", None, None]})


def windows():
    return pd.DataFrame({"date": ["2024-07-01", "2024-07-02", "2024-07-01"],
        "code": ["sh.600001", "sh.600001", "sh.600002"], "source_checks_passed": [True, False, True],
        "window_status": ["valid", "vwap_outside_bar_range", "valid"]})


def test_source_problem_is_distinct_from_no_buy_and_unknown_terminal_value():
    result = attach_quality(rows(), windows())
    assert result.execution_source_category.tolist() == [
        "booked_but_minute_source_inconsistent", "terminal_value_still_unknown", "not_bought"]
    assert result.unknown_after_buy.tolist() == [False, True, False]
    assert pd.isna(result.sell_source_passed.iloc[1])


def test_missing_booked_leg_cannot_be_disguised_as_consistent():
    with pytest.raises(ValueError, match="lost"):
        attach_quality(rows(), windows().iloc[:1])
