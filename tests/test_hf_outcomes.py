"""Execution checks for T+1, estimated limits, and transaction costs."""

import pandas as pd

from trade_research.hf_outcomes import _board_limit_rate, _order_shares, outcomes_for_symbol


DATES = ["2025-09-01", "2025-09-02", "2025-09-03"]


def _inputs(entry=10.0, second=10.0, third=10.0):
    signals = pd.DataFrame([{
        "date": DATES[0], "code": "sh.600000", "isST": 0,
        "reference_gap": False,
    }])
    minute = pd.DataFrame([{
        "date": date, "label": label, "open": price,
        "volume": 250_000, "turnover": price * 250_000,
    } for date, price in zip(DATES, (entry, second, third))
        for label in ("1452", "1453", "1454", "1455")])
    daily = pd.DataFrame([{
        "date": date, "tradestatus": 1, "isST": 0,
        "preclose": prior, "close": close,
    } for date, prior, close in zip(DATES,
                                    (10.0, 10.0, 10.0),
                                    (10.0, 10.0, 10.0))])
    return signals, minute, daily


def test_t_plus_one_exit_and_costs():
    signals, minute, daily = _inputs()
    outcomes = outcomes_for_symbol(signals, minute, daily, DATES)
    one = outcomes.loc[outcomes["horizon"] == 1].iloc[0]
    assert one["entry_status"] == "filled"
    assert one["exit_status"] == "filled"
    assert one["exit_date"] == DATES[1]
    assert one["gross_return"] < 0  # Slippage on both sides.
    assert one["net_return"] < one["gross_return"]


def test_limit_up_entry_and_limit_down_exit_are_not_filled():
    signals, minute, daily = _inputs(entry=11.0)
    outcomes = outcomes_for_symbol(signals, minute, daily, DATES)
    assert set(outcomes["entry_status"]) == {"estimated_upper_limit"}
    assert set(outcomes["exit_status"]) == {"entry_not_filled", "right_censored"}

    signals, minute, daily = _inputs(second=9.0)
    outcomes = outcomes_for_symbol(signals, minute, daily, DATES)
    one = outcomes.loc[outcomes["horizon"] == 1].iloc[0]
    assert one["exit_date"] == DATES[2]
    assert one["exit_delay_sessions"] == 1


def test_corporate_action_crossing_invalidates_raw_return():
    signals, minute, daily = _inputs()
    daily.loc[daily["date"] == DATES[1], "preclose"] = 9.0
    outcomes = outcomes_for_symbol(signals, minute, daily, DATES)
    one = outcomes.loc[outcomes["horizon"] == 1].iloc[0]
    assert one["exit_status"] == "corporate_action_unadjusted"
    assert pd.isna(one["net_return"])


def test_historical_chinext_rule_and_star_order_quantity():
    assert _board_limit_rate("sz.300001", 0, "2020-08-21") == 0.1
    assert _board_limit_rate("sz.300001", 0, "2020-08-24") == 0.2
    assert _order_shares("sh.688001", 10.0, 20_050) == 2005


def test_new_listing_window_is_excluded_from_estimated_fills():
    signals, minute, daily = _inputs()
    signals["listing_age_sessions"] = 2
    outcomes = outcomes_for_symbol(signals, minute, daily, DATES)
    assert set(outcomes["entry_status"]) == {"new_listing_window"}
