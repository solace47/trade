"""Execution checks for T+1, estimated limits, and transaction costs."""

import pandas as pd

from trade_research.hf_outcomes import (
    Assumptions, EXIT_WINDOWS, _board_limit_rate, _fees, _order_shares,
    outcomes_for_symbol,
)


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


def test_next_morning_exit_keeps_same_tail_entry() -> None:
    signals, minute, daily = _inputs()
    morning = pd.DataFrame([{
        "date": DATES[1], "label": label,
        "volume": 250_000, "turnover": 10.2 * 250_000,
    } for label in EXIT_WINDOWS["morning"]])
    minute = pd.concat([minute, morning], ignore_index=True)

    next_morning = outcomes_for_symbol(
        signals, minute, daily, DATES,
        exit_labels=EXIT_WINDOWS["morning"],
    ).loc[lambda frame: frame.horizon.eq(1)].iloc[0]
    next_close = outcomes_for_symbol(signals, minute, daily, DATES).loc[
        lambda frame: frame.horizon.eq(1)
    ].iloc[0]

    assert next_morning["entry_price"] == next_close["entry_price"]
    assert next_morning["exit_date"] == DATES[1]
    assert next_morning["exit_delay_sessions"] == 0
    assert next_morning["exit_label"] == "0935-0938"
    assert next_morning["net_return"] > next_close["net_return"]


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


def test_mainboard_risk_warning_limit_changes_in_july_2026():
    for code in ("sh.600000", "sz.000001"):
        assert _board_limit_rate(code, 1, "2026-07-03") == 0.05
        assert _board_limit_rate(code, 1, "2026-07-06") == 0.1


def test_new_listing_window_is_excluded_from_estimated_fills():
    signals, minute, daily = _inputs()
    signals["listing_age_sessions"] = 2
    outcomes = outcomes_for_symbol(signals, minute, daily, DATES)
    assert set(outcomes["entry_status"]) == {"new_listing_window"}


def test_stamp_tax_uses_the_actual_sale_date():
    assumptions = Assumptions()
    before = _fees(100_000, "sell", assumptions, "2023-08-25")
    after = _fees(100_000, "sell", assumptions, "2023-08-28")
    assert round(before - after, 2) == 50.00


def test_transfer_fee_uses_each_execution_date():
    assumptions = Assumptions()
    for side in ("buy", "sell"):
        before = _fees(100_000, side, assumptions, "2022-04-28")
        after = _fees(100_000, side, assumptions, "2022-04-29")
        assert round(before - after, 2) == 1.00


def test_optional_twenty_session_exit_uses_actual_twentieth_session():
    calendar = pd.bdate_range("2025-09-01", periods=27).strftime("%Y-%m-%d").tolist()
    signals = pd.DataFrame([{"date": calendar[0], "code": "sh.600000",
                             "isST": 0, "reference_gap": False}])
    minute = pd.DataFrame([{
        "date": date, "label": label, "volume": 250_000,
        "turnover": 10.0 * 250_000,
    } for date in calendar for label in ("1452", "1453", "1454", "1455")])
    daily = pd.DataFrame([{
        "date": date, "tradestatus": 1, "isST": 0,
        "preclose": 10.0, "close": 10.0,
    } for date in calendar])
    result = outcomes_for_symbol(signals, minute, daily, calendar,
                                 horizons=(20,)).iloc[0]
    assert result["target_exit_date"] == calendar[20]
    assert result["exit_date"] == calendar[20]
    assert result["exit_status"] == "filled"
