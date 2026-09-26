"""Decision timing, locked liquidation and corporate reference safeguards."""

import pandas as pd
import pytest

from trade_research.hf_outcomes import Assumptions, outcomes_for_symbol
from trade_research.tail_barrier_exit import decision_price, plan_exit, reference_states


DATES = pd.bdate_range("2025-09-01", periods=11).strftime("%Y-%m-%d").tolist()


def bar(price=10., volume=1000):
    return {"open": price, "high": price, "low": price, "close": price, "volume": volume}


def inputs():
    entry = {"date": DATES[0], "code": "sh.600000", "entry_status": "filled", "entry_price": 10.}
    bars = {d: bar() for d in DATES}
    refs = {d: "ordinary" for d in DATES}
    return entry, bars, refs


@pytest.mark.parametrize("price,reason", [(10.31, "take_profit"), (9.69, "stop_loss")])
def test_first_observable_threshold_stays_locked(price, reason):
    entry, bars, refs = inputs()
    bars[DATES[2]] = bar(price)
    bars[DATES[3]] = bar(10.)
    target, trace = plan_exit(entry, bars, refs, DATES)
    assert target["planned_horizon"] == 2
    assert target["decision_date"] == DATES[2]
    assert target["decision_reason"] == reason
    assert len(trace) == 2


def test_no_same_day_sale_or_future_day_five_trigger():
    entry, bars, refs = inputs()
    bars[DATES[0]] = bar(11.)
    bars[DATES[5]] = bar(11.)
    target, trace = plan_exit(entry, bars, refs, DATES)
    assert target["planned_horizon"] == 5
    assert target["decision_reason"] == "time_exit"
    assert [r["decision_date"] for r in trace] == DATES[1:5]


def test_missing_or_zero_volume_quote_waits_until_valid_day():
    entry, bars, refs = inputs()
    del bars[DATES[1]]
    bars[DATES[2]] = bar(11., 0)
    bars[DATES[3]] = bar(10.31)
    target, trace = plan_exit(entry, bars, refs, DATES)
    assert target["planned_horizon"] == 3
    assert [r["quote_status"] for r in trace] == ["missing_1449", "no_positive_volume", "valid"]


@pytest.mark.parametrize("state", ["reference_gap", "reference_unknown"])
def test_discontinuous_reference_disables_raw_price_barrier(state):
    entry, bars, refs = inputs()
    bars[DATES[1]] = bar(9.)
    refs[DATES[1]] = state
    target, trace = plan_exit(entry, bars, refs, DATES)
    assert target["planned_horizon"] == 5
    assert target["decision_reason"] == "reference_fallback"
    assert len(trace) == 1


def test_future_reference_gap_cannot_cancel_already_triggered_sale():
    entry, bars, refs = inputs()
    bars[DATES[1]] = bar(9.69)
    refs[DATES[2]] = "reference_gap"
    assert plan_exit(entry, bars, refs, DATES)[0]["planned_horizon"] == 1


def test_unbought_attempt_is_retained_without_a_price_decision():
    entry, bars, refs = inputs()
    entry["entry_status"] = "volume_cap"
    target, trace = plan_exit(entry, bars, refs, DATES)
    assert target["decision_reason"] == "not_bought"
    assert not trace


def test_today_close_is_not_used_for_today_reference_state():
    daily = pd.DataFrame({"date": DATES[:3], "tradestatus": [1, 1, 1],
        "preclose": [10., 10., 10.], "close": [10., 10., 10.]})
    before = reference_states(daily)
    daily.loc[1, "close"] = 500.
    after = reference_states(daily)
    assert before[DATES[1]] == after[DATES[1]] == "ordinary"
    assert after[DATES[2]] == "reference_gap"


@pytest.mark.parametrize("patch,status", [({"close": 10.1}, "inconsistent_ohlc"),
    ({"close": 10.001}, "off_tick_ohlc"), ({"low": float("nan")}, "invalid_ohlc")])
def test_invalid_1449_price_does_not_generate_a_decision(patch, status):
    quote = bar(); quote.update(patch)
    assert decision_price(quote) == (None, status)


def test_latched_sale_retries_after_capacity_failure_at_actual_later_price():
    entry, bars, refs = inputs()
    bars[DATES[1]] = bar(9.69)
    target, _ = plan_exit(entry, bars, refs, DATES)
    signals = pd.DataFrame([{**entry, "isST": 0, "reference_gap": False, "price_1449": 10.}])
    minutes = pd.DataFrame([{"date": day, "label": label,
        "volume": 100 if day == DATES[1] else 100000,
        "turnover": (100 if day == DATES[1] else 100000) * (9.5 if day == DATES[1] else 10.)}
        for day in DATES for label in ("1452", "1453", "1454", "1455")])
    daily = pd.DataFrame({"date": DATES, "tradestatus": 1, "isST": 0, "preclose": 10., "close": 10.})
    result = outcomes_for_symbol(signals, minutes, daily, DATES,
        Assumptions(target_notional=20000, maximum_exit_delay_sessions=9),
        horizons=(target["planned_horizon"],), sizing_price_column="price_1449").iloc[0]
    assert result.exit_date == DATES[2]
    assert result.exit_delay_sessions == 1
    assert result.exit_price == 10. * .9995


def test_fourth_day_exit_is_supported_without_changing_default_horizons():
    entry, bars, refs = inputs()
    bars[DATES[4]] = bar(10.31)
    target, _ = plan_exit(entry, bars, refs, DATES)
    assert target["planned_horizon"] == 4
    signals = pd.DataFrame([{**entry, "isST": 0, "reference_gap": False, "price_1449": 10.}])
    minutes = pd.DataFrame([{"date": day, "label": label, "volume": 100000, "turnover": 1e6}
        for day in DATES for label in ("1452", "1453", "1454", "1455")])
    daily = pd.DataFrame({"date": DATES, "tradestatus": 1, "isST": 0, "preclose": 10., "close": 10.})
    result = outcomes_for_symbol(signals, minutes, daily, DATES, horizons=(4,)).iloc[0]
    assert result.exit_date == DATES[4]
