"""Guard pre-outcome capacity/cooldown and date-weighted evaluation."""

import duckdb
import pandas as pd

from trade_research.down_market_rally_eval import _summary
from trade_research.down_market_rally_inputs import select_inputs


def test_input_selection_caps_each_day_and_cools_both_legs() -> None:
    dates = ("2024-01-02", "2024-01-03", "2026-01-05")
    treatment = [f"sh.6000{index:02d}" for index in range(8)]
    controls = [f"sh.6010{index:02d}" for index in range(8)]
    snapshots = pd.DataFrame([
        {"date": date, "code": code,
         "return20_prior_adjusted": 0.0, "return_1450": 0.0,
         "amount_1450": 500_000_000, "price_1450": 10.0,
         "position_1450": .5, "isST": 0,
         "listing_age_sessions": 100, "reference_gap": False,
         "quote_outside_traded_range": False}
        for date in dates for code in treatment + controls
    ])
    intraday = pd.DataFrame([
        {"date": date, "code": code, "price_1450": 10.0,
         "return_last30": .005 if code in treatment else 0.0}
        for date in dates for code in treatment + controls
    ])
    states = pd.DataFrame([
        {"date": date, "market_state": "down"} for date in dates
    ])
    connection = duckdb.connect()
    connection.register("snapshots", snapshots)
    connection.register("intraday", intraday)
    connection.register("states", states)
    signals, audit = select_inputs(connection)
    assert audit["capacity_candidates"] == 8
    assert audit["paired"] == 8
    assert signals.groupby("date").size().to_dict() == {
        "2024-01-02": 10, "2024-01-03": 6,
    }
    assert signals.code.nunique() == len(signals)
    assert set(signals.date.str[:4]) == {"2024"}


def test_summary_weights_days_equally() -> None:
    pairs = pd.DataFrame([
        {"date": "2024-01-02", "cash_rally": -.03,
         "cash_flat": 0.0, "stress_rally": -.04,
         "stress_flat": -.01, "entry_status_rally": "filled",
         "entry_status_flat": "filled", "valid_rally": True,
         "valid_flat": True},
        {"date": "2024-01-02", "cash_rally": -.01,
         "cash_flat": 0.0, "stress_rally": -.02,
         "stress_flat": -.01, "entry_status_rally": "filled",
         "entry_status_flat": "filled", "valid_rally": True,
         "valid_flat": True},
        {"date": "2024-01-03", "cash_rally": .02,
         "cash_flat": 0.0, "stress_rally": .01,
         "stress_flat": -.01, "entry_status_rally": "filled",
         "entry_status_flat": "filled", "valid_rally": True,
         "valid_flat": True},
    ])
    result = _summary(pairs)
    assert result["pairs"] == 3
    assert result["days"] == 2
    assert abs(result["rally_minus_flat_cash"]) < 1e-12
