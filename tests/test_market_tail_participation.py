"""The market participation baseline must exclude the signal day."""

import duckdb
import pandas as pd

from trade_research.market_tail_participation import select_inputs


def test_tail_amount_baseline_uses_only_prior_valid_days() -> None:
    dates = pd.bdate_range("2024-01-02", periods=21).strftime("%Y-%m-%d")
    snapshots = []
    intraday = []
    for day, date in enumerate(dates):
        # The preceding ten 10m and ten 20m bars have a 15m median.
        share = .10 if day < 10 else .20 if day < 20 else .30
        for n in range(1000):
            code = f"sh.60{n:04d}"
            snapshots.append({
                "date": date, "code": code, "isST": 0,
                "listing_age_sessions": 30, "price_1450": 10.0,
                "amount_1450": 100_000_000, "reference_gap": False,
                "quote_outside_traded_range": False,
            })
            intraday.append({
                "date": date, "code": code, "price_1450": 10.0,
                "amount_share_last30": share, "return_last30": 0.0,
            })
    connection = duckdb.connect()
    connection.register("snapshots", pd.DataFrame(snapshots))
    connection.register("intraday", pd.DataFrame(intraday))
    signals = pd.DataFrame([
        {"date": dates[-1], "code": code, "pair_id": "sh.600000",
         "candidate": candidate}
        for code, candidate in (("sh.600000", "late_decline"),
                                ("sh.600001", "late_rally_control"))
    ])
    variance = pd.DataFrame({"date": [dates[-1]], "market_surprise": [1.0]})
    states, audit = select_inputs(connection, signals, variance)
    assert len(states) == 1
    assert states.iloc[0].stocks == 1000
    assert abs(states.iloc[0].market_participation - 2.0) < 1e-12
    assert states.iloc[0].participation_state == "high"
    assert audit["classified_pairs"] == 1
    assert not audit["outcome_gate_passed"]
