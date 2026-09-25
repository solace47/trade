"""Check industry pressure uses only contemporaneous peers, excluding self."""

import duckdb
import pandas as pd

from trade_research.late_sector_pressure_inputs import select_inputs


def test_leave_one_out_peers_and_historical_industry_membership() -> None:
    date = "2024-01-02"
    strong = "sh.600001"
    weak = "sh.600002"
    strong_peers = [f"sh.601{index:03d}" for index in range(10)]
    weak_peers = [f"sh.602{index:03d}" for index in range(10)]
    membership = [
        (strong, "A", "2024-01-01", "2025-01-01"),
        (strong, "B", "2025-01-01", None),
        (weak, "B", "2024-01-01", None),
        *((code, "A", "2024-01-01", None) for code in strong_peers),
        *((code, "B", "2024-01-01", None) for code in weak_peers),
    ]
    history = pd.DataFrame(membership, columns=[
        "code", "industry", "effective_date", "next_effective_date",
    ])
    codes = [strong, weak] + strong_peers + weak_peers
    snapshots = pd.DataFrame([
        {"date": date, "code": code, "return20_prior_adjusted": 0.0,
         "return_1450": -.005 if code in (strong, weak) else 0.0,
         "amount_1450": 500_000_000 if code in (strong, weak)
                        else 30_000_000,
         "price_1450": 10.0, "position_1450": .5, "isST": 0,
         "listing_age_sessions": 100, "reference_gap": False,
         "quote_outside_traded_range": False}
        for code in codes
    ])
    intraday = pd.DataFrame([
        {"date": date, "code": code, "price_1450": 10.0,
         "return_last30": -.005 if code in (strong, weak)
                          else .0012 if code in strong_peers else -.0012}
        for code in codes
    ])
    states = pd.DataFrame([{"date": date, "market_tail_return": 0.0}])
    connection = duckdb.connect()
    connection.register("snapshots", snapshots)
    connection.register("intraday", intraday)
    connection.register("industry_history", history)
    connection.register("market_states", states)
    signals, audit, candidates = select_inputs(connection)
    assert audit["resilient_pool"] == audit["weak_pool"] == 1
    assert audit["paired"] == 1
    assert not audit["outcome_gate_passed"]
    assert set(signals.code) == {strong, weak}
    assert candidates.loc[
        candidates.code.eq(strong), "sector_residual_tail"
    ].iloc[0] > .001
    assert candidates.loc[
        candidates.code.eq(strong), "industry"
    ].iloc[0] == "A"
