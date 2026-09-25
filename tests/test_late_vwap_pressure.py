"""Check point-in-time pairing and path balance for the VWAP input gate."""

import duckdb
import pandas as pd

from trade_research.late_vwap_pressure import CONTROL, TREATMENT, select_inputs


def test_vwap_pair_requires_same_board_and_last15_path() -> None:
    date = "2024-01-02"
    rows = (
        ("sh.600001", -.004, -.004),
        ("sz.300001", -.0005, -.004),
        ("sh.600002", -.0005, -.003),
        ("sh.600003", -.0005, 0.0),
    )
    snapshots = pd.DataFrame([
        {"date": date, "code": code, "return20_prior_adjusted": 0.0,
         "return_1450": 0.0, "amount_1450": 500_000_000,
         "price_1450": 10.0, "position_1450": .5, "isST": 0,
         "listing_age_sessions": 100, "reference_gap": False,
         "quote_outside_traded_range": False}
        for code, _, _ in rows
    ])
    intraday = pd.DataFrame([
        {"date": date, "code": code, "price_1450": 10.0,
         "return_last30": -.005, "return_last15": last15,
         "premium_to_last30_vwap": premium}
        for code, premium, last15 in rows
    ])
    connection = duckdb.connect()
    connection.register("snapshots", snapshots)
    connection.register("intraday", intraday)

    signals, audit = select_inputs(connection)

    assert signals.loc[signals.candidate.eq(TREATMENT), "code"].tolist() == [
        "sh.600001"]
    assert signals.loc[signals.candidate.eq(CONTROL), "code"].tolist() == [
        "sh.600002"]
    assert audit["same_board_fraction"] == 1.0
    assert not audit["outcome_gate_passed"]
