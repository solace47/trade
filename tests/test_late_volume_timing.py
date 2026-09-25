"""Check the daily input cap before volume-timing outcomes are available."""

import hashlib

import duckdb
import pandas as pd

from trade_research.late_volume_timing import select_inputs


def test_volume_timing_does_not_scan_past_five_unmatched_candidates() -> None:
    date = "2024-01-02"
    codes = [f"sh.60000{number}" for number in range(6)]
    ranked = sorted(codes, key=lambda code: hashlib.md5(
        f"late-volume-time-v1{date}{code}".encode()).hexdigest())
    current = {ranked[0]: 0.0, ranked[-1]: .02}
    current.update({code: -.03 for code in ranked[1:-1]})
    current.update({"sh.601001": 0.0, "sh.601002": .02})
    snapshots = pd.DataFrame([
        {"date": date, "code": code, "return20_prior_adjusted": 0.0,
         "return_1450": day_return, "amount_1450": 500_000_000,
         "price_1450": 10.0, "position_1450": .5, "isST": 0,
         "listing_age_sessions": 100, "reference_gap": False,
         "quote_outside_traded_range": False}
        for code, day_return in current.items()
    ])
    intraday = pd.DataFrame([
        {"date": date, "code": code, "price_1450": 10.0,
         "return_last30": -.005, "return_last15": -.001,
         "volume_share_last30": .10,
         "volume_share_last15": .07 if code.startswith("sh.601") else .04}
        for code in current
    ])
    connection = duckdb.connect()
    connection.register("snapshots", snapshots)
    connection.register("intraday", intraday)
    signals, audit = select_inputs(connection)
    assert audit["capacity_candidates"] == 5
    assert audit["paired"] == 1
    assert signals.loc[signals.candidate.eq("dispersed_decline"), "code"].tolist() == [
        ranked[0]]
