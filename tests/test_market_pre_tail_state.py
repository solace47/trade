"""The market trend must end at 14:20 and stay distinct from tail returns."""

import duckdb
import json
import pandas as pd
import pytest

from trade_research.market_pre_tail_eval import evaluate
from trade_research.market_pre_tail_state import select_inputs


def test_pre_tail_return_is_reconstructed_without_last_30_minutes() -> None:
    dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    pre_tail = [-.01, .01, 0.0, -.006]
    tail = [0.0, .001, -.001, .002]
    snapshots = []
    intraday = []
    for date, before, final in zip(dates, pre_tail, tail):
        for n in range(1000):
            code = f"sh.60{n:04d}"
            snapshots.append({
                "date": date, "code": code, "return_1450":
                (1 + before) * (1 + final) - 1,
                "isST": 0, "listing_age_sessions": 30,
                "reference_gap": False, "quote_outside_traded_range": False,
                "amount_1450": 100_000_000, "price_1450": 10.0,
            })
            intraday.append({"date": date, "code": code,
                             "return_last30": final, "price_1450": 10.0})
    connection = duckdb.connect()
    connection.register("snapshots", pd.DataFrame(snapshots))
    connection.register("intraday", pd.DataFrame(intraday))
    signals = pd.DataFrame([
        {"date": date, "code": code, "pair_id": 1,
         "candidate": candidate}
        for date in dates
        for code, candidate in (("sh.600000", "late_decline"),
                                ("sh.600001", "late_rally_control"))
    ])
    states, audit = select_inputs(connection, signals)
    assert states.pre_tail_state.tolist() == ["down", "up", "flat", "down"]
    for expected, actual in zip(pre_tail, states.market_pre_tail_return):
        assert abs(expected - actual) < 1e-12
    assert audit["frozen_pairs"] == 4
    assert not audit["outcome_gate_passed"]


def test_failed_input_gate_keeps_outcomes_closed(tmp_path) -> None:
    (tmp_path / "input_audit.json").write_text(
        json.dumps({"outcome_gate_passed": False}), encoding="utf-8")
    with pytest.raises(ValueError, match="input gate failed"):
        evaluate(tmp_path, tmp_path)
