"""A missing funding lower bound must never become a low-funding signal."""

import pandas as pd

from trade_research.buyback_funding_eval import _attach_funding, _daily
from trade_research.buyback_inputs import CONTROL, EVENT


def test_unknown_plan_and_its_peer_are_both_excluded() -> None:
    rows = pd.DataFrame([
        ("2024-02-01", "sh.600001", EVENT, "sh.600001", .05, .04),
        ("2024-02-01", "sh.600002", CONTROL, "sh.600001", .01, .00),
        ("2024-02-01", "sh.600003", EVENT, "sh.600003", .10, .09),
        ("2024-02-01", "sh.600004", CONTROL, "sh.600003", -.04, -.05),
    ], columns=["date", "code", "candidate", "pair_code",
                "cash_return", "cash_stress10"])
    funding = pd.DataFrame([
        ("2024-02-01", "sh.600001", .012),
        ("2024-02-01", "sh.600003", None),
    ], columns=["date", "code", "floor_float_ratio"])
    kept = _attach_funding(rows, funding)
    assert set(kept.code) == {"sh.600001", "sh.600002"}
    assert set(kept.funding_group) == {"high"}
    daily = _daily(kept)
    assert len(daily) == 1
    assert abs(daily.edge.iloc[0] - .04) < 1e-12
