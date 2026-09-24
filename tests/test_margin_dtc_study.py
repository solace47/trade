"""Keep the frozen margin matching on same-day observed inputs."""

from __future__ import annotations

import pandas as pd

from trade_research.margin_dtc_study import CONTROL, TREATED, match_controls


def _row(code: str, board: str = "sz_main", size_bucket: int = 3,
         avg20_amount: float = 100_000_000) -> dict:
    return {"date": "2024-06-04", "trade_date": "2024-06-03",
            "code": code, "board": board, "size_bucket": size_bucket,
            "margin_dtc": 10.0 if code == "sz.000001" else .5,
            "avg20_amount": avg20_amount, "amount_1450": 80_000_000.0,
            "float_mv": 10_000_000_000.0,
            "return20_prior_adjusted": .02, "return_1450": .01,
            "open_gap": 0.0, "price_1450": 10.0}


def test_margin_match_requires_same_day_board_and_size_bucket() -> None:
    high = pd.DataFrame([_row("sz.000001")])
    low = pd.DataFrame([
        _row("sh.600001", board="sh_main"),
        _row("sz.300001", size_bucket=1),
        _row("sz.000002"),
    ])
    picked, controls, unmatched = match_controls(high, low)
    assert unmatched == 0
    assert picked.iloc[0].candidate == TREATED
    assert controls.iloc[0].candidate == CONTROL
    assert controls.iloc[0].code == "sz.000002"


def test_margin_match_rejects_outside_market_cap_caliper() -> None:
    high = pd.DataFrame([_row("sz.000001")])
    low = pd.DataFrame([_row("sz.000002")])
    low.loc[0, "float_mv"] = 1_000_000_000.0
    try:
        match_controls(high, low)
    except ValueError as exc:
        assert "No high-margin-DTC signal" in str(exc)
    else:
        raise AssertionError("A tenfold size mismatch was accepted")
