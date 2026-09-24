"""Keep the frozen margin matching on same-day observed inputs."""

from __future__ import annotations

import pandas as pd

from pathlib import Path

from trade_research.margin_dtc_study import (
    CONTROL, TREATED, match_controls, write_reprice_signals,
)


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


def test_raw_repricing_keeps_entry_quality_fields(tmp_path: Path) -> None:
    selected = pd.DataFrame([{
        "date": "2024-06-04", "code": "sz.000001", "isST": 0,
        "reference_gap": False, "quote_outside_traded_range": False,
        "listing_age_sessions": 200,
    }])
    source = tmp_path / "selected.parquet"
    output = tmp_path / "signals.parquet"
    selected.to_parquet(source)
    write_reprice_signals(source, output)
    signal = pd.read_parquet(output).iloc[0]
    assert signal.isST == 0
    assert not signal.reference_gap
    assert signal.listing_age_sessions == 200
