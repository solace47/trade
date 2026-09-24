"""Guard the frozen short-interest time join, matching and cooldown."""

from __future__ import annotations

import pandas as pd

from trade_research.short_interest_study import CONTROL, TREATED, select_pairs


def _row(day: str, code: str, quintile: int, ratio: float,
         board: str = "sz_main") -> dict:
    return {"date": day, "trade_date": "2024-07-22", "code": code,
            "board": board, "size_bucket": 2, "quintile": quintile,
            "short_ratio": ratio, "float_mv": 10_000_000_000.0,
            "avg20_amount": 200_000_000.0, "amount_1450": 180_000_000.0,
            "return20_prior_adjusted": .02, "return_1450": .01,
            "open_gap": .002, "price_1450": 12.0}


def test_selection_uses_same_day_control_and_ten_full_session_cooldown() -> None:
    rows = []
    days = ["2024-07-23", "2024-07-24", "2024-08-07"]
    for day in days:
        rows.extend([_row(day, "sz.000001", 5, .002),
                     _row(day, "sz.000002", 1, .0001),
                     _row(day, "sh.600002", 1, .0001, "sh_main")])
    frame = pd.DataFrame(rows)
    selected, report = select_pairs(
        frame, {days[0]: 10, days[1]: 11, days[2]: 21})
    high = selected.loc[selected.candidate.eq(TREATED)]
    low = selected.loc[selected.candidate.eq(CONTROL)]
    assert high.date.tolist() == [days[0], days[2]]
    assert low.code.tolist() == ["sz.000002", "sz.000002"]
    assert report["matched_pairs"] == 2


def test_pair_rejects_future_or_same_day_margin_record() -> None:
    frame = pd.DataFrame([_row("2024-07-23", "sz.000001", 5, .002),
                          _row("2024-07-23", "sz.000002", 1, .0001)])
    frame["trade_date"] = "2024-07-23"
    try:
        select_pairs(frame, {"2024-07-23": 10})
    except ValueError as exc:
        assert "future-informed" in str(exc)
    else:
        raise AssertionError("Same-session margin balance reached the signal")
