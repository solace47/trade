"""Guard the frozen short-interest time join, matching and cooldown."""

from __future__ import annotations

import pandas as pd

from trade_research.short_interest_study import (
    CONTROL, TREATED, _quintiles, select_pairs,
)


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


def test_short_interest_quintiles_rank_short_ratio_within_stratum() -> None:
    frame = pd.DataFrame([
        _row("2024-07-23", f"sz.{number:06d}", 0, number / 1_000_000)
        for number in range(1, 26)
    ]).drop(columns="quintile")
    ranked = _quintiles(frame).sort_values("short_ratio")
    assert ranked.quintile.tolist() == [1] * 5 + [2] * 5 + [3] * 5 + [4] * 5 + [5] * 5


def test_net_flow_pair_requires_positive_high_and_negative_low() -> None:
    high = _row("2025-03-04", "sz.000001", 5, .002)
    high["net_short_flow"] = .003
    low = _row("2025-03-04", "sz.000002", 1, .0001)
    low["net_short_flow"] = -.002
    neutral = _row("2025-03-04", "sz.000003", 1, .0001)
    neutral["net_short_flow"] = 0.0
    selected, report = select_pairs(
        pd.DataFrame([high, neutral, low]), {"2025-03-04": 10},
        score_field="net_short_flow", treated="positive", control="negative",
        signed_flows=True)
    assert report["matched_pairs"] == 1
    assert selected.loc[selected.candidate.eq("positive"),
                        "net_short_flow"].gt(0).all()
    assert selected.loc[selected.candidate.eq("negative"),
                        "net_short_flow"].lt(0).all()


def test_net_flow_pair_matches_starting_short_position() -> None:
    high = _row("2025-03-04", "sz.000001", 5, .002)
    high.update(net_short_flow=.003, prior_short_interest=.001)
    far = _row("2025-03-04", "sz.000002", 1, .0001)
    far.update(net_short_flow=-.002, prior_short_interest=.004)
    near = _row("2025-03-04", "sz.000003", 1, .0001)
    near.update(net_short_flow=-.001, prior_short_interest=.0012)
    selected, _ = select_pairs(
        pd.DataFrame([high, far, near]), {"2025-03-04": 10},
        score_field="net_short_flow", signed_flows=True,
        prior_level_caliper=(.5, 2))
    assert selected.loc[selected.candidate.eq(CONTROL),
                        "code"].tolist() == ["sz.000003"]
