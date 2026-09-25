from __future__ import annotations

import pandas as pd
import pytest

from trade_research.morning_path import audit


def test_path_groups_ignore_old_burst_label_and_future_fields():
    rows = []
    for date in ("2024-05-06", "2024-10-08", "2025-05-06", "2025-10-08"):
        for index in range(10):
            rows.append({
                "date": date, "code": f"sh.600{index:03d}",
                "board": "main", "max5": .012,
                "morning_pp": 2.0 if index < 5 else .25,
                "day_pp": 1.25, "tail_pp": 0.0,
                "prior20_pp": 0.0, "price_1449": 10.0,
                "amount_1449": 200_000_000.0,
                "group": "burst", "future_return": 99.0,
            })
    arms, matched, report = audit(pd.DataFrame(rows))

    assert set(arms.arm) == {"morning", "afternoon"}
    assert "future_return" not in arms.columns
    assert "future_return" not in matched.columns
    assert len(matched) == 40
    assert not report["outcome_gate_passed"]
    assert all(part["comparable_strata"] == 1
               for part in report["by_half"].values())

    reduced = pd.DataFrame(rows)
    reduced = reduced.loc[~(
        reduced.date.eq("2024-05-06") & reduced.code.eq("sh.600009"))]
    _, _, reduced_report = audit(reduced)
    assert reduced_report["by_half"]["2024H1"]["comparable_strata"] == 0


def test_path_rejects_2026_development_inputs():
    frame = pd.DataFrame([{
        "date": "2026-01-05", "code": "sh.600000", "board": "main",
        "max5": .01, "morning_pp": 2.0, "day_pp": 1.0,
        "tail_pp": 0.0, "prior20_pp": 0.0, "price_1449": 10.0,
        "amount_1449": 200_000_000.0,
    }])
    with pytest.raises(ValueError, match="out-of-period"):
        audit(frame)
