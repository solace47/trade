"""Keep the new path contrast balanced and limited to decision-time fields."""

import pandas as pd
import pytest

from trade_research.morning_path_continuous import select


def _inputs() -> pd.DataFrame:
    rows = []
    for year in (2024, 2025):
        for month in (3, 9):
            for index in range(8):
                rows.append({
                    "date": f"{year}-{month:02d}-12",
                    "code": f"sh.60{index:04d}", "board": "main",
                    "max5": .01, "morning_pp": index * .3,
                    "day_pp": 1.1, "gap_pp": .1,
                    "tail_pp": .1, "prior20_pp": 2.,
                    "price_1449": 10., "amount_1449": 200_000_000.,
                    "listing_age_sessions": 30, "isST": 0,
                    "reference_gap": False,
                    "quote_outside_traded_range": False,
                    "future_price": index * 1000.,
                })
    return pd.DataFrame(rows)


def test_quartile_contrast_balances_each_stratum_and_drops_future_data():
    selected, report = select(_inputs())
    assert len(selected) == 16
    assert "future_price" not in selected
    assert report["outcome_gate_passed"] is False
    for (half, date), rows in selected.groupby(["half", "date"]):
        assert rows.arm.value_counts().to_dict() == {
            "morning": 2, "afternoon": 2,
        }
        assert rows.loc[rows.arm.eq("morning"), "morning_pp"].min() == pytest.approx(1.8)
        assert rows.loc[rows.arm.eq("afternoon"), "morning_pp"].max() == pytest.approx(.3)


def test_out_of_period_or_duplicate_inputs_are_rejected():
    frame = _inputs()
    frame.loc[0, "date"] = "2023-12-12"
    with pytest.raises(ValueError, match="out-of-period"):
        select(frame)
    frame = _inputs()
    with pytest.raises(ValueError, match="duplicate"):
        select(pd.concat([frame, frame.iloc[[0]]], ignore_index=True))
