"""Frozen percentile and same-day comparison safeguards."""

import pandas as pd
import pytest

from trade_research.absolute_ridge_extreme_tail import assign_groups, summarize_daily


def test_score_groups_are_stable_and_separated_by_date() -> None:
    frame = pd.DataFrame([
        {"date": date, "code": f"sh.{number:06d}",
         "score": float(number), "half": "2024H2"}
        for date in ("2024-07-01", "2024-07-02")
        for number in range(1000)
    ])
    first = assign_groups(frame.sample(frac=1, random_state=31))
    second = assign_groups(frame.sample(frac=1, random_state=47))
    pd.testing.assert_frame_equal(first, second)
    counts = first.groupby(["date", "group"]).size().unstack()
    assert counts.loc["2024-07-01"].to_dict() == {
        "extreme": 5, "high": 45, "raised": 150}
    assert counts.loc["2024-07-02"].to_dict() == {
        "extreme": 5, "high": 45, "raised": 150}
    assert first.loc[first.group.eq("extreme"), "score"].min() == 995


def test_daily_comparison_requires_all_frozen_groups() -> None:
    rows = pd.DataFrame([
        {"date": "2024-07-01", "group": group, "stocks": 5,
         "cash": value, "stress15": value - .001, "entry": 1.0,
         "clean_exit": 1.0, "score": 0.1}
        for group, value in (("raised", .01), ("high", .005),
                             ("extreme", -.01))
    ])
    report = summarize_daily(rows)
    assert report["extreme_minus_raised"] == pytest.approx(-.02)
    assert report["extreme_minus_high"] == pytest.approx(-.015)
    with pytest.raises(ValueError, match="three"):
        summarize_daily(rows.loc[rows.group.ne("high")])
