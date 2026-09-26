import pandas as pd
import pytest

from trade_research.month_turn_1449 import calendar_schedule
from trade_research.month_turn_eval import slot_mean, month_interval


def test_same_month_comparison_is_five_actual_sessions_before_month_end():
    calendar = pd.bdate_range("2024-01-01", "2024-03-29").strftime("%Y-%m-%d").tolist()
    calendar.remove("2024-01-29")
    result = calendar_schedule(calendar)
    assert set(result.month) == {"2024-01", "2024-02"}
    for _, part in result.groupby("month"):
        end = part.loc[part.arm.eq("high"), "date"].iloc[0]
        prior = part.loc[part.arm.eq("low"), "date"].iloc[0]
        assert calendar.index(end) - calendar.index(prior) == 5
        assert end[:7] == prior[:7]


def test_2024_year_end_is_kept_but_2025_holdout_prices_are_not_needed():
    calendar = pd.bdate_range("2024-12-01", "2026-01-31").strftime("%Y-%m-%d").tolist()
    result = calendar_schedule(calendar)
    assert "2024-12" in set(result.month)
    assert "2025-12" not in set(result.month)
    assert result.last_observation_day.max() <= "2025-12-31"
    assert result.loc[result.date.eq("2024-12-31"), "planned_T5"].iloc[0] > "2024-12-31"


def test_duplicates_are_not_accepted_as_extra_trading_sessions():
    with pytest.raises(ValueError, match="ordered and unique"):
        calendar_schedule(["2024-01-02", "2024-01-02"])


def test_unselected_slots_are_cash_but_unknown_bought_returns_remain_unknown():
    assert slot_mean(pd.Series([.1])) == pytest.approx(.02)
    assert slot_mean(pd.Series(dtype=float)) == 0
    assert slot_mean(pd.Series([.1, float("nan")])) is None
    with pytest.raises(ValueError, match="five fixed slots"):
        slot_mean(pd.Series([0.] * 6))


def test_interval_does_not_delete_an_unknown_month_or_treat_halfyear_as_large_sample():
    assert month_interval(pd.Series([.1] * 6)) is None
    assert month_interval(pd.Series([.1] * 11 + [float("nan")])) is None
    assert month_interval(pd.Series([.03] * 12)) == pytest.approx([.03, .03])
