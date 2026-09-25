import pandas as pd
import pytest

from trade_research.pledge_release_inputs import (
    _events, _exclude_announcers, _signals,
)


def test_pledge_signals_lag_notice_and_late_year_is_reserved() -> None:
    calendar = ["2024-12-17", "2024-12-31", "2025-01-02",
                "2025-12-17", "2025-12-18"]
    review = pd.DataFrame([
        {"notice_date": "2024-12-16", "date": "", "code": "sh.600001",
         "pdf_url": "first", "decision": "verified", "planned_repledge": "no"},
        {"notice_date": "2024-12-31", "date": "", "code": "sh.600002",
         "pdf_url": "rollover", "decision": "verified", "planned_repledge": "no"},
        {"notice_date": "2025-12-17", "date": "", "code": "sh.600003",
         "pdf_url": "late", "decision": "verified", "planned_repledge": "yes"},
    ])
    events, audit = _events(review, calendar)
    assert events[["date", "code"]].to_records(index=False).tolist() == [
        ("2024-12-17", "sh.600001")]
    assert audit["year_rollover_excluded"] == 1
    assert audit["late_signal_excluded"] == 1


def test_same_day_pledge_announcer_is_not_a_control() -> None:
    universe = pd.DataFrame({"date": ["2024-06-05"] * 3,
                             "code": ["sh.600001", "sh.600002", "sh.600003"]})
    announced = pd.DataFrame({"date": ["2024-06-05"] * 2,
                              "code": ["sh.600001", "sh.600002"]})
    event = pd.DataFrame({"date": ["2024-06-05"],
                          "code": ["sh.600001"]})
    kept, removed = _exclude_announcers(universe, announced, event)
    assert kept.code.tolist() == ["sh.600001", "sh.600003"]
    assert removed == 1


def test_reprice_signals_reject_conflicting_point_in_time_flags() -> None:
    row = {"date": "2024-06-05", "code": "sh.600001", "isST": 0,
           "reference_gap": False, "quote_outside_traded_range": False,
           "listing_age_sessions": 200}
    first = pd.DataFrame([row])
    second = pd.DataFrame([{**row, "isST": 1}])
    assert len(_signals(first, first)) == 1
    with pytest.raises(ValueError, match="Inconsistent"):
        _signals(first, second)
