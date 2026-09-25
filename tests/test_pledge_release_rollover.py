import pandas as pd

from trade_research.pledge_release_rollover import _late_events


def test_year_end_supplement_only_uses_late_2024_signals() -> None:
    calendar = ["2024-12-17", "2024-12-18", "2024-12-30",
                "2025-01-02", "2025-12-17", "2025-12-18"]
    rows = [
        ("2024-12-17", "sh.600001", "verified"),
        ("2024-12-29", "sh.600002", "verified"),
        ("2024-12-30", "sh.600003", "verified"),
        ("2025-12-17", "sh.600004", "verified"),
        ("2024-12-17", "sh.600005", "reject"),
    ]
    review = pd.DataFrame([
        {"notice_date": day, "code": code, "decision": decision,
         "pdf_url": code, "planned_repledge": "no"}
        for day, code, decision in rows
    ])
    events = _late_events(review, calendar)
    assert events[["date", "code"]].to_records(index=False).tolist() == [
        ("2024-12-18", "sh.600001"),
        ("2024-12-30", "sh.600002"),
    ]
