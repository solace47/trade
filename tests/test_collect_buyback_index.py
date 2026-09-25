from datetime import date

import pytest

from scripts import collect_buyback_index as collector


@pytest.mark.parametrize("drift", ["count", "duplicate"])
def test_page_drift_refetches_disjoint_dates(monkeypatch, tmp_path, drift):
    left = {"adjunctUrl": "first.pdf"}
    right = {"adjunctUrl": "second.pdf"}
    calls = []

    def fetch(start, end, page, cache, searchkey):
        calls.append((start, end, page))
        if start != end:
            if drift == "count":
                return {"totalAnnouncement": 3, "announcements": [left, right]}
            return {"totalAnnouncement": 2, "announcements": [left, left]}
        row = left if start.day == 1 else right
        return {"totalAnnouncement": 1, "announcements": [row]}

    monkeypatch.setattr(collector, "_fetch", fetch)
    rows = collector._range_rows(date(2025, 1, 1), date(2025, 1, 2),
                                 tmp_path, "回购进展")
    assert rows == [left, right]
    assert len(calls) == 3
