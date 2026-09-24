from datetime import date

import pytest

from scripts import collect_sell_completion_index as collector


def test_large_cninfo_query_is_split_before_later_pages(monkeypatch, tmp_path) -> None:
    requested = []

    def fake_fetch(start, end, page, cache, term):
        assert page == 1
        return {"totalAnnouncement": 210 if start == date(2025, 1, 1)
                and end == date(2025, 1, 4) else 105}

    def fake_range(start, end, cache, term):
        requested.append((start, end))
        return [{"range": str(start), "row": n} for n in range(105)]

    monkeypatch.setattr(collector, "_fetch", fake_fetch)
    monkeypatch.setattr(collector, "_range_rows", fake_range)
    rows = collector._bounded_rows(date(2025, 1, 1), date(2025, 1, 4),
                                   tmp_path, "减持计划实施完成")
    assert len(rows) == 210
    assert requested == [(date(2025, 1, 1), date(2025, 1, 2)),
                         (date(2025, 1, 3), date(2025, 1, 4))]


def test_single_day_above_safe_page_limit_is_rejected(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(collector, "_fetch", lambda *_: {
        "totalAnnouncement": collector.MAX_SAFE_PAGES * collector.PAGE_SIZE + 1})
    with pytest.raises(ValueError, match="exceeds"):
        collector._bounded_rows(date(2025, 1, 1), date(2025, 1, 1),
                                tmp_path, "减持计划实施完成")
