from datetime import date
from pathlib import Path

from scripts import collect_original_annual_index as index


def test_partial_final_page_is_fetched(monkeypatch) -> None:
    calls = []

    def fetch(start: date, end: date, page: int, cache: Path) -> dict:
        calls.append(page)
        return {"totalpages": 1, "totalAnnouncement": 31,
                "announcements": [{"id": n} for n in range(30)]
                if page == 1 else [{"id": 30}]}

    monkeypatch.setattr(index, "_fetch", fetch)
    rows = index._range_rows(date(2024, 4, 27), date(2024, 4, 27), Path("x"))
    assert calls == [1, 2]
    assert len(rows) == 31


def test_large_ranges_split_before_capped_page(monkeypatch) -> None:
    calls = []

    def fetch(start: date, end: date, page: int, cache: Path) -> dict:
        calls.append((start, end, page))
        if start != end:
            return {"totalpages": 100, "totalAnnouncement": 3001,
                    "announcements": []}
        return {"totalpages": 0, "totalAnnouncement": 1,
                "announcements": [{"day": str(start)}]}

    monkeypatch.setattr(index, "_fetch", fetch)
    rows = index._range_rows(date(2024, 4, 1), date(2024, 4, 2), Path("x"))
    assert [r["day"] for r in rows] == ["2024-04-01", "2024-04-02"]
    assert all(page == 1 for _, _, page in calls)


def test_keep_first_original_report_and_summary() -> None:
    def row(title: str, code: str, timestamp: int, pdf: str) -> dict:
        return {"announcementTitle": title, "secCode": code,
                "announcementTime": timestamp, "adjunctUrl": pdf}

    frame = index._annual_rows([
        row("白云机场2023年年度报告", "600004", 1714147200000,
            "finalpage/2024-04-27/first.PDF"),
        row("白云机场2023年年度报告摘要", "600004", 1714147200000,
            "finalpage/2024-04-27/summary.PDF"),
        row("白云机场2023年年度报告（更正版）", "600004", 1715000000000,
            "finalpage/2024-05-06/revised.PDF"),
        row("白云机场2023年年度报告", "600004", 1716000000000,
            "finalpage/2024-05-18/later.PDF"),
        row("某B股2023年年度报告", "200468", 1714147200000,
            "finalpage/2024-04-27/b.PDF"),
    ], 2023)
    assert len(frame) == 2
    assert set(frame.kind) == {"full", "summary"}
    assert frame.notice_date.eq("2024-04-27").all()
    assert frame.loc[frame.kind.eq("full"), "pdf_url"].item().endswith(
        "/first.PDF")
