from pathlib import Path

import pytest

from scripts import collect_fund_xbrl_index as index


def _row(number: int) -> dict:
    return {"reportYear": "2024", "reportDesp": "第一季度报告",
            "fundCode": f"{number:06d}", "uploadInfoId": number,
            "uploadDate": "2024-04-18", "reportSendDate": "2024-04-19"}


def test_complete_pagination_and_conservative_visibility(monkeypatch) -> None:
    seen = []

    def page(year: int, quarter: int, kind: str, start: int,
             cache: Path) -> dict:
        seen.append(start)
        rows = [_row(n) for n in range(start, min(start + 20, 21))]
        return {"iTotalRecords": 21, "aaData": rows}

    monkeypatch.setattr(index, "_page", page)
    rows = index._category(2024, 1, "mixed", Path("unused"), 1)
    assert seen == [0, 20]
    assert len(rows) == 21
    assert {row["available_after"] for row in rows} == {"2024-04-19"}
    assert {row["fund_type_query"] for row in rows} == {"mixed"}


def test_missing_final_page_is_rejected(monkeypatch) -> None:
    def page(year: int, quarter: int, kind: str, start: int,
             cache: Path) -> dict:
        return {"iTotalRecords": 21,
                "aaData": [_row(n) for n in range(20)] if start == 0 else []}

    monkeypatch.setattr(index, "_page", page)
    with pytest.raises(ValueError, match="incomplete"):
        index._category(2024, 1, "stock", Path("unused"), 1)
