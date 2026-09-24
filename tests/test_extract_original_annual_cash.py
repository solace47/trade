"""Guard the wrapped figures common in annual-report summary tables."""

from pathlib import Path

import pandas as pd
import pytest

from scripts import extract_original_annual_cash as annual


_number = annual._number
_table_values = annual._table_values


def test_wrapped_annual_rows_keep_current_year_amount() -> None:
    table = [
        ["", "归属于上市", "", "30,910,620.19", "14,333,240.9\n8"],
        [None, "公司股东的", None, None, None],
        [None, "净利润", None, None, None],
        ["", "经营活动产", "", "108,928,550.3\n2", "-\n10,022,131.9\n9"],
        [None, "生的现金流", None, None, None],
        [None, "量净额", None, None, None],
    ]
    rows = list(_table_values(table))
    assert rows == [
        ("归属于上市公司股东的净利润", 30910620.19),
        ("经营活动产生的现金流量净额", 108928550.32),
    ]
    assert _number("(5,644,746,827.65)") == -5644746827.65


def test_quarterly_profit_cannot_complete_missing_annual_table(monkeypatch) -> None:
    class Page:
        def extract_text(self):
            return "报告期分季度的主要会计数据\n第一季度 第二季度"

        def extract_tables(self):
            return [
                [["经营活动产生的现金流量净额", "100", "90"]],
                [["第一季度", "第二季度", "第三季度", "第四季度"],
                 ["归属于上市公司股东的净利润", "25", "35"]],
            ]

    class Document:
        pages = [Page()]

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    monkeypatch.setattr(annual.pdfplumber, "open", lambda _: Document())
    with pytest.raises(ValueError, match="unreadable"):
        annual.extract(Path("summary.pdf"))
    assert list(annual._text_values([])) == []


def test_split_annual_text_label_can_cross_pdf_page() -> None:
    lines = [(4, "归属于上市公司股东 415,465,384.94 782,599,694.87"),
             (5, "的净利润"),
             (5, "经营活动产生的现金 2,095,748,225.61 -8,179,987,359.73"),
             (5, "流量净额")]
    assert list(annual._text_values(lines)) == [
        ("归属于上市公司股东的净利润", 415465384.94, 4),
        ("经营活动产生的现金流量净额", 2095748225.61, 5),
    ]


def test_failed_report_is_skipped_until_explicit_retry(tmp_path, monkeypatch) -> None:
    index = tmp_path / "index.parquet"
    universe = tmp_path / "universe.parquet"
    output = tmp_path / "cash.jsonl"
    pd.DataFrame([
        {"report_year": 2023, "code": "sh.600001", "kind": "summary",
         "notice_date": "2024-04-27", "pdf_url": "https://example/a.pdf"},
        {"report_year": 2023, "code": "sh.600002", "kind": "summary",
         "notice_date": "2024-04-27", "pdf_url": "https://example/b.pdf"},
    ]).to_parquet(index)
    pd.DataFrame({"code": ["sh.600001", "sh.600002"]}).to_parquet(universe)
    monkeypatch.setattr(annual, "_download", lambda *_: None)
    attempts = []

    def extract(path: Path) -> dict:
        attempts.append(path.stem)
        if path.stem == "sh.600001" and attempts.count(path.stem) == 1:
            raise ValueError("Unreadable")
        return {"parent_profit_raw": 2, "operating_cash_raw": 3,
                "cash_to_parent_profit": 1.5}

    monkeypatch.setattr(annual, "extract", extract)
    cache = tmp_path / "pdfs"
    assert annual.collect(index, output, cache, universe,
                          max_reports=1)["failures"] == 1
    assert annual.collect(index, output, cache, universe,
                          max_reports=1)["success"] == 1
    assert attempts == ["sh.600001", "sh.600002"]
    assert annual.collect(index, output, cache, universe,
                          retry_failed=True)["success"] == 1
    assert attempts == ["sh.600001", "sh.600002", "sh.600001"]
