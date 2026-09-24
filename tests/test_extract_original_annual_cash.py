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


def test_separated_financial_tables_are_rejected(monkeypatch) -> None:
    class Page:
        def __init__(self, table):
            self.table = table

        def extract_text(self):
            return "主要会计数据和财务指标"

        def extract_tables(self):
            return [self.table]

    class Document:
        pages = [
            Page([["归属于上市公司股东的净利润", "100"]]),
            Page([["其他项目", "20"]]),
            Page([["经营活动产生的现金流量净额", "150"]]),
        ]

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    monkeypatch.setattr(annual.pdfplumber, "open", lambda _: Document())
    with pytest.raises(ValueError, match="table order invalid"):
        annual.extract(Path("summary.pdf"))


def test_annual_table_after_page_25_and_split_label(monkeypatch) -> None:
    class Page:
        def __init__(self, table=None):
            self.table = table

        def extract_text(self):
            return "主要会计数据和财务指标" if self.table is not None else ""

        def extract_tables(self):
            return [self.table] if self.table is not None else []

    class Document:
        pages = [Page() for _ in range(26)] + [
            Page([["归属于上市", "161,819,588.65", "305,069,713.68"]]),
            Page([["公司股东的净利润", "", ""],
                  ["经营活动产生的现金流量净额", "-1,943,054,851.89",
                   "-391,133,201.40"]]),
        ]

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    monkeypatch.setattr(annual.pdfplumber, "open", lambda _: Document())
    result = annual.extract(Path("summary.pdf"))
    assert result["parent_profit_raw"] == 161819588.65
    assert result["operating_cash_raw"] == -1943054851.89
    assert result["parent_profit_page"] == 27
    assert result["operating_cash_page"] == 28


def test_current_annual_profit_precedes_later_adjustment(monkeypatch) -> None:
    class Page:
        def __init__(self, text, table):
            self.text = text
            self.table = table

        def extract_text(self):
            return self.text

        def extract_tables(self):
            return [self.table]

    class Document:
        pages = [
            Page("主要会计数据和财务指标\n归属于上市公司股东\n"
                 "25,118,302.46 41,489,959.40\n的净利润\n"
                 "经营活动产生的现金\n104,174,515.04 111,937,684.40\n流量净额",
                 [["经营活动产生的现金流量净额", "104,174,515.04"]]),
            Page("2022年度利润表项目\n归属于母公司所有者的净利润 17,506.71",
                 [["归属于母公司所有者的净利润", "17,506.71"]]),
        ]

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    monkeypatch.setattr(annual.pdfplumber, "open", lambda _: Document())
    result = annual.extract(Path("summary.pdf"))
    assert result["parent_profit_raw"] == 25118302.46
    assert result["operating_cash_raw"] == 104174515.04
    assert result["parent_profit_page"] == result["operating_cash_page"] == 1


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
