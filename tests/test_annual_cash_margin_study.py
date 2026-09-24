"""Annual inputs must be complete original PDFs before any outcome read."""

import json

import pandas as pd
import pytest

from trade_research.annual_cash_margin_study import _original_reports, _window


def test_original_cash_inputs_require_complete_matching_source(tmp_path) -> None:
    indices = []
    extracted = []
    for year, code in ((2023, "sh.600004"), (2024, "sz.300225")):
        day = f"{year + 1}-04-27"
        url = f"https://static.cninfo.com.cn/finalpage/{day}/first.PDF"
        index_path = tmp_path / f"index_{year}.parquet"
        pd.DataFrame([{ "report_year": year, "code": code,
                        "kind": "summary", "notice_date": day,
                        "pdf_url": url}]).to_parquet(index_path)
        indices.append(index_path)
        extracted_path = tmp_path / f"cash_{year}.jsonl"
        item = {"report_year": year, "code": code, "status": "ok",
                "notice_date": day, "pdf_url": url,
                "parent_profit_raw": 100, "operating_cash_raw": 150,
                "cash_to_parent_profit": 1.5,
                "parent_profit_page": 4, "operating_cash_page": 5}
        if year == 2024:
            item["parent_profit_raw"] = -100
            item["cash_to_parent_profit"] = -1.5
        extracted_path.write_text(json.dumps(item) + "\n")
        extracted.append(extracted_path)
    codes = {"sh.600004", "sz.300225"}
    reports, audit = _original_reports(tuple(indices), tuple(extracted), codes)
    assert len(reports) == 1
    assert reports.code.item() == "sh.600004"
    assert audit["indexed_margin_stock_years"] == 2
    assert audit["extracted_stock_years"] == 2

    extracted[1].write_text(json.dumps({"report_year": 2024,
                                        "code": "sz.300999",
                                        "status": "failed"}) + "\n")
    with pytest.raises(ValueError, match="incomplete"):
        _original_reports(tuple(indices), tuple(extracted), codes)


def test_window_has_no_2023_or_2026_signals() -> None:
    assert _window("2024-05-15") == "2024_early"
    assert _window("2025-12-17") == "2025_late"
    with pytest.raises(ValueError):
        _window("2023-12-31")
    with pytest.raises(ValueError):
        _window("2026-01-05")


def test_later_profit_cannot_pair_with_earlier_annual_cash(tmp_path) -> None:
    indices = []
    extracted = []
    for year in (2023, 2024):
        day = f"{year + 1}-04-27"
        url = f"https://static.cninfo.com.cn/finalpage/{day}/first.PDF"
        index_path = tmp_path / f"index_{year}.parquet"
        pd.DataFrame([{"report_year": year, "code": "sz.301055",
                       "kind": "summary", "notice_date": day,
                       "pdf_url": url}]).to_parquet(index_path)
        indices.append(index_path)
        extracted_path = tmp_path / f"cash_{year}.jsonl"
        extracted_path.write_text(json.dumps({
            "report_year": year, "code": "sz.301055", "status": "ok",
            "notice_date": day, "pdf_url": url,
            "parent_profit_raw": 17506.71,
            "operating_cash_raw": 104174515.04,
            "cash_to_parent_profit": 104174515.04 / 17506.71,
            "parent_profit_page": 4, "operating_cash_page": 3,
        }) + "\n")
        extracted.append(extracted_path)
    reports, audit = _original_reports(
        tuple(indices), tuple(extracted), {"sz.301055"})
    assert reports.empty
    assert audit["page_gap_excluded"] == 2
