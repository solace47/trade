"""Check source completeness and the public-date guard."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from trade_research.exchange_public_events import (
    _jsonp, _validate_sse_rows, load_complete_notices, read_szse_export,
)


def _tiny_export(path: Path) -> None:
    header = ["公告日期", "证券代码", "证券简称", "成交金额(元)",
              "成交量(股/份)", "披露原因"]
    rows = [header, ["2024-06-03", "000525", "样本股票", "1,000", "100",
                     "日价格涨幅偏离值达到7%"]]
    body = "".join(
        f'<row r="{number}">' + "".join(
            f'<c r="{chr(65 + index)}{number}" t="inlineStr"><is><t>{value}</t></is></c>'
            for index, value in enumerate(row)
        ) + "</row>"
        for number, row in enumerate(rows, start=1)
    )
    sheet = ('<worksheet xmlns="http://schemas.openxmlformats.org/'
             'spreadsheetml/2006/main"><sheetData>' + body
             + '</sheetData></worksheet>')
    with ZipFile(path, "w") as archive:
        archive.writestr("xl/worksheets/sheet1.xml", sheet)


def test_official_export_preserves_zero_padded_stock_code(tmp_path: Path) -> None:
    workbook = tmp_path / "official.xlsx"
    _tiny_export(workbook)
    frame = read_szse_export(workbook, "2024-01-01", "2024-12-31")
    assert frame.code.tolist() == ["sz.000525"]
    assert frame.amount_yuan.tolist() == [1000]
    with pytest.raises(ValueError, match="outside requested window"):
        read_szse_export(workbook, "2025-01-01", "2025-12-31")


def test_complete_loader_rejects_missing_exchange_day(tmp_path: Path) -> None:
    calendar = tmp_path / "calendar.parquet"
    pd.DataFrame([{"calendar_date": "2024-06-03", "is_trading_day": "1"}]).to_parquet(calendar)
    workbook = tmp_path / "official.xlsx"
    _tiny_export(workbook)
    with pytest.raises(FileNotFoundError):
        load_complete_notices(calendar, tmp_path / "sse", workbook,
                              "2024-06-03", "2024-06-03")


def test_sse_wrapper_and_notice_keys_are_checked() -> None:
    assert _jsonp('cb({"actionErrors":[],"pageHelp":{"data":[]}})')["pageHelp"]["data"] == []
    with pytest.raises(ValueError, match="wrapper"):
        _jsonp('other({"pageHelp":{"data":[]}})')
    row = {"tradeDate": "20240603", "secCode": "600001", "refType": "11"}
    with pytest.raises(ValueError, match="Duplicate"):
        _validate_sse_rows([row, row], "20240603", "main")
