"""Guard exchange margin schemas and incomplete daily responses."""

from __future__ import annotations

import gzip
import json
from io import BytesIO
from zipfile import ZipFile

import pandas as pd
import pytest

from trade_research.margin_public_data import (
    SZSE_HEADER, _validate_sse, load_margin, read_szse_day,
)


def _workbook(rows: list[list[str]]) -> bytes:
    body = "".join(
        f'<row r="{number}">' + "".join(
            f'<c r="{chr(65 + index)}{number}" t="inlineStr">'
            f'<is><t>{value}</t></is></c>'
            for index, value in enumerate(row)
        ) + "</row>"
        for number, row in enumerate(rows, start=1)
    )
    sheet = ('<worksheet xmlns="http://schemas.openxmlformats.org/'
             'spreadsheetml/2006/main"><sheetData>' + body
             + '</sheetData></worksheet>')
    stream = BytesIO()
    with ZipFile(stream, "w") as archive:
        archive.writestr("xl/worksheets/sheet1.xml", sheet)
    return stream.getvalue()


def test_szse_daily_balance_arithmetic_and_zero_padded_code() -> None:
    workbook = _workbook([list(SZSE_HEADER),
                          ["000001", "样本", "1,000", "20,000", "10",
                           "20", "500", "20,500"]])
    row = read_szse_day(workbook, "2024-06-03").iloc[0]
    assert row.code == "000001"
    assert row.buy_yuan == 1000
    assert row.total_balance_yuan == 20500
    bad = _workbook([list(SZSE_HEADER),
                     ["000001", "样本", "1,000", "20,000", "10",
                      "20", "500", "21,000"]])
    with pytest.raises(ValueError, match="totals disagree"):
        read_szse_day(bad, "2024-06-03")


def test_sse_margin_day_rejects_missing_page_or_wrong_day() -> None:
    row = {"stockCode": "600000", "opDate": "20240603", "rzye": 100,
           "rzmre": 20, "rzche": 10, "rqyl": 0, "rqmcl": 0}
    _validate_sse([row], "20240603", 1)
    with pytest.raises(ValueError, match="Incomplete"):
        _validate_sse([row], "20240603", 2)
    with pytest.raises(ValueError, match="code/date"):
        _validate_sse([row], "20240604", 1)
    with pytest.raises(ValueError, match="Duplicate"):
        _validate_sse([row, row], "20240603", 2)


def test_merge_preserves_exchange_short_balances_and_previous_day(tmp_path) -> None:
    day = "2024-06-03"
    calendar = tmp_path / "calendar.parquet"
    pd.DataFrame({"calendar_date": [day], "is_trading_day": ["1"]}).to_parquet(
        calendar, index=False)
    archive = tmp_path / "margin"
    (archive / "sse_daily").mkdir(parents=True)
    (archive / "szse_daily").mkdir(parents=True)
    sse_row = {"stockCode": "600000", "opDate": "20240603",
               "rzye": "1,000", "rzmre": "100", "rzche": "20",
               "rqyl": "30", "rqmcl": "10"}
    record = {"date": day, "source": "sse", "total": 1, "pages": 1,
              "rows": [sse_row]}
    with gzip.open(archive / "sse_daily" / f"{day}.json.gz", "wt") as file:
        json.dump(record, file)
    workbook = _workbook([list(SZSE_HEADER),
                          ["000001", "样本", "200", "2,000", "5",
                           "40", "600", "2,600"]])
    (archive / "szse_daily" / f"{day}.xlsx").write_bytes(workbook)
    merged = load_margin(calendar, archive, day, day).set_index("code")
    assert list(merged.index) == ["sh.600000", "sz.000001"]
    assert merged.trade_date.eq(day).all()
    assert merged.loc["sh.600000", "short_balance_shares"] == 30
    assert merged.loc["sh.600000", "short_sell_shares"] == 10
    assert pd.isna(merged.loc["sh.600000", "short_balance_yuan"])
    assert merged.loc["sz.000001", "short_balance_shares"] == 40
    assert merged.loc["sz.000001", "short_sell_shares"] == 5
    assert merged.loc["sz.000001", "short_balance_yuan"] == 600
