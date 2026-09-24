"""Archive the exchanges' post-close trading-public-information notices.

The SZSE workbook is downloaded from its official report page with ego-browser.
SSE daily JSONP endpoints provide main-board and STAR-market records.  Neither
source is available by the 14:50 snapshot on its own trade date; downstream
signals must use an earlier notice date.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import pandas as pd
import requests


SSE_URL = "https://query.sse.com.cn/marketdata/tradedata/"
SSE_ENDPOINTS = {
    "main": ("queryAllTradeOpenDate.do", {"token": "QUERY", "flag": "1"}),
    "star": ("queryKCBTradeInfo.do", {"flag": "1"}),
}
SSE_HEADERS = {"Referer": "https://www.sse.com.cn/", "User-Agent": "Mozilla/5.0"}
XLSX_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
SZSE_HEADER = ("公告日期", "证券代码", "证券简称", "成交金额(元)",
               "成交量(股/份)", "披露原因")


def trading_dates(calendar: Path, start: str, end: str) -> list[str]:
    frame = pd.read_parquet(calendar)
    required = {"calendar_date", "is_trading_day"}
    if not required.issubset(frame.columns):
        raise ValueError("Calendar schema changed")
    dates = frame.loc[frame.is_trading_day.astype(str).eq("1"), "calendar_date"]
    result = sorted(d for d in dates if start <= d <= end)
    if not result or len(result) != len(set(result)):
        raise ValueError("No trading dates or duplicate dates")
    return result


def _jsonp(response: str) -> dict:
    match = re.fullmatch(r"cb\((.*)\);?\s*", response, flags=re.DOTALL)
    if match is None:
        raise ValueError("Unexpected SSE JSONP wrapper")
    data = json.loads(match.group(1))
    if data.get("actionErrors") or not isinstance(data.get("pageHelp", {}).get("data"), list):
        raise ValueError("SSE endpoint returned an error or no data list")
    return data


def _validate_sse_rows(rows: list[dict], day: str, board: str) -> None:
    seen: set[tuple[str, str]] = set()
    for row in rows:
        code = row.get("secCode", "")
        if not re.fullmatch(r"\d{6}", code) or row.get("tradeDate") != day:
            raise ValueError(f"Invalid SSE {board} code/date on {day}")
        if board == "star" and not code.startswith("688"):
            raise ValueError(f"Unexpected STAR code {code}")
        if board == "main" and code.startswith("688"):
            raise ValueError(f"STAR code on main-board endpoint {code}")
        key = code, row.get("refType", "")
        if key in seen or not key[1]:
            raise ValueError(f"Duplicate or missing SSE notice type on {day}: {key}")
        seen.add(key)


def fetch_sse_day(day: str, session: requests.Session | None = None) -> dict:
    if not re.fullmatch(r"20\d{6}", day):
        raise ValueError("SSE trade date must be YYYYMMDD")
    client = session or requests.Session()
    result = {"trade_date": date.fromisoformat(f"{day[:4]}-{day[4:6]}-{day[6:]}" ).isoformat()}
    for board, (endpoint, extra) in SSE_ENDPOINTS.items():
        for attempt in range(3):
            try:
                response = client.get(SSE_URL + endpoint,
                                      params={"jsonCallBack": "cb", "tradeDate": day, **extra},
                                      headers=SSE_HEADERS, timeout=25)
                response.raise_for_status()
                rows = _jsonp(response.text)["pageHelp"]["data"]
                _validate_sse_rows(rows, day, board)
                result[board] = rows
                break
            except (requests.RequestException, ValueError):
                if attempt == 2:
                    raise
                time.sleep(1.5 * (attempt + 1))
    return result


def _check_saved_sse(path: Path, day: str) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("trade_date") != f"{day[:4]}-{day[4:6]}-{day[6:]}":
        raise ValueError(f"Wrong date in {path}")
    for board in SSE_ENDPOINTS:
        if not isinstance(record.get(board), list):
            raise ValueError(f"Missing board in {path}: {board}")
        _validate_sse_rows(record[board], day, board)
    return record


def archive_sse(calendar: Path, output_dir: Path, start: str, end: str,
                workers: int = 4) -> dict:
    if workers < 1 or workers > 6:
        raise ValueError("Use 1 to 6 workers for the exchange endpoint")
    dates = [d.replace("-", "") for d in trading_dates(calendar, start, end)]
    output_dir.mkdir(parents=True, exist_ok=True)
    cached = 0
    pending = []
    for day in dates:
        path = output_dir / f"sse_{day}.json"
        if path.exists():
            _check_saved_sse(path, day)
            cached += 1
        else:
            pending.append(day)
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_sse_day, day): day for day in pending}
        for future in as_completed(futures):
            day = futures[future]
            record = future.result()
            path = output_dir / f"sse_{day}.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(record, ensure_ascii=False) + "\n",
                                 encoding="utf-8")
            temporary.replace(path)
            completed += 1
            if completed % 25 == 0 or completed == len(pending):
                print(f"SSE days: {completed}/{len(pending)} newly archived; "
                      f"{cached} cached", flush=True)
    return {"expected_days": len(dates), "cached_days": cached,
            "new_days": completed}


def _xlsx_cell(cell: ET.Element, shared: list[str]) -> str:
    kind = cell.attrib.get("t")
    if kind == "inlineStr":
        return "".join(t.text or "" for t in cell.findall(".//m:is/m:t", XLSX_NS))
    value = cell.findtext("m:v", default="", namespaces=XLSX_NS)
    return shared[int(value)] if kind == "s" and value else value


def read_szse_export(path: Path, start: str, end: str) -> pd.DataFrame:
    """Read the exchange's compact XLSX without adding an Excel dependency."""
    with ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(t.text or "" for t in item.findall(".//m:t", XLSX_NS))
                      for item in root.findall("m:si", XLSX_NS)]
        root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        xml_rows = root.find("m:sheetData", XLSX_NS)
        if xml_rows is None:
            raise ValueError("SZSE workbook contains no rows")
        rows: list[list[str]] = []
        for row in xml_rows.findall("m:row", XLSX_NS):
            values = [""] * len(SZSE_HEADER)
            for cell in row.findall("m:c", XLSX_NS):
                column = re.match(r"[A-Z]+", cell.attrib["r"])
                if column is None or len(column.group()) != 1 or not "A" <= column.group() <= "F":
                    continue
                index = ord(column.group()) - ord("A")
                values[index] = _xlsx_cell(cell, shared)
            rows.append(values)
    if not rows or tuple(rows[0]) != SZSE_HEADER:
        raise ValueError("SZSE export header changed")
    frame = pd.DataFrame(rows[1:], columns=["trade_date", "code", "name",
                                              "amount_yuan", "volume_shares", "reason"])
    if frame.empty:
        raise ValueError("SZSE workbook has no notices")
    if not frame.trade_date.between(start, end).all():
        raise ValueError("SZSE export contains dates outside requested window")
    if not frame.code.str.fullmatch(r"\d{6}").all():
        raise ValueError("SZSE export contains malformed codes")
    if frame[["trade_date", "code", "reason"]].duplicated().any():
        raise ValueError("SZSE export contains duplicate notice keys")
    frame["code"] = "sz." + frame.code
    for field in ("amount_yuan", "volume_shares"):
        frame[field] = pd.to_numeric(frame[field].str.replace(",", "", regex=False),
                                     errors="raise")
    if (frame[["amount_yuan", "volume_shares"]] < 0).any().any():
        raise ValueError("SZSE export contains negative turnover")
    return frame


def load_complete_notices(calendar: Path, sse_dir: Path, szse_export: Path,
                          start: str = "2024-01-01",
                          end: str = "2025-12-31") -> pd.DataFrame:
    """Combine all exchange notices, preserving each disclosure reason."""
    dates = trading_dates(calendar, start, end)
    szse = read_szse_export(szse_export, start, end)
    if set(szse.trade_date) != set(dates):
        raise ValueError("SZSE export has missing or extra trading dates")
    szse = szse.assign(
        board="szse",
        positive_daily=szse.reason.str.startswith("日价格涨幅"),
        notice_type=szse.reason,
    )[["trade_date", "code", "board", "notice_type", "positive_daily"]]
    sse_rows = []
    for day in dates:
        compact = day.replace("-", "")
        path = sse_dir / f"sse_{compact}.json"
        saved = _check_saved_sse(path, compact)
        for board in SSE_ENDPOINTS:
            for row in saved[board]:
                reason = row["refType"]
                sse_rows.append({
                    "trade_date": day, "code": "sh." + row["secCode"],
                    "board": board, "notice_type": reason,
                    # Official SSE page maps main 11 to daily positive
                    # deviation and STAR 1 to daily positive price change.
                    "positive_daily": reason == ("11" if board == "main" else "1"),
                })
    notices = pd.concat([pd.DataFrame(sse_rows), szse], ignore_index=True)
    if notices.duplicated(["trade_date", "code", "board", "notice_type"]).any():
        raise ValueError("Duplicate exchange notice key")
    if not notices.trade_date.between(start, end).all():
        raise ValueError("Exchange notice escaped requested window")
    return notices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calendar", type=Path, default=Path(
        "data/baostock/market_2020_2026/metadata/calendar.parquet"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/research/lhb/sse_daily"))
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(archive_sse(args.calendar, args.output_dir, args.start, args.end,
                      args.workers))


if __name__ == "__main__":
    main()
