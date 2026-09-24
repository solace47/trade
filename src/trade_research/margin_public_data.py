"""Archive official SSE/SZSE stock-level margin balances by trade date.

The exchange publishes a day's balances before the following session.  A
downstream 14:50 signal must join each record to that *following* session.
The archive preserves the response rows and XLSX; it is local research data.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import pandas as pd
import requests

from .exchange_public_events import XLSX_NS, _xlsx_cell, trading_dates


SSE_URL = "https://query.sse.com.cn/marketdata/tradedata/queryMargin.do"
SZSE_URL = "https://www.szse.cn/api/report/ShowReport"
SSE_HEADERS = {"Referer": "https://www.sse.com.cn/", "User-Agent": "Mozilla/5.0"}
SZSE_HEADERS = {"Referer": "https://www.szse.cn/", "User-Agent": "Mozilla/5.0"}
SZSE_HEADER = ("证券代码", "证券简称", "融资买入额(元)", "融资余额(元)",
               "融券卖出量(股/份)", "融券余量(股/份)", "融券余额(元)",
               "融资融券余额(元)")


def _numeric(value: str | int | None) -> int:
    if value is None:
        raise ValueError("Missing exchange numeric value")
    if isinstance(value, bool):
        raise ValueError("Boolean exchange numeric value")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str) and re.fullmatch(r"\d[\d,]*", value):
        result = int(value.replace(",", ""))
    else:
        raise ValueError(f"Unexpected exchange numeric value: {value!r}")
    if result < 0:
        raise ValueError("Negative exchange numeric value")
    return result


def _validate_sse(rows: list[dict], day: str, expected: int) -> None:
    if not rows or len(rows) != expected:
        raise ValueError(f"Incomplete SSE margin date {day}: {len(rows)}/{expected}")
    codes: set[str] = set()
    for row in rows:
        code = row.get("stockCode", "")
        if row.get("opDate") != day or not re.fullmatch(r"\d{6}", code):
            raise ValueError(f"Invalid SSE margin code/date on {day}")
        if code in codes:
            raise ValueError(f"Duplicate SSE margin code on {day}: {code}")
        codes.add(code)
        for field in ("rzye", "rzmre", "rzche", "rqyl", "rqmcl"):
            _numeric(row.get(field))


def fetch_sse_day(day: str, session: requests.Session | None = None) -> dict:
    """Read every page; the endpoint currently caps a page at 2,000 rows."""
    if not re.fullmatch(r"20\d{6}", day):
        raise ValueError("SSE trade date must be YYYYMMDD")
    client = session or requests.Session()
    rows: list[dict] = []
    expected: int | None = None
    pages: int | None = None
    page = 1
    while True:
        response = client.get(
            SSE_URL,
            params={"isPagination": "true", "tabType": "mxtype", "detailsDate": day,
                    "pageHelp.pageSize": 5000, "pageHelp.pageNo": page,
                    "pageHelp.beginPage": page, "pageHelp.endPage": page,
                    "pageHelp.cacheSize": 1},
            headers=SSE_HEADERS, timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("actionErrors") or not isinstance(data.get("pageHelp"), dict):
            raise ValueError(f"SSE margin response error on {day}")
        helper = data["pageHelp"]
        if expected is None:
            expected = int(helper["total"])
            pages = int(helper["pageCount"])
            if expected <= 0 or pages <= 0:
                raise ValueError(f"Empty SSE margin date {day}")
        elif int(helper["total"]) != expected or int(helper["pageCount"]) != pages:
            raise ValueError(f"SSE pagination changed mid-request on {day}")
        if int(helper["pageNo"]) != page or not isinstance(helper.get("data"), list):
            raise ValueError(f"Wrong SSE margin page on {day}")
        rows.extend(helper["data"])
        if page == pages:
            break
        page += 1
    assert expected is not None and pages is not None
    _validate_sse(rows, day, expected)
    return {"date": f"{day[:4]}-{day[4:6]}-{day[6:]}",
            "source": "sse", "total": expected, "pages": pages, "rows": rows}


def read_szse_day(content: bytes, day: str) -> pd.DataFrame:
    """Parse a single official daily export and reject schema/row corruption."""
    with ZipFile(io.BytesIO(content)) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(t.text or "" for t in item.findall(".//m:t", XLSX_NS))
                      for item in root.findall("m:si", XLSX_NS)]
        root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        sheet = root.find("m:sheetData", XLSX_NS)
        if sheet is None:
            raise ValueError(f"SZSE margin worksheet empty on {day}")
        values = []
        for row in sheet.findall("m:row", XLSX_NS):
            cells = [""] * len(SZSE_HEADER)
            for cell in row.findall("m:c", XLSX_NS):
                match = re.fullmatch(r"([A-H])\d+", cell.get("r", ""))
                if match:
                    cells[ord(match.group(1)) - ord("A")] = _xlsx_cell(cell, shared)
            values.append(cells)
    if not values or tuple(values[0]) != SZSE_HEADER or len(values) < 2:
        raise ValueError(f"SZSE margin header/rows changed on {day}")
    frame = pd.DataFrame(values[1:], columns=("code", "name", "buy_yuan",
                                                    "balance_yuan", "short_sell_shares",
                                                    "short_balance_shares", "short_balance_yuan",
                                                    "total_balance_yuan"))
    if not frame.code.str.fullmatch(r"\d{6}").all() or frame.code.duplicated().any():
        raise ValueError(f"Malformed/duplicate SZSE margin code on {day}")
    for column in frame.columns[2:]:
        frame[column] = frame[column].map(_numeric)
    if not (frame.balance_yuan + frame.short_balance_yuan ==
            frame.total_balance_yuan).all():
        raise ValueError(f"SZSE margin totals disagree on {day}")
    frame.insert(0, "date", day)
    return frame


def fetch_szse_day(day: str, session: requests.Session | None = None) -> bytes:
    client = session or requests.Session()
    response = client.get(SZSE_URL,
                          params={"SHOWTYPE": "xlsx", "CATALOGID": "1837_xxpl",
                                  "TABKEY": "tab2", "txtDate": day},
                          headers=SZSE_HEADERS, timeout=30)
    response.raise_for_status()
    read_szse_day(response.content, day)
    return response.content


def _read_saved_sse(path: Path, day: str) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        record = json.load(handle)
    if record.get("date") != day or record.get("source") != "sse":
        raise ValueError(f"Wrong SSE archive date/source: {path}")
    _validate_sse(record["rows"], day.replace("-", ""), int(record["total"]))
    return record


def archive_margin(calendar: Path, output_dir: Path, start: str, end: str,
                   workers: int = 4) -> dict:
    if workers < 1 or workers > 6:
        raise ValueError("Use 1 to 6 workers for the exchange endpoints")
    days = trading_dates(calendar, start, end)
    sse_dir = output_dir / "sse_daily"
    szse_dir = output_dir / "szse_daily"
    sse_dir.mkdir(parents=True, exist_ok=True)
    szse_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[str, str, Path]] = []
    cached = {"sse": 0, "szse": 0}
    for day in days:
        for exchange, path in (("sse", sse_dir / f"{day}.json.gz"),
                               ("szse", szse_dir / f"{day}.xlsx")):
            if path.exists():
                if exchange == "sse":
                    _read_saved_sse(path, day)
                else:
                    read_szse_day(path.read_bytes(), day)
                cached[exchange] += 1
            else:
                jobs.append((exchange, day, path))

    def fetch(job: tuple[str, str, Path]) -> tuple[tuple[str, str, Path], bytes]:
        exchange, day, _ = job
        for attempt in range(3):
            try:
                if exchange == "szse":
                    return job, fetch_szse_day(day)
                record = fetch_sse_day(day.replace("-", ""))
                data = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                return job, gzip.compress((data + "\n").encode("utf-8"))
            except (requests.RequestException, ValueError, OSError):
                if attempt == 2:
                    raise
                time.sleep(1.5 * (attempt + 1))
        raise AssertionError("unreachable")

    completed = {"sse": 0, "szse": 0}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch, job): job for job in jobs}
        for future in as_completed(futures):
            (exchange, _, path), content = future.result()
            temporary = path.with_name(path.name + ".tmp")
            temporary.write_bytes(content)
            temporary.replace(path)
            completed[exchange] += 1
            count = completed["sse"] + completed["szse"]
            if count % 50 == 0 or count == len(jobs):
                print(f"Margin files {count}/{len(jobs)} new; {cached} cached",
                      flush=True)
    return {"expected_days": len(days), "cached": cached, "new": completed}


def load_margin(calendar: Path, archive_dir: Path, start: str,
                end: str) -> pd.DataFrame:
    """Load every expected date; missing files/rows stop downstream research."""
    result = []
    for day in trading_dates(calendar, start, end):
        sse = _read_saved_sse(archive_dir / "sse_daily" / f"{day}.json.gz", day)
        sh = pd.DataFrame(sse["rows"])[
            ["stockCode", "rzmre", "rzye", "rqyl", "rqmcl"]
        ]
        sh.columns = ["code", "buy_yuan", "balance_yuan",
                      "short_balance_shares", "short_sell_shares"]
        sh["code"] = "sh." + sh.code
        for column in ("buy_yuan", "balance_yuan", "short_balance_shares",
                       "short_sell_shares"):
            sh[column] = sh[column].map(_numeric)
        sh["short_balance_yuan"] = pd.Series([pd.NA] * len(sh), dtype="Int64")
        sh.insert(0, "trade_date", day)
        sz = read_szse_day((archive_dir / "szse_daily" / f"{day}.xlsx").read_bytes(), day)
        sz = sz[["code", "buy_yuan", "balance_yuan",
                 "short_balance_shares", "short_sell_shares",
                 "short_balance_yuan"]].copy()
        sz["code"] = "sz." + sz.code
        sz.insert(0, "trade_date", day)
        result.extend((sh, sz))
    frame = pd.concat(result, ignore_index=True)
    if frame.duplicated(["trade_date", "code"]).any():
        raise ValueError("Duplicate market-wide margin stock-day")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calendar", type=Path, default=Path(
        "data/baostock/market_2020_2026/metadata/calendar.parquet"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/research/margin"))
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--merged-output", type=Path,
                        default=Path("data/research/margin/margin_2024_2025.parquet"))
    parser.add_argument("--merge-only", action="store_true",
                        help="Validate and merge already archived exchange days")
    args = parser.parse_args()
    if not args.merge_only:
        print(archive_margin(args.calendar, args.output_dir, args.start, args.end,
                             args.workers))
    merged = load_margin(args.calendar, args.output_dir, args.start, args.end)
    args.merged_output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.merged_output, index=False, compression="zstd")
    print({"merged_rows": len(merged), "merged_dates": merged.trade_date.nunique(),
           "merged_output": str(args.merged_output)})


if __name__ == "__main__":
    main()
