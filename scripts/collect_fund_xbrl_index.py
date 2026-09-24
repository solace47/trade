"""Archive official fund-quarter report metadata without reading market outcomes.

Only stock and mixed funds are indexed. The first date eligible for a 14:50
signal must be strictly later than both the platform upload date and the
report's stated send date; neither is treated as an intraday timestamp.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import pandas as pd
import requests


API = "http://eid.csrc.gov.cn/fund/disclose/advanced_search_xbrl.do"
REFERER = "http://eid.csrc.gov.cn/fund/disclose/index.html"
PAGE_SIZE = 20  # The disclosure platform caps each response at 20 rows.
FUND_TYPES = {"stock": "6020-6010", "mixed": "6020-6040"}
QUARTERS = {1: "FB030010", 2: "FB030020", 3: "FB030030", 4: "FB030040"}
DESCRIPTIONS = {1: "第一季度报告", 2: "第二季度报告",
                3: "第三季度报告", 4: "第四季度报告"}


def _request(year: int, quarter: int, fund_type: str, start: int) -> dict:
    params = [
        {"name": "sEcho", "value": 1},
        {"name": "iDisplayStart", "value": start},
        {"name": "iDisplayLength", "value": PAGE_SIZE},
        {"name": "fundType", "value": FUND_TYPES[fund_type]},
        {"name": "reportTypeCode", "value": QUARTERS[quarter]},
        {"name": "reportYear", "value": str(year)},
    ]
    params.extend({"name": name, "value": ""} for name in (
        "fundCompanyShortName", "fundCode", "fundShortName",
        "startUploadDate", "endUploadDate"))
    headers = {"User-Agent": "Mozilla/5.0", "Referer": REFERER,
               "Accept": "application/json"}
    for attempt in range(3):
        try:
            response = requests.get(API, params={"aoData": json.dumps(params)},
                                    headers=headers, timeout=20)
            response.raise_for_status()
            payload = response.json()
            if (isinstance(payload.get("iTotalRecords"), int)
                    and isinstance(payload.get("aaData"), list)
                    and len(payload["aaData"]) <= PAGE_SIZE):
                return payload
        except (requests.RequestException, ValueError):
            pass
        if attempt < 2:
            time.sleep(attempt + 1)
    raise RuntimeError(f"Fund XBRL query failed: {year} Q{quarter} "
                       f"{fund_type} offset {start}")


def _page(year: int, quarter: int, fund_type: str, start: int,
          cache: Path) -> dict:
    path = cache / f"{year}q{quarter}_{fund_type}_{start:05d}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    payload = _request(year, quarter, fund_type, start)
    cache.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False),
                         encoding="utf-8")
    temporary.replace(path)
    return payload


def _category(year: int, quarter: int, fund_type: str,
              cache: Path, workers: int) -> list[dict]:
    first = _page(year, quarter, fund_type, 0, cache)
    total = first["iTotalRecords"]
    if total <= 0:
        raise ValueError(f"No official fund reports: {year} Q{quarter} {fund_type}")
    offsets = range(PAGE_SIZE, total, PAGE_SIZE)
    pages = {0: first}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_page, year, quarter, fund_type, start, cache): start
                   for start in offsets}
        for future in as_completed(futures):
            pages[futures[future]] = future.result()
    rows = []
    for start in range(0, total, PAGE_SIZE):
        payload = pages[start]
        if payload["iTotalRecords"] != total:
            raise ValueError("Fund report count changed during pagination")
        current = payload["aaData"]
        if len(current) != min(PAGE_SIZE, total - start):
            raise ValueError(f"Fund report page incomplete at offset {start}")
        rows.extend(current)
    if len(rows) != total:
        raise ValueError("Incomplete fund report category")
    for row in rows:
        if (row.get("reportYear") != str(year)
                or row.get("reportDesp") != DESCRIPTIONS[quarter]
                or not str(row.get("fundCode", "")).isdigit()
                or len(str(row["fundCode"])) != 6
                or not isinstance(row.get("uploadInfoId"), int)):
            raise ValueError("Unexpected fund report metadata")
        try:
            upload = date.fromisoformat(row["uploadDate"])
            sent = date.fromisoformat(row["reportSendDate"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Missing or invalid fund report dates") from error
        row["available_after"] = max(upload, sent).isoformat()
        row["fund_type_query"] = fund_type
        row["report_quarter"] = quarter
    return rows


def collect(year: int, quarter: int, cache: Path, output: Path,
            workers: int = 2) -> pd.DataFrame:
    if year not in (2023, 2024, 2025) or quarter not in QUARTERS:
        raise ValueError("Study index is restricted to 2023-2025 fund quarters")
    if workers < 1 or workers > 4:
        raise ValueError("Use 1-4 workers for the public disclosure platform")
    rows = []
    for fund_type in FUND_TYPES:
        rows.extend(_category(year, quarter, fund_type, cache, workers))
    frame = pd.DataFrame(rows)
    if frame.duplicated(["uploadInfoId", "fundCode"]).any():
        raise ValueError("Duplicate fund report metadata across categories")
    frame = frame.sort_values(["fundCode", "uploadInfoId"]).reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False, compression="zstd")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True, choices=(2023, 2024, 2025))
    parser.add_argument("--quarter", type=int, required=True, choices=tuple(QUARTERS))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cache", type=Path,
                        default=Path("data/research/fund_xbrl_index/cache"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path(
        f"data/research/fund_xbrl_index/{args.year}q{args.quarter}.parquet")
    frame = collect(args.year, args.quarter, args.cache, output, args.workers)
    print({"quarter": f"{args.year}Q{args.quarter}", "reports": len(frame),
           "stock": int(frame.fund_type_query.eq("stock").sum()),
           "mixed": int(frame.fund_type_query.eq("mixed").sum()),
           "earliest_usable_after": frame.available_after.min(),
           "latest_usable_after": frame.available_after.max(),
           "output": str(output)})


if __name__ == "__main__":
    main()
