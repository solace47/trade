"""Index original 2023/2024 annual-report announcements from CNINFO.

The official announcement URL and disclosure day are kept separately from
vendor financial values. This script does not read prices or future returns.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd


API = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
PAGE_SIZE = 30
MAX_WORKING_PAGE = 100  # CNINFO wraps larger page numbers back to page one.
STOCK = re.compile(r"^(?:00|30|60|68)\d{4}$")
LATER_VERSION = re.compile(r"更正|修订|补充|已取消|取消|英文|更新后|修正")


def _fetch(start: date, end: date, page: int, cache: Path) -> dict:
    path = cache / f"{start}_{end}_{page:03d}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    command = [
        "curl", "-fLsS", "--max-time", "20", "-X", "POST", API,
        "-H", "Content-Type: application/x-www-form-urlencoded; charset=UTF-8",
        "--data-urlencode", "stock=",
        "--data-urlencode", "tabName=fulltext",
        "--data-urlencode", f"pageSize={PAGE_SIZE}",
        "--data-urlencode", f"pageNum={page}",
        "--data-urlencode", "column=sse",
        "--data-urlencode", "category=category_ndbg_szsh",
        "--data-urlencode", f"seDate={start}~{end}",
        "--data-urlencode", "isHLtitle=true",
    ]
    for attempt in range(3):
        result = subprocess.run(command, capture_output=True, text=True,
                                check=False)
        if result.returncode == 0:
            try:
                payload = json.loads(result.stdout)
                if (isinstance(payload.get("totalpages"), int)
                        and isinstance(payload.get("announcements"), list)):
                    cache.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(payload, ensure_ascii=False),
                                    encoding="utf-8")
                    return payload
            except json.JSONDecodeError:
                pass
        if attempt < 2:
            time.sleep(attempt + 1)
    raise RuntimeError(f"CNINFO query failed for {start} to {end}, page {page}")


def _range_rows(start: date, end: date, cache: Path) -> list[dict]:
    first = _fetch(start, end, 1, cache)
    # CNINFO's totalpages is floored when the final page is partial.
    pages = math.ceil(first["totalAnnouncement"] / PAGE_SIZE)
    if pages > MAX_WORKING_PAGE:
        if start == end:
            raise RuntimeError(f"A single day exceeds pagination limit: {start}")
        middle = start + timedelta(days=(end - start).days // 2)
        return (_range_rows(start, middle, cache)
                + _range_rows(middle + timedelta(days=1), end, cache))
    rows = first["announcements"][:]
    for page in range(2, pages + 1):
        rows.extend(_fetch(start, end, page, cache)["announcements"])
    if len(rows) != first["totalAnnouncement"]:
        raise RuntimeError(f"Announcement pagination incomplete: {start} to {end}")
    return rows


def _annual_rows(rows: list[dict], report_year: int) -> pd.DataFrame:
    full = []
    marker = f"{report_year}年年度报告"
    for row in rows:
        code = row.get("secCode", "")
        title = re.sub(r"<[^>]*>", "", row.get("announcementTitle", ""))
        if (not STOCK.fullmatch(code) or marker not in title
                or LATER_VERSION.search(title)):
            continue
        kind = "summary" if "摘要" in title else "full"
        if kind == "full" and not title.endswith((marker, marker + "全文")):
            continue
        url = row.get("adjunctUrl", "")
        if not url or not url.lower().endswith(".pdf"):
            continue
        timestamp = pd.to_datetime(row["announcementTime"], unit="ms", utc=True)
        timestamp = timestamp.tz_convert("Asia/Shanghai")
        full.append({
            "code": ("sh." if code.startswith(("60", "68")) else "sz.") + code,
            "report_year": report_year,
            "notice_date": timestamp.strftime("%Y-%m-%d"),
            "kind": kind,
            "title": title,
            "pdf_url": "https://static.cninfo.com.cn/" + url,
            "announcement_time_ms": row["announcementTime"],
        })
    result = pd.DataFrame(full)
    if result.empty:
        raise ValueError(f"No original annual reports for {report_year}")
    result = result.sort_values(["code", "kind", "announcement_time_ms",
                                 "pdf_url"])
    return result.drop_duplicates(["code", "kind"], keep="first").reset_index(
        drop=True)


def collect(report_year: int, cache: Path, output: Path) -> pd.DataFrame:
    if report_year not in (2023, 2024):
        raise ValueError("Only 2023/2024 annual reports are in the study")
    disclosure_year = report_year + 1
    rows = _range_rows(date(disclosure_year, 1, 1),
                       date(disclosure_year, 12, 31), cache)
    result = _annual_rows(rows, report_year)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False, compression="zstd")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True, choices=(2023, 2024))
    parser.add_argument("--cache", type=Path, default=Path(
        "data/research/annual_cash_quality/cninfo_cache"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path(
        f"data/research/annual_cash_quality/cninfo_original_{args.year}.parquet")
    frame = collect(args.year, args.cache, output)
    print({"report_year": args.year, "rows": len(frame),
           "full_reports": int(frame.kind.eq("full").sum()),
           "summaries": int(frame.kind.eq("summary").sum()),
           "unique_stocks": frame.code.nunique(), "output": str(output)})


if __name__ == "__main__":
    main()
