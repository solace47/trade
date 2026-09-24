"""Archive CNINFO search results for A-share repurchase notices, without returns.

Search results are candidates, not validated new plans. Raw page responses
remain in the ignored cache for later title and original-PDF review.
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
SEARCH = "回购方案"
PAGE_SIZE = 30
MAX_WORKING_PAGE = 100
STOCK = re.compile(r"^(?:00|30|60|68)\d{4}$")


def _fetch(start: date, end: date, page: int, cache: Path,
           searchkey: str = SEARCH) -> dict:
    tag = "" if searchkey == SEARCH else re.sub(r"\W+", "_", searchkey) + "_"
    path = cache / f"{tag}{start}_{end}_{page:03d}.json"
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
        "--data-urlencode", "category=",
        "--data-urlencode", f"seDate={start}~{end}",
        "--data-urlencode", f"searchkey={searchkey}",
        "--data-urlencode", "isHLtitle=true",
    ]
    for attempt in range(3):
        result = subprocess.run(command, capture_output=True, text=True,
                                check=False)
        if result.returncode == 0:
            try:
                payload = json.loads(result.stdout)
                if (isinstance(payload.get("totalAnnouncement"), int)
                        and isinstance(payload.get("announcements"), list)):
                    cache.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(payload, ensure_ascii=False),
                                    encoding="utf-8")
                    return payload
            except json.JSONDecodeError:
                pass
        if attempt < 2:
            time.sleep(attempt + 1)
    raise RuntimeError(f"CNINFO repurchase query failed: {start}–{end}, page {page}")


def _range_rows(start: date, end: date, cache: Path,
                searchkey: str = SEARCH) -> list[dict]:
    first = _fetch(start, end, 1, cache, searchkey)
    pages = math.ceil(first["totalAnnouncement"] / PAGE_SIZE)
    if pages > MAX_WORKING_PAGE:
        if start == end:
            raise RuntimeError(f"A single disclosure day exceeds page limit: {start}")
        middle = start + timedelta(days=(end - start).days // 2)
        return (_range_rows(start, middle, cache, searchkey)
                + _range_rows(middle + timedelta(days=1), end, cache,
                              searchkey))
    rows = first["announcements"][:]
    for page in range(2, pages + 1):
        rows.extend(_fetch(start, end, page, cache, searchkey)["announcements"])
    if len(rows) != first["totalAnnouncement"]:
        raise ValueError(f"Incomplete CNINFO repurchase pages: {start}–{end}")
    return rows


def _candidates(rows: list[dict]) -> pd.DataFrame:
    records = []
    for row in rows:
        code = row.get("secCode", "")
        url = row.get("adjunctUrl", "")
        if not STOCK.fullmatch(code) or not url.lower().endswith(".pdf"):
            continue
        timestamp = pd.to_datetime(row["announcementTime"], unit="ms",
                                   utc=True).tz_convert("Asia/Shanghai")
        records.append({
            "code": ("sh." if code.startswith(("60", "68")) else "sz.") + code,
            "notice_date": timestamp.strftime("%Y-%m-%d"),
            "title": re.sub(r"<[^>]*>", "", row.get("announcementTitle", "")),
            "pdf_url": "https://static.cninfo.com.cn/" + url,
            "announcement_time_ms": row["announcementTime"],
        })
    frame = pd.DataFrame(records)
    if frame.empty or frame.duplicated(["pdf_url"]).any():
        raise ValueError("No unique A-share CNINFO repurchase PDFs")
    return frame.sort_values(["notice_date", "code", "pdf_url"])


def collect(year: int, cache: Path, output: Path,
            searchkey: str = SEARCH) -> dict:
    if year not in (2024, 2025):
        raise ValueError("Only 2024/2025 exploratory announcement years")
    rows = _range_rows(date(year, 1, 1), date(year, 12, 31), cache,
                       searchkey)
    frame = _candidates(rows)
    if not frame.notice_date.str.startswith(str(year)).all():
        raise ValueError("Repurchase search yielded wrong-year rows")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False, compression="zstd")
    return {"year": year, "raw_search_rows": len(rows),
            "a_share_pdf_rows": len(frame),
            "same_code_date_title_pdf_rows": int(frame.duplicated(
                ["code", "notice_date", "title"], keep=False).sum()),
            "unique_stocks": int(frame.code.nunique()),
            "notice_days": int(frame.notice_date.nunique()),
            "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, choices=(2024, 2025), required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--searchkey", default=SEARCH)
    args = parser.parse_args()
    if args.searchkey != SEARCH and (args.cache is None or args.output is None):
        parser.error("A different search word needs an explicit cache and output")
    cache = args.cache or Path(f"data/research/buyback/cninfo_cache_{args.year}")
    output = args.output or Path(f"data/research/buyback/search_{args.year}.parquet")
    print(collect(args.year, cache, output, args.searchkey))


if __name__ == "__main__":
    main()
