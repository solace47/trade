"""Extract publicly disclosed fund top-ten stocks from original XBRL HTML.

This is a source audit, not an outcome test or a complete ownership measure.
Original HTML is cached under ignored data/ for repeatable parsing.
"""

from __future__ import annotations

import argparse
import gzip
import html
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd
import requests


BASE = "http://eid.csrc.gov.cn"
REFERER = BASE + "/fund/disclose/index.html"
SECTION = re.compile(r'<a\s+name="tabItem4_topTenStockDetal"[^>]*>.*?'
                     r'(?=<a\s+name=|\Z)', re.I | re.S)
DATA_ROW = re.compile(r'<tr\s+class="dd"[^>]*>(.*?)</tr>', re.I | re.S)
CELL = re.compile(r'<td\b[^>]*>(.*?)</td>', re.I | re.S)
TAG = re.compile(r"<[^>]*>")
A_SHARE = re.compile(r"^(?:00|30|60|68)\d{4}$")
SEND_DATE = re.compile(r"报告送出日期\s*[：:]\s*(\d{4}-\d{2}-\d{2})")


def _plain(raw: str) -> str:
    return " ".join(html.unescape(TAG.sub("", raw)).split())


def parse_holdings(document: str, report_sent: str) -> tuple[list[dict], int]:
    """Return A-share rows and total top-ten rows; reject unknown layouts."""
    cover = _plain(document[:8000])
    dates = SEND_DATE.findall(cover)
    if len(dates) != 1 or dates[0] != report_sent:
        raise ValueError("Fund report cover and indexed send dates differ")
    sections = SECTION.findall(document)
    if len(sections) != 1:
        raise ValueError("Missing or duplicated top-ten stock section")
    section = sections[0]
    if not all(label in section for label in (
            "股票代码", "数量（股）", "公允价值（元）")):
        raise ValueError("Unknown fund top-ten table header")
    rows = DATA_ROW.findall(section)
    if not rows:
        if "注：无" in _plain(section) or "未持有股票" in _plain(section):
            return [], 0
        raise ValueError("Top-ten stock table is empty without an explicit note")
    if len(rows) > 20:
        raise ValueError("Unexpected number of top-ten stock rows")
    holdings = []
    ranks = []
    for raw in rows:
        fields = [_plain(cell) for cell in CELL.findall(raw)]
        if len(fields) != 6:
            raise ValueError("Unknown fund top-ten stock row layout")
        rank, code, name, shares, value, nav_weight = fields
        if not rank.isdigit() or not name:
            raise ValueError("Invalid fund stock rank or name")
        ranks.append(int(rank))
        try:
            shares_decimal = Decimal(shares.replace(",", ""))
            if shares_decimal != shares_decimal.to_integral_value():
                raise ValueError("Fractional fund stock shares")
            shares_int = int(shares_decimal)
            value_float = float(value.replace(",", ""))
            weight_float = float(nav_weight.replace(",", "").rstrip("%"))
        except (ValueError, InvalidOperation) as error:
            raise ValueError("Invalid fund stock quantity or value") from error
        if shares_int < 0 or value_float < 0 or weight_float < 0:
            raise ValueError("Negative fund stock quantity or value")
        if A_SHARE.fullmatch(code):
            holdings.append({
                "code": ("sh." if code.startswith(("60", "68")) else "sz.") + code,
                "rank": int(rank), "shares": shares_int,
                "fair_value_yuan": value_float, "nav_weight_pct": weight_float,
            })
    if ranks != list(range(1, len(ranks) + 1)):
        raise ValueError("Unexpected top-ten stock ordering")
    return holdings, len(rows)


def _html_source(report_id: int, cache: Path) -> str:
    path = cache / f"{report_id}.html.gz"
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as source:
            return source.read()
    url = f"{BASE}/fund/disclose/instance_html_view.do?instanceid={report_id}"
    for attempt in range(3):
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0",
                                                  "Referer": REFERER}, timeout=25)
            response.raise_for_status()
            if ("/xbrl/REPORT/HTML/" in response.url
                    and len(response.content) > 10000
                    and "tabItem4_topTenStockDetal" in response.text):
                cache.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".tmp")
                with gzip.open(temporary, "wt", encoding="utf-8") as target:
                    target.write(response.text)
                temporary.replace(path)
                return response.text
        except (requests.RequestException, OSError):
            pass
        if attempt < 2:
            time.sleep(attempt + 1)
    raise RuntimeError(f"Original fund report unavailable: {report_id}")


def _one(row: dict, cache: Path) -> tuple[list[dict], dict]:
    report_id = int(row["uploadInfoId"])
    base = {"uploadInfoId": report_id, "fundCode": row["fundCode"],
            "reportYear": int(row["reportYear"]),
            "report_quarter": int(row["report_quarter"]),
            "available_after": row["available_after"],
            "fund_type_query": row["fund_type_query"]}
    try:
        holdings, total = parse_holdings(
            _html_source(report_id, cache), row["reportSendDate"])
        report = {**base, "status": "parsed", "top_rows": total,
                  "a_share_rows": len(holdings)}
        return [{**base, **item} for item in holdings], report
    except (ValueError, RuntimeError, OSError,
            requests.RequestException) as error:
        return [], {**base, "status": "rejected", "reason": str(error),
                    "top_rows": None, "a_share_rows": None}


def extract(index: Path, cache: Path, output: Path, audit: Path,
            workers: int = 2, limit: int | None = None) -> dict:
    if workers < 1 or workers > 4:
        raise ValueError("Use 1-4 workers for the public disclosure platform")
    metadata = pd.read_parquet(index)
    if metadata.duplicated("uploadInfoId").any():
        raise ValueError("Duplicate original fund report IDs in index")
    if limit is not None:
        if limit < 1:
            raise ValueError("Limit must be positive")
        metadata = metadata.head(limit)
    holdings, audits = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_one, row, cache): row["uploadInfoId"]
                   for row in metadata.to_dict("records")}
        for future in as_completed(futures):
            rows, audit_row = future.result()
            holdings.extend(rows)
            audits.append(audit_row)
    if len(audits) != len(metadata):
        raise ValueError("Fund report extraction incomplete")
    output.parent.mkdir(parents=True, exist_ok=True)
    audit.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(holdings).to_parquet(output, index=False, compression="zstd")
    pd.DataFrame(audits).sort_values("uploadInfoId").to_parquet(
        audit, index=False, compression="zstd")
    return {"reports": len(audits),
            "parsed": sum(row["status"] == "parsed" for row in audits),
            "rejected": sum(row["status"] == "rejected" for row in audits),
            "a_share_holdings": len(holdings), "limited": limit is not None,
            "output": str(output), "audit": str(audit)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cache", type=Path,
                        default=Path("data/research/fund_xbrl_holdings/cache"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    stem = args.index.stem + (f"_pilot{args.limit}" if args.limit else "")
    base = Path("data/research/fund_xbrl_holdings")
    result = extract(args.index, args.cache,
                     args.output or base / f"{stem}.parquet",
                     args.audit or base / f"{stem}_audit.parquet",
                     args.workers, args.limit)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
