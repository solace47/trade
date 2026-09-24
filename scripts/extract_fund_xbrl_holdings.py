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
INDEX_SECTION = re.compile(r'<a\s+name="tabItem4_QMZSTZAGYJZTZMX"[^>]*>.*?'
                           r'(?=<a\s+name=|\Z)', re.I | re.S)
ACTIVE_SECTION = re.compile(r'<a\s+name="tabItem4_QMJJTZAGYJZTZMX"[^>]*>.*?'
                            r'(?=<a\s+name=|\Z)', re.I | re.S)
ACTIVE_INDUSTRY_SECTION = re.compile(r'<a\s+name="tabItem4_BBQMJJTZ"[^>]*>.*?'
                                     r'(?=<a\s+name=|\Z)', re.I | re.S)
ASSET_SECTION = re.compile(r'<a\s+name="tabItem4_assetsCircs"[^>]*>.*?'
                           r'(?=<a\s+name=|\Z)', re.I | re.S)
FOREIGN_SECTION = re.compile(
    r'<a\s+name="tabItem7_topTen(?:JJTZ)?StockDetal"[^>]*>.*?'
    r'(?=<a\s+name=|\Z)', re.I | re.S)
DATA_ROW = re.compile(r'<tr\s+class="dd"[^>]*>(.*?)</tr>', re.I | re.S)
CELL = re.compile(r'<td\b[^>]*>(.*?)</td>', re.I | re.S)
TAG = re.compile(r"<[^>]*>")
A_SHARE = re.compile(r"^(?:00|30|60|68)\d{4}$")
SEND_DATE = re.compile(r"报告送出日期\s*[：:]\s*(\d{4}-\d{2}-\d{2})")


def _plain(raw: str) -> str:
    return " ".join(html.unescape(TAG.sub("", raw)).split())


def _stocks_explicitly_absent(document: str) -> bool:
    """Verify a blank top-ten table against the separate asset summary."""
    sections = ASSET_SECTION.findall(document)
    if len(sections) != 1:
        return False
    for raw in DATA_ROW.findall(sections[0]):
        fields = [_plain(cell) for cell in CELL.findall(raw)]
        if "其中：股票" in fields or "权益投资" in fields:
            return fields[-2:] == ["-", "-"]
    return False


def _active_stocks_explicitly_absent(document: str) -> bool:
    """Cross-check an unannotated empty active table against industry totals."""
    sections = ACTIVE_INDUSTRY_SECTION.findall(document)
    if len(sections) != 1 or "积极投资按行业分类" not in sections[0]:
        return False
    rows = [[_plain(cell) for cell in CELL.findall(raw)]
            for raw in DATA_ROW.findall(sections[0])]
    return (len(rows) >= 2 and rows[-1] == ["", "合计", "-", "-"]
            and all(len(row) == 4 and row[-2:] == ["-", "-"]
                    for row in rows))


def _foreign_only_report(document: str) -> bool:
    """Recognize the separate template only with verified HK exchange rows."""
    sections = FOREIGN_SECTION.findall(document)
    if not sections:
        return False
    checked = 0
    for section in sections:
        if "证券代码" not in section or "所在证券市场" not in section:
            return False
        for raw in DATA_ROW.findall(section):
            fields = [_plain(cell) for cell in CELL.findall(raw)]
            if len(fields) != 9:
                return False
            code, market = fields[3:5]
            if A_SHARE.fullmatch(code) or "香港联合交易所" not in market:
                return False
            if code and code != "-":
                checked += 1
    return checked > 0


def _stock_rows(section: str) -> list[dict]:
    """Read every reported stock, including non-A shares needed for ranking."""
    if not all(label in section for label in (
            "股票代码", "数量（股）", "公允价值（元）")):
        raise ValueError("Unknown fund top-ten table header")
    rows = DATA_ROW.findall(section)
    if len(rows) > 20:
        raise ValueError("Unexpected number of top-ten stock rows")
    stocks = []
    ranks = []
    previous_code = ""
    for raw in rows:
        fields = [_plain(cell) for cell in CELL.findall(raw)]
        if len(fields) != 6:
            raise ValueError("Unknown fund top-ten stock row layout")
        rank, code, name, shares, value, nav_weight = fields
        if [code, name, shares, value, nav_weight] == ["-"] * 5:
            continue  # Some reports fill unused ranks with dashes.
        if rank == "-" and ranks and (
                bool(A_SHARE.fullmatch(code)) !=
                bool(A_SHARE.fullmatch(previous_code))):
            rank_number = ranks[-1]  # A+H split may omit the second rank.
        elif rank.isdigit():
            rank_number = int(rank)
        else:
            raise ValueError("Invalid fund stock rank or name")
        if not name:
            raise ValueError("Invalid fund stock rank or name")
        ranks.append(rank_number)
        previous_code = code
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
        stocks.append({
                "raw_code": code,
                "rank": rank_number, "shares": shares_int,
                "fair_value_yuan": value_float, "nav_weight_pct": weight_float,
            })
    if not ranks:
        return []
    # Split securities and tied positions use either dense or competition ranks.
    if ranks[0] != 1:
        raise ValueError("Unexpected top-ten stock ordering")
    current, run = ranks[0], 1
    for later in ranks[1:]:
        if later == current:
            run += 1
        elif later in (current + 1, current + run):
            current, run = later, 1
        else:
            raise ValueError("Unexpected top-ten stock ordering")
    return stocks


def _a_share_rows(stocks: list[dict]) -> list[dict]:
    return [{"code": ("sh." if item["raw_code"].startswith(("60", "68"))
                      else "sz.") + item["raw_code"],
             **{key: value for key, value in item.items() if key != "raw_code"}}
            for item in stocks if A_SHARE.fullmatch(item["raw_code"])]


def _split_top_ten(document: str) -> tuple[list[dict], int]:
    """Reconstruct combined top ten only where capped source tables suffice."""
    # Older index-fund reports use the generic top-ten anchor for the index
    # book, with a separate active-investment table. Their active positions
    # can exceed the generic table's tenth value, so that table is not the
    # combined fund top ten.
    index_sections = INDEX_SECTION.findall(document) + SECTION.findall(document)
    active_sections = ACTIVE_SECTION.findall(document)
    if len(index_sections) != 1 or len(active_sections) != 1:
        raise ValueError("Missing or duplicated index/active stock section")
    index_section, active_section = index_sections[0], active_sections[0]
    if not ("前十名股票投资明细" in index_section
            and "积极投资" in active_section
            and "前五名股票投资明细" in active_section):
        raise ValueError("Unknown index/active stock section heading")
    index_stocks = _stock_rows(index_section)
    active_stocks = _stock_rows(active_section)
    books = []
    for kind, section, stocks in (("index", index_section, index_stocks),
                                  ("active", active_section, active_stocks)):
        if not stocks and not (
                "注：无" in _plain(section)
                or f"未持有{'指数投资' if kind == 'index' else '积极投资'}"
                in _plain(section)
                or (kind == "index" and "未持有股票资产" in _plain(section))
                or (kind == "active" and _active_stocks_explicitly_absent(document))
                or _stocks_explicitly_absent(document)):
            raise ValueError(f"Empty {kind} stock table without a source note")
        ranks = list(dict.fromkeys(item["rank"] for item in stocks))
        groups = [
            {"stocks": [item for item in stocks if item["rank"] == rank],
             "value": sum(Decimal(str(item["fair_value_yuan"]))
                          for item in stocks if item["rank"] == rank)}
            for rank in ranks
        ]
        if any(len(group["stocks"]) > 1 and
               (len(group["stocks"]) != 2 or
                sum(bool(A_SHARE.fullmatch(item["raw_code"]))
                    for item in group["stocks"]) != 1)
               for group in groups):
            raise ValueError(f"Unresolved tied {kind} stock rank")
        if any(left["value"] < right["value"]
               for left, right in zip(groups, groups[1:])):
            raise ValueError(f"Unsorted {kind} stock table")
        books.append(groups)
    index_groups, active_groups = books
    combined = index_groups + active_groups
    codes = [item["raw_code"] for group in combined
             for item in group["stocks"]]
    if len(codes) != len(set(codes)):
        raise ValueError("Duplicate stock across index/active tables")
    combined.sort(key=lambda group: group["value"], reverse=True)
    if len(combined) > 10 and combined[9]["value"] == combined[10]["value"]:
        raise ValueError("Combined stock tenth-place tie is unresolved")
    if len(active_groups) >= 5 and (len(combined) < 10
                                    or active_groups[-1]["value"]
                                    >= combined[9]["value"]):
        raise ValueError("Capped active table cannot establish combined top ten")
    selected = [{**item, "rank": rank}
                for rank, group in enumerate(combined[:10], start=1)
                for item in group["stocks"]]
    return _a_share_rows(selected), len(selected)


def parse_holdings(document: str, report_sent: str) -> tuple[list[dict], int]:
    """Return A-share rows and total top-ten rows; reject unknown layouts."""
    cover = _plain(document[:8000])
    dates = SEND_DATE.findall(cover)
    if len(dates) != 1 or dates[0] != report_sent:
        raise ValueError("Fund report cover and indexed send dates differ")
    sections = SECTION.findall(document)
    if INDEX_SECTION.search(document) or ACTIVE_SECTION.search(document):
        return _split_top_ten(document)
    if not sections and _foreign_only_report(document):
        return [], 0
    foreign = FOREIGN_SECTION.findall(document)
    if (not sections and len(foreign) == 1
            and not DATA_ROW.findall(foreign[0])
            and "未持有股票及存托凭证" in _plain(foreign[0])
            and _stocks_explicitly_absent(document)):
        return [], 0
    if len(sections) != 1:
        raise ValueError("Missing or duplicated top-ten stock section")
    section = sections[0]
    stocks = _stock_rows(section)
    if not stocks and not ("注：无" in _plain(section)
                           or "未持有股票" in _plain(section)
                           or _stocks_explicitly_absent(document)):
        raise ValueError("Top-ten stock table is empty without an explicit note")
    return _a_share_rows(stocks), len(stocks)


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
                    and ("tabItem4_topTenStockDetal" in response.text
                         or "tabItem7_topTenStockDetal" in response.text
                         or ("tabItem4_QMZSTZAGYJZTZMX" in response.text
                             and "tabItem4_QMJJTZAGYJZTZMX" in response.text))):
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
    base = {"uploadInfoId": report_id, "fundId": int(row["fundId"]),
            "fundCode": row["fundCode"],
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
