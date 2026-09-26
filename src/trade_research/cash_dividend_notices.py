"""Index historical issuer distribution notices without prices or PDF bulk work.

The search index is evidence of disclosure dates, not proof of the distribution
terms. Each used event still needs its original implementation notice checked.
"""

from __future__ import annotations

import argparse
import calendar
from datetime import date, timedelta
import json
import math
from pathlib import Path
import re
import time

import pandas as pd

from .cash_dividend_catalog import ROOT
from .corporate_cash import API, curl, save_json, sha


KEYWORDS = ("权益分派实施公告", "分红派息实施公告", "利润分配实施公告", "分红实施公告", "派息实施公告",
            "实施公告", "实施的公告", "权益分派公告", "权益分派的公告", "权益分派实施")
PAGE_SIZE = 30
MAX_PAGE = 100
MAINBOARD = re.compile(r"^(?:60|00)\d{4}$")
TITLE = re.compile(
    r"(?:权益分[派配]|权益派发|利润分[配派]|分红(?:派息)?|派息|股息分派|红利分派|现金红利|资本公积(?:金)?转增股本)"
    r".*实施(?:方案)?的?公告"
    r"(?:[（(].*[）)])?(?:\.docx)?(?:V\d+|_\d{4}-\d{2}-\d{2}|\d{4}-\d+)?$"
    r"|权益分[派配]的?公告$"
    r"|权益分派实施$")
REVISION = re.compile(r"更正|修订|补充|已取消|取消|更新|修正")
NON_IMPLEMENTATION = re.compile(r"提议|拟|推迟|参股公司|不实施")


def fetch_page(start: date, end: date, keyword: str, page: int, cache: Path) -> dict:
    if start.year not in (2024, 2025) or end.year != start.year or end < start:
        raise ValueError("Only 2024–2025 announcement inputs are allowed")
    path = cache / keyword / f"{start}_{end}_{page:03d}.json"
    if path.exists():
        payload = json.loads(path.read_text())
    else:
        payload = json.loads(curl(API, {"stock": "", "tabName": "fulltext",
            "pageSize": str(PAGE_SIZE), "pageNum": str(page), "column": "sse",
            "seDate": f"{start}~{end}", "searchkey": keyword, "isHLtitle": "true"}))
    total = payload.get("totalAnnouncement")
    rows = payload.get("announcements")
    if (not isinstance(total, int) or total < 0
            or not (isinstance(rows, list) or (total == 0 and rows is None))):
        raise ValueError("Malformed announcement page is not an empty response")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        save_json(path, payload)
        time.sleep(.1)
    return payload


def range_rows(start: date, end: date, keyword: str, cache: Path) -> list[dict]:
    def split(reason: str) -> list[dict]:
        if start == end:
            raise ValueError(f"Single-day search is not complete ({reason}): {start}")
        middle = start + timedelta(days=(end - start).days // 2)
        return (range_rows(start, middle, keyword, cache)
                + range_rows(middle + timedelta(days=1), end, keyword, cache))

    first = fetch_page(start, end, keyword, 1, cache)
    # The API's totalpages can be floored; page numbers above 100 can wrap.
    pages = math.ceil(first["totalAnnouncement"] / PAGE_SIZE)
    if pages > MAX_PAGE:
        return split("pagination limit")
    rows = list(first.get("announcements") or [])
    for page in range(2, pages + 1):
        payload = fetch_page(start, end, keyword, page, cache)
        if payload["totalAnnouncement"] != first["totalAnnouncement"]:
            # Preserve both inconsistent parent pages, and rebuild disjoint
            # smaller ranges. Never accept a short page as a complete index.
            return split("population changed within pagination")
        rows.extend(payload.get("announcements") or [])
    identities = [(r.get("secCode"), r.get("announcementId"), r.get("adjunctUrl")) for r in rows]
    if len(rows) != first["totalAnnouncement"] or len(set(identities)) != len(rows):
        raise ValueError("Incomplete or repeated announcement search pages")
    for row in rows:
        disclosed = pd.Timestamp(row["announcementTime"], unit="ms", tz="UTC").tz_convert(
            "Asia/Shanghai").date()
        if not start <= disclosed <= end:
            raise ValueError("Search returned an announcement outside the frozen date range")
    return rows


def notice_rows(rows: list[dict]) -> pd.DataFrame:
    records = []
    for row in rows:
        code = row.get("secCode", "")
        title = re.sub(r"<[^>]*>", "", row.get("announcementTitle", "")).strip()
        if (not MAINBOARD.fullmatch(code) or NON_IMPLEMENTATION.search(title)
                or not TITLE.search(re.sub(r"\s+", "", title))):
            continue
        adjunct = row.get("adjunctUrl", "")
        if not adjunct.lower().endswith(".pdf") or not adjunct.startswith("finalpage/"):
            raise ValueError("Distribution notice has no original PDF URL")
        timestamp = pd.Timestamp(row["announcementTime"], unit="ms", tz="UTC").tz_convert(
            "Asia/Shanghai")
        if timestamp.year not in (2024, 2025):
            raise ValueError("Notice is outside the input years")
        records.append({"code": ("sh." if code.startswith("60") else "sz.") + code,
            "notice_date": timestamp.strftime("%Y-%m-%d"),
            "announcement_id": str(row["announcementId"]), "title": title,
            "pdf_url": "https://static.cninfo.com.cn/" + adjunct,
            "revision_flag": bool(REVISION.search(title)),
            "reorganization_flag": "重整" in title, "terms_verified": False})
    columns = ["code", "notice_date", "announcement_id", "title", "pdf_url",
               "revision_flag", "reorganization_flag", "terms_verified"]
    frame = pd.DataFrame(records, columns=columns).drop_duplicates()
    if frame.duplicated(["code", "announcement_id"]).any():
        raise ValueError("The same disclosure identity has conflicting metadata")
    return frame.sort_values(["notice_date", "code", "announcement_id"]).reset_index(drop=True)


def collect(output: Path = ROOT) -> dict:
    cache = output / "notice_index"
    rows, coverage = [], []
    for year in (2024, 2025):
        for month in range(1, 13):
            start = date(year, month, 1)
            end = date(year, month, calendar.monthrange(year, month)[1])
            for keyword in KEYWORDS:
                part = range_rows(start, end, keyword, cache)
                rows.extend(part)
                coverage.append({"start": str(start), "end": str(end),
                                 "keyword": keyword, "search_rows": len(part)})
            print({"through_month": f"{year}-{month:02d}", "search_rows": len(rows)}, flush=True)
    # Issuer/date queries cover unusual titles found in two-way reconciliation.
    supplements = output / "notice_gaps" / "gap_announcement_rows.json"
    if supplements.exists():
        rows.extend(json.loads(supplements.read_text()))
    notices = notice_rows(rows)
    if notices.empty:
        raise ValueError("No implementation notices found")
    notices.to_parquet(output / "notices.parquet", index=False)
    result = {"notices": len(notices), "securities": int(notices.code.nunique()),
              "revisions": int(notices.revision_flag.sum()), "queries": coverage,
              "notices_sha256": sha(output / "notices.parquet"),
              "page_sha256": {str(p): sha(p) for p in sorted(cache.rglob("*.json"))},
              "issuer_supplement_sha256": sha(supplements) if supplements.exists() else None,
              "all_market_completeness_proven": False, "terms_verified": False,
              "holdout_read": False, "strategy_returns_read": False}
    save_json(output / "notice_index_report.json", result)
    return result


def compare_catalog(events: pd.DataFrame, notices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if events.duplicated(["code", "dividOperateDate"]).any():
        raise ValueError("Event dates must be reconciled before notice comparison")
    if notices.duplicated(["code", "announcement_id"]).any():
        raise ValueError("Duplicate primary disclosure identities")
    index = {(code, day): part for (code, day), part in notices.groupby(["code", "notice_date"])}
    output, used = [], set()
    for row in events.to_dict("records"):
        match = index.get((row["code"], row["dividPlanDate"]))
        if match is None:
            ids, urls, revisions = [], [], 0
        else:
            ids, urls = match.announcement_id.tolist(), match.pdf_url.tolist()
            revisions = int(match.revision_flag.sum())
            used.update(zip(match.code, match.announcement_id))
        output.append({**row, "notice_candidates": len(ids), "notice_ids": ids,
            "notice_urls": urls, "notice_revision_candidates": revisions,
            "notice_precedes_action": bool(row["dividPlanDate"] and
                                            row["dividPlanDate"] < row["dividOperateDate"]),
            "terms_verified": False})
    unmatched = notices.loc[[key not in used for key in zip(notices.code, notices.announcement_id)]]
    return pd.DataFrame(output), unmatched.reset_index(drop=True)


def reconcile(output: Path = ROOT) -> dict:
    catalog = json.loads((output / "catalog_report.json").read_text())
    primary = json.loads((output / "notice_index_report.json").read_text())
    if (sha(output / "events.parquet") != catalog["events_sha256"]
            or sha(output / "notices.parquet") != primary["notices_sha256"]):
        raise ValueError("A frozen catalog or disclosure index changed")
    events = pd.read_parquet(output / "events.parquet")
    notices = pd.read_parquet(output / "notices.parquet")
    linked, unmatched = compare_catalog(events, notices)
    linked.to_parquet(output / "events_notices.parquet", index=False)
    unmatched.to_parquet(output / "unmatched_notices.parquet", index=False)
    rows = []
    for (year, kind), part in linked.groupby(["implementation_year", "action_type"]):
        rows.append({"year": year, "action_type": kind, "events": len(part),
            "exact_unique_original_notice": int(((part.notice_candidates == 1)
                & (part.notice_revision_candidates == 0) & part.notice_precedes_action).sum()),
            "no_exact_notice": int((part.notice_candidates == 0).sum()),
            "ambiguous_notice": int((part.notice_candidates > 1).sum()),
            "revision_candidates": int((part.notice_revision_candidates > 0).sum()),
            "missing_vendor_notice_date": int((part.dividPlanDate == "").sum())})
    report = {"by_year_action": rows, "unmatched_primary_notices": len(unmatched),
              "notice_only_keys": int((~unmatched.code.isin(events.code)).sum()),
              "events_sha256": catalog["events_sha256"],
              "notices_sha256": primary["notices_sha256"],
              "linked_sha256": sha(output / "events_notices.parquet"),
              "full_event_coverage_proven": False, "terms_verified": False,
              "holdout_read": False, "strategy_returns_read": False}
    save_json(output / "notice_reconciliation.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", default="collect", choices=("collect", "reconcile"))
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    result = {"collect": collect, "reconcile": reconcile}[args.stage](args.output)
    print({k: v for k, v in result.items() if k not in {"queries", "page_sha256"}})


if __name__ == "__main__":
    main()
