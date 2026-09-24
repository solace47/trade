"""Archive complete CNINFO sell-plan completion searches by bounded date ranges.

The official query may stop returning later pages for large year-long searches.
This collector partitions until each range has at most six pages, checks all
page totals, and compares the combined count with the full-year first page.
It only collects original notice metadata and never opens price outcomes.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
from pathlib import Path

import pandas as pd

from scripts.collect_buyback_index import PAGE_SIZE, _candidates, _fetch, _range_rows


TERMS = ("减持计划实施完毕", "减持计划实施完成")
MAX_SAFE_PAGES = 6


def _bounded_rows(start: date, end: date, cache: Path,
                  term: str) -> list[dict]:
    first = _fetch(start, end, 1, cache, term)
    if first["totalAnnouncement"] <= MAX_SAFE_PAGES * PAGE_SIZE:
        return _range_rows(start, end, cache, term)
    if start == end:
        raise ValueError("One CNINFO disclosure day exceeds the safe page cap")
    middle = start + timedelta(days=(end - start).days // 2)
    return (_bounded_rows(start, middle, cache, term)
            + _bounded_rows(middle + timedelta(days=1), end, cache, term))


def collect(year: int, base: Path) -> dict:
    if year not in (2024, 2025):
        raise ValueError("Only 2024/2025 exploratory completion notices")
    base.mkdir(parents=True, exist_ok=True)
    cache = base / f"query_cache_{year}"
    sources = []
    counts = {}
    for term in TERMS:
        start, end = date(year, 1, 1), date(year, 12, 31)
        expected = _fetch(start, end, 1, cache, term)["totalAnnouncement"]
        rows = _bounded_rows(start, end, cache, term)
        if len(rows) != expected:
            raise ValueError("Date-partitioned CNINFO count differs from year")
        frame = _candidates(rows)
        counts[term] = {"query_rows": expected, "a_share_pdf": len(frame)}
        sources.append(frame)
    combined = pd.concat(sources, ignore_index=True)
    duplicate = combined.loc[combined.pdf_url.duplicated(keep=False)]
    if (not duplicate.empty and duplicate.groupby("pdf_url")
            [["code", "notice_date", "title"]].nunique().gt(1).any().any()):
        raise ValueError("CNINFO searches disagree about the same PDF")
    combined = combined.drop_duplicates("pdf_url")
    if (combined.empty or combined.pdf_url.duplicated().any()
            or not combined.notice_date.str.startswith(str(year)).all()):
        raise ValueError("Malformed original sell-completion union")
    combined.sort_values(["notice_date", "code", "pdf_url"]).to_parquet(
        base / f"search_{year}.parquet", index=False, compression="zstd")
    report = {"year": year, "terms": counts,
              "unique_a_share_pdf": len(combined)}
    (base / f"search_audit_{year}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, choices=(2024, 2025), required=True)
    parser.add_argument("--base", type=Path, default=Path(
        "data/research/insider_sell_complete"))
    args = parser.parse_args()
    print(collect(args.year, args.base))


if __name__ == "__main__":
    main()
