"""Archive complete CNINFO controller pledge-release notice titles, no returns.

The conservative title screen requires one pledge action in the title. The
original PDF must still prove an executed release by the direct controller.
"""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import unicodedata

from scripts.collect_buyback_index import _candidates, _fetch
from scripts.collect_sell_completion_index import _bounded_rows


SEARCH = "解除质押"
EXCLUDED = ("一致行动人", "可转换公司债券", "可交换公司债券",
            "可转债", "可交债", "拟办理", "拟解除")


def candidate_title(title: str) -> bool:
    normalized = unicodedata.normalize("NFKC", title)
    return (
        "控股股东" in normalized
        and "解除质押" in normalized
        and normalized.count("质押") == 1
        and not any(word in normalized for word in EXCLUDED)
    )


def collect(year: int, root: Path) -> dict:
    if year not in (2024, 2025):
        raise ValueError("Only 2024/2025 exploratory pledge releases")
    root.mkdir(parents=True, exist_ok=True)
    cache = root / f"query_cache_{year}"
    start, end = date(year, 1, 1), date(year, 12, 31)
    expected = _fetch(start, end, 1, cache, SEARCH)["totalAnnouncement"]
    rows = _bounded_rows(start, end, cache, SEARCH)
    if len(rows) != expected:
        raise ValueError("Incomplete original pledge-release search")
    indexed = _candidates(rows)
    if not indexed.notice_date.str.startswith(str(year)).all():
        raise ValueError("Wrong-year pledge-release original")
    titles = indexed.loc[indexed.title.map(candidate_title)].copy()
    if titles.empty or titles.pdf_url.duplicated().any():
        raise ValueError("Missing or duplicate direct-controller titles")
    indexed.to_parquet(root / f"search_{year}.parquet", index=False,
                       compression="zstd")
    titles.to_parquet(root / f"title_candidates_{year}.parquet", index=False,
                      compression="zstd")
    report = {"year": year, "fulltext_query_rows": expected,
              "a_share_pdf": len(indexed),
              "conservative_controller_titles": len(titles),
              "title_stocks": int(titles.code.nunique()),
              "outcomes_opened": False}
    (root / f"search_audit_{year}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, choices=(2024, 2025), required=True)
    parser.add_argument("--output", type=Path, default=Path(
        "data/research/pledge"))
    args = parser.parse_args()
    print(collect(args.year, args.output))


if __name__ == "__main__":
    main()
