"""Validate complete CNINFO investigation searches before reading returns.

Titles are deliberately not used to decide whether the issuer was investigated.
Every distinct broad-search PDF must be checked against its original document.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import pandas as pd


SEARCHES = ("notice", "investigation", "broad")
YEARS = (2024, 2025)
FOLLOWUP_TITLE = re.compile(
    r"进展|处罚|结案|撤销|终止|诉讼|仲裁|律师|月报|结果|警示函|"
    r"澄清|回复|问询|受托管理|关注"
)
INITIAL_TITLE = re.compile(r"收到|被|遭|立案告知书|立案决定书")
DISCLOSURE_CAUSE = re.compile(
    r"信息披露违法违规|涉嫌信息披露违法|"
    r"未及时履行信息披露义务|未按规定披露|未按时披露|"
    r"未在(?:法定|规定)期限内披露|未按规定期限披露|"
    r"未按期披露|虚假披露|财务数据.{0,5}虚假记载|"
    r"未及时披露"
)


def initial_title(title: str) -> bool:
    return bool(INITIAL_TITLE.search(title) and not FOLLOWUP_TITLE.search(title))


def screen(source_dir: Path) -> dict:
    report = {}
    for year in YEARS:
        sources = {
            name: pd.read_parquet(source_dir / f"search_{name}_{year}.parquet")
            for name in SEARCHES
        }
        for name, frame in sources.items():
            if (frame.empty or frame.pdf_url.duplicated().any()
                    or not frame.notice_date.str.startswith(str(year)).all()
                    or not frame.title.str.contains("立案").all()):
                raise ValueError(f"Malformed {name} CNINFO investigation search")
        combined = pd.concat(sources.values(), ignore_index=True)
        disagreement = (combined.loc[combined.pdf_url.duplicated(keep=False)]
                        .groupby("pdf_url")[["code", "notice_date", "title",
                                              "announcement_time_ms"]]
                        .nunique().gt(1).any().any())
        if disagreement:
            raise ValueError("Searches disagree on an original investigation PDF")
        broad = sources["broad"]
        for name in ("notice", "investigation"):
            if not set(sources[name].pdf_url).issubset(broad.pdf_url):
                raise ValueError(f"Broad search missed {name} PDF rows")
        broad.sort_values(["notice_date", "code", "pdf_url"]).to_parquet(
            source_dir / f"original_candidates_{year}.parquet", index=False,
            compression="zstd")
        report[str(year)] = {
            "search_pdf_rows": {name: len(frame)
                                for name, frame in sources.items()},
            "unique_original_pdfs_to_audit": len(broad),
            "same_code_date_multiple_pdfs": int(broad.duplicated(
                ["code", "notice_date"], keep=False).sum()),
            "months": int(broad.notice_date.str[:7].nunique()),
        }
    (source_dir / "search_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return report


def audit(source_dir: Path, review_path: Path) -> dict:
    """Require one explicit decision for each initial-title original PDF.

    Positive decisions require matching issuer code and a disclosure cause in
    the original PDF. The review ledger resolves subject and duplicate cases.
    """
    reviews = pd.read_csv(review_path, dtype={"pdf_url": str})
    required = {"pdf_url", "accepted", "reason"}
    if (set(reviews.columns) != required or reviews.pdf_url.duplicated().any()
            or not reviews.accepted.isin([0, 1]).all()
            or reviews.reason.isna().any()
            or reviews.reason.astype(str).str.strip().eq("").any()):
        raise ValueError("Malformed investigation original review ledger")
    marked_urls = set()
    report = {}
    for year in YEARS:
        index = pd.read_parquet(source_dir / f"original_candidates_{year}.parquet")
        originals = pd.read_json(source_dir / f"pdf_audit_{year}.jsonl", lines=True)
        if (index.empty or index.pdf_url.duplicated().any()
                or originals.pdf_url.duplicated().any()
                or set(index.pdf_url) != set(originals.pdf_url)):
            raise ValueError("Incomplete full-text investigation PDF audit")
        joined = index.merge(originals[["pdf_url", "status", "text_full_pdf",
                                        "pages"]], on="pdf_url", how="left",
                             validate="one_to_one")
        marked = joined.loc[joined.title.map(initial_title)].copy()
        marked_urls.update(marked.pdf_url)
        marked = marked.merge(reviews, on="pdf_url", how="left",
                              validate="one_to_one")
        if marked.accepted.isna().any():
            raise ValueError("Initial-title PDF lacks a review decision")
        yes = marked.loc[marked.accepted.eq(1)].copy()
        if (yes.status.ne("ok").any() or yes.pages.isna().any()
                or yes.duplicated(["code", "notice_date"]).any()
                or not yes.apply(lambda row: bool(DISCLOSURE_CAUSE.search(
                    re.sub(r"\s+", "", row.text_full_pdf))), axis=1).all()):
            raise ValueError("Accepted notice lacks unique original evidence")
        confirmed = yes[["code", "notice_date", "title", "pdf_url",
                         "announcement_time_ms"]].sort_values(
                             ["notice_date", "code"])
        confirmed.to_parquet(source_dir / f"confirmed_{year}.parquet",
                             index=False, compression="zstd")
        report[str(year)] = {
            "all_broad_pdfs_extracted": len(joined),
            "pdf_identity_unconfirmed": int(joined.status.ne("ok").sum()),
            "initial_title_pdfs_reviewed": len(marked),
            "rejected_initial_titles": int(marked.accepted.eq(0).sum()),
            "confirmed_issuer_disclosure_notices": len(confirmed),
            "confirmed_stocks": int(confirmed.code.nunique()),
            "confirmed_months": int(confirmed.notice_date.str[:7].nunique()),
        }
    if marked_urls != set(reviews.pdf_url):
        raise ValueError("Review ledger includes noninitial or missing PDFs")
    (source_dir / "source_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(
        "data/research/regulatory_investigation"))
    parser.add_argument("--review", type=Path, default=Path(
        "config/regulatory_investigation_reviews.csv"))
    args = parser.parse_args()
    print({"search": screen(args.source),
           "originals": audit(args.source, args.review)})


if __name__ == "__main__":
    main()
