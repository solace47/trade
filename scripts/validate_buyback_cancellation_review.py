"""Check complete, input-only review coverage before buyback-purpose outcomes."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from scripts.probe_buyback_purpose import provisional_purpose


COLUMNS = ("date", "code", "notice_date", "pdf_url",
           "provisional_purpose", "decision", "evidence")
DECISIONS = {"pending", "pure_cancel", "employee", "mixed", "other", "unclear"}


def source_events(source: Path, pairs_path: Path) -> pd.DataFrame:
    pairs = pd.read_parquet(pairs_path)
    events = pairs.loc[pairs.candidate.eq("buyback_plan"),
                       ["date", "code", "notice_date", "pdf_url"]].copy()
    originals = pd.concat([
        pd.read_json(source / f"pdf_audit_{year}.jsonl", lines=True)[
            ["code", "notice_date", "pdf_url", "status",
             "text_first_three_pages"]]
        for year in (2024, 2025)
    ], ignore_index=True)
    if (events.empty or events.pdf_url.duplicated().any()
            or originals.pdf_url.duplicated().any()
            or set(events.date.str[:4]) != {"2024", "2025"}):
        raise ValueError("Incomplete or duplicated frozen buyback originals")
    matched = events.merge(originals, on=["code", "notice_date", "pdf_url"],
                           validate="one_to_one")
    if (len(matched) != len(events) or not matched.status.eq("ok").all()
            or not matched.date.gt(matched.notice_date).all()):
        raise ValueError("Frozen pair lacks its dated original announcement")
    matched["provisional_purpose"] = matched.text_first_three_pages.map(
        provisional_purpose)
    return matched


def validate_rows(review: pd.DataFrame, originals: pd.DataFrame,
                  source: Path | None = None) -> dict:
    if (tuple(review.columns) != COLUMNS or review.pdf_url.duplicated().any()
            or originals.pdf_url.duplicated().any()):
        raise ValueError("Malformed or duplicate purpose review ledger")
    expected = originals.set_index("pdf_url")
    ledger = review.set_index("pdf_url")
    if set(expected.index) != set(ledger.index):
        raise ValueError("Purpose review does not cover exact frozen originals")
    for url, row in ledger.iterrows():
        origin = expected.loc[url]
        if tuple(row[key] for key in COLUMNS[:3] + COLUMNS[4:5]) != tuple(
                origin[key] for key in COLUMNS[:3] + COLUMNS[4:5]):
            raise ValueError("Purpose review source metadata changed")
        if row.decision not in DECISIONS:
            raise ValueError("Unknown purpose review decision")
        if row.decision == "pending":
            if row.evidence.strip():
                raise ValueError("Pending original cannot contain review evidence")
        elif not row.evidence.strip():
            raise ValueError("Reviewed original needs a source-based reason")
        if source is not None:
            name = Path(urlparse(url).path).name
            pdf = source / "pdfs" / row.notice_date[:4] / name
            if not name.lower().endswith(".pdf") or not pdf.is_file():
                raise ValueError("Purpose review original PDF missing locally")

    by_year = {}
    ready = True
    for year in ("2024", "2025"):
        group = review.loc[review.date.str.startswith(year)]
        counts = group.decision.value_counts().to_dict()
        pure = group.loc[group.decision.eq("pure_cancel")]
        by_year[year] = {
            "originals": len(group),
            "decisions": {key: int(counts.get(key, 0))
                          for key in sorted(DECISIONS)},
            "pure_cancel_days": int(pure.date.nunique()),
            "pure_cancel_months": int(pure.date.str[:7].nunique()),
        }
        ready &= (counts.get("pending", 0) == 0
                  and counts.get("unclear", 0) == 0
                  and len(pure) >= 30
                  and pure.date.str[:7].nunique() >= 6)
    return {"by_year": by_year, "ready_for_matching": bool(ready)}


def validate(review_path: Path, source: Path, pairs_path: Path) -> dict:
    review = pd.read_csv(review_path, dtype=str, keep_default_na=False)
    return validate_rows(review, source_events(source, pairs_path), source)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, default=Path(
        "research/buyback_cancellation_review.csv"))
    parser.add_argument("--source", type=Path, default=Path(
        "data/research/buyback"))
    parser.add_argument("--pairs", type=Path, default=Path(
        "data/research/buyback/pairs.parquet"))
    args = parser.parse_args()
    print(validate(args.review, args.source, args.pairs))


if __name__ == "__main__":
    main()
