"""Create an input-only, full-membership audit of buyback-plan purpose."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.probe_buyback_purpose import provisional_purpose


def initialize(source: Path, pairs_path: Path, output: Path) -> pd.DataFrame:
    if output.exists():
        raise FileExistsError("Purpose review already exists; do not reset decisions")
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
        raise ValueError("Frozen buyback pair lacks its dated original PDF")
    matched["provisional_purpose"] = matched.text_first_three_pages.map(
        provisional_purpose)
    matched["decision"] = "pending"
    matched["evidence"] = ""
    review = matched[["date", "code", "notice_date", "pdf_url",
                      "provisional_purpose", "decision", "evidence"]]
    review = review.sort_values(["date", "code", "pdf_url"])
    output.parent.mkdir(parents=True, exist_ok=True)
    review.to_csv(output, index=False)
    return review


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("data/research/buyback"))
    parser.add_argument("--pairs", type=Path,
                        default=Path("data/research/buyback/pairs.parquet"))
    parser.add_argument("--output", type=Path,
                        default=Path("research/buyback_cancellation_review.csv"))
    args = parser.parse_args()
    review = initialize(args.source, args.pairs, args.output)
    print({"originals": len(review),
           "provisional": review.provisional_purpose.value_counts().to_dict()})


if __name__ == "__main__":
    main()
