"""Validate original-notice decisions before any sell-completion outcome read.

The input is the already matched title-only sample. Removing rejected titles
preserves its original no-replacement controls and never rematches on returns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import pandas as pd

from .insider_sell_completion_inputs import EVENT


REQUIRED = {"signal_date", "notice_date", "code", "pdf_url", "decision",
            "actor", "sale_date", "actual_shares", "evidence", "support_url"}
SOURCE_DAY = re.compile(r"/((?:202[3-6])-[01]\d-[0-3]\d)/")


def validate_review(pairs: pd.DataFrame, review: pd.DataFrame,
                    year: int) -> tuple[pd.DataFrame, dict]:
    """Require one complete PDF decision per title-matched signal in a year."""
    if year not in (2024, 2025) or not REQUIRED.issubset(review.columns):
        raise ValueError("Invalid sell-completion review schema or year")
    event = pairs.loc[(pairs.candidate == EVENT)
                      & pairs.date.str.startswith(str(year))].copy()
    controls = pairs.loc[(pairs.candidate == "same_day_nonannouncer")
                         & pairs.date.str.startswith(str(year))].copy()
    decisions = review.loc[review.signal_date.str.startswith(str(year))].copy()
    if (event.empty or event.pdf_url.isna().any()
            or event.pdf_url.duplicated().any()
            or decisions.pdf_url.duplicated().any()
            or set(event.pdf_url) != set(decisions.pdf_url)):
        raise ValueError("Incomplete or duplicate original-PDF review")
    checked = event.merge(decisions, on="pdf_url", how="left",
                          validate="one_to_one", suffixes=("", "_review"))
    for field, other in (("date", "signal_date"),
                         ("notice_date", "notice_date_review"),
                         ("code", "code_review")):
        if not checked[field].eq(checked[other]).all():
            raise ValueError("Review key differs from matched original notice")
    if not checked.decision.str.startswith(("verified_", "reject_")).all():
        raise ValueError("A sell-completion original remains undecided")
    if (checked.actor.fillna("").str.strip().eq("").any()
            or checked.evidence.fillna("").str.len().lt(12).any()):
        raise ValueError("A review lacks actor or original evidence")
    for row in checked.itertuples():
        for url in str(row.support_url or "").split("|"):
            url = url.strip()
            if not url:
                continue
            match = SOURCE_DAY.search(url)
            if match is None or match.group(1) >= row.date:
                raise ValueError("Supporting original was not public by signal day")
    verified = checked.loc[checked.decision.str.startswith("verified_")].copy()
    if (verified.sale_date.fillna("").eq("").any()
            or pd.to_numeric(verified.actual_shares,
                             errors="coerce").le(0).any()
            or pd.to_numeric(verified.actual_shares,
                             errors="coerce").isna().any()):
        raise ValueError("Verified original lacks positive sale and date")
    assigned = controls.merge(
        verified[["date", "code"]].rename(columns={"code": "pair_code"}),
        on=["date", "pair_code"], how="inner", validate="one_to_one")
    if (len(assigned) != len(verified)
            or verified.duplicated(["date", "code"]).any()
            or assigned.duplicated(["date", "code"]).any()):
        raise ValueError("Verified event has missing or reused title control")
    selected = pd.concat([verified[pairs.columns], assigned[pairs.columns]],
                         ignore_index=True)
    months = int(verified.date.str[:7].nunique())
    report = {"year": year, "title_pairs": len(event),
              "verified_pairs": len(verified), "rejected": len(event) - len(verified),
              "signal_days": int(verified.date.nunique()),
              "calendar_months": months,
              "input_gate_passed": len(verified) >= 30 and months >= 6,
              "outcomes_opened": False}
    return selected, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, choices=(2024, 2025), required=True)
    parser.add_argument("--pairs", type=Path, default=Path(
        "data/research/insider_sell_complete/title_only_pairs.parquet"))
    parser.add_argument("--review", type=Path, default=Path(
        "research/insider_sell_completion_review.csv"))
    parser.add_argument("--output", type=Path, default=Path(
        "data/research/insider_sell_complete"))
    args = parser.parse_args()
    pairs = pd.read_parquet(args.pairs)
    review = pd.read_csv(args.review, dtype=str, keep_default_na=False)
    selected, report = validate_review(pairs, review, args.year)
    args.output.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(args.output / f"verified_pairs_{args.year}.parquet",
                        index=False, compression="zstd")
    (args.output / f"review_audit_{args.year}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
