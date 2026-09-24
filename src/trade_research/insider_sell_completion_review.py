"""Verify every eligible original notice, then rebuild outcome-blind pairs.

Title-only capacity and controls are diagnostic: a rejected title must not use
one of the five daily slots, trigger cooldown, or become a nonannouncer peer.
This module reads only notices, prior daily bars and same-day 14:50 snapshots.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import pandas as pd

from .buyback_inputs import CONTROL, _match, _universe
from .exchange_public_events import trading_dates
from .insider_sell_completion_inputs import GOOD_IDENTITY, _events


EVENT = "verified_sell_completion"
REQUIRED = {"signal_date", "notice_date", "code", "pdf_url", "decision",
            "actor", "sale_date", "actual_shares", "evidence", "support_url"}
SOURCE_DAY = re.compile(r"/((?:202[3-6])-[01]\d-[0-3]\d)/")


def validate_review(eligible: pd.DataFrame, review: pd.DataFrame,
                    year: int) -> tuple[pd.DataFrame, dict]:
    """Require one original-PDF decision per 14:50-eligible title in a year."""
    if year not in (2024, 2025) or not REQUIRED.issubset(review.columns):
        raise ValueError("Invalid sell-completion review schema or year")
    event = eligible.loc[eligible.date.str.startswith(str(year))].copy()
    decisions = review.loc[review.signal_date.str.startswith(str(year))].copy()
    if (event.empty or event.pdf_url.isna().any()
            or event.pdf_url.duplicated().any()
            or event.duplicated(["date", "code"]).any()
            or decisions.pdf_url.duplicated().any()
            or set(event.pdf_url) != set(decisions.pdf_url)):
        raise ValueError("Incomplete or duplicate original-PDF review")
    if "status" in event and not event.status.isin(GOOD_IDENTITY).all():
        raise ValueError("Unverified original-PDF issuer identity")
    checked = event.merge(decisions, on="pdf_url", how="left",
                          validate="one_to_one", suffixes=("", "_review"))
    for field, other in (("date", "signal_date"),
                         ("notice_date", "notice_date_review"),
                         ("code", "code_review")):
        if not checked[field].eq(checked[other]).all():
            raise ValueError("Review key differs from eligible original notice")
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
    shares = pd.to_numeric(verified.actual_shares, errors="coerce")
    if (not verified.sale_date.str.match(r"^202[45]").all()
            or verified.sale_date.str[:4].gt(str(year)).any()
            or shares.isna().any() or shares.le(0).any()
            or shares.mod(1).ne(0).any()
            or verified.duplicated(["date", "code"]).any()):
        raise ValueError("Verified original lacks 2024+ positive integer sale")
    report = {"year": year, "eligible_title_quotes": len(event),
              "verified_originals": len(verified),
              "rejected_originals": len(event) - len(verified),
              "outcomes_opened": False}
    return verified[eligible.columns], report


def exclude_unverified_announcers(universe: pd.DataFrame,
                                  title_events: pd.DataFrame,
                                  verified: pd.DataFrame) -> pd.DataFrame:
    """Keep verified signals; prevent any other title notice becoming a peer."""
    keys = pd.MultiIndex.from_frame(universe[["date", "code"]])
    titles = pd.MultiIndex.from_frame(title_events[["date", "code"]])
    signal = pd.MultiIndex.from_frame(verified[["date", "code"]])
    return universe.loc[~keys.isin(titles) | keys.isin(signal)].copy()


def build(source_dir: Path, snapshot_dir: Path, daily_dir: Path,
          calendar_path: Path, industry_path: Path, review_path: Path,
          output_dir: Path) -> dict:
    calendar = trading_dates(calendar_path, "2024-01-01", "2026-01-15")
    titles, source = _events(source_dir, calendar)
    universe = _universe(snapshot_dir, daily_dir, industry_path, titles, calendar)
    eligible = titles.merge(universe[["date", "code"]], on=["date", "code"],
                            how="inner", validate="one_to_one")
    review = pd.read_csv(review_path, dtype=str, keep_default_na=False)
    if review.pdf_url.duplicated().any():
        raise ValueError("Duplicate original-PDF review")
    verified_by_year = {}
    reports = {}
    for year in (2024, 2025):
        verified_by_year[year], reports[str(year)] = validate_review(
            eligible, review, year)
    verified = pd.concat(verified_by_year.values(), ignore_index=True)
    comparison = exclude_unverified_announcers(universe, titles, verified)
    pairs, main = _match(comparison, verified, calendar, False, EVENT)
    industry_pairs, industry = _match(
        comparison, verified, calendar, True, EVENT)
    title_keys = pd.MultiIndex.from_frame(titles[["date", "code"]])
    for selected in (pairs, industry_pairs):
        controls = selected.loc[selected.candidate.eq(CONTROL)]
        if pd.MultiIndex.from_frame(controls[["date", "code"]]).isin(
                title_keys).any():
            raise ValueError("Title announcer leaked into nonannouncer controls")
    for year in (2024, 2025):
        chosen = pairs.loc[pairs.candidate.eq(EVENT)
                           & pairs.date.str.startswith(str(year))]
        months = int(chosen.date.str[:7].nunique())
        reports[str(year)].update({
            "main_pairs": len(chosen), "main_calendar_months": months,
            "input_gate_passed": len(chosen) >= 30 and months >= 6,
        })
    report = {"source": source, "universe_stock_days": len(universe),
              "eligible_title_quotes": len(eligible),
              "excluded_unverified_title_quotes": len(universe) - len(comparison),
              "review": reports, "main": main, "same_industry": industry,
              "outcomes_opened": False}
    output_dir.mkdir(parents=True, exist_ok=True)
    for year in (2024, 2025):
        pairs.loc[pairs.date.str.startswith(str(year))].to_parquet(
            output_dir / f"verified_pairs_{year}.parquet", index=False,
            compression="zstd")
        industry_pairs.loc[industry_pairs.date.str.startswith(str(year))].to_parquet(
            output_dir / f"verified_industry_pairs_{year}.parquet", index=False,
            compression="zstd")
    (output_dir / "verified_input_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(
        "data/research/insider_sell_complete"))
    parser.add_argument("--snapshots", type=Path, default=Path(
        "data/research/market_snapshots_ci"))
    parser.add_argument("--daily", type=Path, default=Path(
        "data/baostock/market_2020_2026/daily"))
    parser.add_argument("--calendar", type=Path, default=Path(
        "data/baostock/market_2020_2026/metadata/calendar.parquet"))
    parser.add_argument("--industry", type=Path, default=Path(
        "data/research/industry_intervals.parquet"))
    parser.add_argument("--review", type=Path, default=Path(
        "research/insider_sell_completion_review.csv"))
    parser.add_argument("--output", type=Path, default=Path(
        "data/research/insider_sell_complete"))
    args = parser.parse_args()
    report = build(args.source, args.snapshots, args.daily, args.calendar,
                   args.industry, args.review, args.output)
    print({"review": report["review"],
           "main": report["main"]["by_year"],
           "same_industry": report["same_industry"]["by_year"]})


if __name__ == "__main__":
    main()
