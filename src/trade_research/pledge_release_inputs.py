"""Freeze point-in-time pairs for registered controller pledge releases.

Only original notices, prior daily bars, and 14:50 snapshots are read here.
All announcement search hits are barred from the same-day control pool.
"""

from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path

import pandas as pd

from scripts.validate_pledge_release_review import validate
from .buyback_inputs import CONTROL, _match, _universe
from .exchange_public_events import trading_dates


EVENT = "controller_pledge_release"
LATEST_SIGNAL = "12-17"


def _next_session(day: str, calendar: list[str]) -> str:
    position = bisect.bisect_right(calendar, day)
    if position >= len(calendar):
        raise ValueError("No session after original pledge notice")
    return calendar[position]


def _events(review: pd.DataFrame, calendar: list[str]) -> tuple[pd.DataFrame, dict]:
    source = review.loc[review.decision.eq("verified")].copy()
    if source.empty or source.pdf_url.duplicated().any():
        raise ValueError("No unique verified pledge originals")
    source["date"] = source.notice_date.map(lambda d: _next_session(d, calendar))
    source = source.sort_values(["date", "code", "notice_date", "pdf_url"])
    same_session = int(source.duplicated(["date", "code"]).sum())
    source = source.drop_duplicates(["date", "code"], keep="first")
    within_year = source.date.str[:4].eq(source.notice_date.str[:4])
    timely = source.date.str[5:].le(LATEST_SIGNAL)
    events = source.loc[within_year & timely].copy()
    if (events.empty or not events.date.gt(events.notice_date).all()
            or events.duplicated(["date", "code"]).any()):
        raise ValueError("Pledge signal date leaked or repeated")
    report = {
        "verified_originals": len(review.loc[review.decision.eq("verified")]),
        "same_stock_signal_day_removed": same_session,
        "year_rollover_excluded": int((~within_year).sum()),
        "late_signal_excluded": int((within_year & ~timely).sum()),
        "mapped_main_events": len(events),
    }
    return events, report


def _announcers(root: Path, calendar: list[str]) -> pd.DataFrame:
    search = pd.concat([
        pd.read_parquet(root / f"search_{year}.parquet",
                        columns=["code", "notice_date"])
        for year in (2024, 2025)
    ], ignore_index=True)
    if search.empty or not search.notice_date.str[:4].isin(("2024", "2025")).all():
        raise ValueError("Incomplete pledge search for control exclusion")
    search["date"] = search.notice_date.map(
        lambda d: _next_session(d, calendar))
    return search[["date", "code"]].drop_duplicates()


def _exclude_announcers(universe: pd.DataFrame,
                       announcers: pd.DataFrame,
                       events: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    keys = pd.MultiIndex.from_frame(universe[["date", "code"]])
    announced = pd.MultiIndex.from_frame(announcers[["date", "code"]])
    verified = pd.MultiIndex.from_frame(events[["date", "code"]])
    excluded = keys.isin(announced) & ~keys.isin(verified)
    return universe.loc[~excluded].copy(), int(excluded.sum())


def _tag_plan(pairs: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    tags = events[["date", "code", "planned_repledge"]].rename(
        columns={"code": "pair_code"})
    tagged = pairs.merge(tags, on=["date", "pair_code"],
                         how="left", validate="many_to_one")
    if (len(tagged) != len(pairs)
            or tagged.planned_repledge.isna().any()):
        raise ValueError("Matched pledge pair lacks a frozen repledge flag")
    return tagged


def _signals(*memberships: pd.DataFrame) -> pd.DataFrame:
    columns = ["date", "code", "isST", "reference_gap",
               "quote_outside_traded_range", "listing_age_sessions"]
    signals = pd.concat([frame[columns] for frame in memberships],
                        ignore_index=True).drop_duplicates()
    if signals.empty or signals.duplicated(["date", "code"]).any():
        raise ValueError("Inconsistent original-minute pledge signal inputs")
    return signals.sort_values(["date", "code"]).reset_index(drop=True)


def build(root: Path, review_path: Path, snapshot_dir: Path,
          daily_dir: Path, calendar_path: Path,
          industry_path: Path) -> dict:
    gate = validate(review_path, root)
    if not gate["ready_for_matching"]:
        raise ValueError("Original pledge reviews must finish before matching")
    calendar = trading_dates(calendar_path, "2024-01-01", "2026-01-15")
    review = pd.read_csv(review_path, dtype=str, keep_default_na=False)
    events, source = _events(review, calendar)
    universe = _universe(snapshot_dir, daily_dir, industry_path,
                         events, calendar)
    announcers = _announcers(root, calendar)
    comparison, excluded = _exclude_announcers(
        universe, announcers, events)
    pairs, main = _match(comparison, events, calendar, False, EVENT)
    industry, same_industry = _match(
        comparison, events, calendar, True, EVENT)
    pairs = _tag_plan(pairs, events)
    industry = _tag_plan(industry, events)
    excluded_keys = pd.MultiIndex.from_frame(announcers[["date", "code"]])
    for selected in (pairs, industry):
        controls = selected.loc[selected.candidate.eq(CONTROL)]
        if pd.MultiIndex.from_frame(controls[["date", "code"]]).isin(
                excluded_keys).any():
            raise ValueError("Pledge announcer leaked into nonannouncer controls")
    coverage = {}
    for year in ("2024", "2025"):
        chosen = pairs.loc[pairs.candidate.eq(EVENT)
                           & pairs.date.str.startswith(year)]
        months = int(chosen.date.str[:7].nunique())
        coverage[year] = {"pairs": len(chosen), "months": months,
                          "input_gate_passed": len(chosen) >= 30 and months >= 6}
    report = {"review": gate, "source": source,
              "universe_stock_days": len(universe),
              "excluded_announcer_stock_days": excluded,
              "main": main, "same_industry": same_industry,
              "coverage": coverage, "outcomes_opened": False}
    root.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(root / "pairs.parquet", index=False, compression="zstd")
    industry.to_parquet(root / "industry_pairs.parquet", index=False,
                       compression="zstd")
    _signals(pairs, industry).to_parquet(
        root / "reprice_signals.parquet", index=False, compression="zstd")
    (root / "input_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(
        "data/research/pledge"))
    parser.add_argument("--review", type=Path, default=Path(
        "research/pledge_release_review.csv"))
    parser.add_argument("--snapshots", type=Path, default=Path(
        "data/research/market_snapshots_ci"))
    parser.add_argument("--daily", type=Path, default=Path(
        "data/baostock/market_2020_2026/daily"))
    parser.add_argument("--calendar", type=Path, default=Path(
        "data/baostock/market_2020_2026/metadata/calendar.parquet"))
    parser.add_argument("--industry", type=Path, default=Path(
        "data/research/industry_intervals.parquet"))
    args = parser.parse_args()
    report = build(args.source, args.review, args.snapshots,
                   args.daily, args.calendar, args.industry)
    print({"source": report["source"], "coverage": report["coverage"],
           "main": report["main"]["by_year"],
           "same_industry": report["same_industry"]["by_year"]})


if __name__ == "__main__":
    main()
