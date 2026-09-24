"""Build next-session actual-first-buyback pairs from original issuer PDFs.

Only announcement metadata, already published text, prior bars and 14:50
snapshots enter this stage. No exit price or return is opened.
"""

from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path

import pandas as pd

from .buyback_inputs import _match, _universe
from .exchange_public_events import trading_dates
from .first_buyback_source import strict_first_title


EVENT = "first_actual_buyback"


def _events(source_dir: Path, calendar: list[str]) -> tuple[pd.DataFrame, dict]:
    frames = []
    counts = {}
    for year in (2024, 2025):
        index = pd.read_parquet(source_dir / f"title_candidates_{year}.parquet")
        checked = pd.read_json(source_dir / f"pdf_audit_{year}.jsonl",
                               lines=True)[["pdf_url", "status"]]
        if (index.empty or index.pdf_url.duplicated().any()
                or checked.pdf_url.duplicated().any()
                or not index.title.map(strict_first_title).all()):
            raise ValueError("Malformed first-buyback original index")
        joined = index.merge(checked, on="pdf_url", how="left",
                             validate="one_to_one")
        if joined.status.isna().any():
            raise ValueError("First-buyback original lacks PDF audit")
        counts[str(year)] = {
            "title_candidates": len(index),
            "original_pdf_confirmed": int(joined.status.eq("ok").sum()),
            "unconfirmed_by_status": joined.status.value_counts().to_dict(),
        }
        frames.append(joined.loc[joined.status.eq("ok")].copy())
    events = pd.concat(frames, ignore_index=True)
    duplicate = events.duplicated(["code", "notice_date"], keep=False)
    removed = int(duplicate.sum())
    events = events.loc[~duplicate].copy()
    events["date"] = events.notice_date.map(
        lambda day: calendar[bisect.bisect_right(calendar, day)])
    events = events.loc[
        events.date.str[:4].eq(events.notice_date.str[:4])
        & events.date.str[5:].le("12-17")
    ].copy()
    if (events.empty or events.duplicated(["date", "code"]).any()
            or not events.date.gt(events.notice_date).all()):
        raise ValueError("First-buyback event did not lag official notice")
    return events, {"source": counts,
                    "duplicate_code_notice_rows_removed": removed,
                    "mapped_events": len(events)}


def _plan_gap(events: pd.DataFrame, calendar: list[str],
              plan_dir: Path) -> pd.DataFrame:
    """Known 2024/25 original-plan announcements preceding each first trade."""
    plans = pd.concat([
        pd.read_json(plan_dir / f"pdf_audit_{year}.jsonl", lines=True)[
            ["code", "notice_date", "status"]]
        for year in (2024, 2025)
    ], ignore_index=True)
    plans = plans.loc[plans.status.eq("ok")]
    dates_by_code = plans.groupby("code").notice_date.apply(
        lambda days: sorted(set(days))).to_dict()
    position = {day: number for number, day in enumerate(calendar)}
    records = []
    for item in events.itertuples(index=False):
        known = dates_by_code.get(item.code, [])
        offset = bisect.bisect_right(known, item.notice_date) - 1
        prior = known[offset] if offset >= 0 else None
        plan_signal = (calendar[bisect.bisect_right(calendar, prior)]
                       if prior is not None else None)
        gap = (position[item.date] - position[plan_signal]
               if plan_signal is not None else None)
        if gap is not None and gap < 0:
            raise ValueError("Future plan notice entered first-buyback signal")
        records.append({"date": item.date, "code": item.code,
                        "prior_plan_notice": prior,
                        "plan_gap_sessions": gap,
                        "recent_plan_10": gap is not None and gap <= 10})
    return pd.DataFrame(records)


def _tag_pairs(pairs: pd.DataFrame, gap: pd.DataFrame) -> pd.DataFrame:
    tags = gap.rename(columns={"code": "pair_code"})
    tagged = pairs.merge(tags, on=["date", "pair_code"],
                         validate="many_to_one")
    if len(tagged) != len(pairs):
        raise ValueError("First-buyback pair lacks a past-plan audit")
    return tagged


def build(source_dir: Path, snapshot_dir: Path, daily_dir: Path,
          calendar_path: Path, industry_path: Path,
          output_dir: Path,
          plan_dir: Path = Path("data/research/buyback")) -> dict:
    calendar = trading_dates(calendar_path, "2024-01-01", "2026-01-15")
    events, source_report = _events(source_dir, calendar)
    universe = _universe(snapshot_dir, daily_dir, industry_path,
                         events, calendar)
    pairs, main = _match(universe, events, calendar, False, EVENT)
    industry, within = _match(universe, events, calendar, True, EVENT)
    gap = _plan_gap(events, calendar, plan_dir)
    pairs = _tag_pairs(pairs, gap)
    industry = _tag_pairs(industry, gap)
    recent_report = {}
    for year in ("2024", "2025"):
        selected = pairs.loc[pairs.candidate.eq(EVENT)
                             & pairs.date.str.startswith(year)]
        recent_report[year] = {
            "known_prior_plan": int(selected.plan_gap_sessions.notna().sum()),
            "known_within_10_sessions": int(selected.recent_plan_10.sum()),
        }
    audit = {"source": source_report, "universe_stock_days": len(universe),
             "main": main, "same_industry": within,
             "prior_plan_overlap": recent_report,
             "note": "First actual buyback inputs only; no future returns opened"}
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(output_dir / "pairs.parquet", index=False,
                     compression="zstd")
    industry.to_parquet(output_dir / "industry_pairs.parquet", index=False,
                        compression="zstd")
    (output_dir / "input_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(
        "data/research/first_buyback"))
    parser.add_argument("--snapshots", type=Path, default=Path(
        "data/research/market_snapshots_ci"))
    parser.add_argument("--daily", type=Path, default=Path(
        "data/baostock/market_2020_2026/daily"))
    parser.add_argument("--calendar", type=Path, default=Path(
        "data/baostock/market_2020_2026/metadata/calendar.parquet"))
    parser.add_argument("--industry", type=Path, default=Path(
        "data/research/industry_intervals.parquet"))
    parser.add_argument("--output", type=Path, default=Path(
        "data/research/first_buyback"))
    parser.add_argument("--plan-source", type=Path, default=Path(
        "data/research/buyback"))
    args = parser.parse_args()
    audit = build(args.source, args.snapshots, args.daily, args.calendar,
                  args.industry, args.output, args.plan_source)
    print({"source": audit["source"], "main": audit["main"],
           "same_industry": audit["same_industry"]})


if __name__ == "__main__":
    main()
