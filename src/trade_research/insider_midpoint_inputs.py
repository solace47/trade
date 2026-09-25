"""Freeze next-session midpoint zero-execution pairs without future returns."""

from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path

import pandas as pd

from .buyback_inputs import _match, _universe
from .exchange_public_events import trading_dates


EVENT = "insider_midpoint_zero"
COOLDOWN = 120


def _events(source_dir: Path, calendar: list[str]) -> tuple[pd.DataFrame, dict]:
    audit = json.loads((source_dir / "source_audit.json").read_text(
        encoding="utf-8"))
    frames = []
    for year in (2024, 2025):
        source = pd.read_parquet(source_dir / f"confirmed_{year}.parquet")
        if (source.empty or len(source) != audit[str(year)]["confirmed_zero_pdfs"]
                or source.duplicated(["code", "notice_date"]).any()
                or not source.notice_date.str.startswith(str(year)).all()):
            raise ValueError("Confirmed midpoint originals changed")
        frames.append(source)
    events = pd.concat(frames, ignore_index=True)
    if events.pdf_url.duplicated().any():
        raise ValueError("Duplicate midpoint original PDF")
    events["date"] = events.notice_date.map(
        lambda day: calendar[bisect.bisect_right(calendar, day)])
    if (not events.date.gt(events.notice_date).all()
            or not events.date.str[:4].isin(("2024", "2025")).all()
            or events.duplicated(["date", "code"]).any()):
        raise ValueError("Midpoint notice is not a unique next-session signal")
    return events, {"confirmed_originals": len(events),
                    "mapped_event_stock_days": len(events),
                    "by_notice_year": events.notice_date.str[:4]
                    .value_counts().sort_index().to_dict(),
                    "note": "Original notices and signal times only; no future outcomes"}


def _tag(pairs: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    tags = events[["date", "code", "notice_date", "pdf_url"]].rename(
        columns={"code": "pair_code", "notice_date": "event_notice_date",
                 "pdf_url": "event_pdf_url"})
    tagged = pairs.merge(tags, on=["date", "pair_code"],
                         validate="many_to_one")
    if (len(tagged) != len(pairs)
            or not tagged.date.gt(tagged.event_notice_date).all()):
        raise ValueError("Pair lacks its pre-signal original notice")
    tagged["event_exchange"] = tagged.pair_code.str[:2]
    return tagged


def build(source_dir: Path, snapshot_dir: Path, daily_dir: Path,
          calendar_path: Path, industry_path: Path,
          output_dir: Path) -> dict:
    calendar = trading_dates(calendar_path, "2024-01-01", "2026-01-15")
    events, source = _events(source_dir, calendar)
    universe = _universe(snapshot_dir, daily_dir, industry_path,
                         events, calendar)
    main, main_report = _match(
        universe, events, calendar, False, EVENT,
        cooldown_sessions=COOLDOWN)
    within, industry_report = _match(
        universe, events, calendar, True, EVENT,
        cooldown_sessions=COOLDOWN)
    main = _tag(main, events)
    within = _tag(within, events)
    main_events = main.loc[main.candidate.eq(EVENT)]
    peak_months = {
        year: main_events.loc[main_events.date.str.startswith(year), "date"]
        .str[:7].value_counts().idxmax()
        for year in ("2024", "2025")
    }
    report = {"source": source, "universe_stock_days": len(universe),
              "main": main_report, "same_industry": industry_report,
              "peak_months": peak_months,
              "exchange_counts": {
                  name: frame.loc[frame.candidate.eq(EVENT),
                                  "event_exchange"].value_counts().to_dict()
                  for name, frame in (("main", main),
                                      ("same_industry", within))},
              "note": "Midpoint input membership only; no future returns opened"}
    output_dir.mkdir(parents=True, exist_ok=True)
    main.to_parquet(output_dir / "pairs.parquet", index=False,
                    compression="zstd")
    within.to_parquet(output_dir / "industry_pairs.parquet", index=False,
                      compression="zstd")
    (output_dir / "input_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(
        "data/research/insider_midpoint"))
    parser.add_argument("--snapshots", type=Path, default=Path(
        "data/research/market_snapshots_ci"))
    parser.add_argument("--daily", type=Path, default=Path(
        "data/baostock/market_2020_2026/daily"))
    parser.add_argument("--calendar", type=Path, default=Path(
        "data/baostock/market_2020_2026/metadata/calendar.parquet"))
    parser.add_argument("--industry", type=Path, default=Path(
        "data/research/industry_intervals.parquet"))
    parser.add_argument("--output", type=Path, default=Path(
        "data/research/insider_midpoint"))
    args = parser.parse_args()
    report = build(args.source, args.snapshots, args.daily, args.calendar,
                   args.industry, args.output)
    print({"source": report["source"], "main": report["main"],
           "same_industry": report["same_industry"],
           "exchange_counts": report["exchange_counts"]})


if __name__ == "__main__":
    main()
