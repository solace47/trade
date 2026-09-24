"""Build next-session insider-increase-plan pairs without future outcomes."""

from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path

import pandas as pd

from .buyback_inputs import _match, _universe
from .exchange_public_events import trading_dates
from .insider_plan_source import strict_plan_title


EVENT = "insider_increase_plan"


def _events(source_dir: Path, calendar: list[str]) -> tuple[pd.DataFrame, dict]:
    frames = []
    counts = {}
    for year in (2024, 2025):
        index = pd.read_parquet(source_dir / f"title_candidates_{year}.parquet")
        checked = pd.read_json(source_dir / f"pdf_audit_{year}.jsonl",
                               lines=True)[["pdf_url", "status"]]
        if (index.empty or index.pdf_url.duplicated().any()
                or checked.pdf_url.duplicated().any()
                or not index.title.map(strict_plan_title).all()):
            raise ValueError("Malformed insider-plan original index")
        joined = index.merge(checked, on="pdf_url", how="left",
                             validate="one_to_one")
        if joined.status.isna().any():
            raise ValueError("Insider-plan original lacks PDF audit")
        counts[str(year)] = {
            "title_candidates": len(index),
            "original_pdf_confirmed": int(joined.status.eq("ok").sum()),
            "unconfirmed_by_status": joined.status.value_counts().to_dict(),
        }
        frames.append(joined.loc[joined.status.eq("ok")].copy())
    events = pd.concat(frames, ignore_index=True)
    duplicates = events.duplicated(["code", "notice_date"], keep=False)
    removed = int(duplicates.sum())
    events = events.loc[~duplicates].copy()
    events["date"] = events.notice_date.map(
        lambda day: calendar[bisect.bisect_right(calendar, day)])
    events = events.loc[
        events.date.str[:4].eq(events.notice_date.str[:4])
        & events.date.str[5:].le("12-17")
    ].copy()
    same_signal_duplicates = events.duplicated(["date", "code"], keep=False)
    same_signal_removed = int(same_signal_duplicates.sum())
    events = events.loc[~same_signal_duplicates].copy()
    if (events.empty or events.duplicated(["date", "code"]).any()
            or not events.date.gt(events.notice_date).all()):
        raise ValueError("Insider-plan event did not lag official notice")
    return events, {"source": counts,
                    "duplicate_code_notice_rows_removed": removed,
                    "duplicate_code_signal_rows_removed": same_signal_removed,
                    "mapped_events": len(events)}


def build(source_dir: Path, snapshot_dir: Path, daily_dir: Path,
          calendar_path: Path, industry_path: Path, output_dir: Path) -> dict:
    calendar = trading_dates(calendar_path, "2024-01-01", "2026-01-15")
    events, source = _events(source_dir, calendar)
    universe = _universe(snapshot_dir, daily_dir, industry_path,
                         events, calendar)
    pairs, main = _match(universe, events, calendar, False, EVENT)
    industry, within = _match(universe, events, calendar, True, EVENT)
    audit = {"source": source, "universe_stock_days": len(universe),
             "main": main, "same_industry": within,
             "note": "Insider increase plan inputs only; no future returns opened"}
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
        "data/research/insider_buy"))
    parser.add_argument("--snapshots", type=Path, default=Path(
        "data/research/market_snapshots_ci"))
    parser.add_argument("--daily", type=Path, default=Path(
        "data/baostock/market_2020_2026/daily"))
    parser.add_argument("--calendar", type=Path, default=Path(
        "data/baostock/market_2020_2026/metadata/calendar.parquet"))
    parser.add_argument("--industry", type=Path, default=Path(
        "data/research/industry_intervals.parquet"))
    parser.add_argument("--output", type=Path, default=Path(
        "data/research/insider_buy"))
    args = parser.parse_args()
    audit = build(args.source, args.snapshots, args.daily, args.calendar,
                  args.industry, args.output)
    print({"source": audit["source"], "main": audit["main"],
           "same_industry": audit["same_industry"]})


if __name__ == "__main__":
    main()
