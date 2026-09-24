"""Diagnose major-holder sale-completion titles using only observable inputs.

These events are unverified title candidates. This module never opens returns
or minute trade outcomes and its pairs must not be used as a strategy result.
"""

from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path

import pandas as pd

from scripts.audit_sell_completion_pdfs import candidate_title
from .buyback_inputs import _match, _universe
from .exchange_public_events import trading_dates


EVENT = "title_only_sell_completion"
GOOD_IDENTITY = {"ok", "verified_header_typo"}


def _events(source_dir: Path, calendar: list[str]) -> tuple[pd.DataFrame, dict]:
    frames = []
    source = {}
    for year in (2024, 2025):
        index = pd.read_parquet(source_dir / f"search_{year}.parquet")
        candidates = index.loc[index.title.map(candidate_title)].copy()
        audit = pd.read_json(source_dir / f"pdf_audit_{year}.jsonl", lines=True)
        if (index.empty or candidates.empty or index.pdf_url.duplicated().any()
                or audit.pdf_url.duplicated().any()
                or set(candidates.pdf_url) != set(audit.pdf_url)
                or not candidates.notice_date.str.startswith(str(year)).all()):
            raise ValueError("Incomplete or inconsistent completion PDF audit")
        checked = candidates.merge(audit[["pdf_url", "status"]], on="pdf_url",
                                   how="left", validate="one_to_one")
        if not checked.status.isin(GOOD_IDENTITY).all():
            raise ValueError("A completion PDF has unverified issuer identity")
        source[str(year)] = {"a_share_search_rows": len(index),
                             "actor_completion_titles": len(candidates),
                             "identity_status": checked.status.value_counts().to_dict()}
        frames.append(checked)
    events = pd.concat(frames, ignore_index=True)
    same_notice = events.duplicated(["code", "notice_date"], keep=False)
    removed_notice = int(same_notice.sum())
    events = events.loc[~same_notice].copy()
    events["date"] = events.notice_date.map(
        lambda day: calendar[bisect.bisect_right(calendar, day)])
    events = events.loc[
        events.date.str[:4].eq(events.notice_date.str[:4])
        & events.date.str[5:].le("12-17")
    ].copy()
    same_signal = events.duplicated(["date", "code"], keep=False)
    removed_signal = int(same_signal.sum())
    events = events.loc[~same_signal].copy()
    if (events.empty or events.duplicated(["date", "code"]).any()
            or not events.date.gt(events.notice_date).all()):
        raise ValueError("Completion titles did not map to unique next sessions")
    return events, {"source": source,
                    "duplicate_code_notice_rows_removed": removed_notice,
                    "duplicate_signal_rows_removed": removed_signal,
                    "mapped_title_events": len(events)}


def build(source_dir: Path, snapshot_dir: Path, daily_dir: Path,
          calendar_path: Path, industry_path: Path, output_dir: Path) -> dict:
    calendar = trading_dates(calendar_path, "2024-01-01", "2026-01-15")
    events, source = _events(source_dir, calendar)
    universe = _universe(snapshot_dir, daily_dir, industry_path, events, calendar)
    pairs, main = _match(universe, events, calendar, False, EVENT)
    industry_pairs, industry = _match(universe, events, calendar, True, EVENT)
    report = {**source, "universe_stock_days": len(universe),
              "main": main, "same_industry": industry,
              "warning": "Titles only; actor, actual sale and completion are unverified",
              "outcomes_opened": False}
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(output_dir / "title_only_pairs.parquet", index=False,
                     compression="zstd")
    industry_pairs.to_parquet(
        output_dir / "title_only_industry_pairs.parquet", index=False,
        compression="zstd")
    (output_dir / "title_only_input_audit.json").write_text(
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
    parser.add_argument("--output", type=Path, default=Path(
        "data/research/insider_sell_complete"))
    args = parser.parse_args()
    report = build(args.source, args.snapshots, args.daily, args.calendar,
                   args.industry, args.output)
    print({"source": report["source"], "main": report["main"]["by_year"],
           "same_industry": report["same_industry"]["by_year"]})


if __name__ == "__main__":
    main()
