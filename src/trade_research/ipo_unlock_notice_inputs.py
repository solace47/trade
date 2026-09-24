"""Freeze first post-disclosure IPO-unlock signals before reading outcomes."""

from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path

import pandas as pd

from .buyback_inputs import _match, _universe
from .exchange_public_events import trading_dates
from .ipo_unlock_inputs import (
    MAX_CURRENT_RETURN_GAP,
    MIN_UNLOCK_PRIOR_FLOAT,
    _confirmed_audits,
    _prior_float,
)


EVENT = "large_ipo_unlock_notice"


def _notices(base: Path, calendar: list[str]
             ) -> tuple[pd.DataFrame, pd.MultiIndex, dict]:
    confirmed, source = _confirmed_audits(base)

    def next_session(day: str) -> str | None:
        offset = bisect.bisect_right(calendar, day)
        return calendar[offset] if offset < len(calendar) else None

    events = confirmed.copy()
    events["date"] = events.notice_date.map(next_session)
    all_notice_keys = pd.MultiIndex.from_frame(events.loc[
        events.date.notna(), ["date", "code"]].drop_duplicates())
    events = events.loc[
        events.date.notna()
        & events.date.str[:4].isin(("2024", "2025"))
        & events.date.str[5:].le("12-17")
        & events.date.lt(events.unlock_date)
        & events.unlock_date.isin(calendar)
    ].copy()
    session = {day: offset for offset, day in enumerate(calendar)}
    events = events.loc[events.apply(
        lambda row: session[row.unlock_date] - session[row.date] >= 2,
        axis=1,
    )].copy()
    if (events.empty or events.duplicated(["date", "code"]).any()
            or not events.date.gt(events.notice_date).all()):
        raise ValueError("Malformed post-notice IPO unlock stock-date")
    return events, all_notice_keys, source


def build(base: Path, snapshot_dir: Path, daily_dir: Path,
          calendar_path: Path, industry_path: Path, output_dir: Path) -> dict:
    calendar = trading_dates(calendar_path, "2024-01-01", "2026-01-15")
    events, all_notice_keys, source = _notices(base, calendar)
    available = _prior_float(events, daily_dir, calendar)
    large = available.loc[available.unlock_prior_float_ratio.ge(
        MIN_UNLOCK_PRIOR_FLOAT)].copy()
    if large.empty:
        raise ValueError("No large post-notice IPO unlock remains")
    universe = _universe(snapshot_dir, daily_dir, industry_path,
                         large, calendar)
    # No confirmed notice can be an untreated control, including notices
    # omitted from the event arm because their unlock is too near.
    event_keys = pd.MultiIndex.from_frame(large[["date", "code"]])
    control_exclusions = all_notice_keys.difference(event_keys)
    universe = universe.loc[~pd.MultiIndex.from_frame(
        universe[["date", "code"]]).isin(control_exclusions)].copy()
    main, main_report = _match(universe, large, calendar, False, EVENT,
                               MAX_CURRENT_RETURN_GAP)
    industry, industry_report = _match(universe, large, calendar, True, EVENT,
                                       MAX_CURRENT_RETURN_GAP)
    audit = {"source": source,
             "all_confirmed_notice_stock_days": len(all_notice_keys),
             "post_notice_before_unlock": len(events),
             "valid_prior_float": len(available),
             "large_prior_float": len(large),
             "large_by_signal_year": {year: len(large.loc[
                 large.date.str.startswith(year)])
                 for year in ("2024", "2025")},
             "universe_stock_days": len(universe),
             "main": main_report, "same_industry": industry_report,
             "note": "Original IPO unlock notice inputs only; no future returns opened"}
    output_dir.mkdir(parents=True, exist_ok=True)
    large.to_parquet(output_dir / "events.parquet", index=False,
                     compression="zstd")
    main.to_parquet(output_dir / "pairs.parquet", index=False,
                    compression="zstd")
    industry.to_parquet(output_dir / "industry_pairs.parquet", index=False,
                        compression="zstd")
    (output_dir / "input_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path(
        "data/research/unlock"))
    parser.add_argument("--snapshots", type=Path, default=Path(
        "data/research/market_snapshots_ci"))
    parser.add_argument("--daily", type=Path, default=Path(
        "data/baostock/market_2020_2026/daily"))
    parser.add_argument("--calendar", type=Path, default=Path(
        "data/baostock/market_2020_2026/metadata/calendar.parquet"))
    parser.add_argument("--industry", type=Path, default=Path(
        "data/research/industry_intervals.parquet"))
    parser.add_argument("--output", type=Path, default=Path(
        "data/research/unlock_notice"))
    args = parser.parse_args()
    print(build(args.base, args.snapshots, args.daily, args.calendar,
                args.industry, args.output))


if __name__ == "__main__":
    main()
