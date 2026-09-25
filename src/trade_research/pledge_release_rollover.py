"""Check 2024 year-end pledge releases with the frozen same-day rules.

The input phase writes membership before any minute outcomes are read. Its
small year-end sample is a supplement, not an independent validation year.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.validate_pledge_release_review import validate
from .buyback_eval import _summary
from .buyback_inputs import CONTROL, _match, _universe
from .exchange_public_events import trading_dates
from .pledge_release_eval import _scored
from .pledge_release_inputs import (
    EVENT, _announcers, _exclude_announcers, _next_session, _signals,
    _tag_plan,
)


def _late_events(review: pd.DataFrame, calendar: list[str]) -> pd.DataFrame:
    source = review.loc[review.decision.eq("verified")].copy()
    source["date"] = source.notice_date.map(
        lambda day: _next_session(day, calendar))
    source = source.sort_values(["date", "code", "notice_date", "pdf_url"])
    source = source.drop_duplicates(["date", "code"], keep="first")
    late = source.loc[
        source.notice_date.str.startswith("2024")
        & source.date.between("2024-12-18", "2024-12-31")
    ].copy()
    if (late.empty or not late.date.gt(late.notice_date).all()
            or late.duplicated(["date", "code"]).any()
            or not late.planned_repledge.isin(("yes", "no")).all()):
        raise ValueError("Malformed 2024 year-end pledge originals")
    return late


def freeze(source_root: Path, output_root: Path, review_path: Path,
           snapshot_dir: Path, daily_dir: Path, calendar_path: Path,
           industry_path: Path) -> dict:
    gate = validate(review_path, source_root)
    if not gate["ready_for_matching"]:
        raise ValueError("Pledge original review is not complete")
    calendar = trading_dates(calendar_path, "2024-01-01", "2026-01-15")
    review = pd.read_csv(review_path, dtype=str, keep_default_na=False)
    events = _late_events(review, calendar)
    universe = _universe(snapshot_dir, daily_dir, industry_path,
                         events, calendar)
    announcers = _announcers(source_root, calendar)
    comparison, excluded = _exclude_announcers(
        universe, announcers, events)
    pairs, match = _match(comparison, events, calendar, False, EVENT)
    pairs = _tag_plan(pairs, events)
    controls = pairs.loc[pairs.candidate.eq(CONTROL)]
    if pd.MultiIndex.from_frame(controls[["date", "code"]]).isin(
            pd.MultiIndex.from_frame(announcers[["date", "code"]])).any():
        raise ValueError("Pledge announcer entered the rollover controls")
    report = {"originals": len(events),
              "excluded_announcer_stock_days": excluded,
              "match": match, "outcomes_opened": False,
              "note": "Frozen 2024 year-end supplement; 2025 exit only"}
    output_root.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(output_root / "pairs.parquet", index=False,
                     compression="zstd")
    _signals(pairs).to_parquet(
        output_root / "signals.parquet", index=False, compression="zstd")
    (output_root / "input_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return report


def evaluate(root: Path, raw_path: Path, report_path: Path) -> dict:
    audit = json.loads((root / "input_audit.json").read_text(
        encoding="utf-8"))
    if (audit.get("outcomes_opened") is not False
            or audit.get("note") != "Frozen 2024 year-end supplement; 2025 exit only"):
        raise ValueError("Rollover membership was not frozen before outcomes")
    pairs = pd.read_parquet(root / "pairs.parquet")
    events = pairs.loc[pairs.candidate.eq(EVENT)]
    if (len(events) != audit["match"]["matched_pairs"]
            or events.date.nunique() != audit["match"]["matched_days"]
            or not events.date.between("2024-12-18", "2024-12-31").all()
            or not events.date.gt(events.notice_date).all()
            or not pairs.groupby(["date", "pair_code"])
            .candidate.nunique().eq(2).all()):
        raise ValueError("Rollover pair list changed after freezing")
    raw = pd.read_parquet(raw_path)
    keys = pairs[["date", "code"]].drop_duplicates()
    raw_keys = set(map(tuple, raw[["date", "code"]].itertuples(index=False,
                                                              name=None)))
    frozen_keys = set(map(tuple, keys.itertuples(index=False, name=None)))
    if (len(raw) != len(keys) * 4 or raw_keys != frozen_keys
            or set(raw.target_notional) != {20_000.0, 100_000.0}
            or set(raw.horizon) != {1, 5}
            or not raw.exit_window.eq("close").all()
            or raw.duplicated(
                ["date", "code", "target_notional", "horizon"]).any()
            or raw.exit_date.dropna().gt("2025-01-31").any()):
        raise ValueError("Incomplete or out-of-window rollover minute outcomes")
    scored = _scored(raw, pairs)
    report = {"note": "Small 2024 year-end supplement; 2025 exit was previously viewed",
              "pairs": len(events), "results": {}, "cross_year_exit_rows": {}}
    for size in (20_000, 100_000):
        for horizon in (1, 5):
            frame = scored.loc[scored.target_notional.eq(size)
                               & scored.horizon.eq(horizon)]
            label = f"{size}_t{horizon}"
            report["results"][label] = _summary(frame, "2024", "full", EVENT)
            report["cross_year_exit_rows"][label] = int(
                frame.exit_date.dropna().str.startswith("2025").sum())
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "evaluate"))
    parser.add_argument("--source", type=Path,
                        default=Path("data/research/pledge"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/research/pledge/rollover_2024"))
    parser.add_argument("--review", type=Path,
                        default=Path("research/pledge_release_review.csv"))
    parser.add_argument("--snapshots", type=Path,
                        default=Path("data/research/market_snapshots_ci"))
    parser.add_argument("--daily", type=Path,
                        default=Path("data/baostock/market_2020_2026/daily"))
    parser.add_argument("--calendar", type=Path, default=Path(
        "data/baostock/market_2020_2026/metadata/calendar.parquet"))
    parser.add_argument("--industry", type=Path,
                        default=Path("data/research/industry_intervals.parquet"))
    args = parser.parse_args()
    if args.phase == "freeze":
        print(freeze(args.source, args.output, args.review, args.snapshots,
                     args.daily, args.calendar, args.industry))
    else:
        print(evaluate(args.output, args.output / "repriced.parquet",
                       args.output / "report.json"))


if __name__ == "__main__":
    main()
