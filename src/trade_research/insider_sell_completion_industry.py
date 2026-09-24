"""Full-coverage industry-basket diagnostic for frozen sale completions.

Build inputs before opening outcomes. This is an exploratory benchmark, not a
replacement for the frozen strict matching test or an independent holdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .annual_cash_direct_eval import _month_bootstrap
from .buyback_eval import _score
from .buyback_inputs import CONTROL, _universe
from .exchange_public_events import trading_dates
from .insider_sell_completion_inputs import _events
from .insider_sell_completion_review import EVENT


MIN_PEERS = 5
GROUP = ["date", "industry", "board"]


def build_inputs(root: Path, snapshot_dir: Path, daily_dir: Path,
                 calendar_path: Path, industry_path: Path) -> dict:
    audit = json.loads((root / "verified_input_audit.json").read_text(
        encoding="utf-8"))
    if audit.get("outcomes_opened") is not False:
        raise ValueError("No frozen original-notice input audit")
    calendar = trading_dates(calendar_path, "2024-01-01", "2026-01-15")
    title_events, _ = _events(root, calendar)
    universe = _universe(snapshot_dir, daily_dir, industry_path,
                         title_events, calendar)
    title_keys = pd.MultiIndex.from_frame(title_events[["date", "code"]])
    pool = universe.loc[
        ~pd.MultiIndex.from_frame(universe[["date", "code"]]).isin(title_keys)
    ].copy()
    event = pd.concat([
        pd.read_parquet(root / f"verified_pairs_{year}.parquet")
        for year in (2024, 2025)
    ], ignore_index=True)
    event = event.loc[event.candidate.eq(EVENT)].copy()
    if len(event) != audit["main"]["matched_pairs"]:
        raise ValueError("Frozen completion event count changed")
    counts = pool.groupby(GROUP).size().rename("peer_count").reset_index()
    event = event.merge(counts, on=GROUP, how="left", validate="many_to_one")
    event["peer_count"] = event.peer_count.fillna(0).astype(int)
    event = event.loc[event.peer_count.ge(MIN_PEERS)].copy()
    groups = event[GROUP].drop_duplicates()
    peers = pool.merge(groups, on=GROUP, how="inner", validate="many_to_one")
    peers["candidate"] = CONTROL
    peers["pair_code"] = peers.code
    peers["notice_date"] = None
    if (peers.empty or peers.duplicated(["date", "code"]).any()
            or not peers.groupby(GROUP).size().ge(MIN_PEERS).all()):
        raise ValueError("Incomplete same-day industry benchmark")
    event.to_parquet(root / "industry_benchmark_events.parquet",
                     index=False, compression="zstd")
    peers.to_parquet(root / "industry_benchmark_peers.parquet",
                     index=False, compression="zstd")
    report = {
        "definition": "same date, historical industry and board; all 14:50 "
                      "eligible non-title stocks; at least five peers",
        "min_peers": MIN_PEERS,
        "events": len(event), "peer_stock_days": len(peers),
        "by_year": {
            year: {
                "events": int(event.date.str.startswith(year).sum()),
                "all_frozen_events": audit["main"]["by_year"][year]["pairs"],
            } for year in ("2024", "2025")
        },
        "outcomes_opened": False,
    }
    (root / "industry_benchmark_input_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return report


def evaluate(root: Path, outcome_dir: Path, issues_dir: Path) -> dict:
    audit = json.loads((root / "industry_benchmark_input_audit.json").read_text(
        encoding="utf-8"))
    if (audit.get("outcomes_opened") is not False
            or audit.get("min_peers") != MIN_PEERS):
        raise ValueError("Industry benchmark inputs were not frozen")
    event = pd.read_parquet(root / "industry_benchmark_events.parquet")
    peers = pd.read_parquet(root / "industry_benchmark_peers.parquet")
    if len(event) != audit["events"] or len(peers) != audit["peer_stock_days"]:
        raise ValueError("Industry basket changed after input audit")
    scored_events = _score(event, outcome_dir, issues_dir, 5)
    scored_peers = _score(peers, outcome_dir, issues_dir, 5)
    basket = scored_peers.groupby(GROUP).agg(
        peer_cash_mean=("cash_return", "mean"),
        peer_clean_exit_rate=("quality_clean_exit", "mean"),
        peer_stock_days=("code", "size"),
    ).reset_index()
    joined = scored_events.merge(basket, on=GROUP, how="left",
                                 validate="many_to_one")
    if (len(joined) != len(event) or joined.peer_cash_mean.isna().any()
            or joined.peer_stock_days.lt(MIN_PEERS).any()):
        raise ValueError("An industry event lacks its frozen benchmark")
    joined["industry_edge"] = joined.cash_return - joined.peer_cash_mean
    report = {"note": "Post-result diagnostic; 2025 not blind, 2026 unopened",
              "by_year": {}}
    for year in ("2024", "2025"):
        sample = joined.loc[joined.date.str.startswith(year)]
        daily = sample.groupby("date")[
            ["cash_return", "peer_cash_mean", "industry_edge"]].mean()
        dates = pd.Series(daily.index.to_list())
        report["by_year"][year] = {
            "events": len(sample), "days": len(daily),
            "event_cash_mean": float(daily.cash_return.mean()),
            "peer_cash_mean": float(daily.peer_cash_mean.mean()),
            "industry_edge_mean": float(daily.industry_edge.mean()),
            "industry_edge_month_ci": _month_bootstrap(
                daily.industry_edge.reset_index(drop=True), dates, 147),
            "event_clean_exit_rate": float(sample.quality_clean_exit.mean()),
            "peer_clean_exit_rate": float(
                scored_peers.loc[scored_peers.date.str.startswith(year),
                                 "quality_clean_exit"].mean()),
        }
    (root / "industry_benchmark_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return report


def evaluate_raw_20k(root: Path, event_raw_path: Path,
                     peer_raw_path: Path) -> dict:
    """Recompute both sides of the frozen basket at the smaller order size."""
    audit = json.loads((root / "industry_benchmark_input_audit.json").read_text(
        encoding="utf-8"))
    event = pd.read_parquet(root / "industry_benchmark_events.parquet")
    peers = pd.read_parquet(root / "industry_benchmark_peers.parquet")
    if (audit.get("outcomes_opened") is not False
            or len(event) != audit["events"]
            or len(peers) != audit["peer_stock_days"]):
        raise ValueError("Industry basket changed before raw repricing")
    event_raw = pd.read_parquet(event_raw_path)
    peer_raw = pd.read_parquet(peer_raw_path)
    event_raw = event_raw.loc[
        event_raw.target_notional.eq(20_000) & event_raw.horizon.eq(5)]
    if (len(peer_raw) != len(peers)
            or not peer_raw.target_notional.eq(20_000).all()
            or not peer_raw.horizon.eq(5).all()
            or peer_raw.duplicated(["date", "code"]).any()):
        raise ValueError("Incomplete original-minute industry peers")
    event_rows = event[["date", "code"] + GROUP[1:]].merge(
        event_raw, on=["date", "code"], validate="one_to_one")
    peer_rows = peers[["date", "code"] + GROUP[1:]].merge(
        peer_raw, on=["date", "code"], validate="one_to_one")
    if len(event_rows) != len(event) or len(peer_rows) != len(peers):
        raise ValueError("Original-minute basket misses a frozen stock-day")
    for rows in (event_rows, peer_rows):
        rows["cash_return"] = np.where(
            rows.quality_clean_exit, rows.net_return, 0.0)
        if not np.isfinite(rows.cash_return).all():
            raise ValueError("Nonfinite original-minute basket cash return")
    basket = peer_rows.groupby(GROUP).cash_return.mean().rename(
        "peer_cash_mean").reset_index()
    joined = event_rows.merge(basket, on=GROUP, how="left",
                              validate="many_to_one")
    if len(joined) != len(event) or joined.peer_cash_mean.isna().any():
        raise ValueError("Raw event lacks an industry basket")
    joined["industry_edge"] = joined.cash_return - joined.peer_cash_mean
    report = {"note": "Post-result 20k minute diagnostic; not blind",
              "by_year": {}}
    for year in ("2024", "2025"):
        sample = joined.loc[joined.date.str.startswith(year)]
        daily = sample.groupby("date")[
            ["cash_return", "peer_cash_mean", "industry_edge"]].mean()
        dates = pd.Series(daily.index.to_list())
        report["by_year"][year] = {
            "events": len(sample), "days": len(daily),
            "event_cash_mean": float(daily.cash_return.mean()),
            "peer_cash_mean": float(daily.peer_cash_mean.mean()),
            "industry_edge_mean": float(daily.industry_edge.mean()),
            "industry_edge_month_ci": _month_bootstrap(
                daily.industry_edge.reset_index(drop=True), dates, 148),
            "event_clean_exit_rate": float(sample.quality_clean_exit.mean()),
            "peer_clean_exit_rate": float(
                peer_rows.loc[peer_rows.date.str.startswith(year),
                              "quality_clean_exit"].mean()),
        }
    (root / "industry_benchmark_20k_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("inputs", "evaluate", "raw20k"))
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
    parser.add_argument("--outcomes", type=Path, default=Path(
        "data/research/market_outcomes_ci"))
    parser.add_argument("--issues", type=Path, default=Path(
        "data/research/market_issues_ci"))
    parser.add_argument("--event-raw", type=Path, default=Path(
        "data/research/insider_sell_complete/repriced.parquet"))
    parser.add_argument("--peer-raw", type=Path, default=Path(
        "data/research/insider_sell_complete/industry_benchmark_peers_20k.parquet"))
    args = parser.parse_args()
    if args.stage == "inputs":
        report = build_inputs(args.source, args.snapshots, args.daily,
                              args.calendar, args.industry)
    elif args.stage == "evaluate":
        report = evaluate(args.source, args.outcomes, args.issues)
    else:
        report = evaluate_raw_20k(args.source, args.event_raw, args.peer_raw)
    print(report)


if __name__ == "__main__":
    main()
