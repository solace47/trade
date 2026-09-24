"""Diagnose industry controls using five nearest same-industry peers.

The distance uses only 14:50-observable inputs and is the existing frozen
distance used by other research. This exploratory benchmark has no hard
caliper and permits a peer to serve more than one event.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .annual_cash_direct_eval import _month_bootstrap
from .margin_dtc_study import _distance


K = 5
GROUP = ["date", "industry", "board"]


def build_inputs(root: Path) -> dict:
    sector_audit = json.loads(
        (root / "industry_benchmark_input_audit.json").read_text(
            encoding="utf-8"))
    events = pd.read_parquet(root / "industry_benchmark_events.parquet")
    peers = pd.read_parquet(root / "industry_benchmark_peers.parquet")
    if (sector_audit.get("outcomes_opened") is not False
            or len(events) != sector_audit["events"]
            or len(peers) != sector_audit["peer_stock_days"]):
        raise ValueError("Unfrozen industry peer universe")
    grouped = {
        key: frame for key, frame in peers.groupby(GROUP, sort=False)
    }
    rows = []
    balance = []
    for item in events.itertuples(index=False):
        signal = pd.Series(item._asdict())
        choices = grouped[(item.date, item.industry, item.board)].copy()
        choices["distance"] = _distance(choices, signal)
        if (len(choices) < K or not np.isfinite(choices.distance).all()):
            raise ValueError("Too few or nonfinite same-industry peers")
        nearest = choices.sort_values(["distance", "code"]).head(K)
        for peer in nearest.itertuples(index=False):
            rows.append((item.date, item.code, peer.code,
                         float(peer.distance)))
        balance.append({
            "date": item.date, "event_code": item.code,
            "fifth_distance": float(nearest.distance.iloc[-1]),
            "float_mv_log_gap": float(np.log(
                nearest.float_mv.median() / item.float_mv)),
            "prior20_gap": float(
                nearest.return20_prior_adjusted.mean()
                - item.return20_prior_adjusted),
            "current_gap": float(
                nearest.return_1450.mean() - item.return_1450),
        })
    membership = pd.DataFrame(rows, columns=[
        "date", "event_code", "peer_code", "distance"])
    if (len(membership) != K * len(events)
            or membership.duplicated(
                ["date", "event_code", "peer_code"]).any()
            or membership.groupby(["date", "event_code"]).size().ne(K).any()):
        raise ValueError("Incomplete five-peer industry membership")
    membership.to_parquet(root / "nearest5_membership.parquet",
                          index=False, compression="zstd")
    b = pd.DataFrame(balance)
    report = {
        "definition": "five nearest 14:50-eligible non-title stocks "
                      "with same date, historical industry and board",
        "distance": "existing seven-covariate absolute distance; no caliper",
        "events": len(events), "legs": len(membership),
        "median_fifth_distance": float(b.fifth_distance.median()),
        "median_abs_float_mv_log_gap": float(b.float_mv_log_gap.abs().median()),
        "median_abs_prior20_gap": float(b.prior20_gap.abs().median()),
        "median_abs_current_gap": float(b.current_gap.abs().median()),
        "outcomes_opened": False,
    }
    (root / "nearest5_input_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return report


def evaluate(root: Path, event_raw_path: Path,
             peer_raw_path: Path) -> dict:
    audit = json.loads((root / "nearest5_input_audit.json").read_text(
        encoding="utf-8"))
    members = pd.read_parquet(root / "nearest5_membership.parquet")
    event_raw = pd.read_parquet(event_raw_path)
    peer_raw = pd.read_parquet(peer_raw_path)
    event_raw = event_raw.loc[
        event_raw.target_notional.eq(20_000) & event_raw.horizon.eq(5)]
    if (audit.get("outcomes_opened") is not False
            or audit.get("legs") != len(members)
            or not peer_raw.target_notional.eq(20_000).all()
            or not peer_raw.horizon.eq(5).all()):
        raise ValueError("Five-peer inputs or original minutes changed")
    peer = peer_raw[["date", "code", "quality_clean_exit", "net_return"]].rename(
        columns={"code": "peer_code", "quality_clean_exit": "peer_clean",
                 "net_return": "peer_return"})
    event = event_raw[["date", "code", "quality_clean_exit", "net_return"]].rename(
        columns={"code": "event_code", "quality_clean_exit": "event_clean",
                 "net_return": "event_return"})
    rows = members.merge(peer, on=["date", "peer_code"], how="left",
                         validate="many_to_one").merge(
        event, on=["date", "event_code"], how="left",
        validate="many_to_one")
    if (len(rows) != len(members) or rows.peer_clean.isna().any()
            or rows.event_clean.isna().any()):
        raise ValueError("Missing five-peer original-minute outcome")
    rows["peer_cash"] = np.where(rows.peer_clean, rows.peer_return, 0.0)
    rows["event_cash"] = np.where(rows.event_clean, rows.event_return, 0.0)
    if (not np.isfinite(rows.peer_cash).all()
            or not np.isfinite(rows.event_cash).all()):
        raise ValueError("Nonfinite nearest-peer cash return")
    per_event = rows.groupby(["date", "event_code"]).agg(
        event_cash=("event_cash", "first"),
        peer_cash=("peer_cash", "mean"),
        event_clean=("event_clean", "first"),
        peer_clean=("peer_clean", "mean"),
    ).reset_index()
    per_event["edge"] = per_event.event_cash - per_event.peer_cash
    report = {"note": "Post-result control diagnostic; not a blind test",
              "by_year": {}}
    for year in ("2024", "2025"):
        sample = per_event.loc[per_event.date.str.startswith(year)]
        daily = sample.groupby("date")[
            ["event_cash", "peer_cash", "edge"]].mean()
        report["by_year"][year] = {
            "events": len(sample), "days": len(daily),
            "event_cash_mean": float(daily.event_cash.mean()),
            "peer_cash_mean": float(daily.peer_cash.mean()),
            "edge_mean": float(daily.edge.mean()),
            "edge_month_ci": _month_bootstrap(
                daily.edge.reset_index(drop=True),
                pd.Series(daily.index.to_list()), 149),
            "event_clean_exit_rate": float(sample.event_clean.mean()),
            "peer_clean_exit_rate": float(sample.peer_clean.mean()),
        }
    (root / "nearest5_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("inputs", "evaluate"))
    parser.add_argument("--source", type=Path, default=Path(
        "data/research/insider_sell_complete"))
    parser.add_argument("--event-raw", type=Path, default=Path(
        "data/research/insider_sell_complete/repriced.parquet"))
    parser.add_argument("--peer-raw", type=Path, default=Path(
        "data/research/insider_sell_complete/industry_benchmark_peers_20k.parquet"))
    args = parser.parse_args()
    report = (build_inputs(args.source) if args.stage == "inputs" else
              evaluate(args.source, args.event_raw, args.peer_raw))
    print(report)


if __name__ == "__main__":
    main()
