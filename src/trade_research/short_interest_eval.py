"""Evaluate the previously frozen post-regulation short-interest groups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .margin_heterogeneity_eval import write_reprice_signals
from .market_study import _quality_keys, _quality_symbols, _week_bootstrap
from .residual_liquidity import _period
from .short_interest_study import CONTROL, TREATED


HORIZONS = (1, 5)
PERIODS = {
    "2024_postreg": ("2024_runoff", "2024_postrunoff"),
    "2024_runoff": ("2024_runoff",),
    "2024_postrunoff": ("2024_postrunoff",),
    "2025": ("2025_H1", "2025_H2"),
    "2025_H1": ("2025_H1",),
    "2025_H2": ("2025_H2",),
}


def evaluate(selected_path: Path, quintiles_path: Path, outcome_dir: Path,
             issues_dir: Path, report_path: Path, trades_path: Path,
             treated: str = TREATED, control: str = CONTROL,
             periods: dict[str, tuple[str, ...]] = PERIODS) -> dict:
    selected = pd.read_parquet(selected_path)
    quintiles = pd.read_parquet(quintiles_path)
    if set(selected.candidate) != {treated, control}:
        raise ValueError("Missing frozen high or low short-interest group")
    if (not selected.date.gt(selected.trade_date).all()
            or not quintiles.date.gt(quintiles.trade_date).all()
            or not selected.date.str[:4].isin(("2024", "2025")).all()):
        raise ValueError("Short-interest result has a future or out-of-range balance")
    pairs = selected.groupby(["date", "pair_code"]).candidate.nunique()
    if not pairs.eq(2).all() or len(selected) != 2 * len(pairs):
        raise ValueError("Incomplete frozen short-interest pairs")
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.register("r", selected)
    connection.read_parquet(str(outcome_dir / "*.parquet")).create_view("o")
    connection.register("bad_days", _quality_keys(issues_dir))
    connection.register("bad_symbols", _quality_symbols(issues_dir))
    quality = """
        o.exit_status = 'filled'
        AND NOT EXISTS (SELECT 1 FROM bad_symbols b WHERE b.code = r.code)
        AND NOT EXISTS (
            SELECT 1 FROM bad_days q WHERE q.code = r.code
              AND q.date >= r.date AND q.date <= o.exit_date
        )
    """
    trades = connection.execute(f"""
        SELECT r.*, o.horizon, o.entry_status, o.entry_price, o.shares,
               o.exit_status, o.exit_date, o.exit_delay_sessions,
               o.exit_price, o.net_return,
               {quality} AS quality_clean_exit
        FROM r JOIN o USING(date, code)
        WHERE o.horizon IN (1, 5)
    """).df()
    if len(trades) != len(selected) * len(HORIZONS):
        raise ValueError("Frozen short-interest pair lacks minute outcomes")
    report = {"primary_horizon": 5, "matched": {}, "full_market": {},
              "note": "2025 is exploratory, not blind; 2026 outcomes untouched"}
    for period, windows in periods.items():
        report["matched"][period] = {}
        for horizon in HORIZONS:
            sample = trades.loc[trades.window.isin(windows)
                                & trades.horizon.eq(horizon)]
            high = sample.loc[sample.candidate.eq(treated)]
            low = sample.loc[sample.candidate.eq(control)]
            if high.empty or len(high) != len(low):
                raise ValueError(f"Missing matched short-interest {period} outcomes")
            summary = _period(high, low)
            summary["same_day_matched_mean"] = summary.pop(
                "same_day_random_mean")
            report["matched"][period][str(horizon)] = summary

    connection.unregister("r")
    connection.register("r", quintiles[["date", "code", "window", "board",
                                        "size_bucket", "quintile"]])
    full = connection.execute(f"""
        SELECT r.window, r.date, r.board, r.size_bucket, r.quintile,
               COUNT(*) AS stock_days,
               AVG(CASE WHEN {quality} THEN o.net_return ELSE 0 END) AS cash_mean,
               SUM(CASE WHEN o.entry_status = 'filled' THEN 1 ELSE 0 END) AS entries,
               SUM(CASE WHEN {quality} THEN 1 ELSE 0 END) AS clean_exits
        FROM r JOIN o USING(date, code)
        WHERE r.quintile IN (1, 5) AND o.horizon = 5
        GROUP BY r.window, r.date, r.board, r.size_bucket, r.quintile
    """).df()
    expected_full = int(quintiles.quintile.isin((1, 5)).sum())
    if int(full.stock_days.sum()) != expected_full:
        raise ValueError("A full-market short-interest stock-day lacks outcome")
    keys = ["window", "date", "board", "size_bucket"]
    means = full.pivot(index=keys, columns="quintile", values="cash_mean")
    if means.isna().any().any() or set(means.columns) != {1, 5}:
        raise ValueError("A short-interest stratum has no high or low group")
    means = means.rename(columns={1: "low", 5: "high"}).reset_index()
    means["edge"] = means.high - means.low
    for period, windows in periods.items():
        report["full_market"][period] = {}
        section = means.loc[means.window.isin(windows)]
        for board in ("all", "sh_main", "sh_star", "sz_main", "sz_gem"):
            part = section if board == "all" else section.loc[section.board.eq(board)]
            daily = part.groupby("date", as_index=False)[
                ["high", "low", "edge"]].mean()
            if daily.empty:
                continue
            coverage = full.loc[full.window.isin(windows)]
            if board != "all":
                coverage = coverage.loc[coverage.board.eq(board)]
            count = coverage.groupby("quintile")[
                ["stock_days", "entries", "clean_exits"]].sum()
            report["full_market"][period][board] = {
                "days": len(daily), "strata": len(part),
                "high_cash_mean": float(daily.high.mean()),
                "low_cash_mean": float(daily.low.mean()),
                "edge_mean": float(daily.edge.mean()),
                "edge_week_ci": _week_bootstrap(daily.edge, daily.date, 247),
                "high_entry_rate": float(count.loc[5, "entries"] /
                                         count.loc[5, "stock_days"]),
                "low_entry_rate": float(count.loc[1, "entries"] /
                                        count.loc[1, "stock_days"]),
                "high_clean_exit_rate": float(count.loc[5, "clean_exits"] /
                                              count.loc[5, "stock_days"]),
                "low_clean_exit_rate": float(count.loc[1, "clean_exits"] /
                                             count.loc[1, "stock_days"]),
            }
    report["full_market_stock_days"] = expected_full
    for path in (report_path, trades_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    trades.to_parquet(trades_path, index=False, compression="zstd")
    return report


def summarize_repriced(selected_path: Path, repriced_path: Path,
                       trades_path: Path, output_path: Path,
                       treated: str = TREATED, control: str = CONTROL,
                       periods: dict[str, tuple[str, ...]] = PERIODS) -> dict:
    membership = pd.read_parquet(selected_path)[
        ["date", "code", "candidate", "pair_code", "window"]]
    raw = pd.read_parquet(repriced_path)
    if (len(raw) != len(membership) * 4
            or set(raw.target_notional) != {20_000.0, 100_000.0}
            or set(raw.horizon) != {1, 5}
            or raw.duplicated(["date", "code", "target_notional", "horizon"]).any()):
        raise ValueError("Incomplete frozen raw-minute short-interest repricing")
    stored = pd.read_parquet(trades_path)
    comparison = stored.merge(raw.loc[raw.target_notional.eq(100_000)],
                              on=["date", "code", "horizon"],
                              suffixes=("_stored", "_raw"),
                              validate="one_to_one")
    if len(comparison) != len(stored):
        raise ValueError("Stored short-interest outcome lacks raw-minute match")
    for field in ("entry_status", "exit_status", "exit_date",
                  "quality_clean_exit"):
        if not comparison[f"{field}_stored"].fillna("missing").eq(
                comparison[f"{field}_raw"].fillna("missing")).all():
            raise ValueError(f"Stored/raw short-interest {field} disagree")
    for field in ("entry_price", "shares", "exit_delay_sessions",
                  "exit_price", "net_return"):
        if not np.isclose(comparison[f"{field}_stored"].to_numpy(dtype=float),
                          comparison[f"{field}_raw"].to_numpy(dtype=float),
                          atol=1e-12, equal_nan=True).all():
            raise ValueError(f"Stored/raw short-interest {field} disagree")
    rows = raw.merge(membership, on=["date", "code"], validate="many_to_one")
    if len(rows) != len(raw):
        raise ValueError("Raw short-interest trade outside frozen membership")
    report = {"exact_100k_rows": len(comparison), "results": {}}
    for size in (20_000, 100_000):
        report["results"][str(size)] = {}
        for period, windows in periods.items():
            report["results"][str(size)][period] = {}
            for horizon in HORIZONS:
                sample = rows.loc[rows.target_notional.eq(size)
                                  & rows.window.isin(windows)
                                  & rows.horizon.eq(horizon)]
                high = sample.loc[sample.candidate.eq(treated)]
                low = sample.loc[sample.candidate.eq(control)]
                summary = _period(high, low)
                summary["same_day_matched_mean"] = summary.pop(
                    "same_day_random_mean")
                report["results"][str(size)][period][str(horizon)] = summary
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", type=Path,
                        default=Path("data/research/short_interest_selected.parquet"))
    parser.add_argument("--quintiles", type=Path,
                        default=Path("data/research/short_interest_quintiles.parquet"))
    parser.add_argument("--snapshots", type=Path,
                        default=Path("data/research/market_snapshots_ci"))
    parser.add_argument("--outcomes", type=Path,
                        default=Path("data/research/market_outcomes_ci"))
    parser.add_argument("--issues", type=Path,
                        default=Path("data/research/market_issues_ci"))
    parser.add_argument("--report", type=Path,
                        default=Path("data/research/short_interest_report.json"))
    parser.add_argument("--trades", type=Path,
                        default=Path("data/research/short_interest_trades.parquet"))
    parser.add_argument("--reprice-signals", type=Path,
                        default=Path("data/research/short_interest_reprice_signals.parquet"))
    parser.add_argument("--repriced", type=Path)
    parser.add_argument("--repriced-report", type=Path,
                        default=Path("data/research/short_interest_reprice_report.json"))
    args = parser.parse_args()
    write_reprice_signals(args.selected, args.snapshots, args.reprice_signals)
    result = evaluate(args.selected, args.quintiles, args.outcomes,
                      args.issues, args.report, args.trades)
    print({"matched_pairs": len(pd.read_parquet(args.selected)) // 2,
           "full_market_stock_days": result["full_market_stock_days"],
           "matched_t5_edges": {key: value["5"]["edge_mean"]
                                for key, value in result["matched"].items()}})
    if args.repriced:
        reprice = summarize_repriced(args.selected, args.repriced,
                                     args.trades, args.repriced_report)
        print({"exact_100k_rows": reprice["exact_100k_rows"],
               "repriced_report": str(args.repriced_report)})


if __name__ == "__main__":
    main()
