"""Evaluate a committed, frozen screen on 2024 and later market data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

from .factor_scan import FACTORS
from .market_study import _quality_keys, _quality_symbols
from .strategy_scan import CANDIDATES, HORIZONS, _summarize


def _freeze(path: Path) -> dict:
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if (frozen.get("candidate") not in CANDIDATES
            or frozen.get("horizon") not in HORIZONS
            or frozen.get("daily_capacity") != 5
            or frozen.get("ranking") != "return_1450_desc_code_asc"):
        raise ValueError("Frozen strategy does not match the registered search space")
    return frozen


def _thresholds(report: dict) -> dict:
    checks = {}
    for period in ("2024", "2025_2026"):
        metrics = report[period]
        if not metrics.get("clean_completed_exits"):
            checks[period] = {"passed": False, "reasons": ["no completed exits"]}
            continue
        reasons = []
        if metrics["clean_completed_exits"] < 100:
            reasons.append("fewer than 100 quality-clean exits")
        if metrics["entry_fills"] / metrics["signals"] < .8:
            reasons.append("entry fill rate below 80%")
        if metrics["clean_completed_exits"] / metrics["signals"] < .7:
            reasons.append("clean completion rate below 70%")
        if metrics["date_weighted_mean_net_return"] <= 0:
            reasons.append("date-weighted net return is not positive")
        if metrics["date_weighted_week_bootstrap_95pct_interval"][0] <= 0:
            reasons.append("weekly bootstrap interval includes zero")
        if metrics["date_weighted_mean_with_10bps_slippage_each_side"] <= 0:
            reasons.append("net return fails the 10 bps per-side slippage stress")
        checks[period] = {"passed": not reasons, "reasons": reasons}
    return {
        "periods": checks,
        "both_periods_pass": all(item["passed"] for item in checks.values()),
        "interpretation": "Prerequisite for publishing a formula, not an estimate of live fills",
    }


def evaluate(snapshot_dir: Path, outcome_dir: Path, issues_dir: Path,
             freeze_file: Path, output: Path, trades_output: Path,
             allow_partial: bool = False) -> dict:
    frozen = _freeze(freeze_file)
    snapshots = sorted(snapshot_dir.glob("shard_*_part_*.parquet"))
    outcomes = sorted(outcome_dir.glob("shard_*_part_*.parquet"))
    shards = {path.name.split("_")[1] for path in snapshots}
    if not allow_partial and len(shards) != 20:
        raise ValueError(f"Expected 20 market shards; found {len(shards)}")
    if not snapshots or len(snapshots) != len(outcomes):
        raise ValueError("Snapshot and outcome partitions are incomplete")
    connection = duckdb.connect()
    connection.read_parquet(str(snapshot_dir / "*.parquet")).create_view("snapshots")
    connection.read_parquet(str(outcome_dir / "*.parquet")).create_view("outcomes")
    connection.register("bad_days", _quality_keys(issues_dir))
    connection.register("bad_symbols", _quality_symbols(issues_dir))
    connection.execute("""
        CREATE TEMP VIEW base AS
        SELECT s.date, s.code, s.return_1450, s.position_1450,
               s.volume_ratio_est, s.price_1450, s.amount_1450,
               s.ma5_prior_adjusted, s.ma20_prior_adjusted,
               s.return5_prior_adjusted, s.return20_prior_adjusted,
               s.preclose,
               o.horizon, o.entry_status, o.entry_price, o.shares,
               o.exit_status, o.exit_date, o.exit_price, o.net_return,
               NOT EXISTS (
                   SELECT 1 FROM bad_days AS x
                   WHERE x.code = s.code AND x.date >= s.date
                     AND x.date <= o.exit_date
               ) AS quality_clean_exit
        FROM snapshots AS s
        JOIN outcomes AS o USING (date, code)
        LEFT JOIN bad_days AS b ON b.date = s.date AND b.code = s.code
        LEFT JOIN bad_symbols AS excluded ON excluded.code = s.code
        WHERE s.date >= '2024-01-01'
          AND b.code IS NULL AND excluded.code IS NULL
          AND s.isST = 0 AND s.listing_age_sessions >= 20
          AND NOT s.reference_gap AND NOT s.quote_outside_traded_range
          AND s.amount_1450 >= 30000000
    """)
    horizon = frozen["horizon"]
    benchmark = connection.execute("""
        SELECT date,
               AVG(net_return) FILTER (
                   WHERE exit_status = 'filled' AND quality_clean_exit
               ) AS baseline_net_return
        FROM base WHERE horizon = ? GROUP BY date
    """, [horizon]).df()
    expression = FACTORS[frozen["candidate"]]
    frame = connection.execute(f"""
        SELECT * FROM (
            SELECT date, code, horizon, entry_status, entry_price, shares,
                   exit_status, exit_date, exit_price, net_return,
                   quality_clean_exit, return_1450, amount_1450,
                   ROW_NUMBER() OVER (
                       PARTITION BY date ORDER BY return_1450 DESC, code
                   ) AS daily_rank
            FROM base
            WHERE horizon = ? AND {expression} = 'selected'
        ) AS ranked WHERE daily_rank <= 5
    """, [horizon]).df()
    periods = {}
    for name, mask in (
        ("2024", frame["date"].str.startswith("2024")),
        ("2025_2026", frame["date"] >= "2025-01-01"),
    ):
        selected = frame.loc[mask]
        baseline = benchmark.loc[
            (benchmark["date"].str.startswith("2024") if name == "2024"
             else benchmark["date"] >= "2025-01-01"),
            ["date", "baseline_net_return"],
        ]
        periods[name] = _summarize(selected, baseline)
    result = {
        "scope": "Shanghai/Shenzhen 2024 holdout and 2025-2026 later test",
        "shards": len(shards), "frozen_strategy": frozen,
        "periods": periods, "publication_thresholds": _thresholds(periods),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    trades_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    frame.to_parquet(trades_output, index=False, compression="zstd")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path,
                        default=Path("data/research/market_snapshots_ci"))
    parser.add_argument("--outcomes", type=Path,
                        default=Path("data/research/market_outcomes_ci"))
    parser.add_argument("--issues", type=Path,
                        default=Path("data/research/market_issues_ci"))
    parser.add_argument("--freeze-file", type=Path,
                        default=Path("config/strategy-freeze.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/research/holdout_eval.json"))
    parser.add_argument("--trades-output", type=Path,
                        default=Path("data/research/holdout_trades.parquet"))
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    result = evaluate(args.snapshots, args.outcomes, args.issues,
                      args.freeze_file, args.output, args.trades_output,
                      args.allow_partial)
    print(json.dumps({"shards": result["shards"],
                      "publication_thresholds": result["publication_thresholds"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
