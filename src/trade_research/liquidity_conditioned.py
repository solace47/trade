"""Diagnose liquidity shocks after controlling prior momentum and board.

This follow-up was designed after seeing the broad liquidity-shock results.
It is exploratory; only a rule frozen before opening 2026 can use that year
as a holdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

from .market_study import _quality_keys, _quality_symbols
from .residual_liquidity import CAPACITY, HORIZONS, _period


RANKS = {
    "adjusted_absolute": "adjusted_absolute DESC",
    "adjusted_relative": "adjusted_relative DESC",
    "unadjusted_absolute": "absolute_shock DESC",
}


def study(features_path: Path, snapshot_dir: Path, outcome_dir: Path,
          issues_dir: Path, report_path: Path, trades_path: Path) -> dict:
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.read_parquet(str(features_path)).create_view("f")
    connection.read_parquet(str(snapshot_dir / "*.parquet")).create_view("s")
    connection.read_parquet(str(outcome_dir / "*.parquet")).create_view("o")
    connection.register("bad_days", _quality_keys(issues_dir))
    connection.register("bad_symbols", _quality_symbols(issues_dir))
    connection.execute("""
        CREATE TEMP TABLE pool AS
        SELECT f.date, f.code, f.absolute_shock,
               f.adjusted_absolute, f.adjusted_relative
        FROM f JOIN s USING (date, code)
        WHERE (f.code LIKE 'sh.60%' OR f.code LIKE 'sz.00%')
          AND s.return20_prior_adjusted BETWEEN -.10 AND .10
    """)
    selected = []
    for name, order in RANKS.items():
        ranked = connection.execute(f"""
            SELECT date, code, daily_rank FROM (
                SELECT date, code, ROW_NUMBER() OVER (
                    PARTITION BY date ORDER BY {order}, code
                ) AS daily_rank FROM pool
            ) WHERE daily_rank <= {CAPACITY}
        """).df()
        ranked["candidate"] = name
        selected.append(ranked)
    random = connection.execute(f"""
        SELECT date, code, daily_rank FROM (
            SELECT date, code, ROW_NUMBER() OVER (
                PARTITION BY date ORDER BY md5(date || code), code
            ) AS daily_rank FROM pool
        ) WHERE daily_rank <= {CAPACITY}
    """).df()
    random["candidate"] = "same_pool_random"
    selected.append(random)
    membership = pd.concat(selected, ignore_index=True)
    if membership.duplicated(["candidate", "date", "code"]).any():
        raise ValueError("Duplicate ranked stock-day")
    connection.register("membership", membership)
    trades = connection.execute("""
        SELECT r.*, o.horizon, o.entry_status, o.entry_price, o.shares,
               o.exit_status, o.exit_date, o.exit_delay_sessions,
               o.exit_price, o.net_return,
               o.exit_status = 'filled'
               AND NOT EXISTS (SELECT 1 FROM bad_symbols b
                               WHERE b.code = r.code)
               AND NOT EXISTS (
                   SELECT 1 FROM bad_days q
                   WHERE q.code = r.code AND q.date >= r.date
                     AND q.date <= o.exit_date
               ) AS quality_clean_exit
        FROM membership r JOIN o USING (date, code)
        WHERE o.horizon IN (1, 2, 3, 5)
    """).df()
    if len(trades) != len(membership) * len(HORIZONS):
        raise ValueError("A ranked stock-day lacks an outcome")
    if not trades.date.str[:4].isin(("2024", "2025")).all():
        raise ValueError("Conditioned study escaped the 2024-2025 window")
    report = {
        "hypotheses": list(RANKS), "capacity": CAPACITY,
        "pool": "Shanghai/Shenzhen main board, prior 20-day adjusted return "
                "between -10% and +10%, base liquidity universe",
        "control": "deterministic same-day random stocks from the same pool",
        "note": "Designed after viewing broad-shock results; exploratory only",
        "results": {},
    }
    for name in RANKS:
        report["results"][name] = {}
        for year in ("2024", "2025"):
            annual = trades.loc[trades.candidate.eq(name)
                                & trades.date.str.startswith(year)]
            control = trades.loc[trades.candidate.eq("same_pool_random")
                                 & trades.date.str.startswith(year)]
            for horizon in HORIZONS:
                subset = annual.loc[annual.horizon.eq(horizon)]
                matched = control.loc[control.horizon.eq(horizon)]
                report["results"][name].setdefault(year, {})[str(horizon)] = {}
                for period, rows in (
                    ("H1", subset.loc[subset.date.str[5:7].astype(int).le(6)]),
                    ("H2", subset.loc[subset.date.str[5:7].astype(int).gt(6)]),
                    ("full", subset),
                ):
                    if not rows.empty:
                        report["results"][name][year][str(horizon)][period] = \
                            _period(rows, matched)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    trades_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    trades.to_parquet(trades_path, index=False, compression="zstd")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path,
                        default=Path("data/research/liquidity_shock_features.parquet"))
    parser.add_argument("--snapshots", type=Path,
                        default=Path("data/research/market_snapshots_ci"))
    parser.add_argument("--outcomes", type=Path,
                        default=Path("data/research/market_outcomes_ci"))
    parser.add_argument("--issues", type=Path,
                        default=Path("data/research/market_issues_ci"))
    parser.add_argument("--report", type=Path,
                        default=Path("data/research/liquidity_conditioned_report.json"))
    parser.add_argument("--trades", type=Path,
                        default=Path("data/research/liquidity_conditioned_trades.parquet"))
    args = parser.parse_args()
    report = study(args.features, args.snapshots, args.outcomes, args.issues,
                   args.report, args.trades)
    print({"hypotheses": list(report["results"]), "report": str(args.report)})


if __name__ == "__main__":
    main()
