"""Compare registered screens and exits using 2024-2025 only.

Queries are restricted to 2024-2025. The selected screen, horizon, and daily
capacity must be frozen before evaluating the 2026 holdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .factor_scan import FACTORS
from .hf_outcomes import Assumptions
from .market_study import _quality_keys, _quality_symbols, _week_bootstrap
from .study_periods import DEVELOPMENT_YEAR, VALIDATION_YEAR, HOLDOUT_YEAR


CANDIDATES = tuple(name for name in FACTORS if name.startswith("candidate_"))
HORIZONS = (1, 2, 3, 5)


def _last_safe_entry(connection: duckdb.DuckDBPyConnection,
                     first_date: str, last_date: str,
                     reserve_sessions: int = 10) -> str:
    """Leave enough market sessions for T+5 and a five-session exit delay."""
    dates = [row[0] for row in connection.execute("""
        SELECT DISTINCT date FROM snapshots
        WHERE date >= ? AND date < ? ORDER BY date
    """, [first_date, last_date]).fetchall()]
    if len(dates) <= reserve_sessions:
        raise ValueError("Too few trading dates to reserve an exit window")
    return dates[-(reserve_sessions + 1)]


def _stressed_returns(frame: pd.DataFrame, slippage_bps_each_side: float) -> np.ndarray:
    """Reprice completed trades at wider slip; keep original fill decisions."""
    terms = Assumptions()
    original_slip = terms.slippage_bps_each_side / 10_000
    stressed_slip = slippage_bps_each_side / 10_000
    shares = frame["shares"].to_numpy(dtype=float)
    buy_value = shares * frame["entry_price"].to_numpy(dtype=float) / \
        (1 + original_slip) * (1 + stressed_slip)
    sell_value = shares * frame["exit_price"].to_numpy(dtype=float) / \
        (1 - original_slip) * (1 - stressed_slip)
    transfer_buy = np.where(
        frame["date"].to_numpy() < terms.transfer_cutover_date,
        terms.transfer_bps_each_side_before_cutover, terms.transfer_bps_each_side,
    ) / 10_000
    transfer_sell = np.where(
        frame["exit_date"].to_numpy() < terms.transfer_cutover_date,
        terms.transfer_bps_each_side_before_cutover, terms.transfer_bps_each_side,
    ) / 10_000
    stamp_sell = np.where(
        frame["exit_date"].to_numpy() < terms.stamp_tax_cutover_date,
        terms.stamp_tax_bps_on_sale_before_cutover, terms.stamp_tax_bps_on_sale,
    ) / 10_000
    commission_buy = np.maximum(
        terms.commission_minimum_each_side,
        buy_value * terms.commission_bps_each_side / 10_000,
    )
    commission_sell = np.maximum(
        terms.commission_minimum_each_side,
        sell_value * terms.commission_bps_each_side / 10_000,
    )
    return (sell_value * (1 - transfer_sell - stamp_sell) - commission_sell) / \
        (buy_value * (1 + transfer_buy) + commission_buy) - 1


def _summarize(frame: pd.DataFrame, baseline: pd.DataFrame) -> dict:
    entry_filled = frame["entry_status"].eq("filled")
    completed = frame["exit_status"].eq("filled") & frame["quality_clean_exit"]
    valid = frame.loc[completed].copy()
    result = {
        "signals": len(frame),
        "signal_days": int(frame["date"].nunique()),
        "entry_fills": int(entry_filled.sum()),
        "clean_completed_exits": len(valid),
        "delayed_clean_exits": int(valid["exit_delay_sessions"].gt(0).sum()),
        "exit_source_quality_excluded": int((
            frame["exit_status"].eq("filled") & ~frame["quality_clean_exit"]
        ).sum()),
        "corporate_action_exits": int(frame["exit_status"].eq(
            "corporate_action_unadjusted"
        ).sum()),
    }
    if valid.empty:
        return result
    returns = valid["net_return"].to_numpy(dtype=float)
    if not np.allclose(_stressed_returns(valid, 5), returns, atol=1e-12):
        raise ValueError("Stored net returns disagree with the execution cost model")
    stress_10 = _stressed_returns(valid, 10)
    stress_20 = _stressed_returns(valid, 20)
    valid["stress_10"] = stress_10
    valid["stress_20"] = stress_20
    daily = valid.groupby("date", as_index=False)[
        ["net_return", "stress_10", "stress_20"]
    ].mean()
    daily = daily.merge(baseline, on="date", how="left", validate="one_to_one")
    if daily["baseline_net_return"].isna().any():
        raise ValueError("Missing same-day baseline for a completed candidate")
    edge = daily["net_return"] - daily["baseline_net_return"]
    trim = len(returns) // 10
    middle = np.sort(returns)[trim:len(returns) - trim] if trim else returns
    result.update({
        "completed_days": len(daily),
        "win_rate": float((returns > 0).mean()),
        "mean_net_return_per_trade": float(returns.mean()),
        "median_net_return_per_trade": float(np.median(returns)),
        "trimmed_10pct_mean_net_return": float(middle.mean()),
        "mean_net_return_with_10bps_slippage_each_side": float(stress_10.mean()),
        "mean_net_return_with_20bps_slippage_each_side": float(stress_20.mean()),
        "loss_decile_return": float(np.quantile(returns, .1)),
        "date_weighted_mean_net_return": float(daily["net_return"].mean()),
        "date_weighted_mean_with_10bps_slippage_each_side": float(
            daily["stress_10"].mean()
        ),
        "date_weighted_mean_with_20bps_slippage_each_side": float(
            daily["stress_20"].mean()
        ),
        "date_weighted_week_bootstrap_95pct_interval": _week_bootstrap(
            daily["net_return"], daily["date"], 20260924
        ),
        "date_weighted_edge_vs_same_day_universe": float(edge.mean()),
        "edge_week_bootstrap_95pct_interval": _week_bootstrap(
            edge, daily["date"], 20260925
        ),
    })
    return result


def scan(snapshot_dir: Path, outcome_dir: Path, issues_dir: Path,
         output: Path, trades_output: Path,
         allow_partial: bool = False) -> dict:
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
    last_development = _last_safe_entry(
        connection, f"{DEVELOPMENT_YEAR}-01-01", f"{VALIDATION_YEAR}-01-01"
    )
    last_validation = _last_safe_entry(
        connection, f"{VALIDATION_YEAR}-01-01", f"{HOLDOUT_YEAR}-01-01"
    )
    bad_days = _quality_keys(issues_dir)
    bad_symbols = _quality_symbols(issues_dir)
    connection.register("bad_days", bad_days)
    connection.register("bad_symbols", bad_symbols)
    connection.execute(f"""
        CREATE TEMP VIEW base AS
        SELECT s.date, s.code, s.return_1450, s.position_1450,
               s.volume_ratio_est, s.price_1450, s.amount_1450,
               s.ma5_prior_adjusted, s.ma20_prior_adjusted,
               s.return5_prior_adjusted,
               s.return20_prior_adjusted, s.preclose,
               o.horizon, o.entry_status, o.entry_price, o.shares,
               o.exit_status, o.exit_date, o.exit_delay_sessions,
               o.exit_price, o.net_return,
               NOT EXISTS (
                   SELECT 1 FROM bad_days AS x
                   WHERE x.code = s.code AND x.date >= s.date
                     AND x.date <= o.exit_date
               ) AS quality_clean_exit
        FROM snapshots AS s
        JOIN outcomes AS o USING (date, code)
        LEFT JOIN bad_days AS b ON b.date = s.date AND b.code = s.code
        LEFT JOIN bad_symbols AS excluded ON excluded.code = s.code
        WHERE ((s.date >= '{DEVELOPMENT_YEAR}-01-01'
                AND s.date <= '{last_development}')
           OR (s.date >= '{VALIDATION_YEAR}-01-01'
                AND s.date <= '{last_validation}'))
          AND b.code IS NULL AND excluded.code IS NULL
          AND s.isST = 0 AND s.listing_age_sessions >= 20
          AND NOT s.reference_gap AND NOT s.quote_outside_traded_range
          AND s.amount_1450 >= 30000000
    """)
    baseline = connection.execute("""
        SELECT date, horizon,
               AVG(net_return) FILTER (
                   WHERE exit_status = 'filled' AND quality_clean_exit
               ) AS baseline_net_return
        FROM base GROUP BY date, horizon
    """).df()
    reports = {}
    trades = []
    for name in CANDIDATES:
        expression = FACTORS[name]
        frame = connection.execute(f"""
            SELECT * FROM (
                SELECT date, code, horizon, entry_status, entry_price, shares,
                       exit_status, exit_date, exit_delay_sessions,
                       exit_price, net_return,
                       quality_clean_exit,
                       return_1450, amount_1450,
                       ROW_NUMBER() OVER (
                           PARTITION BY date, horizon
                           ORDER BY return_1450 DESC, code
                       ) AS daily_rank
                FROM base WHERE {expression} = 'selected'
            ) AS ranked WHERE daily_rank <= 5
        """).df()
        frame["candidate"] = name
        trades.append(frame)
        periods = {}
        for year in (str(DEVELOPMENT_YEAR), str(VALIDATION_YEAR)):
            annual = frame.loc[frame["date"].str.startswith(year)]
            for horizon in HORIZONS:
                subset = annual.loc[annual["horizon"] == horizon]
                benchmark = baseline.loc[
                    (baseline["date"].str.startswith(year))
                    & (baseline["horizon"] == horizon),
                    ["date", "baseline_net_return"],
                ]
                periods.setdefault(year, {})[str(horizon)] = _summarize(
                    subset, benchmark
                )
        reports[name] = periods
    result = {
        "scope": f"Shanghai/Shenzhen development {DEVELOPMENT_YEAR} and "
                 f"validation {VALIDATION_YEAR} only",
        "shards": len(shards),
        "last_entry_dates": {
            str(DEVELOPMENT_YEAR): last_development,
            str(VALIDATION_YEAR): last_validation,
        },
        "capacity": "at most five ranked signals per trading day",
        "ranking": "14:50 return descending, code ascending tie break",
        "candidate_screens": list(CANDIDATES),
        "horizons": HORIZONS,
        "multiple_comparisons": f"Choose one screen and horizon before opening "
                                f"the {HOLDOUT_YEAR} holdout",
        "reports": reports,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    trades_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    pd.concat(trades, ignore_index=True).to_parquet(
        trades_output, index=False, compression="zstd"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path,
                        default=Path("data/research/market_snapshots_ci"))
    parser.add_argument("--outcomes", type=Path,
                        default=Path("data/research/market_outcomes_ci"))
    parser.add_argument("--issues", type=Path,
                        default=Path("data/research/market_issues_ci"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/research/strategy_scan.json"))
    parser.add_argument("--trades-output", type=Path,
                        default=Path("data/research/strategy_scan_trades.parquet"))
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    result = scan(args.snapshots, args.outcomes, args.issues,
                  args.output, args.trades_output, args.allow_partial)
    print(json.dumps({"shards": result["shards"],
                      "screens": result["candidate_screens"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
