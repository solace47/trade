"""Exploratory T+1 factor bins on development and validation dates only.

The 2024 holdout and later dates are deliberately inaccessible here. Bins
are fixed in code before full-market results are inspected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .market_study import _quality_keys


FACTORS = {
    "intraday_return": """
        CASE WHEN return_1450 < 0 THEN 'loss'
             WHEN return_1450 < .015 THEN '0_to_1_5pct'
             WHEN return_1450 < .03 THEN '1_5_to_3pct'
             WHEN return_1450 < .05 THEN '3_to_5pct'
             WHEN return_1450 < .07 THEN '5_to_7pct'
             ELSE 'at_least_7pct' END
    """,
    "intraday_position": """
        CASE WHEN position_1450 IS NULL THEN 'unknown'
             WHEN position_1450 < .3 THEN 'bottom_30pct'
             WHEN position_1450 < .7 THEN 'middle_40pct'
             ELSE 'top_30pct' END
    """,
    "volume_ratio": """
        CASE WHEN volume_ratio_est IS NULL THEN 'unknown'
             WHEN volume_ratio_est < .8 THEN 'below_0_8'
             WHEN volume_ratio_est < 1.2 THEN '0_8_to_1_2'
             WHEN volume_ratio_est < 2 THEN '1_2_to_2'
             ELSE 'at_least_2' END
    """,
    "distance_to_ma20": """
        CASE WHEN distance_ma20_adjusted IS NULL THEN 'unknown'
             WHEN distance_ma20_adjusted < -.05 THEN 'below_minus_5pct'
             WHEN distance_ma20_adjusted < 0 THEN 'minus_5_to_0pct'
             WHEN distance_ma20_adjusted < .05 THEN '0_to_5pct'
             WHEN distance_ma20_adjusted < .10 THEN '5_to_10pct'
             ELSE 'at_least_10pct' END
    """,
    "prior_five_day_return": """
        CASE WHEN return5_prior_adjusted IS NULL THEN 'unknown'
             WHEN return5_prior_adjusted < -.05 THEN 'below_minus_5pct'
             WHEN return5_prior_adjusted < 0 THEN 'minus_5_to_0pct'
             WHEN return5_prior_adjusted < .05 THEN '0_to_5pct'
             ELSE 'at_least_5pct' END
    """,
    "prior_twenty_day_return": """
        CASE WHEN return20_prior_adjusted IS NULL THEN 'unknown'
             WHEN return20_prior_adjusted < -.10 THEN 'below_minus_10pct'
             WHEN return20_prior_adjusted < 0 THEN 'minus_10_to_0pct'
             WHEN return20_prior_adjusted < .10 THEN '0_to_10pct'
             ELSE 'at_least_10pct' END
    """,
    "turnover_amount": """
        CASE WHEN amount_1450 < 100000000 THEN '30m_to_100m'
             WHEN amount_1450 < 500000000 THEN '100m_to_500m'
             ELSE 'at_least_500m' END
    """,
    "prior_trend": """
        CASE WHEN ma5_prior_adjusted IS NULL OR ma20_prior_adjusted IS NULL
                       OR ma60_prior_adjusted IS NULL THEN 'unknown'
             WHEN ma5_prior_adjusted > ma20_prior_adjusted
                       AND ma20_prior_adjusted > ma60_prior_adjusted THEN 'up'
             WHEN ma5_prior_adjusted < ma20_prior_adjusted
                       AND ma20_prior_adjusted < ma60_prior_adjusted THEN 'down'
             ELSE 'mixed' END
    """,
    "intraday_range": """
        CASE WHEN preclose <= 0 THEN 'unknown'
             WHEN (high_1450 - low_1450) / preclose < .02 THEN 'below_2pct'
             WHEN (high_1450 - low_1450) / preclose < .05 THEN '2_to_5pct'
             WHEN (high_1450 - low_1450) / preclose < .08 THEN '5_to_8pct'
             ELSE 'at_least_8pct' END
    """,
}


def _metrics(frame: pd.DataFrame, baseline: pd.DataFrame) -> dict:
    signals = int(frame["signals"].sum())
    fills = int(frame["entry_fills"].sum())
    completed = int(frame["clean_outcomes"].sum())
    result = {
        "signals": signals, "entry_fills": fills,
        "clean_completed_exits": completed,
        "signal_days": int(len(frame)),
    }
    if completed == 0:
        return result
    active = frame.loc[frame["clean_outcomes"] > 0].copy()
    active["daily_mean"] = active["net_sum"] / active["clean_outcomes"]
    matched = active.merge(baseline[["date", "baseline_daily_mean"]],
                           on="date", validate="one_to_one")
    edge = (matched["daily_mean"] - matched["baseline_daily_mean"]).to_numpy()
    rng = np.random.default_rng(20260924)
    sampled = rng.choice(edge, size=(2000, len(edge)), replace=True).mean(axis=1)
    lo, hi = np.quantile(sampled, [.025, .975])
    result.update({
        "completed_days": int(len(active)),
        "mean_net_return_per_trade": float(active["net_sum"].sum() / completed),
        "date_weighted_mean_net_return": float(active["daily_mean"].mean()),
        "win_rate": float(active["wins"].sum() / completed),
        "date_weighted_edge_vs_same_day_universe": float(edge.mean()),
        "edge_date_bootstrap_95pct_interval": [float(lo), float(hi)],
    })
    return result


def scan(snapshot_dir: Path, outcome_dir: Path, issues_dir: Path,
         output: Path, allow_partial: bool = False) -> dict:
    snapshots = sorted(snapshot_dir.glob("shard_*_part_*.parquet"))
    outcomes = sorted(outcome_dir.glob("shard_*_part_*.parquet"))
    shards = {path.name.split("_")[1] for path in snapshots}
    if not allow_partial and len(shards) != 20:
        raise ValueError(f"Expected 20 market shards; found {len(shards)}")
    if not snapshots or len(snapshots) != len(outcomes):
        raise ValueError("Snapshot and outcome partitions are incomplete")
    bad = _quality_keys(issues_dir)
    connection = duckdb.connect()
    connection.read_parquet(str(snapshot_dir / "*.parquet")).create_view("snapshots")
    connection.read_parquet(str(outcome_dir / "*.parquet")).create_view("outcomes")
    connection.register("bad_quality", bad)
    connection.execute("""
        CREATE TEMP TABLE base AS
        SELECT s.date, s.return_1450, s.position_1450, s.volume_ratio_est,
               s.distance_ma20_adjusted, s.return5_prior_adjusted,
               s.return20_prior_adjusted, s.amount_1450,
               s.ma5_prior_adjusted, s.ma20_prior_adjusted,
               s.ma60_prior_adjusted, s.preclose, s.high_1450, s.low_1450,
               o.entry_status, o.exit_status, o.net_return,
               CASE WHEN s.date < '2023-01-01' THEN '2022_development'
                    ELSE '2023_validation' END AS period,
               x.code IS NULL AS quality_clean_exit
        FROM snapshots AS s
        JOIN outcomes AS o USING (date, code)
        LEFT JOIN bad_quality AS b ON b.date = s.date AND b.code = s.code
        LEFT JOIN bad_quality AS x ON x.date = o.exit_date AND x.code = o.code
        WHERE o.horizon = 1 AND s.date >= '2022-01-01' AND s.date < '2024-01-01'
          AND b.code IS NULL AND s.isST = 0 AND s.listing_age_sessions >= 20
          AND NOT s.reference_gap AND NOT s.quote_outside_traded_range
          AND s.amount_1450 >= 30000000
    """)
    baseline = connection.execute("""
        SELECT period, date,
               AVG(net_return) FILTER (WHERE exit_status = 'filled'
                    AND quality_clean_exit) AS baseline_daily_mean
        FROM base GROUP BY period, date
    """).df()
    universe_days = connection.execute("""
        SELECT period, date, COUNT(*) AS signals,
               COUNT(*) FILTER (WHERE entry_status = 'filled') AS entry_fills,
               COUNT(*) FILTER (WHERE exit_status = 'filled'
                   AND quality_clean_exit) AS clean_outcomes,
               COALESCE(SUM(net_return) FILTER (WHERE exit_status = 'filled'
                   AND quality_clean_exit), 0) AS net_sum,
               COUNT(*) FILTER (WHERE exit_status = 'filled'
                   AND quality_clean_exit AND net_return > 0) AS wins
        FROM base GROUP BY period, date
    """).df()
    universe = {period: _metrics(frame, baseline.loc[baseline["period"] == period])
                for period, frame in universe_days.groupby("period")}
    report = {}
    for name, expression in FACTORS.items():
        by_date = connection.execute(f"""
            SELECT period, date, {expression} AS bucket,
                   COUNT(*) AS signals,
                   COUNT(*) FILTER (WHERE entry_status = 'filled') AS entry_fills,
                   COUNT(*) FILTER (WHERE exit_status = 'filled'
                       AND quality_clean_exit) AS clean_outcomes,
                   COALESCE(SUM(net_return) FILTER (WHERE exit_status = 'filled'
                       AND quality_clean_exit), 0) AS net_sum,
                   COUNT(*) FILTER (WHERE exit_status = 'filled'
                       AND quality_clean_exit AND net_return > 0) AS wins
            FROM base GROUP BY period, date, bucket
        """).df()
        factor = {}
        for (period, bucket), frame in by_date.groupby(["period", "bucket"]):
            factor.setdefault(period, {})[bucket] = _metrics(
                frame, baseline.loc[baseline["period"] == period]
            )
        report[name] = factor
    result = {
        "scope": "Shanghai/Shenzhen 2022 development and 2023 validation, T+1",
        "shards": len(shards), "quality_bad_stock_days": len(bad),
        "base": "non-ST, listed >=20 sessions, no reference gap, clean minute audit, "
                "14:50 turnover >=CNY30m",
        "universe": universe,
        "note": "Exploratory bins; multiple comparisons require a separately frozen strategy and holdout.",
        "factors": report,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
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
                        default=Path("data/research/factor_scan.json"))
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    result = scan(args.snapshots, args.outcomes, args.issues,
                  args.output, args.allow_partial)
    print(json.dumps({"shards": result["shards"],
                      "factors": list(result["factors"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
