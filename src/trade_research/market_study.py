"""Evaluate the frozen tail-entry rule on audited market snapshots and fills."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


SCREEN = """
    s.isST = 0
    AND s.listing_age_sessions >= 20
    AND NOT s.reference_gap
    AND NOT s.quote_outside_traded_range
    AND s.return_1450 BETWEEN 0.015 AND 0.05
    AND s.position_1450 >= 0.7
    AND s.volume_ratio_est >= 1.2
    AND s.price_1450 > s.ma20_prior_adjusted
    AND s.amount_1450 >= 30000000
"""


def _quality_keys(issues_dir: Path) -> pd.DataFrame:
    files = sorted(issues_dir.glob("shard_*.csv"))
    if not files:
        raise FileNotFoundError(f"No shard quality reports in {issues_dir}")
    frames = [pd.read_csv(path, dtype=str) for path in files]
    issues = pd.concat(frames, ignore_index=True)
    bad = issues.loc[issues["kind"].isin((
        "ohlc_disagreement", "partial_minute_day", "unexpected_minute_on_suspended_day",
        "active_no_trade", "missing_active_minute",
    )), ["date", "code"]].drop_duplicates()
    return bad.reset_index(drop=True)


def _summarize(days: pd.DataFrame) -> dict:
    signals = int(days["signals"].sum())
    entries = int(days["entry_fills"].sum())
    completed = int(days["clean_outcomes"].sum())
    result = {
        "signals": signals, "entry_fills": entries,
        "quality_clean_completed_exits": completed,
        "quality_excluded_or_unfilled": signals - completed,
        "trading_dates_with_signals": int(len(days)),
    }
    if not completed:
        return result
    active = days.loc[days["clean_outcomes"] > 0].copy()
    daily_mean = active["net_sum"] / active["clean_outcomes"]
    rng = np.random.default_rng(20260924)
    draw = rng.choice(daily_mean.to_numpy(), size=(2000, len(daily_mean)), replace=True)
    interval = np.quantile(draw.mean(axis=1), [0.025, 0.975])
    result.update({
        "trading_dates_with_completed_exits": int(len(active)),
        "mean_net_return_per_trade": float(days["net_sum"].sum() / completed),
        "date_weighted_mean_net_return": float(daily_mean.mean()),
        "date_bootstrap_95pct_interval": [float(interval[0]), float(interval[1])],
        "win_rate_per_trade": float(days["wins"].sum() / completed),
    })
    return result


def study(snapshot_dir: Path, outcome_dir: Path, issues_dir: Path,
          output: Path, allow_partial: bool = False) -> dict:
    snapshot_files = sorted(snapshot_dir.glob("shard_*_part_*.parquet"))
    outcome_files = sorted(outcome_dir.glob("shard_*_part_*.parquet"))
    shard_ids = {path.name.split("_")[1] for path in snapshot_files}
    if not allow_partial and len(shard_ids) != 20:
        raise ValueError(f"Expected 20 market shards; found {len(shard_ids)}")
    if not snapshot_files or len(snapshot_files) != len(outcome_files):
        raise ValueError("Snapshot and outcome partitions are incomplete")
    bad = _quality_keys(issues_dir)
    connection = duckdb.connect()
    connection.read_parquet(str(snapshot_dir / "*.parquet")).create_view("snapshots_raw")
    connection.read_parquet(str(outcome_dir / "*.parquet")).create_view("outcomes_raw")
    connection.register("bad_quality", bad)
    connection.execute(f"""
        CREATE TEMP VIEW screened AS
        SELECT s.*, CASE WHEN b.code IS NULL AND {SCREEN}
                         THEN TRUE ELSE FALSE END AS candidate
        FROM snapshots_raw AS s
        LEFT JOIN bad_quality AS b ON b.date = s.date AND b.code = s.code
    """)
    connection.execute("""
        CREATE TEMP VIEW ranked AS
        SELECT *, ROW_NUMBER() OVER (
            PARTITION BY date
            ORDER BY CASE WHEN candidate THEN return_1450 ELSE -1000000 END DESC, code
        ) AS tail_rank
        FROM screened
    """)
    connection.execute("""
        CREATE TEMP VIEW joined AS
        SELECT s.date, s.code, s.candidate, s.tail_rank,
               o.horizon, o.entry_status, o.exit_status, o.net_return,
               CASE WHEN s.date < '2023-01-01' THEN '2022_development'
                    WHEN s.date < '2024-01-01' THEN '2023_validation'
                    WHEN s.date < '2025-01-01' THEN '2024_holdout'
                    ELSE '2025_2026_later' END AS period,
               NOT EXISTS (
                   SELECT 1 FROM bad_quality AS x
                   WHERE x.code = s.code AND x.date >= s.date
                     AND x.date <= o.exit_date
               ) AS quality_clean
        FROM ranked AS s
        JOIN outcomes_raw AS o USING (date, code)
    """)
    policies = {
        "all_signals": "TRUE",
        "frozen_rule": "candidate",
        "frozen_rule_top5": "candidate AND tail_rank <= 5",
    }
    report = {}
    for name, condition in policies.items():
        days = connection.execute(f"""
            SELECT period, horizon, date,
                   COUNT(*) AS signals,
                   SUM(CASE WHEN entry_status = 'filled' THEN 1 ELSE 0 END) AS entry_fills,
                   SUM(CASE WHEN exit_status = 'filled' AND quality_clean
                            THEN 1 ELSE 0 END) AS clean_outcomes,
                   SUM(CASE WHEN exit_status = 'filled' AND quality_clean
                            THEN net_return ELSE 0 END) AS net_sum,
                   SUM(CASE WHEN exit_status = 'filled' AND quality_clean
                                 AND net_return > 0 THEN 1 ELSE 0 END) AS wins
            FROM joined WHERE {condition}
            GROUP BY period, horizon, date
        """).df()
        policy = {}
        for (period, horizon), frame in days.groupby(["period", "horizon"]):
            policy.setdefault(period, {})[str(horizon)] = _summarize(frame)
        report[name] = policy
    result = {
        "scope": "historical Shanghai/Shenzhen, quality-screened 2022-2026",
        "shards": len(shard_ids), "quality_bad_stock_days": len(bad),
        "execution_model": "14:52-14:55 VWAP, costs, volume cap, limits, T+1, delayed exit",
        "rule": "non-ST; seasoned >=20 sessions; 14:50 gain 1.5%-5%; position >=0.7; "
                "estimated volume ratio >=1.2; above adjusted prior MA20; turnover >=CNY30m",
        "policies": report,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, default=Path("data/research/market_snapshots_ci"))
    parser.add_argument("--outcomes", type=Path, default=Path("data/research/market_outcomes_ci"))
    parser.add_argument("--issues", type=Path, default=Path("data/research/market_issues_ci"))
    parser.add_argument("--output", type=Path, default=Path("data/research/market_study.json"))
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    report = study(args.snapshots, args.outcomes, args.issues,
                   args.output, args.allow_partial)
    print(json.dumps({
        "shards": report["shards"], "quality_bad_stock_days": report["quality_bad_stock_days"],
        "frozen_rule": report["policies"]["frozen_rule"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
