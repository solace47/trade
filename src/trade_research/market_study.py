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
BAD_STOCK_FIELDS = (
    "invalid_rows", "duplicate_rows", "wrong_identity_rows",
    "volume_over_100_shares_days", "amount_over_0_01pct_days",
    "unexpected_nontrading_minute_days", "active_no_trade_days",
)


def _week_bootstrap(values: pd.Series, dates: pd.Series, seed: int) -> list[float]:
    """Resample weeks so nearby signal days remain together."""
    weekly = pd.DataFrame({
        "value": values.to_numpy(),
        "week": pd.to_datetime(dates).dt.to_period("W-SUN").astype(str).to_numpy(),
    }).groupby("week")["value"].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(weekly), size=(2000, len(weekly)))
    means = weekly["sum"].to_numpy()[draws].sum(axis=1) / \
        weekly["count"].to_numpy()[draws].sum(axis=1)
    return [float(x) for x in np.quantile(means, [.025, .975])]


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


def _quality_symbols(issues_dir: Path) -> pd.DataFrame:
    codes = set()
    for issue_path in issues_dir.glob("shard_*.csv"):
        stock_path = issue_path.resolve().with_name("stocks.csv")
        if not stock_path.exists():
            continue
        stocks = pd.read_csv(stock_path, dtype={"code": str})
        bad = pd.Series(False, index=stocks.index)
        for field in BAD_STOCK_FIELDS:
            if field in stocks:
                bad |= stocks[field].fillna(0).gt(0)
        codes.update(stocks.loc[bad, "code"])
    return pd.DataFrame({"code": sorted(codes)})


def _summarize(days: pd.DataFrame) -> dict:
    signals = int(days["signals"].sum())
    entries = int(days["entry_fills"].sum())
    completed = int(days["clean_outcomes"].sum())
    result = {
        "signals": signals, "entry_fills": entries,
        "quality_clean_completed_exits": completed,
        "quality_excluded_or_unfilled": signals - completed,
        "entry_not_filled": signals - entries,
        "corporate_action_exits": int(days["corporate_actions"].sum()),
        "source_quality_excluded_exits": int(days["bad_exits"].sum()),
        "delayed_completed_exits": int(days["delayed_exits"].sum()),
        "trading_dates_with_signals": int(len(days)),
    }
    if not completed:
        return result
    active = days.loc[days["clean_outcomes"] > 0].copy()
    daily_mean = active["net_sum"] / active["clean_outcomes"]
    interval = _week_bootstrap(daily_mean, active["date"], 20260924)
    result.update({
        "trading_dates_with_completed_exits": int(len(active)),
        "mean_net_return_per_trade": float(days["net_sum"].sum() / completed),
        "date_weighted_mean_net_return": float(daily_mean.mean()),
        "week_bootstrap_95pct_interval": interval,
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
    bad_symbols = _quality_symbols(issues_dir)
    connection = duckdb.connect()
    connection.read_parquet(str(snapshot_dir / "*.parquet")).create_view("snapshots_raw")
    connection.read_parquet(str(outcome_dir / "*.parquet")).create_view("outcomes_raw")
    connection.register("bad_quality", bad)
    connection.register("bad_symbols", bad_symbols)
    connection.execute("""
        CREATE TEMP VIEW clean_snapshots AS
        SELECT s.* FROM snapshots_raw AS s
        LEFT JOIN bad_symbols AS b ON b.code = s.code
        WHERE b.code IS NULL
    """)
    connection.execute("""
        CREATE TEMP VIEW market_environment AS
        SELECT s.date,
               CASE WHEN AVG(CASE WHEN s.return_1450 > 0 THEN 1.0 ELSE 0.0 END) >= .6
                         THEN 'broad_advance'
                    WHEN AVG(CASE WHEN s.return_1450 > 0 THEN 1.0 ELSE 0.0 END) <= .4
                         THEN 'broad_decline'
                    ELSE 'mixed' END AS regime
        FROM clean_snapshots AS s
        LEFT JOIN bad_quality AS b ON b.date = s.date AND b.code = s.code
        WHERE b.code IS NULL AND s.isST = 0 AND s.listing_age_sessions >= 20
        GROUP BY s.date
    """)
    connection.execute(f"""
        CREATE TEMP VIEW screened AS
        SELECT s.*, CASE WHEN b.code IS NULL AND {SCREEN}
                         THEN TRUE ELSE FALSE END AS candidate
        FROM clean_snapshots AS s
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
        SELECT s.date, s.code, s.candidate, s.tail_rank, m.regime,
               o.horizon, o.entry_status, o.exit_status, o.exit_delay_sessions,
               o.net_return,
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
        JOIN market_environment AS m USING (date)
    """)
    policies = {
        "all_signals": "TRUE",
        "frozen_rule": "candidate",
        "frozen_rule_top5": "candidate AND tail_rank <= 5",
    }
    report = {}
    by_regime = {}
    for name, condition in policies.items():
        days = connection.execute(f"""
            SELECT period, horizon, date, regime,
                   COUNT(*) AS signals,
                   SUM(CASE WHEN entry_status = 'filled' THEN 1 ELSE 0 END) AS entry_fills,
                   SUM(CASE WHEN exit_status = 'filled' AND quality_clean
                            THEN 1 ELSE 0 END) AS clean_outcomes,
                   SUM(CASE WHEN exit_status = 'filled' AND quality_clean
                            THEN net_return ELSE 0 END) AS net_sum,
                   SUM(CASE WHEN exit_status = 'filled' AND quality_clean
                                 AND net_return > 0 THEN 1 ELSE 0 END) AS wins,
                   SUM(CASE WHEN exit_status = 'corporate_action_unadjusted'
                            THEN 1 ELSE 0 END) AS corporate_actions,
                   SUM(CASE WHEN exit_status = 'filled' AND NOT quality_clean
                            THEN 1 ELSE 0 END) AS bad_exits,
                   SUM(CASE WHEN exit_status = 'filled' AND quality_clean
                                 AND exit_delay_sessions > 0 THEN 1 ELSE 0 END) AS delayed_exits
            FROM joined WHERE {condition}
            GROUP BY period, horizon, date, regime
        """).df()
        policy = {}
        for (period, horizon), frame in days.groupby(["period", "horizon"]):
            policy.setdefault(period, {})[str(horizon)] = _summarize(frame)
        report[name] = policy
        if name == "frozen_rule_top5":
            for (period, horizon, regime), frame in days.groupby(
                ["period", "horizon", "regime"]
            ):
                by_regime.setdefault(period, {}).setdefault(str(horizon), {})[regime] = \
                    _summarize(frame)
    result = {
        "scope": "historical Shanghai/Shenzhen, quality-screened 2022-2026",
        "shards": len(shard_ids), "quality_bad_stock_days": len(bad),
        "quality_excluded_symbols": len(bad_symbols),
        "execution_model": "14:52-14:55 VWAP, costs, volume cap, limits, T+1, delayed exit",
        "rule": "non-ST; seasoned >=20 sessions; 14:50 gain 1.5%-5%; position >=0.7; "
                "estimated volume ratio >=1.2; above adjusted prior MA20; turnover >=CNY30m",
        "market_environment": "14:50 fraction of seasoned non-ST stocks advancing: "
                              "<=40% decline, >=60% advance, otherwise mixed",
        "policies": report, "frozen_rule_top5_by_regime": by_regime,
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
