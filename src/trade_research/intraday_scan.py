"""Explore six recent-market late-session hypotheses with fixed daily capacity.

The 2024 run is for development. Run 2025 only after reviewing 2024; both
years have already been consulted elsewhere and are not untouched holdouts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

from .market_study import _quality_keys, _quality_symbols, _week_bootstrap
from .strategy_scan import _last_safe_entry, _stressed_returns
from .study_periods import DEVELOPMENT_YEAR, VALIDATION_YEAR


# Conditions use only the snapshot and intraday bars stamped at or before 14:50.
# Ranking chooses at most five new positions on any signal day.
SCREENS = {
    "late_push": (
        "return_last30 >= .007 AND return_1450 BETWEEN 0 AND .05 "
        "AND position_1450 >= .7 AND ma5_prior_adjusted > ma20_prior_adjusted",
        "return_last30 DESC",
    ),
    "late_rebound": (
        "return_last30 >= .007 AND return_1450 BETWEEN -.04 AND 0 "
        "AND position_1450 >= .4",
        "return_last30 DESC",
    ),
    "late_dip": (
        "return_last30 <= -.007 AND return_1450 BETWEEN .01 AND .05 "
        "AND return20_prior_adjusted > 0",
        "return_last30 ASC",
    ),
    "late_volume_push": (
        "volume_share_last30 >= .18 AND return_last30 >= .003 "
        "AND return_1450 BETWEEN 0 AND .05",
        "volume_share_last30 DESC",
    ),
    "quiet_trend": (
        "volume_share_last30 <= .08 AND return_last30 BETWEEN -.002 AND .002 "
        "AND return_1450 BETWEEN 0 AND .02 AND position_1450 >= .7 "
        "AND return20_prior_adjusted > 0",
        "return20_prior_adjusted DESC",
    ),
    "gap_down_recovery": (
        "overnight_gap <= -.01 AND open_to_cutoff >= .01 "
        "AND return_1450 BETWEEN -.01 AND .03 AND position_1450 >= .7 "
        "AND return_last30 >= 0",
        "overnight_gap ASC",
    ),
}
HORIZONS = (1, 2, 3, 5)


def _summary(frame: pd.DataFrame) -> dict:
    signals = len(frame)
    entries = int(frame.entry_status.eq("filled").sum())
    valid = frame.loc[frame.exit_status.eq("filled")
                      & frame.quality_clean_exit].copy()
    result = {
        "signals": signals,
        "signal_days": int(frame.date.nunique()),
        "entry_fills": entries,
        "clean_exits": len(valid),
        "ontime_exits": int(valid.exit_delay_sessions.eq(0).sum()),
    }
    if valid.empty:
        return result
    valid["stress10"] = _stressed_returns(valid, 10)
    dates = sorted(frame.date.unique())
    per_day = frame.groupby("date").size().reindex(dates)
    cash = valid.groupby("date").net_return.sum().reindex(dates, fill_value=0) / per_day
    cash_stress = (valid.groupby("date").stress10.sum().reindex(dates, fill_value=0)
                   / per_day)
    daily = valid.groupby("date").net_return.mean()
    result.update({
        "median_trade_net": float(valid.net_return.median()),
        "date_weighted_net": float(daily.mean()),
        "cash_aware_signal_mean": float(cash.mean()),
        "cash_aware_stress10_mean": float(cash_stress.mean()),
        "cash_aware_week_ci": _week_bootstrap(
            cash.reset_index(drop=True), pd.Series(dates), 20260924
        ),
    })
    return result


def scan(snapshot_dir: Path, outcome_dir: Path, issues_dir: Path,
         feature_file: Path, year: int,
         trades_output: Path, report_output: Path) -> dict:
    if year not in (DEVELOPMENT_YEAR, VALIDATION_YEAR):
        raise ValueError("This research scans 2024-2025 only")
    if len({path.name.split("_")[1] for path in snapshot_dir.glob(
            "shard_*_part_*.parquet")}) != 20:
        raise ValueError("Expected 20 complete market shards")
    connection = duckdb.connect()
    connection.read_parquet(str(snapshot_dir / "*.parquet")).create_view("s")
    connection.read_parquet(str(outcome_dir / "*.parquet")).create_view("o")
    connection.read_parquet(str(feature_file)).create_view("i")
    connection.register("bad_days", _quality_keys(issues_dir))
    connection.register("bad_symbols", _quality_symbols(issues_dir))
    # _last_safe_entry expects a view named snapshots.
    connection.execute("CREATE TEMP VIEW snapshots AS SELECT date FROM s")
    last_entry = _last_safe_entry(
        connection, f"{year}-01-01", f"{year + 1}-01-01"
    )
    connection.execute(f"""
        CREATE TEMP TABLE eligible AS
        SELECT s.date, s.code, s.return_1450, s.position_1450,
               s.ma5_prior_adjusted, s.ma20_prior_adjusted,
               s.return20_prior_adjusted, s.amount_1450,
               i.return_last30, i.return_last15, i.return_afternoon,
               i.volume_share_last30, i.volume_share_last15,
               i.premium_to_last30_vwap,
               s.open_1450 / NULLIF(s.preclose, 0) - 1 AS overnight_gap,
               s.price_1450 / NULLIF(s.open_1450, 0) - 1 AS open_to_cutoff
        FROM s JOIN i USING (date, code)
        WHERE s.date BETWEEN '{year}-01-01' AND '{last_entry}'
          AND s.isST = 0 AND s.listing_age_sessions >= 20
          AND NOT s.reference_gap AND NOT s.quote_outside_traded_range
          AND abs(s.price_1450 - i.price_1450) <= .005
          AND s.amount_1450 BETWEEN 100000000 AND 1000000000
    """)
    pieces = []
    report = {"year": year, "last_entry": last_entry,
              "order_size": 100_000, "screen": {}}
    for name, (condition, ranking) in SCREENS.items():
        trades = connection.execute(f"""
            WITH ranked AS (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY date ORDER BY {ranking}, code
                ) AS daily_rank
                FROM eligible WHERE {condition}
            )
            SELECT r.date, r.code, r.daily_rank, r.return_last30,
                   r.volume_share_last30, r.overnight_gap,
                   r.return_1450, r.amount_1450,
                   o.horizon, o.entry_status, o.entry_price, o.shares,
                   o.exit_status, o.exit_date, o.exit_delay_sessions,
                   o.exit_price, o.net_return,
                   NOT EXISTS (SELECT 1 FROM bad_symbols b
                               WHERE b.code = r.code)
                   AND
                   NOT EXISTS (
                       SELECT 1 FROM bad_days q
                       WHERE q.code = r.code AND q.date >= r.date
                         AND q.date <= o.exit_date
                   ) AS quality_clean_exit
            FROM ranked r JOIN o USING (date, code)
            WHERE r.daily_rank <= 5
        """).df()
        if trades.empty or trades.duplicated(["date", "code", "horizon"]).any():
            raise ValueError(f"Empty or duplicate outcomes for {name}")
        trades["screen"] = name
        pieces.append(trades)
        periods = {}
        half = trades.date.str[5:7].astype(int).le(6).map({True: "H1", False: "H2"})
        for (period, horizon), frame in trades.groupby([half, "horizon"]):
            periods.setdefault(period, {})[str(horizon)] = _summary(frame)
        report["screen"][name] = periods
    output = pd.concat(pieces, ignore_index=True)
    trades_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(trades_output, index=False, compression="zstd")
    report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int,
                        choices=(DEVELOPMENT_YEAR, VALIDATION_YEAR),
                        default=DEVELOPMENT_YEAR)
    parser.add_argument("--snapshots", type=Path,
                        default=Path("data/research/market_snapshots_ci"))
    parser.add_argument("--outcomes", type=Path,
                        default=Path("data/research/market_outcomes_ci"))
    parser.add_argument("--issues", type=Path,
                        default=Path("data/research/market_issues_ci"))
    parser.add_argument("--feature-dir", type=Path,
                        default=Path("data/research/intraday_features"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/research/intraday_scan"))
    args = parser.parse_args()
    report = scan(args.snapshots, args.outcomes, args.issues,
                  args.feature_dir / f"{args.year}.parquet", args.year,
                  args.output_dir / f"trades_{args.year}.parquet",
                  args.output_dir / f"report_{args.year}.json")
    print(json.dumps({"year": args.year, "screens": list(report["screen"]),
                      "last_entry": report["last_entry"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
