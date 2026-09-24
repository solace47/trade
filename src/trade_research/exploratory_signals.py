"""Recreate two recent-market exploratory signal lists for execution checks."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd

from .market_study import _quality_keys, _quality_symbols
from .strategy_scan import _last_safe_entry
from .study_periods import DEVELOPMENT_YEAR, VALIDATION_YEAR


SCREENS = {
    "low_amount_neutral": (
        "s.amount_1450 < 100000000 "
        "AND s.return_1450 >= 0 AND s.return_1450 < .015",
        "s.amount_1450 ASC, s.code ASC",
    ),
    "mid_amount_prior_loser": (
        "s.amount_1450 >= 70000000 AND s.amount_1450 < 100000000 "
        "AND s.return20_prior_adjusted IS NOT NULL",
        "s.return20_prior_adjusted ASC, s.code ASC",
    ),
}


def select(snapshot_dir: Path, issues_dir: Path, candidate: str,
           output: Path, allow_partial: bool = False) -> pd.DataFrame:
    if candidate not in SCREENS:
        raise ValueError(f"Unknown exploratory candidate: {candidate}")
    snapshots = sorted(snapshot_dir.glob("shard_*_part_*.parquet"))
    shards = {path.name.split("_")[1] for path in snapshots}
    if not snapshots or (not allow_partial and len(shards) != 20):
        raise ValueError("Expected 20 market snapshot shards")
    connection = duckdb.connect()
    connection.read_parquet(str(snapshot_dir / "*.parquet")).create_view("snapshots")
    last_development = _last_safe_entry(
        connection, f"{DEVELOPMENT_YEAR}-01-01", f"{VALIDATION_YEAR}-01-01"
    )
    last_validation = _last_safe_entry(
        connection, f"{VALIDATION_YEAR}-01-01", f"{VALIDATION_YEAR + 1}-01-01"
    )
    connection.register("bad_days", _quality_keys(issues_dir))
    connection.register("bad_symbols", _quality_symbols(issues_dir))
    condition, ranking = SCREENS[candidate]
    frame = connection.execute(f"""
        WITH ranked AS (
            SELECT s.*,
                   ROW_NUMBER() OVER (PARTITION BY s.date
                       ORDER BY {ranking}) AS daily_rank
            FROM snapshots AS s
            LEFT JOIN bad_days AS d ON d.date = s.date AND d.code = s.code
            LEFT JOIN bad_symbols AS excluded ON excluded.code = s.code
            WHERE ((s.date >= '{DEVELOPMENT_YEAR}-01-01'
                    AND s.date <= '{last_development}')
                OR (s.date >= '{VALIDATION_YEAR}-01-01'
                    AND s.date <= '{last_validation}'))
              AND d.code IS NULL AND excluded.code IS NULL
              AND s.isST = 0 AND s.listing_age_sessions >= 20
              AND NOT s.reference_gap AND NOT s.quote_outside_traded_range
              AND s.amount_1450 >= 30000000 AND {condition}
        )
        SELECT * EXCLUDE (daily_rank) FROM ranked WHERE daily_rank <= 5
        ORDER BY date, daily_rank
    """).df()
    if frame.empty or frame.duplicated(["date", "code"]).any():
        raise ValueError("No unique exploratory signals")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False, compression="zstd")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=SCREENS, required=True)
    parser.add_argument("--snapshots", type=Path,
                        default=Path("data/research/market_snapshots_ci"))
    parser.add_argument("--issues", type=Path,
                        default=Path("data/research/market_issues_ci"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    frame = select(args.snapshots, args.issues, args.candidate,
                   args.output, args.allow_partial)
    print({"candidate": args.candidate, "signals": len(frame),
           "first": frame["date"].min(), "last": frame["date"].max()})


if __name__ == "__main__":
    main()
