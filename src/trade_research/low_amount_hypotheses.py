"""Select three low-turnover, calm-tape hypotheses for 20k order repricing.

These were proposed after the 2024 late-session factor audit. Their 2024
results are exploratory and must be checked on later dates before use.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd

from .strategy_scan import _last_safe_entry
from .study_periods import DEVELOPMENT_YEAR, VALIDATION_YEAR


SCREENS = {
    "small_calm": (
        "abs(return_last30) <= .003 AND overnight_gap BETWEEN -.01 AND .01",
        "amount_1450 ASC",
    ),
    "small_calm_advance": (
        "abs(return_last30) <= .003 AND overnight_gap BETWEEN -.01 AND .01 "
        "AND breadth >= .6",
        "amount_1450 ASC",
    ),
    "small_loser_advance": (
        "abs(return_last30) <= .003 AND return20_prior_adjusted <= -.1 "
        "AND breadth >= .6",
        "return20_prior_adjusted ASC",
    ),
}


def select(snapshot_dir: Path, feature_file: Path,
           year: int, signals_output: Path,
           membership_output: Path) -> dict:
    if year not in (DEVELOPMENT_YEAR, VALIDATION_YEAR):
        raise ValueError("Only 2024-2025 may be tested")
    if len({path.name.split("_")[1] for path in snapshot_dir.glob(
            "shard_*_part_*.parquet")}) != 20:
        raise ValueError("Expected 20 complete market shards")
    c = duckdb.connect()
    c.read_parquet(str(snapshot_dir / "*.parquet")).create_view("snapshots")
    c.read_parquet(str(feature_file)).create_view("intraday")
    last_entry = _last_safe_entry(c, f"{year}-01-01", f"{year + 1}-01-01")
    c.execute(f"""
        CREATE TEMP TABLE base AS
        SELECT s.*, i.return_last30,
               s.open_1450 / NULLIF(s.preclose, 0) - 1 AS overnight_gap
        FROM snapshots s JOIN intraday i USING (date, code)
        WHERE s.date BETWEEN '{year}-01-01' AND '{last_entry}'
          AND s.isST = 0
          AND s.listing_age_sessions >= 20 AND NOT s.reference_gap
          AND NOT s.quote_outside_traded_range
          AND abs(s.price_1450 - i.price_1450) <= .005
    """)
    c.execute("""
        CREATE TEMP TABLE breadth AS
        SELECT date, AVG((return_1450 > 0)::INT) AS breadth
        FROM base GROUP BY date
    """)
    c.execute("""
        CREATE TEMP VIEW candidates AS
        SELECT b.*, m.breadth
        FROM base b JOIN breadth m USING (date)
        WHERE b.amount_1450 >= 30000000 AND b.amount_1450 < 100000000
    """)
    selected = []
    membership = []
    counts = {}
    for name, (condition, ranking) in SCREENS.items():
        frame = c.execute(f"""
            SELECT * EXCLUDE (daily_rank) FROM (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY date ORDER BY {ranking}, code
                ) AS daily_rank
                FROM candidates WHERE {condition}
            ) WHERE daily_rank <= 5
            ORDER BY date, code
        """).df()
        if frame.empty or frame.duplicated(["date", "code"]).any():
            raise ValueError(f"Empty or duplicate signals for {name}")
        selected.append(frame)
        membership.append(frame[["date", "code"]].assign(candidate=name))
        counts[name] = {"signals": len(frame), "days": frame.date.nunique()}
    signals = pd.concat(selected, ignore_index=True).drop_duplicates(
        ["date", "code"]
    )
    mapping = pd.concat(membership, ignore_index=True)
    signals_output.parent.mkdir(parents=True, exist_ok=True)
    membership_output.parent.mkdir(parents=True, exist_ok=True)
    signals.to_parquet(signals_output, index=False, compression="zstd")
    mapping.to_parquet(membership_output, index=False, compression="zstd")
    return {"last_entry": last_entry, "unique_signals": len(signals),
            "candidates": counts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int,
                        choices=(DEVELOPMENT_YEAR, VALIDATION_YEAR),
                        default=DEVELOPMENT_YEAR)
    parser.add_argument("--snapshots", type=Path,
                        default=Path("data/research/market_snapshots_ci"))
    parser.add_argument("--features", type=Path,
                        default=Path("data/research/intraday_features"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/research/low_amount_hypotheses"))
    args = parser.parse_args()
    result = select(args.snapshots, args.features / f"{args.year}.parquet",
                    args.year,
                    args.output_dir / f"signals_{args.year}.parquet",
                    args.output_dir / f"membership_{args.year}.parquet")
    print(result)


if __name__ == "__main__":
    main()
