"""Draw fixed low-turnover controls within late and prior-return strata.

The draw is deterministic by date and code. It describes factor buckets; it
does not turn a randomly ranked basket into a proposed trading formula.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd

from .strategy_scan import _last_safe_entry
from .study_periods import DEVELOPMENT_YEAR, VALIDATION_YEAR


STRATA = {
    "late_drop": "i.return_last30 < -.01",
    "late_middle": "i.return_last30 BETWEEN -.003 AND .003",
    "late_rise": "i.return_last30 > .01",
    "prior_deep_loss": "s.return20_prior_adjusted < -.1",
    "prior_middle": "s.return20_prior_adjusted BETWEEN -.1 AND .1",
    "prior_strong": "s.return20_prior_adjusted > .1",
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
    c.read_parquet(str(snapshot_dir / "*.parquet")).create_view("s")
    c.read_parquet(str(feature_file)).create_view("i")
    c.execute("CREATE TEMP VIEW snapshots AS SELECT date FROM s")
    last = _last_safe_entry(c, f"{year}-01-01", f"{year + 1}-01-01")
    common = f"""
        FROM s JOIN i USING (date, code)
        WHERE s.date BETWEEN '{year}-01-01' AND '{last}'
          AND s.isST = 0
          AND s.listing_age_sessions >= 20 AND NOT s.reference_gap
          AND NOT s.quote_outside_traded_range
          AND s.amount_1450 >= 30000000 AND s.amount_1450 < 100000000
          AND abs(s.price_1450 - i.price_1450) <= .005
    """
    # Execute each stratum separately so ranking remains exactly five per day.
    lists = []
    membership = []
    counts = {}
    for name, condition in STRATA.items():
        frame = c.execute(f"""
            SELECT * EXCLUDE (daily_rank) FROM (
                SELECT s.*, ROW_NUMBER() OVER (
                    PARTITION BY s.date
                    ORDER BY md5(s.date || s.code || 'low-strata-20260924'), s.code
                ) AS daily_rank
                {common} AND {condition}
            ) WHERE daily_rank <= 5
            ORDER BY date, code
        """).df()
        if frame.empty or frame.duplicated(["date", "code"]).any():
            raise ValueError(f"Empty or duplicate stratum: {name}")
        lists.append(frame)
        membership.append(frame[["date", "code"]].assign(stratum=name))
        counts[name] = {"signals": len(frame), "days": frame.date.nunique()}
    unique = pd.concat(lists, ignore_index=True).drop_duplicates(["date", "code"])
    mapping = pd.concat(membership, ignore_index=True)
    signals_output.parent.mkdir(parents=True, exist_ok=True)
    membership_output.parent.mkdir(parents=True, exist_ok=True)
    unique.to_parquet(signals_output, index=False, compression="zstd")
    mapping.to_parquet(membership_output, index=False, compression="zstd")
    return {"last_entry": last, "unique_signals": len(unique),
            "strata": counts}


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
                        default=Path("data/research/stratified_low_sample"))
    args = parser.parse_args()
    result = select(args.snapshots, args.features / f"{args.year}.parquet",
                    args.year,
                    args.output_dir / f"signals_{args.year}.parquet",
                    args.output_dir / f"membership_{args.year}.parquet")
    print(result)


if __name__ == "__main__":
    main()
