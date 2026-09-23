"""Check that imported 14:50 snapshots have one outcome for each exit horizon."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb


def check(snapshot_dir: Path, outcome_dir: Path, output: Path,
          allow_partial: bool = False) -> dict:
    snapshots = sorted(snapshot_dir.glob("shard_*_part_*.parquet"))
    outcomes = sorted(outcome_dir.glob("shard_*_part_*.parquet"))
    snapshot_names = {path.name for path in snapshots}
    outcome_names = {path.name for path in outcomes}
    shards = {name.split("_")[1] for name in snapshot_names}
    if not snapshots or snapshot_names != outcome_names:
        raise ValueError("Snapshot and outcome partitions differ")
    if not allow_partial and shards != {f"{n:02d}" for n in range(20)}:
        raise ValueError("Expected all 20 market shards")

    db = duckdb.connect()
    db.read_parquet(str(snapshot_dir / "*.parquet")).create_view("snapshots")
    db.read_parquet(str(outcome_dir / "*.parquet")).create_view("outcomes")
    snapshot_rows, snapshot_keys, null_snapshot_keys = db.execute("""
        SELECT COUNT(*), COUNT(DISTINCT (date, code)),
               COUNT(*) FILTER (WHERE date IS NULL OR code IS NULL)
        FROM snapshots
    """).fetchone()
    outcome_rows, outcome_keys, bad_horizons, null_outcome_keys = db.execute("""
        SELECT COUNT(*), COUNT(DISTINCT (date, code, horizon)),
               COUNT(*) FILTER (WHERE horizon NOT IN (1, 2, 3, 5)),
               COUNT(*) FILTER (WHERE date IS NULL OR code IS NULL OR horizon IS NULL)
        FROM outcomes
    """).fetchone()
    orphan_outcomes = db.execute("""
        SELECT COUNT(*) FROM outcomes AS o
        ANTI JOIN snapshots AS s USING (date, code)
    """).fetchone()[0]
    invalid_dates = db.execute("""
        SELECT COUNT(*) FROM outcomes
        WHERE (target_exit_date IS NOT NULL AND target_exit_date <= date)
           OR (exit_date IS NOT NULL AND exit_date <= date)
           OR (exit_status = 'filled' AND (exit_date IS NULL OR net_return IS NULL))
    """).fetchone()[0]
    valid = (
        snapshot_rows == snapshot_keys and null_snapshot_keys == 0
        and outcome_rows == outcome_keys == snapshot_rows * 4
        and bad_horizons == null_outcome_keys == orphan_outcomes == invalid_dates == 0
    )
    result = {
        "shards": len(shards), "snapshot_partitions": len(snapshots),
        "snapshot_rows": snapshot_rows, "unique_snapshot_keys": snapshot_keys,
        "outcome_rows": outcome_rows, "unique_outcome_keys": outcome_keys,
        "invalid_horizons": bad_horizons, "null_keys": null_snapshot_keys + null_outcome_keys,
        "orphan_outcomes": orphan_outcomes, "invalid_outcome_dates": invalid_dates,
        "valid": valid,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    if not valid:
        raise ValueError(f"Imported market partitions failed integrity check: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path,
                        default=Path("data/research/market_snapshots_ci"))
    parser.add_argument("--outcomes", type=Path,
                        default=Path("data/research/market_outcomes_ci"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/research/market_integrity.json"))
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    print(json.dumps(check(args.snapshots, args.outcomes, args.output,
                           args.allow_partial), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
