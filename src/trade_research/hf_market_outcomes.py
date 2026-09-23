"""Build resumable market-wide T+1/T+2/T+3/T+5 execution outcomes."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .hf_audit import read_window
from .hf_outcomes import Assumptions, HORIZONS, outcomes_for_symbol
from .ingest import _atomic_parquet


FIRST_DATE = "2020-01-01"
LAST_DATE = "2026-08-06"


def build(hf_root: Path, bao_root: Path, snapshots: Path, output: Path,
          limit_batches: int | None = None,
          assumptions: Assumptions = Assumptions()) -> dict:
    trading = pd.read_parquet(bao_root / "metadata" / "calendar.parquet")
    calendar = sorted(trading.loc[
        (trading["is_trading_day"] == "1")
        & trading["calendar_date"].between(FIRST_DATE, LAST_DATE), "calendar_date"
    ].tolist())
    partitions = sorted(snapshots.glob("part_*.parquet"))
    if limit_batches is not None:
        partitions = partitions[:limit_batches]
    output.mkdir(parents=True, exist_ok=True)
    for source in partitions:
        target = output / source.name
        if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
            print(f"Cached outcome partition {target.name}", flush=True)
            continue
        signals = pd.read_parquet(source)
        frames = []
        for code, stock_signals in signals.groupby("code", sort=False):
            exchange, symbol = code.split(".")
            minute_file = hf_root / "data" / "stock_1m" / exchange.upper() / f"{symbol}.parquet"
            daily_file = bao_root / "daily" / f"{exchange}_{symbol}.parquet"
            minute = read_window(minute_file, FIRST_DATE, LAST_DATE)
            daily = pd.read_parquet(daily_file)
            frames.append(outcomes_for_symbol(
                stock_signals, minute, daily, calendar, assumptions
            ))
        result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if len(result) != len(signals) * len(HORIZONS):
            raise ValueError(f"Outcome count mismatch in {source.name}")
        _atomic_parquet(result, target)
        print(f"Wrote {target.name}: {len(result)} outcomes", flush=True)
    available = sorted(output.glob("part_*.parquet"))
    summary = {
        "assumptions": asdict(assumptions),
        "first_date": FIRST_DATE, "last_date": LAST_DATE,
        "snapshot_partitions": len(sorted(snapshots.glob("part_*.parquet"))),
        "outcome_partitions": len(available),
        "outcome_rows": sum(pq.ParquetFile(path).metadata.num_rows for path in available),
        "complete_for_available_snapshots": (
            len(available) == len(sorted(snapshots.glob("part_*.parquet")))
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-root", type=Path, default=Path("data/hf/pilot"))
    parser.add_argument("--bao-root", type=Path, default=Path("data/baostock/market_2020_2026"))
    parser.add_argument("--snapshots", type=Path, default=Path("data/research/market_snapshots"))
    parser.add_argument("--output", type=Path, default=Path("data/research/market_outcomes"))
    parser.add_argument("--limit-batches", type=int)
    args = parser.parse_args()
    print(json.dumps(build(args.hf_root, args.bao_root, args.snapshots,
                           args.output, args.limit_batches), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
