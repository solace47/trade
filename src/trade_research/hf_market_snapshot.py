"""Resumable, partitioned 14:50 snapshots for historical Shanghai/Shenzhen stocks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from .hf_audit import read_window
from .hf_download import selected_paths
from .hf_snapshot import snapshot_for_symbol
from .ingest import _atomic_parquet


FIRST_DATE = "2020-01-01"
LAST_DATE = "2026-08-06"


def build(hf_root: Path, bao_root: Path, output: Path,
          first_date: str = FIRST_DATE, last_date: str = LAST_DATE,
          batch_size: int = 50, limit_batches: int | None = None) -> dict:
    symbols_file = bao_root / "metadata" / "symbols.parquet"
    symbols = pd.read_parquet(symbols_file)
    paths = selected_paths(symbols_file)
    output.mkdir(parents=True, exist_ok=True)
    processed = 0
    for offset in range(0, len(symbols), batch_size):
        if limit_batches is not None and processed >= limit_batches:
            break
        target = output / f"part_{offset // batch_size:04d}.parquet"
        missing_file = output / f"part_{offset // batch_size:04d}_missing.json"
        if target.exists():
            prior_missing = json.loads(missing_file.read_text(encoding="utf-8")) \
                if missing_file.exists() else []
            new_sources = any(
                (hf_root / f"data/stock_1m/{code.split('.')[0].upper()}/{code.split('.')[1]}.parquet").exists()
                and (bao_root / "daily" / f"{code.replace('.', '_')}.parquet").exists()
                for code in prior_missing
            )
            if not new_sources:
                print(f"Cached snapshot partition {target.name}", flush=True)
                processed += 1
                continue
        frames = []
        missing_batch = []
        for code, relative in zip(symbols["code"].iloc[offset:offset + batch_size],
                                  paths[offset:offset + batch_size], strict=True):
            minute_path = hf_root / relative
            daily_path = bao_root / "daily" / f"{code.replace('.', '_')}.parquet"
            if not minute_path.exists() or not daily_path.exists():
                missing_batch.append(code)
                continue
            minute = read_window(minute_path, first_date, last_date)
            daily = pd.read_parquet(daily_path)
            frame = snapshot_for_symbol(minute, daily, code)
            if not frame.empty:
                frames.append(frame)
        if frames:
            partition = pd.concat(frames, ignore_index=True)
            partition = partition.sort_values(["date", "code"])
            if partition.duplicated(["date", "code"]).any():
                raise ValueError(f"Duplicate snapshot keys in partition {offset}")
            _atomic_parquet(partition, target)
            print(f"Wrote {target.name}: {len(partition)} snapshots", flush=True)
        missing_file.write_text(
            json.dumps(missing_batch, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        processed += 1
    partitions = sorted(output.glob("part_*.parquet"))
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "first_date": first_date, "last_date": last_date,
        "selected_symbols": len(symbols),
        "partition_files": len(partitions),
        "snapshot_rows": sum(pd.read_parquet(path, columns=["date"]).shape[0]
                             for path in partitions),
        "missing_source_symbols": sorted({
            code for path in output.glob("part_*_missing.json")
            for code in json.loads(path.read_text(encoding="utf-8"))
        }),
        "output": str(output),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-root", type=Path, default=Path("data/hf/pilot"))
    parser.add_argument("--bao-root", type=Path, default=Path("data/baostock/market_2020_2026"))
    parser.add_argument("--output", type=Path, default=Path("data/research/market_snapshots"))
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--limit-batches", type=int)
    args = parser.parse_args()
    print(json.dumps(build(args.hf_root, args.bao_root, args.output,
                           batch_size=args.batch_size,
                           limit_batches=args.limit_batches), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
