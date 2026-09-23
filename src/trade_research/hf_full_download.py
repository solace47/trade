"""Resumable batch download of the pinned full one-minute archive."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import pyarrow.parquet as pq

from .hf_download import REPO, REVISION
from .network import effective_environment


def _valid_parquet(path: Path) -> bool:
    try:
        return path.exists() and pq.ParquetFile(path).metadata.num_rows > 0
    except Exception:
        return False


def download(inventory_file: Path, output: Path, workers: int,
             chunk_size: int, limit_batches: int | None = None) -> dict:
    frozen = json.loads(inventory_file.read_text(encoding="utf-8"))
    if frozen["repository"] != REPO or frozen["revision"] != REVISION:
        raise ValueError("Inventory points to a different dataset version")
    paths = frozen["stock_files"]
    cli = Path(sys.executable).with_name("hf")
    command = str(cli) if cli.exists() else "hf"
    output.mkdir(parents=True, exist_ok=True)
    events = output / "full_download_events.jsonl"
    processed = 0
    failures = []
    for offset in range(0, len(paths), chunk_size):
        if limit_batches is not None and processed >= limit_batches:
            break
        batch = paths[offset:offset + chunk_size]
        missing = [relative for relative in batch if not _valid_parquet(output / relative)]
        if not missing:
            event = {"batch_start": offset, "files": len(batch), "status": "cached"}
        else:
            started = time.monotonic()
            error = None
            for attempt in range(3):
                result = subprocess.run([
                    command, "download", REPO, *missing,
                    "--type", "dataset", "--revision", REVISION,
                    "--local-dir", str(output), "--max-workers", str(workers),
                    "--format", "quiet",
                ], capture_output=True, text=True, env=effective_environment())
                if result.returncode == 0:
                    break
                error = result.stderr[-2000:]
                time.sleep(10 * (attempt + 1))
            broken = [relative for relative in batch if not _valid_parquet(output / relative)]
            event = {
                "batch_start": offset, "files": len(batch), "new_files": len(missing),
                "status": "ok" if not broken else "error",
                "seconds": round(time.monotonic() - started, 2),
                "broken_files": broken, "last_error": error if broken else None,
            }
            failures.extend(broken)
        event["time_utc"] = datetime.now(timezone.utc).isoformat()
        with events.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        print(json.dumps(event, ensure_ascii=False), flush=True)
        processed += 1
        if failures:
            break
    available = sum(_valid_parquet(output / path) for path in paths)
    summary = {
        "repository": REPO, "revision": REVISION,
        "stock_files_in_inventory": len(paths),
        "available_stock_files": available,
        "failed_files": failures,
        "complete": available == len(paths),
    }
    (output / "full_download_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if failures:
        raise RuntimeError(f"Unable to download {len(failures)} files in current batch")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=Path("data/hf/archive_inventory.json"))
    parser.add_argument("--output", type=Path, default=Path("data/hf/pilot"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--limit-batches", type=int)
    args = parser.parse_args()
    print(json.dumps(download(args.inventory, args.output, args.workers,
                              args.chunk_size, args.limit_batches), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
