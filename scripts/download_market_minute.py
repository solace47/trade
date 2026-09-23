"""Resume the pinned public minute archive for the historical SH/SZ universe."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time

from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq

from trade_research.hf_download import REPO, locked_revision, selected_paths
from trade_research.network import effective_environment


def download(selection: Path, output: Path, workers: int) -> dict:
    if workers < 1:
        raise ValueError("workers must be positive")
    os.environ.update(effective_environment())
    revision = locked_revision()
    available = set(HfApi().list_repo_files(REPO, repo_type="dataset", revision=revision))
    requested = selected_paths(selection)
    paths = [path for path in requested if path in available]
    missing = [path for path in requested if path not in available]

    def one(relative: str) -> int:
        for attempt in range(3):
            try:
                path = Path(hf_hub_download(
                    repo_id=REPO, filename=relative, repo_type="dataset",
                    revision=revision, local_dir=output,
                ))
                if pq.ParquetFile(path).metadata.num_rows <= 0:
                    raise ValueError(f"Empty minute file: {relative}")
                return path.stat().st_size
            except (OSError, TimeoutError):
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        raise AssertionError("unreachable")

    total_bytes = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, relative): relative for relative in paths}
        for count, future in enumerate(as_completed(futures), start=1):
            total_bytes += future.result()
            if count % 100 == 0 or count == len(paths):
                print(f"verified {count}/{len(paths)} files", flush=True)
    result = {
        "selected_symbols": len(requested),
        "verified_minute_files": len(paths),
        "missing_source_paths": missing,
        "verified_bytes": total_bytes,
        "revision_lock": "data/hf/revision.lock",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "market_download_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=Path(
        "data/baostock/market_2020_2026/metadata/symbols.parquet"
    ))
    parser.add_argument("--output", type=Path, default=Path("data/hf/pilot"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(download(args.selection, args.output, args.workers),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
