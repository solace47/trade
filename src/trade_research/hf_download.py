"""Download the same pilot stocks from a pinned public one-minute archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
from huggingface_hub import HfApi

from .network import effective_environment


REPO = "neigezhu/china-a-share-1min-ohlcv"
REVISION_LOCK = Path("data/hf/revision.lock")


def locked_revision() -> str:
    """Use one archive revision for all downloads in this local workspace."""
    override = os.environ.get("TRADE_HF_REVISION", "").strip()
    if override:
        return override
    if REVISION_LOCK.exists():
        revision = REVISION_LOCK.read_text(encoding="utf-8").strip()
        if revision:
            return revision
        raise ValueError(f"Empty archive revision lock: {REVISION_LOCK}")
    os.environ.update(effective_environment())
    revision = HfApi().repo_info(REPO, repo_type="dataset").sha
    if not revision:
        raise RuntimeError("Unable to resolve an archive revision")
    REVISION_LOCK.parent.mkdir(parents=True, exist_ok=True)
    REVISION_LOCK.write_text(revision + "\n", encoding="utf-8")
    return revision


def selected_paths(selection_file: Path) -> list[str]:
    pilot = pd.read_parquet(selection_file)
    paths = []
    for code in pilot["code"]:
        exchange, symbol = code.split(".")
        paths.append(f"data/stock_1m/{exchange.upper()}/{symbol}.parquet")
    return paths


def download(selection_file: Path, output: Path, dry_run: bool, workers: int) -> None:
    revision = locked_revision()
    files = selected_paths(selection_file)
    files.extend(("metadata/source_provenance.json", "metadata/summary.json"))
    local_cli = Path(sys.executable).with_name("hf")
    command = [
        str(local_cli) if local_cli.exists() else "hf", "download", REPO, *files,
        "--type", "dataset", "--revision", revision,
        "--local-dir", str(output), "--max-workers", str(workers),
        "--format", "json",
    ]
    if dry_run:
        command.append("--dry-run")
    result = subprocess.run(command, check=True, capture_output=True, text=True,
                            env=effective_environment())
    if dry_run:
        planned = json.loads(result.stdout)
        print(json.dumps({"planned_files": len(planned), "examples": planned[:3]}, ensure_ascii=False))
        return
    print(result.stdout)
    checksums = {}
    for file in files:
        path = output / file
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        checksums[file] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    manifest = {"repository": REPO, "revision": revision, "files": checksums}
    (output / "download_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Verified {len(files)} files", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=Path(
        "data/baostock/pilot_2025_2026/metadata/pilot_symbols.parquet"
    ))
    parser.add_argument("--output", type=Path, default=Path("data/hf/pilot"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    download(args.selection, args.output, args.dry_run, args.workers)


if __name__ == "__main__":
    main()
