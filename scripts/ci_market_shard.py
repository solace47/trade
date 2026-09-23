"""Run one bounded historical market shard on a standard hosted runner."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout
import json
from pathlib import Path
import tarfile
import time

from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq

from trade_research.hf_download import REPO, selected_paths
from trade_research.hf_market_audit import audit as audit_minute
from trade_research.hf_market_outcomes import build as build_outcomes
from trade_research.hf_market_snapshot import build as build_snapshots
from trade_research.ingest import _atomic_parquet
from trade_research.market_daily_audit import audit as audit_daily
from trade_research.market_daily_ingest import Config, download as download_daily, prepare


def run(shard: int, shards: int, limit_symbols: int) -> dict:
    if not 0 <= shard < shards or limit_symbols < 0:
        raise ValueError("Invalid shard configuration")
    started = time.monotonic()
    cfg = Config()
    bao_root = Path(cfg.root)
    all_symbols = prepare(cfg)
    symbols = all_symbols.iloc[shard::shards].copy()
    if limit_symbols:
        symbols = symbols.head(limit_symbols)
    symbols_file = bao_root / "metadata" / "symbols.parquet"
    _atomic_parquet(symbols, symbols_file)

    with (bao_root / "metadata" / "daily_download.log").open("w") as log:
        with redirect_stdout(log):
            download_daily(cfg, symbols)
    daily = audit_daily(bao_root)
    print("daily_complete", len(symbols), daily["accepted"],
          "seconds", round(time.monotonic() - started, 1), flush=True)
    if not daily["accepted"]:
        raise RuntimeError("Daily history failed its completeness audit")

    revision = HfApi().repo_info(REPO, repo_type="dataset").sha
    if not revision:
        raise RuntimeError("Unable to resolve minute archive version")
    hf_root = Path("data/hf/ci")
    paths = selected_paths(symbols_file)
    def download_one(relative: str) -> tuple[str, int]:
        local = Path(hf_hub_download(
            repo_id=REPO, filename=relative, repo_type="dataset",
            revision=revision, local_dir=hf_root,
        ))
        if pq.ParquetFile(local).metadata.num_rows == 0:
            raise ValueError(f"Empty minute file: {relative}")
        return relative, local.stat().st_size
    downloaded_bytes = 0
    with ThreadPoolExecutor(max_workers=12) as pool:
        for future in as_completed(pool.submit(download_one, path) for path in paths):
            _, size = future.result()
            downloaded_bytes += size
    print("minute_download_complete", len(paths), downloaded_bytes,
          "seconds", round(time.monotonic() - started, 1), flush=True)

    minute = audit_minute(hf_root, bao_root, Path("data/research/market_audit"))
    snapshots = build_snapshots(
        hf_root, bao_root, Path("data/research/market_snapshots")
    )
    outcomes = build_outcomes(
        hf_root, bao_root, Path("data/research/market_snapshots"),
        Path("data/research/market_outcomes"),
    )
    result = {
        "shard": shard, "shards": shards, "selected_symbols": len(symbols),
        "daily_rows": daily["daily_rows"],
        "minute_files": len(paths), "minute_bytes": downloaded_bytes,
        "minute_complete_days": minute["complete_minute_days"],
        "minute_active_days": minute["active_daily_days"],
        "minute_nonopen_mismatch_days": minute["unexplained_ohlc_mismatch_days"],
        "snapshots": snapshots["snapshot_rows"],
        "outcomes": outcomes["outcome_rows"],
        "seconds": round(time.monotonic() - started, 1),
    }
    provenance = Path(f"data/research/shard_{shard}_source.lock")
    provenance.write_text(revision + "\n", encoding="utf-8")
    report = Path(f"data/research/shard_{shard}_result.json")
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    archive = Path(f"data/research/shard_{shard}.tar")
    files = [
        bao_root / "metadata" / name for name in (
            "selection.json", "symbols.parquet", "calendar.parquet", "audit_summary.json"
        )
    ]
    files.extend(sorted((bao_root / "daily").glob("*.parquet")))
    files.extend(sorted(Path("data/research/market_audit").glob("*")))
    files.extend(sorted(Path("data/research/market_snapshots").glob("*.parquet")))
    files.extend(sorted(Path("data/research/market_outcomes").glob("*.parquet")))
    files.extend((provenance, report))
    with tarfile.open(archive, "w") as stream:
        for path in files:
            stream.add(path, arcname=path.as_posix())
    print("archive_bytes", archive.stat().st_size, flush=True)
    print("CI_SHARD_SUMMARY=" + json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--limit-symbols", type=int, default=0)
    args = parser.parse_args()
    run(args.shard, args.shards, args.limit_symbols)


if __name__ == "__main__":
    main()
