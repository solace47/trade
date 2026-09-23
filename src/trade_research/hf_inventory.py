"""Freeze the complete pinned one-minute archive file list and Bao coverage."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from huggingface_hub import HfApi

from .hf_download import REPO, REVISION, selected_paths
from .network import effective_environment


def inventory(output: Path, bao_symbols: Path) -> dict:
    os.environ.update(effective_environment())
    files = HfApi().list_repo_files(REPO, repo_type="dataset", revision=REVISION)
    stocks = sorted(path for path in files
                    if path.startswith("data/stock_1m/") and path.endswith(".parquet"))
    selected = set(selected_paths(bao_symbols))
    archived = set(stocks)
    result = {
        "repository": REPO, "revision": REVISION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stock_files": stocks,
        "counts_by_exchange": {
            exchange: sum(path.startswith(f"data/stock_1m/{exchange}/") for path in stocks)
            for exchange in ("BJ", "SH", "SZ")
        },
        "bao_symbols": len(selected),
        "bao_missing_from_archive": sorted(selected - archived),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {key: value for key, value in result.items() if key != "stock_files"} | {
        "stock_files": len(stocks)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/hf/archive_inventory.json"))
    parser.add_argument("--bao-symbols", type=Path, default=Path(
        "data/baostock/market_2020_2026/metadata/symbols.parquet"
    ))
    args = parser.parse_args()
    print(json.dumps(inventory(args.output, args.bao_symbols), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
