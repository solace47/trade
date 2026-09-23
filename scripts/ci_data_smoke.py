"""Check that the hosted runner can reach both research data sources."""

from __future__ import annotations

from pathlib import Path
import time

import baostock as bs
from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq


REPO = "neigezhu/china-a-share-1min-ohlcv"
FILE = "data/stock_1m/SH/600055.parquet"


def main() -> None:
    started = time.monotonic()
    revision = HfApi().repo_info(REPO, repo_type="dataset").sha
    if not revision:
        raise RuntimeError("Unable to resolve minute archive version")
    filename = hf_hub_download(
        repo_id=REPO, filename=FILE, repo_type="dataset", revision=revision,
        local_dir="data/smoke",
    )
    path = Path(filename)
    rows = pq.ParquetFile(path).metadata.num_rows
    print("minute_file_bytes", path.stat().st_size, "rows", rows,
          "seconds", round(time.monotonic() - started, 2), flush=True)
    if rows < 200:
        raise RuntimeError("Minute file is unexpectedly small")

    logged = bs.login()
    if logged.error_code != "0":
        raise RuntimeError(f"Daily source login failed: {logged.error_code}")
    try:
        result = bs.query_history_k_data_plus(
            "sh.600055", "date,close,preclose,tradestatus,isST",
            start_date="2025-09-01", end_date="2025-09-30", frequency="d", adjustflag="3",
        )
        if result.error_code != "0":
            raise RuntimeError(f"Daily query failed: {result.error_code}")
        daily_rows = 0
        while result.next():
            result.get_row_data()
            daily_rows += 1
    finally:
        bs.logout()
    print("daily_rows", daily_rows, flush=True)
    if daily_rows < 10:
        raise RuntimeError("Daily source returned too few rows")


if __name__ == "__main__":
    main()
