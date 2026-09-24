"""Download historical A-share earnings notices in resumable stock batches.

BaoStock exposes the publication date of each earnings forecast and express
report. Raw events are kept outside Git so the later strategy can insist on
a publication date strictly before its 14:50 signal day. No outcome data are
read while acquiring or auditing announcements.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import baostock as bs
import pandas as pd


FIRST_PUBLICATION = "2023-12-01"
LAST_PUBLICATION = "2025-12-31"
BATCH_SIZE = 100
FIELDS = {
    "forecast": ("code", "profitForcastExpPubDate", "profitForcastExpStatDate",
                 "profitForcastType", "profitForcastAbstract",
                 "profitForcastChgPctUp", "profitForcastChgPctDwn"),
    "express": ("code", "performanceExpPubDate", "performanceExpStatDate",
                "performanceExpUpdateDate", "performanceExpressTotalAsset",
                "performanceExpressNetAsset", "performanceExpressEPSChgPct",
                "performanceExpressROEWa", "performanceExpressEPSDiluted",
                "performanceExpressGRYOY", "performanceExpressOPYOY"),
}
PUBLICATION_FIELD = {
    "forecast": "profitForcastExpPubDate",
    "express": "performanceExpPubDate",
}
AVAILABLE_FIELD = {
    "forecast": "profitForcastExpPubDate",
    "express": "performanceExpUpdateDate",
}


def stock_codes(stock_basic: Path) -> list[str]:
    basic = pd.read_parquet(stock_basic)
    selected = basic.loc[basic.type.astype(str).eq("1"), "code"]
    codes = sorted(selected.tolist())
    if not codes or len(codes) != len(set(codes)):
        raise ValueError("Missing or duplicated stock codes")
    if any(not code.startswith(("sh.", "sz.")) for code in codes):
        raise ValueError("Only Shanghai and Shenzhen stock codes are supported")
    return codes


def _validate(frame: pd.DataFrame, kind: str, requested: set[str]) -> pd.DataFrame:
    if tuple(frame.columns) != FIELDS[kind]:
        raise ValueError(f"Unexpected {kind} schema")
    if not set(frame.code).issubset(requested):
        raise ValueError(f"Unexpected {kind} stock code")
    published = frame[PUBLICATION_FIELD[kind]]
    if published.isna().any() or published.eq("").any():
        raise ValueError(f"Missing {kind} publication date")
    if kind == "express":
        updated = frame.performanceExpUpdateDate
        if updated.isna().any() or updated.eq("").any():
            raise ValueError("Missing express update date")
        if updated.lt(published).any():
            raise ValueError("Express update predates publication")
    available = frame[AVAILABLE_FIELD[kind]]
    if not available.between(FIRST_PUBLICATION, LAST_PUBLICATION).all():
        raise ValueError(f"{kind} availability outside requested window")
    return frame


def _select_available(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Filter server results by the first date this version could be seen."""
    available = frame[AVAILABLE_FIELD[kind]]
    return frame.loc[available.between(FIRST_PUBLICATION, LAST_PUBLICATION)].reset_index(drop=True)


def _query(code: str, kind: str) -> list[list[str]]:
    call = (bs.query_forecast_report if kind == "forecast"
            else bs.query_performance_express_report)
    response = call(code, start_date=FIRST_PUBLICATION,
                    end_date=LAST_PUBLICATION)
    if response.error_code != "0" or tuple(response.fields) != FIELDS[kind]:
        raise RuntimeError(f"BaoStock {kind} query failed for {code}: "
                           f"{response.error_code} {response.error_msg}")
    rows = []
    while response.next():
        rows.append(response.get_row_data())
    if response.error_code != "0":
        raise RuntimeError(f"BaoStock {kind} pagination failed for {code}: "
                           f"{response.error_code} {response.error_msg}")
    return rows


def download(stock_basic: Path, output_dir: Path, start_batch: int,
             max_batches: int) -> dict:
    if start_batch < 0 or max_batches < 1:
        raise ValueError("Batch offset and count must be positive")
    codes = stock_codes(stock_basic)
    batches = [codes[i:i + BATCH_SIZE] for i in range(0, len(codes), BATCH_SIZE)]
    if start_batch >= len(batches):
        raise ValueError("Batch offset exceeds all stock codes")
    output_dir.mkdir(parents=True, exist_ok=True)
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"BaoStock login failed: {login.error_msg}")
    last_login = time.monotonic()
    finished = []
    try:
        for index in range(start_batch,
                           min(start_batch + max_batches, len(batches))):
            if time.monotonic() - last_login >= 480:
                bs.logout()
                login = bs.login()
                if login.error_code != "0":
                    raise RuntimeError(f"BaoStock re-login failed: {login.error_msg}")
                last_login = time.monotonic()
            batch = batches[index]
            paths = {kind: output_dir / f"{kind}_{index:03d}.parquet"
                     for kind in FIELDS}
            manifest = output_dir / f"batch_{index:03d}.json"
            if manifest.exists() and all(path.exists() for path in paths.values()):
                saved = json.loads(manifest.read_text(encoding="utf-8"))
                if saved.get("codes") != batch:
                    raise ValueError(f"Batch {index} manifest differs from stock list")
                if (saved.get("first_publication") != FIRST_PUBLICATION or
                        saved.get("last_publication") != LAST_PUBLICATION):
                    raise ValueError(f"Batch {index} manifest differs from date window")
                for kind, path in paths.items():
                    frame = _validate(pd.read_parquet(path), kind, set(batch))
                    if saved.get("rows", {}).get(kind) != len(frame):
                        raise ValueError(f"Batch {index} {kind} row count differs")
                finished.append(index)
                print(f"Batch {index}/{len(batches)} cached", flush=True)
                continue
            rows = {kind: [] for kind in FIELDS}
            for count, code in enumerate(batch, start=1):
                for kind in FIELDS:
                    rows[kind].extend(_query(code, kind))
                if count % 25 == 0:
                    print(f"Batch {index}: {count}/{len(batch)} stocks", flush=True)
            counts = {}
            queried_counts = {}
            for kind, path in paths.items():
                queried_counts[kind] = len(rows[kind])
                frame = _select_available(
                    pd.DataFrame(rows[kind], columns=FIELDS[kind]), kind)
                frame = _validate(frame, kind, set(batch))
                temporary = path.with_suffix(".tmp.parquet")
                frame.to_parquet(temporary, index=False, compression="zstd")
                temporary.replace(path)
                counts[kind] = len(frame)
            manifest_body = json.dumps({
                "batch": index, "codes": batch,
                "first_publication": FIRST_PUBLICATION,
                "last_publication": LAST_PUBLICATION,
                "rows": counts,
                "queried_rows": queried_counts,
            }, ensure_ascii=False) + "\n"
            temporary_manifest = manifest.with_suffix(".tmp.json")
            temporary_manifest.write_text(manifest_body, encoding="utf-8")
            temporary_manifest.replace(manifest)
            finished.append(index)
            print(f"Batch {index}/{len(batches)} complete: {counts}", flush=True)
    finally:
        bs.logout()
    return {"completed_batches": finished, "total_batches": len(batches),
            "output_dir": str(output_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock-basic", type=Path, default=Path(
        "data/baostock/market_2020_2026/metadata/stock_basic.parquet"
    ))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/research/earnings_events"))
    parser.add_argument("--start-batch", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=1)
    args = parser.parse_args()
    print(download(args.stock_basic, args.output_dir,
                   args.start_batch, args.max_batches))


if __name__ == "__main__":
    main()
