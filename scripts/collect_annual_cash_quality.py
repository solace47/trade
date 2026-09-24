"""Checkpoint BaoStock annual profit/cash-flow inputs for 2024-25 research.

Only 2023 and 2024 annual reports are requested. Their publication dates
must be checked again at signal construction; no future returns are accessed.
Run separate processes with --shard 0..N-1 to collect disjoint stocks.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import baostock as bs
import duckdb


YEARS = (2023, 2024)


def universe_codes(path: Path) -> list[str]:
    connection = duckdb.connect()
    rows = connection.execute(f"""
        SELECT DISTINCT code FROM read_parquet('{path}') ORDER BY code
    """).fetchall()
    return [code for (code,) in rows]


def one_result(api, code: str, year: int) -> dict | None:
    result = api(code=code, year=year, quarter=4)
    if result.error_code != "0":
        raise RuntimeError(f"{api.__name__} {code} {year}: "
                           f"{result.error_code} {result.error_msg}")
    rows = []
    while result.next():
        rows.append(dict(zip(result.fields, result.get_row_data())))
    if len(rows) > 1:
        raise ValueError(f"Multiple annual report rows: {api.__name__} {code} {year}")
    if rows and (rows[0].get("code") != code
                 or rows[0].get("statDate") != f"{year}-12-31"):
        raise ValueError(f"Unexpected annual report key: {api.__name__} {code} {year}")
    return rows[0] if rows else None


def login_with_retry(attempts: int = 3) -> None:
    for attempt in range(attempts):
        try:
            login = bs.login()
            if login.error_code == "0":
                return
            error = login.error_msg
        except OSError as exc:
            error = str(exc)
        if attempt + 1 < attempts:
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"BaoStock login failed after {attempts} attempts: {error}")


def fetch_pair(code: str, year: int) -> tuple[dict | None, dict | None]:
    for attempt in range(3):
        try:
            profit = one_result(bs.query_profit_data, code, year)
            cash = one_result(bs.query_cash_flow_data, code, year)
            return profit, cash
        except (RuntimeError, OSError):
            if attempt == 2:
                raise
            try:
                bs.logout()
            except OSError:
                pass
            time.sleep(10 * (attempt + 1))
            login_with_retry()
    raise AssertionError("Unreachable financial query retry state")


def collect(universe: Path, output_dir: Path, shard: int, shards: int,
            max_codes: int | None = None) -> dict:
    if shards < 1 or not 0 <= shard < shards:
        raise ValueError("Invalid shard index/count")
    codes = universe_codes(universe)[shard::shards]
    if max_codes is not None:
        codes = codes[:max_codes]
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"annual_cash_{shard:02d}_of_{shards:02d}.jsonl"
    completed = set()
    if output.exists():
        with output.open(encoding="utf-8") as source:
            for line in source:
                item = json.loads(line)
                key = (item["code"], item["year"])
                if key in completed:
                    raise ValueError(f"Duplicate financial checkpoint: {key}")
                completed.add(key)
    login_with_retry()
    fetched = 0
    try:
        with output.open("a", encoding="utf-8") as destination:
            for code in codes:
                for year in YEARS:
                    if (code, year) in completed:
                        continue
                    profit, cash = fetch_pair(code, year)
                    item = {
                        "code": code, "year": year,
                        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                        "profit": profit, "cash": cash,
                    }
                    destination.write(json.dumps(item, ensure_ascii=False) + "\n")
                    destination.flush()
                    fetched += 1
                if fetched and fetched % 100 == 0:
                    print({"shard": shard, "fetched": fetched,
                           "total_codes": len(codes)}, flush=True)
    finally:
        try:
            bs.logout()
        except OSError:
            pass
    return {"shard": shard, "codes": len(codes),
            "already_completed": len(completed), "fetched": fetched,
            "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path,
                        default=Path("data/research/margin_universe_quintiles.parquet"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/research/annual_cash_quality"))
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--max-codes", type=int)
    args = parser.parse_args()
    print(collect(args.universe, args.output_dir, args.shard,
                  args.shards, args.max_codes), flush=True)


if __name__ == "__main__":
    main()
