"""Resumable Shanghai/Shenzhen A-share daily history, including delisted stocks."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import baostock as bs
import pandas as pd

from .ingest import DAILY_FIELDS, _atomic_parquet, _login, _normalize_kline, _query_with_retry, _rows


@dataclass(frozen=True)
class Config:
    root: str = "data/baostock/market_2020_2026"
    first_date: str = "2019-01-01"  # One year of feature warmup.
    research_start: str = "2020-01-01"
    last_date: str = "2026-08-06"
    workers: int = 4


def prepare(cfg: Config) -> pd.DataFrame:
    root = Path(cfg.root)
    metadata = root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    selection_file = metadata / "selection.json"
    if selection_file.exists():
        saved = json.loads(selection_file.read_text(encoding="utf-8"))
        for key in ("first_date", "research_start", "last_date"):
            if saved[key] != getattr(cfg, key):
                raise ValueError(f"Existing market dataset uses different {key}")
        return pd.read_parquet(metadata / "symbols.parquet")
    _login()
    try:
        basic = _rows(bs.query_stock_basic())
        calendar = _rows(bs.query_trade_dates(
            start_date=cfg.first_date, end_date=cfg.last_date
        ))
    finally:
        bs.logout()
    symbols = basic.loc[
        basic["type"].eq("1")
        & basic["code"].str.startswith(("sh.", "sz."))
        & basic["ipoDate"].le(cfg.last_date)
        & (basic["outDate"].eq("") | basic["outDate"].ge(cfg.research_start))
    ].copy().sort_values("code")
    _atomic_parquet(basic, metadata / "stock_basic.parquet")
    _atomic_parquet(calendar, metadata / "calendar.parquet")
    _atomic_parquet(symbols, metadata / "symbols.parquet")
    selection_file.write_text(json.dumps({
        **asdict(cfg), "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "BaoStock", "package_version": "0.9.4",
        "selected_symbols": len(symbols),
        "universe_note": "Shanghai/Shenzhen stocks, historical IPO/delisting dates; Beijing excluded.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return symbols


def _worker_init() -> None:
    for attempt in range(10):
        try:
            _login()
            return
        except Exception:
            if attempt == 9:
                raise
            time.sleep(2 + attempt * 2)


def _download_one(code: str, first: str, last: str, root: str) -> dict:
    path = Path(root) / "daily" / f"{code.replace('.', '_')}.parquet"
    if path.exists():
        return {"code": code, "status": "cached", "rows": pd.read_parquet(path, columns=["date"]).shape[0]}
    started = time.monotonic()
    frame = _query_with_retry(code, DAILY_FIELDS, first, last, "d")
    if frame.empty:
        return {"code": code, "status": "no_bars", "rows": 0}
    frame = _normalize_kline(frame, minute=False)
    _atomic_parquet(frame, path)
    return {
        "code": code, "status": "ok", "rows": len(frame),
        "first_date": frame["date"].iloc[0], "last_date": frame["date"].iloc[-1],
        "seconds": round(time.monotonic() - started, 2),
    }


def download(cfg: Config, symbols: pd.DataFrame, limit: int | None = None) -> dict:
    root = Path(cfg.root)
    selected = symbols.head(limit) if limit is not None else symbols
    failures = []
    statuses = {}
    records = list(selected.itertuples(index=False))
    for offset in range(0, len(records), 200):
        pending = records[offset:offset + 200]
        for attempt in range(3):
            failed = []
            with ProcessPoolExecutor(max_workers=cfg.workers, initializer=_worker_init) as pool:
                futures = {}
                for stock in pending:
                    start = max(cfg.first_date, stock.ipoDate)
                    end = min(cfg.last_date, stock.outDate) if stock.outDate else cfg.last_date
                    futures[pool.submit(_download_one, stock.code, start, end, cfg.root)] = stock
                for future in as_completed(futures):
                    stock = futures[future]
                    try:
                        result = future.result()
                    except Exception as error:
                        result = {"code": stock.code, "status": "error", "error": repr(error)}
                        failed.append(stock)
                    statuses[result["status"]] = statuses.get(result["status"], 0) + 1
                    with (root / "metadata" / "download_events.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                    print(json.dumps(result, ensure_ascii=False), flush=True)
            if not failed:
                break
            pending = failed
            time.sleep(3 * (attempt + 1))
        failures.extend(stock.code for stock in pending if not (
            root / "daily" / f"{stock.code.replace('.', '_')}.parquet"
        ).exists())
    summary = {"attempted": len(selected), "statuses": statuses, "failures": failures}
    (root / "metadata" / "download_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if failures:
        raise RuntimeError(f"Daily download failed for {len(failures)} symbols")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=Config.root)
    parser.add_argument("--workers", type=int, default=Config.workers)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    cfg = Config(root=args.root, workers=args.workers)
    symbols = prepare(cfg)
    print(f"Prepared {len(symbols)} historical Shanghai/Shenzhen stocks", flush=True)
    if not args.prepare_only:
        print(json.dumps(download(cfg, symbols, args.limit), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
