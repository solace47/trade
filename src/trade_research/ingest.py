"""Download a reproducible, resumable BaoStock five-minute pilot dataset.

The pilot uses a historical as-of universe. It is a data pipeline acceptance
sample, not a representative universe for evaluating a trading strategy.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import baostock as bs
import pandas as pd


MINUTE_FIELDS = "date,time,code,open,high,low,close,volume,amount,adjustflag"
DAILY_FIELDS = (
    "date,code,open,high,low,close,preclose,volume,amount,adjustflag,"
    "turn,tradestatus,pctChg,isST"
)
PRICE_COLUMNS = ("open", "high", "low", "close")
BOARD_QUOTAS = {"sh.60": 35, "sz.00": 35, "sz.30": 20, "sh.68": 10}


@dataclass(frozen=True)
class Config:
    root: str = "data/baostock/pilot_2025_2026"
    asof_date: str = "2025-09-01"
    minute_start: str = "2025-09-01"
    minute_end: str = "2026-08-31"
    daily_start: str = "2025-03-01"
    daily_end: str = "2026-09-15"
    seed: str = "trade-pilot-20260923"
    workers: int = 4


def _login() -> None:
    result = bs.login()
    if result.error_code != "0":
        raise RuntimeError(f"BaoStock login: {result.error_code} {result.error_msg}")


def _rows(result) -> pd.DataFrame:
    if result.error_code != "0":
        raise RuntimeError(f"BaoStock query: {result.error_code} {result.error_msg}")
    rows = []
    while result.next():
        rows.append(result.get_row_data())
    if result.error_code != "0":
        raise RuntimeError(f"BaoStock paging: {result.error_code} {result.error_msg}")
    return pd.DataFrame(rows, columns=result.fields)


def _atomic_parquet(frame: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.{os.getpid()}.tmp.parquet")
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _normalize_kline(frame: pd.DataFrame, minute: bool) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("BaoStock returned no bars for an as-of-listed stock")
    frame = frame.copy()
    numeric = [*PRICE_COLUMNS, "volume", "amount"]
    if not minute:
        numeric.extend(("preclose", "turn", "pctChg"))
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["volume"] = frame["volume"].astype("Int64")
    frame["adjustflag"] = pd.to_numeric(frame["adjustflag"], errors="coerce").astype("Int8")
    if minute:
        frame["bar_end"] = pd.to_datetime(
            frame["time"].str[:14], format="%Y%m%d%H%M%S", errors="raise"
        ).dt.tz_localize("Asia/Shanghai")
        if frame.duplicated(["code", "time"]).any():
            raise ValueError("Duplicate minute bar keys")
        if (frame["time"].str[:8] != frame["date"].str.replace("-", "", regex=False)).any():
            raise ValueError("Minute date and time disagree")
        frame = frame.sort_values(["code", "time"])
    else:
        for column in ("tradestatus", "isST"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int8")
        if frame.duplicated(["code", "date"]).any():
            raise ValueError("Duplicate daily bar keys")
        frame = frame.sort_values(["code", "date"])
    if (frame["adjustflag"] != 3).any():
        raise ValueError("Unexpected adjusted bar")
    active = pd.Series(True, index=frame.index) if minute else frame["tradestatus"].eq(1).fillna(False)
    if frame.loc[active, [*PRICE_COLUMNS, "volume", "amount"]].isna().any().any():
        raise ValueError("Missing critical OHLCV values")
    zero_placeholder = frame[[*PRICE_COLUMNS, "volume", "amount"]].eq(0).all(axis=1)
    if minute:
        # BaoStock emits 48 all-zero records on some suspended dates. Keep
        # them as source evidence; the audit checks them against daily status.
        frame["zero_placeholder"] = zero_placeholder
    invalid = ((frame["high"] < frame[["open", "close", "low"]].max(axis=1)) |
        (frame["low"] > frame[["open", "close", "high"]].min(axis=1)) |
        (frame["low"] <= 0) | (frame["volume"] < 0) | (frame["amount"] < 0))
    if (invalid.loc[active].fillna(True) & ~zero_placeholder.loc[active]).any():
        raise ValueError("Invalid OHLCV values")
    return frame.reset_index(drop=True)


def _select_pilot(asof: pd.DataFrame, seed: str) -> pd.DataFrame:
    selected = []
    for board, count in BOARD_QUOTAS.items():
        cohort = asof.loc[asof["code"].str.startswith(board)].copy()
        cohort["selection_key"] = cohort["code"].map(
            lambda code: hashlib.sha256(f"{seed}:{code}".encode()).hexdigest()
        )
        if len(cohort) < count:
            raise ValueError(f"Not enough {board} stocks: {len(cohort)} < {count}")
        selected.append(cohort.sort_values("selection_key").head(count))
    return pd.concat(selected, ignore_index=True).drop(columns="selection_key")


def prepare(cfg: Config) -> list[str]:
    root = Path(cfg.root)
    root.mkdir(parents=True, exist_ok=True)
    frozen = root / "metadata" / "selection.json"
    if frozen.exists():
        recorded = json.loads(frozen.read_text(encoding="utf-8"))
        for key in ("asof_date", "minute_start", "minute_end", "daily_start", "daily_end", "seed"):
            if recorded[key] != getattr(cfg, key):
                raise ValueError(f"Existing pilot selection uses a different {key}; choose another root")
        return pd.read_parquet(root / "metadata" / "pilot_symbols.parquet")["code"].tolist()
    _login()
    try:
        basic = _rows(bs.query_stock_basic())
        asof = _rows(bs.query_all_stock(day=cfg.asof_date))
        calendar = _rows(bs.query_trade_dates(
            start_date=cfg.daily_start, end_date=cfg.daily_end
        ))
    finally:
        bs.logout()
    stocks = asof.merge(basic, on="code", how="inner", suffixes=("_asof", ""))
    stocks = stocks.loc[stocks["type"] == "1"].copy()
    pilot = _select_pilot(stocks, cfg.seed)
    _atomic_parquet(basic, root / "metadata" / "stock_basic.parquet")
    _atomic_parquet(asof, root / "metadata" / "all_stock_asof.parquet")
    _atomic_parquet(stocks, root / "metadata" / "a_share_asof.parquet")
    _atomic_parquet(pilot, root / "metadata" / "pilot_symbols.parquet")
    _atomic_parquet(calendar, root / "metadata" / "calendar.parquet")
    manifest = {
        **asdict(cfg),
        "provider": "BaoStock",
        "package_version": "0.9.4",
        "frequency": "5",
        "adjustflag": "3",
        "universe_count": len(stocks),
        "pilot_count": len(pilot),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "board_quotas": BOARD_QUOTAS,
        "scope_note": "Historical Shanghai/Shenzhen A-share sample; excludes Beijing exchange.",
    }
    (root / "metadata" / "selection.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return pilot["code"].tolist()


def _worker_init() -> None:
    _login()


def _query_with_retry(code: str, fields: str, start: str, end: str, frequency: str) -> pd.DataFrame:
    for attempt in range(3):
        try:
            result = bs.query_history_k_data_plus(
                code, fields, start_date=start, end_date=end,
                frequency=frequency, adjustflag="3"
            )
            return _rows(result)
        except Exception:
            if attempt == 2:
                raise
            try:
                bs.logout()
            except Exception:
                pass
            time.sleep(1 + attempt * 2)
            _login()
    raise AssertionError("unreachable")


def _download_one(code: str, cfg: Config) -> dict:
    root = Path(cfg.root)
    stem = code.replace(".", "_")
    minute_file = root / "minute_5m" / f"{stem}.parquet"
    daily_file = root / "daily" / f"{stem}.parquet"
    started = time.monotonic()
    if not minute_file.exists():
        minute = _normalize_kline(_query_with_retry(
            code, MINUTE_FIELDS, cfg.minute_start, cfg.minute_end, "5"
        ), minute=True)
        _atomic_parquet(minute, minute_file)
    else:
        minute = pd.read_parquet(minute_file, columns=["date", "time"])
    if not daily_file.exists():
        daily = _normalize_kline(_query_with_retry(
            code, DAILY_FIELDS, cfg.daily_start, cfg.daily_end, "d"
        ), minute=False)
        _atomic_parquet(daily, daily_file)
    else:
        daily = pd.read_parquet(daily_file, columns=["date"])
    return {
        "code": code,
        "minute_rows": len(minute),
        "minute_first": minute["time"].iloc[0],
        "minute_last": minute["time"].iloc[-1],
        "minute_days": int(minute["date"].nunique()),
        "daily_rows": len(daily),
        "seconds": round(time.monotonic() - started, 2),
        "status": "ok",
    }


def download(cfg: Config, codes: list[str]) -> None:
    root = Path(cfg.root)
    failures = []
    with ProcessPoolExecutor(max_workers=cfg.workers, initializer=_worker_init) as pool:
        futures = {pool.submit(_download_one, code, cfg): code for code in codes}
        for future in as_completed(futures):
            code = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"code": code, "status": "error", "error": repr(exc)}
                failures.append(code)
            with (root / "metadata" / "download_events.jsonl").open("a", encoding="utf-8") as out:
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(json.dumps(result, ensure_ascii=False), flush=True)
    if failures:
        raise RuntimeError(f"Failed to download {len(failures)} symbols: {', '.join(failures)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=Config.root)
    parser.add_argument("--workers", type=int, default=Config.workers)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    cfg = Config(root=args.root, workers=args.workers)
    codes = prepare(cfg)
    print(f"Prepared {len(codes)} historically listed symbols", flush=True)
    if not args.prepare_only:
        download(cfg, codes)


if __name__ == "__main__":
    main()
