"""Audit minute price consistency of old fixed economic accounting events.

The legacy cohort remains rejected. This module adds source-quality flags but
never reprices a holding, changes a trade, or fills an unknown terminal value.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .corporate_cash import MINUTES, save_json, sha
from .fill_accounting import KEY
from .orderbook_funding import BOOK, ROOT as FUNDING, SOURCE, execution_events, verify
from .round_number_entry import LABELS, _connection, validate_window


ROOT = Path("data/research/fixed_execution_quality")
ROW_COLUMNS = list(dict.fromkeys([*KEY, *BOOK, "half", "entry_status", "entry_price", "shares",
    "unknown_after_buy", "sold_shares", "accounting_exit_date", "accounting_exit_price", "accounting_category"]))


def freeze(output: Path = ROOT) -> dict:
    verify(FUNDING)
    audit = json.loads((FUNDING / "quote_audit.json").read_text())
    if sha(FUNDING / "quotes.parquet") != audit["quotes_sha256"]:
        raise ValueError("Old verified quote table changed")
    rows = pd.read_parquet(SOURCE, columns=ROW_COLUMNS)
    events = execution_events(rows)
    events = events.merge(rows[[*KEY, "half"]].rename(columns={"date": "signal_date"}),
                          on=["signal_date", "code", "target_notional", "horizon"], validate="many_to_one")
    windows = events[["date", "code"]].drop_duplicates().sort_values(["date", "code"])
    if (len(rows) != 12648 or len(events) != audit["events"] or len(events) != 22210
            or len(windows) != audit["stock_days"] or len(windows) != 8551
            or int(rows.unknown_after_buy.sum()) != 2
            or not windows.date.between("2024-01-01", "2025-12-31").all()):
        raise ValueError("Old fixed accounting population changed")
    sources = [SOURCE, SOURCE.with_name("report.json"), FUNDING / "quote_audit.json", FUNDING / "quotes.parquet"]
    manifest = {"rule_commit": "4a686b4", "sha256": {str(p): sha(p) for p in sources},
        "original_rows": len(rows), "execution_events": len(events), "execution_windows": len(windows),
        "legacy_history_warmup_inherited": True, "returns_recomputed": False, "holdout_read": False}
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    if path.exists():
        old = json.loads(path.read_text())
        if {k: v for k, v in old.items() if k != "frozen_sha256"} != manifest:
            raise ValueError("Cannot replace old execution audit scope")
        for name, digest in old["frozen_sha256"].items():
            if sha(output / name) != digest:
                raise ValueError("Existing frozen audit keys changed")
    for name, frame in (("original_rows", rows), ("events", events), ("windows", windows)):
        frame.to_parquet(output / f"{name}.parquet", index=False)
    manifest["frozen_sha256"] = {name + ".parquet": sha(output / (name + ".parquet"))
                                 for name in ("original_rows", "events", "windows")}
    save_json(path, manifest)
    return manifest


def window_quality(code: str, date: str, bars: pd.DataFrame) -> dict:
    status, _ = validate_window(bars)
    exact = (len(bars) == 4 and bars.timestamp.dt.strftime("%H%M").sort_values().tolist() == list(LABELS))
    numeric = exact and np.isfinite(bars[["volume", "turnover"]].to_numpy(dtype=float)).all()
    volume = float(bars.volume.to_numpy(dtype=float).sum()) if numeric else None
    amount = float(bars.turnover.to_numpy(dtype=float).sum()) if numeric else None
    raw_vwap = amount / volume if numeric and volume > 0 and amount > 0 else None
    positive = bars.loc[bars.volume.gt(0)]
    low = float(positive.low.min()) if len(positive) else None
    high = float(positive.high.max()) if len(positive) else None
    outside = (raw_vwap < low - .0101 or raw_vwap > high + .0101
               if raw_vwap is not None and low is not None and high is not None else None)
    return {"date": date, "code": code, "window_status": status, "source_checks_passed": status == "valid",
            "raw_vwap": raw_vwap, "volume": volume, "window_low": low, "window_high": high,
            "aggregate_vwap_outside_window_tolerance": outside}


def _raw_one(job) -> tuple[list[dict], pd.DataFrame, dict]:
    code, dates, expected = job
    exchange, symbol = code.split(".")
    path = MINUTES / exchange.upper() / f"{symbol}.parquet"
    digest = sha(path)
    if digest != expected:
        raise ValueError(f"Previously audited source changed: {code}")
    connection = _connection()
    connection.register("execution_dates", pd.DataFrame({"date": dates}))
    raw = connection.execute("""
        SELECT m.timestamp,m.open,m.high,m.low,m.close,m.volume,m.turnover
        FROM read_parquet(?) m JOIN execution_dates k
          ON cast(m.timestamp AS DATE)=cast(k.date AS DATE)
        WHERE m.timestamp >= ? AND m.timestamp < ?
          AND strftime(m.timestamp,'%H%M') IN ('1452','1453','1454','1455')
        ORDER BY timestamp
    """, [str(path), pd.Timestamp(min(dates)), pd.Timestamp(max(dates)) + pd.Timedelta(days=1)]).df()
    raw["date"] = raw.timestamp.dt.strftime("%Y-%m-%d")
    raw["code"] = code
    parts = {date: part for date, part in raw.groupby("date", sort=False)}
    results = [window_quality(code, date, parts.get(date, raw.iloc[:0])) for date in dates]
    return results, raw, {"code": code, "path": str(path), "sha256": digest,
                          "matches_prior_audit": True, "windows": len(dates)}


def attach_quality(rows: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    if rows.duplicated(KEY).any() or windows.duplicated(["date", "code"]).any():
        raise ValueError("Quality attachment requires unique source and trade keys")
    fields = ["date", "code", "source_checks_passed", "window_status"]
    buy = windows[fields].rename(columns={"source_checks_passed": "buy_source_passed", "window_status": "buy_window_status"})
    sell = windows[fields].rename(columns={"date": "accounting_exit_date", "source_checks_passed": "sell_source_passed", "window_status": "sell_window_status"})
    result = rows.merge(buy, on=["date", "code"], how="left", validate="many_to_one")
    result = result.merge(sell, on=["accounting_exit_date", "code"], how="left", validate="many_to_one")
    bought = result.entry_status.eq("filled")
    known = bought & ~result.unknown_after_buy
    if result.loc[bought, "buy_source_passed"].isna().any() or result.loc[known, "sell_source_passed"].isna().any():
        raise ValueError("Booked execution leg lost its raw source status")
    result["execution_source_category"] = np.select([
        ~bought, result.unknown_after_buy,
        known & result.buy_source_passed.eq(True) & result.sell_source_passed.eq(True)],
        ["not_bought", "terminal_value_still_unknown", "both_windows_consistent"],
        default="booked_but_minute_source_inconsistent")
    return result


def evaluate(output: Path = ROOT, workers: int = 4) -> dict:
    manifest = json.loads((output / "manifest.json").read_text())
    for name, digest in manifest["sha256"].items():
        if sha(Path(name)) != digest:
            raise ValueError(f"Fixed accounting source changed: {name}")
    for name, digest in manifest["frozen_sha256"].items():
        if sha(output / name) != digest:
            raise ValueError(f"Frozen execution scope changed: {name}")
    events = pd.read_parquet(output / "events.parquet")
    needed = pd.read_parquet(output / "windows.parquet")
    prior = json.loads((FUNDING / "quote_audit.json").read_text())
    jobs = []
    for code, part in needed.groupby("code", sort=True):
        exchange, symbol = code.split(".")
        path = MINUTES / exchange.upper() / f"{symbol}.parquet"
        jobs.append((code, sorted(part.date.tolist()), prior["raw_sha256"][str(path)]))
    results, bars, sources = [], [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for count, (rows, raw, source) in enumerate(pool.map(_raw_one, jobs), 1):
            results.extend(rows)
            bars.append(raw)
            sources.append(source)
            if count % 150 == 0 or count == len(jobs):
                print(f"Fixed execution sources: {count}/{len(jobs)} stocks", flush=True)
    windows = pd.DataFrame(results).sort_values(["date", "code"])
    raw = pd.concat(bars, ignore_index=True).sort_values(["date", "code", "timestamp"])
    linked = events.merge(windows, on=["date", "code"], how="left", validate="many_to_one")
    reproduced = linked.raw_vwap * np.where(linked.side.eq("buy"), 1.0005, .9995)
    error = (reproduced - linked.price).abs()
    if len(windows) != len(needed) or len(linked) != len(events) or error.isna().any() or error.gt(1e-10).any():
        raise ValueError("Old raw point prices did not reproduce")
    accounted = attach_quality(pd.read_parquet(output / "original_rows.parquet"), windows)
    for name, frame in (("window_quality", windows), ("raw_windows", raw), ("event_quality", linked), ("accounting_quality", accounted)):
        frame.to_parquet(output / f"{name}.parquet", index=False)
    cells = []
    for key, part in linked.groupby(["half", *BOOK, "side"], sort=True):
        cells.append({**dict(zip(["half", *BOOK, "side"], key)), "execution_events": len(part),
            "consistent_events": int(part.source_checks_passed.sum()),
            "consistent_rate": float(part.source_checks_passed.mean()),
            "window_status": part.window_status.value_counts().to_dict()})
    report = {"rule_commit": manifest["rule_commit"], "original_rows": len(accounted),
        "execution_events": len(linked), "unique_windows": len(windows), "source_files": len(sources),
        "maximum_old_point_price_error": float(error.max()),
        "window_status": windows.window_status.value_counts().to_dict(),
        "event_status": linked.window_status.value_counts().to_dict(),
        "aggregate_vwap_outside_window_tolerance": int(windows.aggregate_vwap_outside_window_tolerance.eq(True).sum()),
        "accounting_source_categories": accounted.execution_source_category.value_counts().to_dict(),
        "by_original_half_book_side": cells,
        "all_prior_source_hashes_matched": all(s["matches_prior_audit"] for s in sources),
        "returns_recomputed": False, "holdout_read": False,
        "sha256": {p.name: sha(p) for p in sorted(output.glob("*.parquet"))}}
    save_json(output / "source_index.json", sources)
    save_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "evaluate"))
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    report = freeze(args.output) if args.stage == "freeze" else evaluate(args.output, args.workers)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
