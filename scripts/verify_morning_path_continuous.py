"""Independently recalculate frozen sample inputs and trade VWAPs from source rows."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("data/research/morning_path_continuous")
MINUTES = Path("data/hf/pilot/data/stock_1m")
DAILY = Path("data/baostock/market_2020_2026/daily")


def _check(row: pd.Series, opening_anchor: str = "source_0930_close",
           decision_time_sizing: bool = False) -> dict:
    exchange, symbol = row.code.split(".")
    end = row.target_exit_date or row.date
    bars = pd.read_parquet(
        MINUTES / exchange.upper() / f"{symbol}.parquet",
        columns=["timestamp", "close", "volume", "turnover"],
        filters=[("timestamp", ">=", pd.Timestamp(row.date)),
                 ("timestamp", "<", pd.Timestamp(end) + pd.Timedelta(days=1))],
    )
    bars["date"] = bars.timestamp.dt.strftime("%Y-%m-%d")
    bars["label"] = bars.timestamp.dt.strftime("%H%M")
    today = bars.loc[bars.date.eq(row.date)]
    close = {}
    for label in ("0930", "1130", "1449"):
        found = today.loc[today.label.eq(label)]
        if len(found) != 1:
            return {"input_complete": False, "entry_check": False,
                    "exit_check": False}
        close[label] = float(found.close.iloc[0])
    daily = pd.read_parquet(
        DAILY / f"{exchange}_{symbol}.parquet",
        columns=["date", "preclose", "open"],
        filters=[("date", "==", row.date)],
    )
    if len(daily) != 1:
        return {"input_complete": False, "entry_check": False,
                "exit_check": False}
    previous = float(daily.preclose.iloc[0])
    anchor = (close["0930"] if opening_anchor == "source_0930_close"
              else float(daily.open.iloc[0]))
    input_errors = [
        abs(close["1449"] - row.price_1449),
        abs(100 * (close["1130"] / anchor - 1) - row.morning_pp),
        abs(100 * (anchor / previous - 1) - row.gap_pp),
        abs(100 * (close["1449"] / previous - 1) - row.day_pp),
    ]
    shares_check = True
    if decision_time_sizing:
        affordable = int(100000 // close["1449"])
        planned = (affordable if affordable >= 200 else 0) if row.code.startswith(
            "sh.68") else affordable // 100 * 100
        shares_check = (int(row.shares) == planned if row.entry_status == "filled"
                        else int(row.shares) == 0)
    entry = today.loc[today.label.isin(("1452", "1453", "1454", "1455"))]
    entry_complete = entry.label.tolist() == ["1452", "1453", "1454", "1455"]
    if row.entry_status == "filled":
        entry_price = entry.turnover.sum() / entry.volume.sum() * 1.0005
        entry_error = abs(entry_price - row.entry_price)
        entry_check = entry_complete and np.isfinite(entry_error) and entry_error < 1e-8
    else:
        entry_error = None
        entry_check = entry_complete
    exit_error = None
    exit_check = True
    if row.exit_status == "filled" and row.exit_delay_sessions == 0:
        exit_bars = bars.loc[bars.date.eq(row.target_exit_date)
                             & bars.label.isin(("0935", "0936", "0937", "0938"))]
        exit_check = exit_bars.label.tolist() == [
            "0935", "0936", "0937", "0938"]
        if exit_check:
            exit_price = exit_bars.turnover.sum() / exit_bars.volume.sum() * .9995
            exit_error = abs(exit_price - row.exit_price)
            exit_check = np.isfinite(exit_error) and exit_error < 1e-8
    return {"input_complete": True,
            "input_max_abs_error": float(max(input_errors)),
            "input_check": max(input_errors) < 1e-8,
            "entry_check": bool(entry_check),
            "shares_check": bool(shares_check),
            "entry_abs_error": entry_error,
            "exit_check": bool(exit_check),
            "exit_abs_error": exit_error}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--opening-anchor",
                        choices=("source_0930_close", "daily_open"),
                        default="source_0930_close")
    parser.add_argument("--decision-time-sizing", action="store_true")
    args = parser.parse_args()
    sample = pd.read_parquet(args.root / "raw_signals.parquet")
    priced = pd.read_parquet(args.root / "repriced.parquet")
    priced = priced.loc[priced.target_notional.eq(100000)
                        & priced.exit_window.eq("morning")]
    rows = sample.merge(priced, on=["date", "code"],
                        validate="one_to_one")
    if len(rows) != 320 or rows.duplicated(["date", "code"]).any():
        raise ValueError("Frozen sample or repricing grid is incomplete")
    with ThreadPoolExecutor(max_workers=8) as pool:
        checks = list(pool.map(
            partial(_check, opening_anchor=args.opening_anchor,
                    decision_time_sizing=args.decision_time_sizing),
            (row for _, row in rows.iterrows()),
        ))
    frame = pd.DataFrame(checks)
    report = {"sample_stock_days": len(frame),
              "opening_anchor": args.opening_anchor,
              "decision_time_sizing": args.decision_time_sizing,
              "input_complete": int(frame.input_complete.sum()),
              "input_equal": int(frame.input_check.fillna(False).sum()),
              "shares_equal": (int(frame.shares_check.sum())
                               if args.decision_time_sizing else None),
              "entry_equal": int(frame.entry_check.sum()),
              "on_time_exit_equal_or_not_applicable": int(frame.exit_check.sum()),
              "maximum_input_abs_error": float(frame.input_max_abs_error.max()),
              "maximum_entry_abs_error": float(frame.entry_abs_error.max()),
              "maximum_exit_abs_error": float(frame.exit_abs_error.max()),
              "all_passed": bool(frame.input_complete.all()
                                 and frame.input_check.all()
                                 and frame.shares_check.all()
                                 and frame.entry_check.all()
                                 and frame.exit_check.all()),
              "note": "Same source files, independent field and VWAP arithmetic; not external timestamp validation"}
    (args.root / "raw_sample_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
