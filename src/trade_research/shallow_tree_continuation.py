"""Trace existing unresolved positions, without changing model selections."""

from __future__ import annotations

import json
from math import isfinite
from pathlib import Path
import shutil

import duckdb
import pandas as pd

from .absolute_ridge_1449_eval import apply_period_quality
from .corporate_cash import save_json, sha
from .hf_outcomes import Assumptions, outcomes_for_symbol
from .reference_gain_eval import MINUTES, DAILY, account_with_windows
from .round_number_entry import validate_window
from .shallow_tree_1449 import ROOT
from .turnover_reference import CALENDAR

KEY = ["date", "code", "horizon"]


def continue_model(source: Path, output: Path) -> dict:
    baseline_report = json.loads((source / "execution_report.json").read_text())
    for name, expected in baseline_report["output_sha256"].items():
        if sha(source / (name + ".parquet")) != expected:
            raise ValueError("Original execution changed before continuation")
    signals = pd.read_parquet(source / "signals.parquet")
    original = pd.read_parquet(source / "repriced.parquet")
    notionals = original.target_notional.unique()
    if len(notionals) != 1 or not isfinite(notionals[0]) or notionals[0] <= 0:
        raise ValueError("Continuation requires one positive fixed order size")
    notional = notionals[0]
    pending = original.loc[original.entry_status.eq("filled") & original.exit_price.isna()]
    table = pd.read_parquet(CALENDAR)
    calendar = sorted(table.loc[table.is_trading_day.eq("1")
        & table.calendar_date.between("2024-01-01", "2025-12-31"), "calendar_date"].tolist())
    changed, new_bars, new_quality, sources = [], [], [], []
    checks = ["entry_status", "entry_price", "shares", "target_exit_date",
        "exit_status", "exit_date", "exit_delay_sessions", "exit_price",
        "net_return", "corporate_action_crossed"]
    for code, group in pending.groupby("code"):
        exchange, symbol = code.split(".")
        minute_path, daily_path = MINUTES / exchange.upper() / (symbol + ".parquet"), DAILY / (code.replace(".", "_") + ".parquet")
        c = duckdb.connect()
        c.read_parquet(str(minute_path)).create_view("minutes")
        minute = c.execute("""SELECT timestamp,open,high,low,close,volume,turnover
            FROM minutes WHERE timestamp>=?::TIMESTAMP AND timestamp<'2026-01-01'::TIMESTAMP
              AND strftime(timestamp,'%H%M') IN ('1452','1453','1454','1455')
            ORDER BY timestamp""", [group.date.min()]).df()
        c.read_parquet(str(daily_path)).create_view("daily")
        previous = c.execute("SELECT max(date) FROM daily WHERE date<? AND tradestatus=1", [group.date.min()]).fetchone()[0]
        daily = c.execute("SELECT * FROM daily WHERE date BETWEEN ? AND '2025-12-31' ORDER BY date", [previous]).df()
        c.close()
        minute["date"], minute["label"], minute["code"] = minute.timestamp.dt.strftime("%Y-%m-%d"), minute.timestamp.dt.strftime("%H%M"), code
        new_bars.append(minute)
        for date, bars in minute.groupby("date"):
            status, _ = validate_window(bars)
            positive = bars.loc[bars.volume.gt(0)]
            new_quality.append({"date": date, "code": code, "window_status": status,
                "raw_vwap": bars.turnover.sum() / bars.volume.sum() if bars.volume.sum() > 0 else None,
                "window_low": positive.low.min() if len(positive) else None,
                "window_high": positive.high.max() if len(positive) else None})
        for horizon, part in group.groupby("horizon"):
            chosen = signals.loc[signals.code.eq(code) & signals.date.isin(part.date)]
            baseline = outcomes_for_symbol(chosen, minute, daily, calendar,
                Assumptions(target_notional=notional), horizons=(int(horizon),), sizing_price_column="price_1449")
            left = baseline.set_index(KEY)[checks].sort_index().astype(object)
            right = part.set_index(KEY)[checks].sort_index().astype(object)
            pd.testing.assert_frame_equal(left.where(left.notna(), None), right.where(right.notna(), None),
                check_dtype=False, check_exact=False, atol=1e-12, rtol=0)
            extended = outcomes_for_symbol(chosen, minute, daily, calendar,
                Assumptions(target_notional=notional, maximum_exit_delay_sessions=len(calendar)),
                horizons=(int(horizon),), sizing_price_column="price_1449")
            extended["target_notional"], extended["entry_window"], extended["exit_window"] = notional, "baseline", "close"
            extended["quality_clean_exit"] = False
            extended, _ = apply_period_quality(extended)
            changed.append(extended)
        sources.append({"code": code, "minute_sha256": sha(minute_path), "daily_sha256": sha(daily_path)})
    traced = pd.concat(changed, ignore_index=True) if changed else original.iloc[:0].copy()
    remaining = original.loc[~original.set_index(KEY).index.isin(pending.set_index(KEY).index)]
    raw = pd.concat([remaining, traced], ignore_index=True).sort_values(KEY).reset_index(drop=True)
    bars = pd.concat([pd.read_parquet(source / "raw_windows.parquet"), *new_bars], ignore_index=True)
    bars = bars.drop_duplicates().sort_values(["code", "timestamp"])
    if bars.duplicated(["code", "timestamp"]).any():
        raise ValueError("A reloaded raw execution bar differs from the original")
    windows = pd.concat([pd.read_parquet(source / "window_quality.parquet"), pd.DataFrame(new_quality)], ignore_index=True)
    windows = windows.drop_duplicates().sort_values(["code", "date"])
    if windows.duplicated(["code", "date"]).any():
        raise ValueError("Recomputed window diagnostics differ")
    accounted = account_with_windows(raw, signals, windows, calendar)
    old = pd.read_parquet(source / "accounted_initial.parquet")
    for field in ("old_score5", "old_score15", "buy_cost5", "buy_cost15"):
        pd.testing.assert_series_equal(accounted.set_index(KEY)[field].sort_index(),
            old.set_index(KEY)[field].sort_index(), check_dtype=False, check_exact=False, atol=1e-12, rtol=0)
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / "signals.parquet", output / "signals.parquet")
    shutil.copyfile(source / "input_report.json", output / "input_report.json")
    for name, frame in (("repriced", raw), ("accounted_initial", accounted), ("raw_windows", bars), ("window_quality", windows)):
        frame.to_parquet(output / (name + ".parquet"), index=False, compression="zstd")
    result = {"interpretation": "continued_accounting_diagnostic_not_the_original_five_day_exit_delay_rule",
        "source_directory": str(source), "source_execution_sha256": sha(source / "execution_report.json"),
        "continued_rows": len(traced), "remaining_unresolved": int((accounted.entry_status.eq("filled") & accounted.exit_price.isna()).sum()),
        "last_allowed_date": calendar[-1], "holdout_read": False, "sources": sources,
        "output_sha256": {name: sha(output / (name + ".parquet")) for name in ("repriced", "accounted_initial", "raw_windows", "window_quality")}}
    save_json(output / "execution_report.json", result)
    return result


if __name__ == "__main__":
    output = ROOT / "continued"
    output.mkdir(exist_ok=True)
    shutil.copyfile(ROOT / "input_report.json", output / "input_report.json")
    for name in ("ridge", "tree"):
        result = continue_model(ROOT / name, output / name)
        print(name, {k: result[k] for k in ("continued_rows", "remaining_unresolved")}, flush=True)
