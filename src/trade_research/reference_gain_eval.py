"""Raw-minute execution and explicit unresolved accounting for fixed pairs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from math import isfinite
from pathlib import Path

import duckdb
import pandas as pd

from .absolute_ridge_1449_eval import PERIOD_QUALITY, apply_period_quality
from .corporate_cash import save_json, sha
from .fill_accounting import account_rows
from .hf_outcomes import Assumptions, outcomes_for_symbol, EXECUTION_LABELS, EXIT_WINDOWS
from .reference_gain_pairs import ROOT
from .round_number_entry import validate_window
from .turnover_reference import CALENDAR

MINUTES = Path("data/hf/pilot/data/stock_1m")
DAILY = Path("data/baostock/market_2020_2026/daily")
SIGNAL_SHA = "c23556798937d28472cb909dc52947cc3f8e41a8771adcef971e134ae6abf214"


def account_with_windows(raw: pd.DataFrame, signals: pd.DataFrame,
                         win: pd.DataFrame, calendar: list[str], *,
                         first_date: str = "2024-01-01", last_date: str = "2025-12-31") -> pd.DataFrame:
    accounted = account_rows(raw, calendar, first_date=first_date, last_date=last_date)
    accounted = accounted.merge(signals[["date", "code", "arm", "pair_id", "price_1449"]],
        on=["date", "code"], validate="many_to_one")
    for side, key in (("entry", "date"), ("exit", "exit_date")):
        selected = (win.loc[win.window_role.eq(side)].drop(columns="window_role")
                    if "window_role" in win else win)
        fields = selected.rename(columns={"date": key,
            **{k: side + "_" + k for k in selected.columns if k not in ("date", "code")}})
        accounted = accounted.merge(fields, on=[key, "code"], how="left", validate="many_to_one")
    accounted["execution_source_valid"] = accounted.entry_window_status.eq("valid") & (
        accounted.exit_window_status.eq("valid") | accounted.entry_status.ne("filled"))
    for slip in (5, 15):
        accounted[f"known_return{slip}"] = accounted[f"known_return{slip}"].where(accounted.execution_source_valid)
    accounted["unknown_after_buy"] |= accounted.entry_status.eq("filled") & ~accounted.execution_source_valid
    return accounted


def quality_windows(minute: pd.DataFrame, dates: list[str],
                    exit_labels: tuple[str, ...] = EXECUTION_LABELS) -> pd.DataFrame:
    """Keep morning and afternoon prices/volume separate even on the same date."""
    separate = exit_labels != EXECUTION_LABELS
    roles = (("entry", EXECUTION_LABELS), ("exit", exit_labels)) if separate else (("shared", EXECUTION_LABELS),)
    groups = {date: bars for date, bars in minute.groupby("date")}
    records = []
    code = minute.code.iloc[0] if len(minute) else None
    for date in dates:
        daily = groups.get(date, minute.iloc[:0])
        for role, labels in roles:
            bars = daily.loc[daily.label.isin(labels)]
            status, _ = validate_window(bars, expected_labels=labels)
            positive = bars.loc[bars.volume.gt(0)]
            row = {"date": date, "code": code, "window_status": status,
                "raw_vwap": float(bars.turnover.sum() / bars.volume.sum()) if bars.volume.sum() > 0 else None,
                "window_low": float(positive.low.min()) if len(positive) else None,
                "window_high": float(positive.high.max()) if len(positive) else None}
            if separate:
                row["window_role"] = role
            records.append(row)
    return pd.DataFrame(records)


def reprice(output: Path = ROOT, *, expected_signal_sha: str = SIGNAL_SHA,
            rule_commit: str = "8c75ac3", notional: float = 20000,
            first_date: str = "2024-01-01", last_date: str = "2025-12-31",
            quality_report: Path = PERIOD_QUALITY, horizons: tuple[int, ...] = (1, 5),
            exit_window: str = "close") -> dict:
    if not isfinite(notional) or notional <= 0:
        raise ValueError("The requested notional must be positive")
    if exit_window not in ("close", "morning") or not horizons or any(h not in (1, 5) for h in horizons):
        raise ValueError("Only explicitly supported holding windows can be repriced")
    exit_labels = EXIT_WINDOWS[exit_window]
    needed_labels = sorted(set(EXECUTION_LABELS + exit_labels))
    signal_file = output / "signals.parquet"
    if sha(signal_file) != expected_signal_sha:
        raise ValueError("Frozen execution list changed")
    signals = pd.read_parquet(signal_file)
    raw_calendar = pd.read_parquet(CALENDAR)
    calendar = sorted(raw_calendar.loc[raw_calendar.is_trading_day.eq("1")
        & raw_calendar.calendar_date.between(first_date, last_date), "calendar_date"].tolist())
    positions = {d: i for i, d in enumerate(calendar)}
    if (not len(signals) or not set(signals.date).issubset(positions)
            or any(positions[d] + 10 >= len(calendar) for d in signals.date)):
        raise ValueError("Every signal needs a complete ten-session observation window")
    manifest = {"rule_commit": rule_commit, "signals_sha256": expected_signal_sha,
        "calendar_sha256": sha(CALENDAR), "notional": notional, "horizons": list(horizons),
        "execution_labels": ["1452", "1453", "1454", "1455"], "holdout_read": last_date > "2025-12-31",
        "exit_window": exit_window, "exit_labels": list(exit_labels),
        "first_date": first_date, "last_date": last_date,
        "quality_report": str(quality_report), "quality_report_sha256": sha(quality_report),
        "verified_entry_reference_rows": int(signals.get("entry_reference_verified", pd.Series(False, index=signals.index)).eq(True).sum())}
    save_json(output / "execution_manifest.json", manifest)

    def one(item):
        code, group = item
        exchange, symbol = code.split(".")
        minute_path = MINUTES / exchange.upper() / (symbol + ".parquet")
        daily_path = DAILY / (code.replace(".", "_") + ".parquet")
        dates = sorted({d for signal in group.date for d in calendar[
            positions[signal]:positions[signal] + 11]})
        c = duckdb.connect()
        c.execute("SET threads=1")
        c.read_parquet(str(minute_path)).create_view("minute_source")
        c.register("needed", pd.DataFrame({"date": dates}))
        minute = c.execute("""SELECT timestamp,open,high,low,close,volume,turnover
            FROM minute_source WHERE timestamp>=?::TIMESTAMP AND timestamp<?::DATE+INTERVAL 1 DAY
              AND strftime(timestamp,'%H%M') IN (SELECT unnest(?))
              AND strftime(timestamp,'%Y-%m-%d') IN (SELECT date FROM needed)
            ORDER BY timestamp""", [dates[0], dates[-1], needed_labels]).df()
        c.read_parquet(str(daily_path)).create_view("daily_source")
        prior_day = c.execute("""SELECT max(date) FROM daily_source
            WHERE date<? AND tradestatus=1""", [dates[0]]).fetchone()[0]
        if prior_day is None:
            raise ValueError("No historical normal trading day before a selected signal")
        daily_start = (min(group.previous_traded_date) if "previous_traded_date" in group else prior_day)
        daily = c.execute("""SELECT * FROM daily_source WHERE date BETWEEN ? AND ?
            ORDER BY date""", [daily_start, dates[-1]]).df()
        c.close()
        minute["date"] = minute.timestamp.dt.strftime("%Y-%m-%d")
        minute["label"] = minute.timestamp.dt.strftime("%H%M")
        minute["code"] = code
        trades = outcomes_for_symbol(group, minute, daily, calendar,
            Assumptions(target_notional=notional), horizons=horizons, exit_labels=exit_labels,
            sizing_price_column="price_1449")
        trades["target_notional"] = notional
        trades["entry_window"] = "baseline"
        trades["exit_window"] = exit_window
        quality = quality_windows(minute, dates, exit_labels)
        quality["code"] = code
        return trades, minute, quality, {"code": code,
            "minute_sha256": sha(minute_path), "daily_sha256": sha(daily_path)}

    grouped = list(signals.groupby("code", sort=True))
    trades, bars, windows, sources = [], [], [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, (t, b, w, s) in enumerate(pool.map(one, grouped), 1):
            trades.append(t); bars.append(b); windows.append(w); sources.append(s)
            if i % 200 == 0 or i == len(grouped):
                print(f"Repriced raw minutes {i}/{len(grouped)} stocks", flush=True)
    raw = pd.concat(trades, ignore_index=True)
    raw["quality_clean_exit"] = False  # Replaced by the established period-only audit below.
    raw, _ = apply_period_quality(raw, report_path=quality_report, first_date=first_date, last_date=last_date)
    win = pd.concat(windows, ignore_index=True)
    accounted = account_with_windows(raw, signals, win, calendar, first_date=first_date, last_date=last_date)
    for name, table in (("repriced", raw), ("accounted_initial", accounted),
                        ("raw_windows", pd.concat(bars, ignore_index=True)), ("window_quality", win)):
        table.to_parquet(output / f"{name}.parquet", index=False, compression="zstd")
    save_json(output / "execution_sources.json", sources)
    result = {**manifest, "rows": len(accounted), "signals": len(signals),
        "entry_statuses": accounted.entry_status.value_counts().to_dict(),
        "exit_statuses": accounted.exit_status.value_counts().to_dict(),
        "accounting_categories": accounted.category.value_counts().to_dict(),
        "unknown_after_buy": int(accounted.unknown_after_buy.sum()),
        "source_invalid_rows": int((~accounted.execution_source_valid).sum()),
        "output_sha256": {k: sha(output / (k + ".parquet")) for k in
            ("repriced", "accounted_initial", "raw_windows", "window_quality")}}
    save_json(output / "execution_report.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(reprice(), ensure_ascii=False, indent=2))
