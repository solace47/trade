"""Audit downloaded BaoStock bars against their daily records and calendar."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd


def expected_bar_labels() -> tuple[str, ...]:
    morning = range(9 * 60 + 35, 11 * 60 + 31, 5)
    afternoon = range(13 * 60 + 5, 15 * 60 + 1, 5)
    return tuple(f"{minute // 60:02d}{minute % 60:02d}" for minute in (*morning, *afternoon))


EXPECTED_LABELS = expected_bar_labels()
assert len(EXPECTED_LABELS) == 48 and EXPECTED_LABELS[45] == "1450"


def audit_symbol(
    code: str,
    root: Path,
    first_date: str,
    last_date: str,
    calendar_dates: set[str],
    ipo_date: str,
    out_date: str,
) -> tuple[dict, list[dict]]:
    stem = code.replace(".", "_")
    minute_file = root / "minute_5m" / f"{stem}.parquet"
    daily_file = root / "daily" / f"{stem}.parquet"
    if not minute_file.exists() or not daily_file.exists():
        return {"code": code, "status": "missing_file"}, []
    minute = pd.read_parquet(minute_file)
    daily = pd.read_parquet(daily_file)
    minute = minute.loc[minute["date"].between(first_date, last_date)].copy()
    daily = daily.loc[daily["date"].between(first_date, last_date)].copy()
    minute["label"] = minute["time"].str[8:12]
    active_dates = set(daily.loc[daily["tradestatus"] == 1, "date"])
    daily_dates = set(daily["date"])
    listed_dates = {
        date for date in calendar_dates
        if date >= max(first_date, ipo_date) and (not out_date or date < out_date)
    }
    day_groups = minute.groupby("date", sort=True)
    observed_dates = set(day_groups.groups)
    partial = []
    zero_placeholder_dates = set()
    mixed_zero_dates = set()
    offgrid = 0
    wrong_order = 0
    for date, group in day_groups:
        labels = group["label"].tolist()
        zero = group[["open", "high", "low", "close", "volume", "amount"]].eq(0).all(axis=1)
        if zero.all():
            zero_placeholder_dates.add(date)
        elif zero.any():
            mixed_zero_dates.add(date)
        if labels != list(EXPECTED_LABELS):
            partial.append(date)
        offgrid += int((~group["label"].isin(EXPECTED_LABELS)).sum())
        wrong_order += int(labels != sorted(labels))
    observed_active_dates = observed_dates - zero_placeholder_dates
    issues = []
    for kind, dates in (
        ("missing_daily", listed_dates - daily_dates),
        ("missing_active_minute", active_dates - observed_active_dates),
        ("partial_minute_day", set(partial)),
        ("mixed_zero_minute_day", mixed_zero_dates),
        ("zero_minute_on_active_day", zero_placeholder_dates & active_dates),
        ("nonzero_minute_on_suspended_day", observed_active_dates - active_dates),
        ("minute_on_nontrading_day", observed_dates - calendar_dates),
    ):
        issues.extend({"code": code, "date": date, "kind": kind} for date in sorted(dates))
    duplicate_minutes = int(minute.duplicated(["code", "time"]).sum())
    duplicate_daily = int(daily.duplicated(["code", "date"]).sum())

    # Only complete 48-bar days are compared to daily aggregates. BaoStock
    # sometimes differs by a few shares or yuan due to source rounding.
    complete_dates = (observed_active_dates & active_dates) - set(partial) - mixed_zero_dates
    complete = minute.loc[minute["date"].isin(complete_dates)].sort_values("time")
    aggregates = complete.groupby("date").agg(
        minute_open=("open", "first"), minute_high=("high", "max"),
        minute_low=("low", "min"), minute_close=("close", "last"),
        minute_volume=("volume", "sum"), minute_amount=("amount", "sum"),
    )
    comparison = daily.set_index("date").join(aggregates, how="inner")
    mismatches = {}
    for field in ("open", "high", "low", "close"):
        mismatches[field] = int(((comparison[field] - comparison[f"minute_{field}"]).abs() > 0.0001).sum())
    ohlc_disagree = pd.concat(
        [(comparison[field] - comparison[f"minute_{field}"]).abs() > 0.0001
         for field in ("open", "high", "low", "close")], axis=1
    ).any(axis=1)
    volume_delta = (comparison["volume"] - comparison["minute_volume"]).abs()
    amount_delta = (comparison["amount"] - comparison["minute_amount"]).abs()
    mismatches["volume_over_10_shares"] = int((volume_delta > 10).sum())
    mismatches["amount_over_0_01pct"] = int((amount_delta > comparison["amount"].abs() * 0.0001).sum())

    row = {
        "code": code,
        "status": "ok",
        "minute_rows": len(minute),
        "minute_days": len(observed_dates),
        "active_daily_days": len(active_dates),
        "complete_days": len(complete_dates),
        "zero_placeholder_days": len(zero_placeholder_dates),
        "mixed_zero_days": len(mixed_zero_dates),
        "partial_days": len(partial),
        "missing_active_minute_days": len(active_dates - observed_active_dates),
        "missing_daily_days": len(listed_dates - daily_dates),
        "duplicate_minutes": duplicate_minutes,
        "duplicate_daily": duplicate_daily,
        "offgrid_minutes": offgrid,
        "unsorted_days": wrong_order,
        "first_minute_date": minute["date"].min(),
        "last_minute_date": minute["date"].max(),
        "complete_ratio": round(len(complete_dates) / len(active_dates), 6) if active_dates else 0,
        "daily_comparison_days": len(comparison),
        "mismatch_any_ohlc_days": int(ohlc_disagree.sum()),
        **{f"mismatch_{key}": val for key, val in mismatches.items()},
    }
    return row, issues


def audit(root: Path) -> dict:
    metadata = root / "metadata"
    selection = json.loads((metadata / "selection.json").read_text(encoding="utf-8"))
    pilot = pd.read_parquet(metadata / "pilot_symbols.parquet")
    calendar = pd.read_parquet(metadata / "calendar.parquet")
    trading_dates = set(calendar.loc[calendar["is_trading_day"] == "1", "calendar_date"])
    first_date, last_date = selection["minute_start"], selection["minute_end"]
    trading_dates = {date for date in trading_dates if first_date <= date <= last_date}
    rows, issues = [], []
    for stock in pilot.itertuples(index=False):
        row, issue_rows = audit_symbol(
            stock.code, root, first_date, last_date, trading_dates,
            stock.ipoDate, stock.outDate,
        )
        rows.append(row)
        issues.extend(issue_rows)
    stock_report = pd.DataFrame(rows)
    stock_report.to_csv(metadata / "audit_stocks.csv", index=False)
    pd.DataFrame(issues, columns=["code", "date", "kind"]).to_csv(
        metadata / "audit_issues.csv", index=False
    )
    valid = stock_report.loc[stock_report["status"] == "ok"]
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "BaoStock",
        "scope": "100-stock historical as-of pilot, Shanghai/Shenzhen only",
        "first_date": first_date,
        "last_date": last_date,
        "selected_symbols": len(pilot),
        "downloaded_symbols": len(valid),
        "minute_rows": int(valid["minute_rows"].sum()) if len(valid) else 0,
        "active_daily_days": int(valid["active_daily_days"].sum()) if len(valid) else 0,
        "complete_minute_days": int(valid["complete_days"].sum()) if len(valid) else 0,
        "zero_placeholder_days": int(valid["zero_placeholder_days"].sum()) if len(valid) else 0,
        "mixed_zero_days": int(valid["mixed_zero_days"].sum()) if len(valid) else 0,
        "partial_minute_days": int(valid["partial_days"].sum()) if len(valid) else 0,
        "missing_active_minute_days": int(valid["missing_active_minute_days"].sum()) if len(valid) else 0,
        "missing_daily_days": int(valid["missing_daily_days"].sum()) if len(valid) else 0,
        "duplicate_minute_rows": int(valid["duplicate_minutes"].sum()) if len(valid) else 0,
        "offgrid_minute_rows": int(valid["offgrid_minutes"].sum()) if len(valid) else 0,
        "ohlc_mismatch_days": int(valid["mismatch_any_ohlc_days"].sum()) if len(valid) else 0,
        "ohlc_mismatch_fields": int(valid[[f"mismatch_{x}" for x in ("open", "high", "low", "close")]].sum().sum()) if len(valid) else 0,
    }
    summary["complete_ratio"] = round(
        summary["complete_minute_days"] / summary["active_daily_days"], 6
    ) if summary["active_daily_days"] else 0
    summary["pilot_accepted"] = (
        summary["downloaded_symbols"] == 100
        and summary["complete_ratio"] >= 0.95
        and summary["duplicate_minute_rows"] == 0
        and summary["offgrid_minute_rows"] == 0
        and summary["mixed_zero_days"] == 0
        and summary["ohlc_mismatch_days"] == 0
        and summary["missing_daily_days"] == 0
    )
    (metadata / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/baostock/pilot_2025_2026")
    args = parser.parse_args()
    print(json.dumps(audit(Path(args.root)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
