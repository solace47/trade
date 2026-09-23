"""Audit a pinned one-minute archive against BaoStock daily bars.

The archive uses naive timestamps interpreted as Asia/Shanghai bar-end labels.
09:30 is the opening-auction record; 09:31--11:30 and 13:01--15:00 are
the 240 regular-session minute records. The last complete archive date is
2026-08-06; the repository's 2026-08-07 records are intraday only.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pandas as pd

from .hf_download import REVISION, selected_paths


FIRST_DATE = "2025-08-07"
LAST_DATE = "2026-08-06"
EXPECTED_LABELS = (
    "0930",
    *(f"{minute // 60:02d}{minute % 60:02d}" for minute in range(9 * 60 + 31, 11 * 60 + 31)),
    *(f"{minute // 60:02d}{minute % 60:02d}" for minute in range(13 * 60 + 1, 15 * 60 + 1)),
)
assert len(EXPECTED_LABELS) == 241 and EXPECTED_LABELS[230] == "1450"


def read_window(path: Path, first_date: str = FIRST_DATE, last_date: str = LAST_DATE) -> pd.DataFrame:
    since = datetime.fromisoformat(first_date)
    until = datetime.fromisoformat(last_date) + timedelta(days=1)
    result = pd.read_parquet(path, filters=[
        ("timestamp", ">=", since), ("timestamp", "<", until)
    ])
    result = result.loc[result["timestamp"].between(
        pd.Timestamp(since), pd.Timestamp(until), inclusive="left"
    )].copy()
    result["date"] = result["timestamp"].dt.strftime("%Y-%m-%d")
    result["label"] = result["timestamp"].dt.strftime("%H%M")
    return result.sort_values("timestamp").reset_index(drop=True)


def audit_symbol(code: str, minute_path: Path, daily_path: Path,
                 first_date: str, last_date: str) -> tuple[dict, list[dict]]:
    if not minute_path.exists() or not daily_path.exists():
        return {"code": code, "status": "missing_file"}, []
    minute = read_window(minute_path, first_date, last_date)
    daily = pd.read_parquet(daily_path)
    daily = daily.loc[daily["date"].between(first_date, last_date)].copy()
    exchange, symbol = code.split(".")
    wrong_identity = int(((minute["exchange"].str.upper() != exchange.upper()) |
                          (minute["symbol"].str.zfill(6) != symbol)).sum())
    active_dates = set(daily.loc[daily["tradestatus"] == 1, "date"])
    day_groups = minute.groupby("date", sort=True)
    observed_dates = set(day_groups.groups)
    partial = {date for date, group in day_groups
               if group["label"].tolist() != list(EXPECTED_LABELS)}
    missing = active_dates - observed_dates
    nontrading = observed_dates - active_dates
    suspension_placeholders = {
        date for date in nontrading if day_groups.get_group(date)["volume"].sum() == 0
    }
    unexpected_nontrading = nontrading - suspension_placeholders
    complete_dates = (active_dates & observed_dates) - partial
    invalid_rows = int((
        (minute["low"] <= 0) |
        (minute["high"] < minute[["open", "low", "close"]].max(axis=1)) |
        (minute["low"] > minute[["open", "high", "close"]].min(axis=1)) |
        (minute["volume"] < 0) | (minute["turnover"] < 0) |
        minute[["open", "high", "low", "close", "volume", "turnover"]].isna().any(axis=1)
    ).sum())
    complete = minute.loc[minute["date"].isin(complete_dates)]
    # Quote-only bars can have zero volume and a price outside the day's
    # executed range (especially the 09:30 auction placeholder).
    traded = complete.loc[complete["volume"] > 0]
    aggregates = traded.groupby("date").agg(
        minute_open=("open", "first"), minute_high=("high", "max"),
        minute_low=("low", "min"), minute_close=("close", "last"),
        minute_volume=("volume", "sum"), minute_amount=("turnover", "sum"),
    )
    # The 15:00 official closing price can be carried in a zero-volume bar.
    aggregates["minute_close"] = complete.groupby("date")["close"].last()
    active_no_trade = complete_dates - set(aggregates.index)
    comparison = daily.set_index("date").join(aggregates, how="inner")
    mismatch_by_field = {
        field: (comparison[field] - comparison[f"minute_{field}"]).abs() > 0.0001
        for field in ("open", "high", "low", "close")
    }
    mismatch_days = pd.DataFrame(mismatch_by_field).any(axis=1)
    opening_volume = complete.groupby("date")["volume"].first()
    auction_placeholder_mismatch = (
        mismatch_by_field["open"]
        & ~pd.DataFrame({field: mismatch_by_field[field]
                         for field in ("high", "low", "close")}).any(axis=1)
        & opening_volume.reindex(comparison.index).eq(0)
    )
    unexplained_mismatch = mismatch_days & ~auction_placeholder_mismatch
    volume_delta = (comparison["volume"] - comparison["minute_volume"]).abs()
    amount_delta = (comparison["amount"] - comparison["minute_amount"]).abs()
    issues = [
        {"code": code, "date": date, "kind": kind}
        for kind, dates in (
            ("missing_active_minute", missing),
            ("partial_minute_day", partial),
            ("suspension_placeholder_day", suspension_placeholders),
            ("unexpected_minute_on_suspended_day", unexpected_nontrading),
            ("ohlc_disagreement", set(mismatch_days.index[mismatch_days])),
        ) for date in sorted(dates)
    ]
    issues.extend({"code": code, "date": date, "kind": "active_no_trade"}
                  for date in sorted(active_no_trade))
    row = {
        "code": code, "status": "ok", "minute_rows": len(minute),
        "active_daily_days": len(active_dates), "observed_days": len(observed_dates),
        "complete_days": len(complete_dates), "partial_days": len(partial),
        "missing_active_days": len(missing), "nontrading_minute_days": len(nontrading),
        "suspension_placeholder_days": len(suspension_placeholders),
        "unexpected_nontrading_minute_days": len(unexpected_nontrading),
        "duplicate_rows": int(minute.duplicated(["timestamp"]).sum()),
        "invalid_rows": invalid_rows, "wrong_identity_rows": wrong_identity,
        "comparison_days": len(comparison),
        "active_no_trade_days": len(active_no_trade),
        "zero_volume_auction_days": int((complete.loc[complete["label"] == "0930", "volume"] == 0).sum()),
        "ohlc_mismatch_days": int(mismatch_days.sum()),
        "auction_placeholder_open_mismatch_days": int(auction_placeholder_mismatch.sum()),
        "unexplained_ohlc_mismatch_days": int(unexplained_mismatch.sum()),
        **{f"mismatch_{field}": int(mask.sum()) for field, mask in mismatch_by_field.items()},
        "volume_over_100_shares_days": int((volume_delta > 100).sum()),
        "amount_over_0_01pct_days": int((amount_delta > comparison["amount"].abs() * 0.0001).sum()),
        "first_date": minute["date"].min() if len(minute) else None,
        "last_date": minute["date"].max() if len(minute) else None,
    }
    return row, issues


def audit(hf_root: Path, bao_root: Path, first_date: str = FIRST_DATE,
          last_date: str = LAST_DATE) -> dict:
    metadata = bao_root / "metadata"
    pilot_file = metadata / "pilot_symbols.parquet"
    pilot = pd.read_parquet(pilot_file)
    paths = selected_paths(pilot_file)
    rows, issues = [], []
    for code, relative in zip(pilot["code"], paths, strict=True):
        row, issue_rows = audit_symbol(
            code, hf_root / relative,
            bao_root / "daily" / f"{code.replace('.', '_')}.parquet",
            first_date, last_date,
        )
        rows.append(row)
        issues.extend(issue_rows)
    output = bao_root / "hf_audit"
    output.mkdir(parents=True, exist_ok=True)
    report = pd.DataFrame(rows)
    report.to_csv(output / "stocks.csv", index=False)
    pd.DataFrame(issues, columns=["code", "date", "kind"]).to_csv(
        output / "issues.csv", index=False
    )
    valid = report.loc[report["status"] == "ok"]
    def total(column: str) -> int:
        return int(valid[column].sum()) if len(valid) else 0
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": REVISION,
        "first_date": first_date, "last_date": last_date,
        "selected_symbols": len(pilot), "downloaded_symbols": len(valid),
        "minute_rows": total("minute_rows"),
        "active_daily_days": total("active_daily_days"),
        "complete_days": total("complete_days"),
        "partial_days": total("partial_days"),
        "missing_active_days": total("missing_active_days"),
        "nontrading_minute_days": total("nontrading_minute_days"),
        "suspension_placeholder_days": total("suspension_placeholder_days"),
        "unexpected_nontrading_minute_days": total("unexpected_nontrading_minute_days"),
        "duplicate_rows": total("duplicate_rows"),
        "invalid_rows": total("invalid_rows"),
        "wrong_identity_rows": total("wrong_identity_rows"),
        "comparison_days": total("comparison_days"),
        "active_no_trade_days": total("active_no_trade_days"),
        "zero_volume_auction_days": total("zero_volume_auction_days"),
        "ohlc_mismatch_days": total("ohlc_mismatch_days"),
        "auction_placeholder_open_mismatch_days": total("auction_placeholder_open_mismatch_days"),
        "unexplained_ohlc_mismatch_days": total("unexplained_ohlc_mismatch_days"),
        "volume_over_100_shares_days": total("volume_over_100_shares_days"),
        "amount_over_0_01pct_days": total("amount_over_0_01pct_days"),
    }
    summary["complete_ratio"] = (
        round(summary["complete_days"] / summary["active_daily_days"], 6)
        if summary["active_daily_days"] else 0
    )
    summary["pilot_accepted"] = (
        summary["downloaded_symbols"] == summary["selected_symbols"]
        and summary["complete_ratio"] >= 0.99
        and summary["duplicate_rows"] == 0
        and summary["invalid_rows"] == 0
        and summary["wrong_identity_rows"] == 0
        and summary["active_no_trade_days"] == 0
        and summary["unexpected_nontrading_minute_days"] == 0
        and summary["unexplained_ohlc_mismatch_days"] == 0
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-root", type=Path, default=Path("data/hf/pilot"))
    parser.add_argument("--bao-root", type=Path, default=Path("data/baostock/pilot_2025_2026"))
    parser.add_argument("--first-date", default=FIRST_DATE)
    parser.add_argument("--last-date", default=LAST_DATE)
    args = parser.parse_args()
    print(json.dumps(audit(args.hf_root, args.bao_root, args.first_date, args.last_date),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
