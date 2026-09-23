"""Coverage and state audit for the historical Shanghai/Shenzhen universe."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd


def audit(root: Path) -> dict:
    metadata = root / "metadata"
    selection = json.loads((metadata / "selection.json").read_text(encoding="utf-8"))
    symbols = pd.read_parquet(metadata / "symbols.parquet")
    calendar = pd.read_parquet(metadata / "calendar.parquet")
    all_trading = set(calendar.loc[calendar["is_trading_day"] == "1", "calendar_date"])
    first_date, last_date = selection["first_date"], selection["last_date"]
    rows = []
    for stock in symbols.itertuples(index=False):
        path = root / "daily" / f"{stock.code.replace('.', '_')}.parquet"
        if not path.exists():
            rows.append({"code": stock.code, "status": "missing_file"})
            continue
        daily = pd.read_parquet(path)
        expected = {
            date for date in all_trading
            if max(first_date, stock.ipoDate) <= date <=
            min(last_date, stock.outDate or last_date)
        }
        observed = set(daily["date"])
        active = daily.loc[daily["tradestatus"] == 1]
        suspended = daily.loc[daily["tradestatus"] == 0]
        invalid_prices = (
            (active["low"] <= 0)
            | (active["high"] < active[["open", "low", "close"]].max(axis=1))
            | (active["low"] > active[["open", "high", "close"]].min(axis=1))
            | active[["open", "high", "low", "close", "volume", "amount"]].isna().any(axis=1)
        )
        rows.append({
            "code": stock.code, "status": "ok", "daily_rows": len(daily),
            "expected_days": len(expected), "observed_days": len(observed),
            "missing_days": len(expected - observed),
            "extra_days": len(observed - expected),
            "active_days": len(active), "suspended_days": len(suspended),
            "st_active_days": int(active["isST"].eq(1).sum()),
            "invalid_active_rows": int(invalid_prices.sum()),
            "duplicate_keys": int(daily.duplicated(["code", "date"]).sum()),
            "first_date": daily["date"].min(), "last_date": daily["date"].max(),
        })
    report = pd.DataFrame(rows)
    report.to_csv(metadata / "audit_stocks.csv", index=False)
    valid = report.loc[report["status"] == "ok"]
    def total(field: str) -> int:
        return int(valid[field].sum()) if len(valid) else 0
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_symbols": len(symbols), "downloaded_symbols": len(valid),
        "daily_rows": total("daily_rows"), "expected_days": total("expected_days"),
        "missing_days": total("missing_days"), "extra_days": total("extra_days"),
        "active_days": total("active_days"), "suspended_days": total("suspended_days"),
        "st_active_days": total("st_active_days"),
        "invalid_active_rows": total("invalid_active_rows"),
        "duplicate_keys": total("duplicate_keys"),
    }
    summary["coverage"] = round(1 - summary["missing_days"] / summary["expected_days"], 6) \
        if summary["expected_days"] else 0
    summary["accepted"] = (
        summary["downloaded_symbols"] == summary["selected_symbols"]
        and summary["missing_days"] == 0
        and summary["extra_days"] == 0
        and summary["invalid_active_rows"] == 0
        and summary["duplicate_keys"] == 0
    )
    (metadata / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/baostock/market_2020_2026"))
    args = parser.parse_args()
    print(json.dumps(audit(args.root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
