"""Cross-check the full Shanghai/Shenzhen minute archive against daily history."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from .hf_audit import audit_symbol
from .hf_download import selected_paths


FIRST_DATE = "2020-01-01"
LAST_DATE = "2026-08-06"


def audit(hf_root: Path, bao_root: Path, output: Path,
          first_date: str = FIRST_DATE, last_date: str = LAST_DATE) -> dict:
    symbols_file = bao_root / "metadata" / "symbols.parquet"
    symbols = pd.read_parquet(symbols_file)
    paths = selected_paths(symbols_file)
    rows, issues = [], []
    for number, (code, relative) in enumerate(zip(symbols["code"], paths, strict=True), 1):
        row, found = audit_symbol(
            code, hf_root / relative,
            bao_root / "daily" / f"{code.replace('.', '_')}.parquet",
            first_date, last_date,
        )
        rows.append(row)
        issues.extend(found)
        if number % 100 == 0:
            print(f"Audited {number}/{len(symbols)} symbols", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    report = pd.DataFrame(rows)
    report.to_csv(output / "stocks.csv", index=False)
    pd.DataFrame(issues, columns=["code", "date", "kind"]).to_csv(
        output / "issues.csv", index=False
    )
    valid = report.loc[report["status"] == "ok"]
    def total(field: str) -> int:
        return int(valid[field].sum()) if len(valid) else 0
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "first_date": first_date, "last_date": last_date,
        "selected_symbols": len(symbols), "audited_symbols": len(valid),
        "active_daily_days": total("active_daily_days"),
        "complete_minute_days": total("complete_days"),
        "missing_active_days": total("missing_active_days"),
        "partial_days": total("partial_days"),
        "suspension_placeholder_days": total("suspension_placeholder_days"),
        "unexpected_nontrading_minute_days": total("unexpected_nontrading_minute_days"),
        "invalid_rows": total("invalid_rows"),
        "duplicate_rows": total("duplicate_rows"),
        "ohlc_mismatch_days": total("ohlc_mismatch_days"),
        "opening_only_mismatch_days": total("opening_only_mismatch_days"),
        "traded_auction_open_mismatch_days": total("traded_auction_open_mismatch_days"),
        "unexplained_ohlc_mismatch_days": total("unexplained_ohlc_mismatch_days"),
        "missing_symbols": report.loc[report["status"] != "ok", "code"].tolist(),
    }
    summary["complete_ratio_among_audited"] = (
        round(summary["complete_minute_days"] / summary["active_daily_days"], 6)
        if summary["active_daily_days"] else 0
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-root", type=Path, default=Path("data/hf/pilot"))
    parser.add_argument("--bao-root", type=Path, default=Path("data/baostock/market_2020_2026"))
    parser.add_argument("--output", type=Path, default=Path("data/research/market_audit"))
    args = parser.parse_args()
    print(json.dumps(audit(args.hf_root, args.bao_root, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
