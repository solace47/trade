"""Recheck source-wide quality flags within an explicit research period.

The market audit conservatively flags a whole symbol if it has a severe
minute-source problem anywhere in 2022–2026. A defect outside the evaluated
years must not silently remove its otherwise valid 2024–2025 observations.
This audit only revisits those globally flagged symbols; ordinary bad-day
keys remain a separate exclusion at evaluation time.
"""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path

import pandas as pd

from .hf_audit import audit_symbol
from .market_study import BAD_STOCK_FIELDS, _quality_symbols


ROOT = Path("data/research")


def audit_period(issues_dir: Path, minute_root: Path, daily_root: Path,
                 first_date: str, last_date: str, output: Path) -> dict:
    if date.fromisoformat(first_date) > date.fromisoformat(last_date):
        raise ValueError("First date must not follow last date")
    issue_files = sorted(issues_dir.glob("shard_*.csv"))
    source_audits = sorted((issues_dir.parent / "market_audit_ci").glob(
        "shard_*.json"))
    if (len(issue_files) != 20 or len(source_audits) != 20
            or {path.stem for path in issue_files}
            != {path.stem for path in source_audits}):
        raise FileNotFoundError("The 20 source-wide audit shards are required")
    for path in source_audits:
        source = json.loads(path.read_text(encoding="utf-8"))
        if source["first_date"] > first_date or source["last_date"] < last_date:
            raise ValueError("Requested period exceeds the source-wide audit")
    globally_flagged = _quality_symbols(issues_dir).code.tolist()
    rows = []
    period_bad = []
    for code in globally_flagged:
        exchange, symbol = code.split(".")
        minute_file = minute_root / exchange.upper() / f"{symbol}.parquet"
        daily_file = daily_root / f"{exchange}_{symbol}.parquet"
        audit, _ = audit_symbol(code, minute_file, daily_file,
                                first_date, last_date)
        severe = audit.get("status") != "ok" or any(
            int(audit.get(field, 0)) > 0 for field in BAD_STOCK_FIELDS
        )
        if severe:
            period_bad.append(code)
        rows.append({"code": code, "severe_in_period": bool(severe),
                     "status": audit.get("status"),
                     "severe_fields": {
                         field: int(audit.get(field, 0))
                         for field in BAD_STOCK_FIELDS
                     }})
    report = {
        "first_date": first_date, "last_date": last_date,
        "globally_flagged_count": len(globally_flagged),
        "period_bad_symbols": period_bad,
        "period_bad_count": len(period_bad),
        "releasable_count": len(globally_flagged) - len(period_bad),
        "rows": rows,
        "note": "Keep excluding issue-day keys independently; no returns were read",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return report


def load_period_bad_symbols(report_path: Path, first_date: str,
                            last_date: str) -> pd.DataFrame:
    """Return flags only if an audit covers the entire requested interval."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (date.fromisoformat(report["first_date"]) > date.fromisoformat(first_date)
            or date.fromisoformat(report["last_date"]) < date.fromisoformat(last_date)):
        raise ValueError("Period audit does not cover the evaluation interval")
    return pd.DataFrame({"code": pd.Series(report["period_bad_symbols"],
                                            dtype="string")})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issues", type=Path,
                        default=ROOT / "market_issues_ci")
    parser.add_argument("--minute-root", type=Path,
                        default=Path("data/hf/pilot/data/stock_1m"))
    parser.add_argument("--daily-root", type=Path,
                        default=Path("data/baostock/market_2020_2026/daily"))
    parser.add_argument("--first-date", default="2024-01-01")
    parser.add_argument("--last-date", default="2025-12-31")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "quality_period_2024_2025.json")
    args = parser.parse_args()
    report = audit_period(args.issues, args.minute_root, args.daily_root,
                          args.first_date, args.last_date, args.output)
    print(json.dumps({key: report[key] for key in (
        "first_date", "last_date", "globally_flagged_count",
        "period_bad_count", "releasable_count",
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
