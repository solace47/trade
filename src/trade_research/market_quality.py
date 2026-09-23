"""Summarize imported market shards and identify unusable stock histories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


TOTAL_FIELDS = (
    "selected_symbols", "audited_symbols", "active_daily_days", "complete_minute_days",
    "missing_active_days", "partial_days", "invalid_rows", "duplicate_rows",
    "wrong_identity_rows", "volume_over_100_shares_days",
    "amount_over_0_01pct_days",
    "unexpected_nontrading_minute_days", "active_no_trade_days",
    "ohlc_mismatch_days", "opening_only_mismatch_days",
    "unexplained_ohlc_mismatch_days",
)
STOCK_RISK_FIELDS = (
    "invalid_rows", "duplicate_rows", "wrong_identity_rows",
    "volume_over_100_shares_days", "amount_over_0_01pct_days",
    "active_no_trade_days",
)


def summarize(audit_dir: Path, symbols_dir: Path, output: Path,
              allow_partial: bool = False) -> dict:
    audit_paths = sorted(audit_dir.glob("shard_*.json"))
    shards = [path.stem.split("_")[1] for path in audit_paths]
    if not allow_partial and len(shards) != 20:
        raise ValueError(f"Expected 20 market shards; found {len(shards)}")
    if not audit_paths:
        raise FileNotFoundError("No imported audit summaries")
    all_symbols = []
    stocks = []
    totals = {field: 0 for field in TOTAL_FIELDS}
    shard_reports = []
    for path in audit_paths:
        shard = path.stem.split("_")[1]
        symbol_path = symbols_dir / f"shard_{shard}.parquet"
        if not symbol_path.exists():
            raise FileNotFoundError(symbol_path)
        codes = pd.read_parquet(symbol_path, columns=["code"])["code"].tolist()
        all_symbols.extend(codes)
        audit = json.loads(path.read_text(encoding="utf-8"))
        stock_path = path.resolve().parent / "stocks.csv"
        frame = pd.read_csv(stock_path, dtype={"code": str})
        if set(frame["code"]) != set(codes):
            raise ValueError(f"Stock audit and selected universe disagree for shard {shard}")
        for field in TOTAL_FIELDS:
            if field in audit:
                totals[field] += int(audit[field])
            elif field in frame:
                totals[field] += int(frame[field].fillna(0).sum())
        for field in STOCK_RISK_FIELDS:
            if field in frame:
                affected = frame.loc[frame[field].fillna(0).gt(0), "code"].tolist()
                for code in affected:
                    stocks.append({"code": code, "shard": int(shard), "problem": field})
        shard_reports.append({
            "shard": int(shard), "symbols": len(codes),
            "active_days": audit["active_daily_days"],
            "complete_days": audit["complete_minute_days"],
            "unexplained_price_days": audit["unexplained_ohlc_mismatch_days"],
            "missing_symbols": audit["missing_symbols"],
        })
    duplicate_symbols = sorted(pd.Series(all_symbols).loc[
        pd.Series(all_symbols).duplicated()].unique().tolist())
    result = {
        "shards": len(shards), "selected_unique_symbols": len(set(all_symbols)),
        "duplicate_symbols": duplicate_symbols, "totals": totals,
        "complete_ratio_among_audited": (
            totals["complete_minute_days"] / totals["active_daily_days"]
            if totals["active_daily_days"] else 0
        ),
        "stock_quality_problems": stocks,
        "shard_reports": shard_reports,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audits", type=Path,
                        default=Path("data/research/market_audit_ci"))
    parser.add_argument("--symbols", type=Path,
                        default=Path("data/research/market_symbols_ci"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/research/market_quality.json"))
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    result = summarize(args.audits, args.symbols, args.output, args.allow_partial)
    print(json.dumps({key: result[key] for key in (
        "shards", "selected_unique_symbols", "duplicate_symbols", "totals",
        "complete_ratio_among_audited",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
