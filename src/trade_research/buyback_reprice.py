"""Verify and stress the frozen buyback pairs with original minute files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .buyback_eval import _score, _summary
from .buyback_inputs import CONTROL, EVENT
from .strategy_scan import _stressed_returns


def _exact(stored: pd.DataFrame, raw: pd.DataFrame) -> int:
    fields = ["date", "code", "horizon"]
    comparison = stored.merge(raw, on=fields, suffixes=("_stored", "_raw"),
                              validate="one_to_one")
    if len(comparison) != len(stored):
        raise ValueError("Stored 100k buyback outcomes lack raw-minute matches")
    for field in ("entry_status", "exit_status", "exit_date",
                  "quality_clean_exit"):
        if not comparison[f"{field}_stored"].fillna("missing").eq(
                comparison[f"{field}_raw"].fillna("missing")).all():
            raise ValueError(f"Buyback stored/raw {field} differs")
    for field in ("entry_price", "shares", "exit_delay_sessions",
                  "exit_price", "net_return"):
        left = comparison[f"{field}_stored"].to_numpy(dtype=float)
        right = comparison[f"{field}_raw"].to_numpy(dtype=float)
        if not np.isclose(left, right, atol=1e-12, equal_nan=True).all():
            raise ValueError(f"Buyback stored/raw {field} differs")
    return len(comparison)


def evaluate(selected_path: Path, repriced_path: Path,
             outcome_dir: Path, issues_dir: Path,
             report_path: Path) -> dict:
    membership = pd.read_parquet(selected_path)
    raw = pd.read_parquet(repriced_path)
    expected = len(membership) * 4
    if (len(raw) != expected
            or set(raw.target_notional) != {20_000.0, 100_000.0}
            or set(raw.horizon) != {1, 5}
            or raw.duplicated(["date", "code", "target_notional",
                               "horizon"]).any()):
        raise ValueError("Incomplete original-minute buyback repricing")
    stored = pd.concat([
        _score(membership, outcome_dir, issues_dir, horizon)
        for horizon in (1, 5)
    ], ignore_index=True)
    exact = _exact(stored, raw.loc[raw.target_notional.eq(100_000)])
    rows = raw.merge(membership[["date", "code", "candidate", "pair_code"]],
                     on=["date", "code"], validate="many_to_one")
    if len(rows) != len(raw) or set(rows.candidate) != {EVENT, CONTROL}:
        raise ValueError("Raw-minute trade is outside frozen event pairs")
    rows["cash_return"] = np.where(rows.quality_clean_exit,
                                   rows.net_return, 0.0)
    rows["cash_stress10"] = 0.0
    clean = rows.quality_clean_exit
    rows.loc[clean, "cash_stress10"] = _stressed_returns(rows.loc[clean], 10)
    if (not np.isfinite(rows.cash_return).all()
            or not np.isfinite(rows.cash_stress10).all()):
        raise ValueError("Nonfinite original-minute cash return")
    report = {"exact_100k_rows": exact, "results": {},
              "note": "Original-minute 2024/2025 only; 2025 not blind"}
    for size in (20_000, 100_000):
        report["results"][str(size)] = {}
        for horizon in (1, 5):
            sample = rows.loc[rows.target_notional.eq(size)
                              & rows.horizon.eq(horizon)]
            report["results"][str(size)][str(horizon)] = {
                year: {
                    segment: _summary(sample, year, segment)
                    for segment in ("full", "H1", "H2",
                                    "without_peak_month")
                } for year in ("2024", "2025")
            }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", type=Path, default=Path(
        "data/research/buyback/pairs.parquet"))
    parser.add_argument("--repriced", type=Path, default=Path(
        "data/research/buyback/repriced.parquet"))
    parser.add_argument("--outcomes", type=Path, default=Path(
        "data/research/market_outcomes_ci"))
    parser.add_argument("--issues", type=Path, default=Path(
        "data/research/market_issues_ci"))
    parser.add_argument("--report", type=Path, default=Path(
        "data/research/buyback/reprice_report.json"))
    args = parser.parse_args()
    report = evaluate(args.selected, args.repriced, args.outcomes,
                      args.issues, args.report)
    print({"exact_100k_rows": report["exact_100k_rows"],
           "main_20k_t5": report["results"]["20000"]["5"]})


if __name__ == "__main__":
    main()
