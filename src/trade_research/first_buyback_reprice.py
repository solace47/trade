"""Verify frozen first-buyback pairs against original minute executions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .buyback_eval import _score, _summary
from .buyback_inputs import CONTROL
from .buyback_reprice import _exact
from .first_buyback_inputs import EVENT
from .strategy_scan import _stressed_returns


def _sections(rows: pd.DataFrame, year: str) -> dict:
    result = {
        segment: _summary(rows, year, segment, EVENT)
        for segment in ("full", "H1", "H2", "without_peak_month")
    }
    result["without_recent_plan"] = _summary(
        rows.loc[~rows.recent_plan_10], year, "full", EVENT)
    if year == "2024":
        result["without_january"] = _summary(
            rows.loc[~rows.date.str.startswith("2024-01")],
            year, "full", EVENT)
    return result


def evaluate(source_dir: Path, raw_path: Path, outcome_dir: Path,
             issues_dir: Path, report_path: Path) -> dict:
    membership = pd.read_parquet(source_dir / "pairs.parquet")
    raw = pd.read_parquet(raw_path)
    if (membership.empty or len(raw) != len(membership) * 4
            or set(raw.target_notional) != {20_000.0, 100_000.0}
            or set(raw.horizon) != {1, 5}
            or raw.duplicated(["date", "code", "target_notional",
                               "horizon"]).any()):
        raise ValueError("Incomplete original-minute first-buyback repricing")
    stored = pd.concat([
        _score(membership, outcome_dir, issues_dir, horizon)
        for horizon in (1, 5)
    ], ignore_index=True)
    exact = _exact(stored, raw.loc[raw.target_notional.eq(100_000)])
    rows = raw.merge(
        membership[["date", "code", "candidate", "pair_code",
                    "recent_plan_10"]],
        on=["date", "code"], validate="many_to_one")
    if len(rows) != len(raw) or set(rows.candidate) != {EVENT, CONTROL}:
        raise ValueError("Original-minute first-buyback leg is not frozen")
    rows["cash_return"] = np.where(rows.quality_clean_exit,
                                   rows.net_return, 0.0)
    rows["cash_stress10"] = 0.0
    clean = rows.quality_clean_exit
    rows.loc[clean, "cash_stress10"] = _stressed_returns(rows.loc[clean], 10)
    if (not np.isfinite(rows.cash_return).all()
            or not np.isfinite(rows.cash_stress10).all()):
        raise ValueError("Nonfinite original-minute first-buyback cash return")
    report = {
        "exact_100k_rows": exact,
        "note": "Original minutes 2024/2025 only; 2025 not blind",
        "results": {},
    }
    for size in (20_000, 100_000):
        report["results"][str(size)] = {}
        for horizon in (1, 5):
            frame = rows.loc[rows.target_notional.eq(size)
                             & rows.horizon.eq(horizon)]
            report["results"][str(size)][str(horizon)] = {
                year: _sections(frame, year)
                for year in ("2024", "2025")
            }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("data/research/first_buyback"))
    parser.add_argument("--raw", type=Path,
                        default=Path("data/research/first_buyback/repriced.parquet"))
    parser.add_argument("--outcomes", type=Path,
                        default=Path("data/research/market_outcomes_ci"))
    parser.add_argument("--issues", type=Path,
                        default=Path("data/research/market_issues_ci"))
    parser.add_argument("--report", type=Path,
                        default=Path("data/research/first_buyback/reprice_report.json"))
    args = parser.parse_args()
    report = evaluate(args.source, args.raw, args.outcomes, args.issues,
                      args.report)
    print({"exact_100k_rows": report["exact_100k_rows"],
           "main_20k_t5": report["results"]["20000"]["5"]})


if __name__ == "__main__":
    main()
