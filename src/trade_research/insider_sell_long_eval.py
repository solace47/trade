"""Evaluate the frozen exploratory T+20 major-holder-sale follow-up."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .buyback_eval import _summary
from .buyback_inputs import CONTROL
from .insider_sell_eval import _segments
from .insider_sell_inputs import EVENT
from .strategy_scan import _stressed_returns


def _check(pairs: pd.DataFrame, counts: dict) -> None:
    if (pairs.empty or pairs.duplicated(["date", "code"]).any()
            or set(pairs.candidate) != {EVENT, CONTROL}):
        raise ValueError("Long-horizon sale-plan membership malformed")
    groups = pairs.groupby(["date", "pair_code"]).candidate.agg(
        ["size", "nunique"])
    if not groups["size"].eq(2).all() or not groups["nunique"].eq(2).all():
        raise ValueError("Long-horizon sale-plan pair is incomplete")
    treated = pairs.loc[pairs.candidate.eq(EVENT)]
    if (not treated.date.gt(treated.notice_date).all()
            or treated.groupby("date").size().gt(5).any()):
        raise ValueError("Long-horizon sale-plan timing changed")
    for year in ("2024", "2025"):
        if len(treated.loc[treated.date.str.startswith(year)]) != counts[year]:
            raise ValueError("Long-horizon sale-plan sample changed by year")


def _score(raw: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    joined = raw.merge(pairs[["date", "code", "candidate", "pair_code"]],
                       on=["date", "code"], validate="many_to_one")
    if len(joined) != len(pairs) * 2:
        raise ValueError("T+20 original-minute pair leg is absent")
    joined["cash_return"] = np.where(joined.quality_clean_exit,
                                     joined.net_return, 0.0)
    joined["cash_stress10"] = 0.0
    clean = joined.quality_clean_exit
    joined.loc[clean, "cash_stress10"] = _stressed_returns(joined.loc[clean], 10)
    if (not np.isfinite(joined.cash_return).all()
            or not np.isfinite(joined.cash_stress10).all()):
        raise ValueError("T+20 original-minute return is nonfinite")
    return joined


def evaluate(source_dir: Path, raw_path: Path, report_path: Path) -> dict:
    audit = json.loads((source_dir / "long_input_audit.json").read_text(
        encoding="utf-8"))
    if (audit.get("note") != "Post-T+5 mechanism follow-up; inputs only"
            or audit.get("horizon") != 20 or audit.get("max_exit_delay") != 5):
        raise ValueError("T+20 sale-plan inputs were not frozen")
    main = pd.read_parquet(source_dir / "long_pairs.parquet")
    industry = pd.read_parquet(source_dir / "long_industry_pairs.parquet")
    _check(main, audit["main"])
    _check(industry, audit["same_industry"])
    raw = pd.read_parquet(raw_path)
    keys = pd.concat([main, industry], ignore_index=True).drop_duplicates(
        ["date", "code"])
    if (len(raw) != 2 * len(keys)
            or len(keys) != audit["unique_stock_days"]
            or set(raw.target_notional) != {20_000.0, 100_000.0}
            or set(raw.horizon) != {20}
            or raw.duplicated(["date", "code", "target_notional"]).any()
            or raw.loc[raw.exit_date.notna()].apply(
                lambda row: row.date[:4] != row.exit_date[:4], axis=1).any()):
        raise ValueError("T+20 original-minute follow-up is incomplete")
    treated = _score(raw, main)
    within = _score(raw, industry)
    report = {"note": "Post-T+5 exploratory 2024/2025; 2026 not read",
              "unique_stock_days": len(keys), "main": {}, "same_industry": {}}
    for size in (20_000, 100_000):
        selected = treated.loc[treated.target_notional.eq(size)]
        industry_selected = within.loc[within.target_notional.eq(size)]
        report["main"][str(size)] = {
            year: _segments(selected, year) for year in ("2024", "2025")
        }
        report["same_industry"][str(size)] = {
            year: _summary(industry_selected, year, "full", EVENT)
            for year in ("2024", "2025")
        }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(
        "data/research/insider_sell"))
    parser.add_argument("--raw", type=Path, default=Path(
        "data/research/insider_sell/long_repriced.parquet"))
    parser.add_argument("--report", type=Path, default=Path(
        "data/research/insider_sell/long_report.json"))
    args = parser.parse_args()
    report = evaluate(args.source, args.raw, args.report)
    print({"main_20k": {year: report["main"]["20000"][year]["full"]
                         for year in ("2024", "2025")},
           "unique_stock_days": report["unique_stock_days"]})


if __name__ == "__main__":
    main()
