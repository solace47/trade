"""Score frozen major-holder-sale plans using original minute executions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .buyback_eval import _score, _summary
from .buyback_inputs import CONTROL
from .buyback_reprice import _exact
from .insider_sell_inputs import EVENT
from .strategy_scan import _stressed_returns


PEAK = {"2024": "2024-10", "2025": "2025-09"}


def _check_pairs(pairs: pd.DataFrame, expected: dict) -> None:
    if (pairs.empty or len(pairs) != 2 * expected["matched_pairs"]
            or pairs.duplicated(["date", "code"]).any()
            or set(pairs.candidate) != {EVENT, CONTROL}
            or not pairs.date.str[:4].isin(("2024", "2025")).all()):
        raise ValueError("Major-holder-sale pair membership changed")
    groups = pairs.groupby(["date", "pair_code"]).candidate.agg(
        ["size", "nunique"])
    if (len(groups) != expected["matched_pairs"]
            or not groups["size"].eq(2).all()
            or not groups["nunique"].eq(2).all()):
        raise ValueError("Major-holder-sale pair lacks one same-day peer")
    event = pairs.loc[pairs.candidate.eq(EVENT)]
    if (event.notice_date.isna().any()
            or not event.date.gt(event.notice_date).all()
            or event.groupby("date").size().gt(5).any()
            or event.date.nunique() != expected["matched_days"]):
        raise ValueError("Major-holder-sale event timing or capacity changed")
    for year in ("2024", "2025"):
        period = event.loc[event.date.str.startswith(year)]
        if (len(period) != expected["by_year"][year]["pairs"]
                or period.date.nunique() != expected["by_year"][year]["days"]):
            raise ValueError("Major-holder-sale frozen year sample changed")


def _membership(raw: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    joined = raw.merge(
        pairs[["date", "code", "candidate", "pair_code", "notice_date"]],
        on=["date", "code"], validate="many_to_one")
    if len(joined) != len(pairs) * 4:
        raise ValueError("Raw-minute major-holder-sale legs incomplete")
    joined["cash_return"] = np.where(joined.quality_clean_exit,
                                     joined.net_return, 0.0)
    joined["cash_stress10"] = 0.0
    clean = joined.quality_clean_exit
    joined.loc[clean, "cash_stress10"] = _stressed_returns(joined.loc[clean], 10)
    if (not np.isfinite(joined.cash_return).all()
            or not np.isfinite(joined.cash_stress10).all()):
        raise ValueError("Nonfinite original-minute sale-plan cash return")
    return joined


def _segments(rows: pd.DataFrame, year: str) -> dict:
    return {
        "full": _summary(rows, year, "full", EVENT),
        "H1": _summary(rows, year, "H1", EVENT),
        "H2": _summary(rows, year, "H2", EVENT),
        "without_peak_month": _summary(
            rows.loc[~rows.date.str.startswith(PEAK[year])],
            year, "full", EVENT),
    }


def evaluate(source_dir: Path, raw_path: Path, outcome_dir: Path,
             issues_dir: Path, report_path: Path) -> dict:
    audit = json.loads((source_dir / "input_audit.json").read_text(
        encoding="utf-8"))
    if audit.get("note") != (
            "Major holder sale plan inputs only; no future returns opened"):
        raise ValueError("Major-holder-sale input list was not frozen")
    main = pd.read_parquet(source_dir / "pairs.parquet")
    industry = pd.read_parquet(source_dir / "industry_pairs.parquet")
    _check_pairs(main, audit["main"])
    _check_pairs(industry, audit["same_industry"])
    unique = pd.concat([main, industry], ignore_index=True).drop_duplicates(
        ["date", "code"])
    raw = pd.read_parquet(raw_path)
    if (len(raw) != len(unique) * 4
            or set(raw.target_notional) != {20_000.0, 100_000.0}
            or set(raw.horizon) != {1, 5}
            or raw.duplicated(["date", "code", "target_notional",
                               "horizon"]).any()):
        raise ValueError("Raw-minute major-holder-sale list is incomplete")
    stored = pd.concat([
        _score(unique, outcome_dir, issues_dir, horizon)
        for horizon in (1, 5)
    ], ignore_index=True)
    exact = _exact(stored, raw.loc[raw.target_notional.eq(100_000)])
    main_raw = _membership(raw, main)
    within_raw = _membership(raw, industry)
    report = {
        "note": "Exploratory 2024/2025; 2025 not blind; 2026 not read",
        "exact_100k_rows": exact,
        "main": {}, "same_industry": {},
    }
    for size in (20_000, 100_000):
        report["main"][str(size)] = {}
        report["same_industry"][str(size)] = {}
        for horizon in (1, 5):
            selected = main_raw.loc[main_raw.target_notional.eq(size)
                                    & main_raw.horizon.eq(horizon)]
            within = within_raw.loc[within_raw.target_notional.eq(size)
                                    & within_raw.horizon.eq(horizon)]
            report["main"][str(size)][str(horizon)] = {
                year: _segments(selected, year) for year in ("2024", "2025")
            }
            report["same_industry"][str(size)][str(horizon)] = {
                year: _summary(within, year, "full", EVENT)
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
        "data/research/insider_sell/repriced.parquet"))
    parser.add_argument("--outcomes", type=Path, default=Path(
        "data/research/market_outcomes_ci"))
    parser.add_argument("--issues", type=Path, default=Path(
        "data/research/market_issues_ci"))
    parser.add_argument("--report", type=Path, default=Path(
        "data/research/insider_sell/report.json"))
    args = parser.parse_args()
    report = evaluate(args.source, args.raw, args.outcomes,
                      args.issues, args.report)
    print({"exact_100k_rows": report["exact_100k_rows"],
           "main_20k_t5": {
               year: report["main"]["20000"]["5"][year]["full"]
               for year in ("2024", "2025")}})


if __name__ == "__main__":
    main()
