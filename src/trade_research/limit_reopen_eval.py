"""Score frozen 14:50 failed limit-up pairs with raw-minute trades."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .buyback_eval import _score, _summary
from .buyback_inputs import CONTROL
from .buyback_reprice import _exact
from .limit_reopen_inputs import EVENT
from .strategy_scan import _stressed_returns


def _check(pairs: pd.DataFrame, expected: dict) -> None:
    if (pairs.empty or pairs.duplicated(["date", "code"]).any()
            or len(pairs) != 2 * expected["matched_pairs"]
            or set(pairs.candidate) != {EVENT, CONTROL}):
        raise ValueError("Frozen reopened-limit pair membership changed")
    group = pairs.groupby(["date", "pair_code"]).candidate.agg(
        ["size", "nunique"])
    if (not group["size"].eq(2).all()
            or not group["nunique"].eq(2).all()):
        raise ValueError("Reopened-limit event lacks exactly one control")
    event = pairs.loc[pairs.candidate.eq(EVENT)]
    if (event.date.nunique() != expected["matched_days"]
            or event.groupby("date").size().gt(5).any()
            or event.date.str[:4].isin(("2024", "2025")).eq(False).any()
            or event.date.str[5:].gt("12-17").any()):
        raise ValueError("Reopened-limit date or capacity changed")
    for year in ("2024", "2025"):
        part = event.loc[event.date.str.startswith(year)]
        if (len(part) != expected["by_year"][year]["pairs"]
                or part.date.nunique() != expected["by_year"][year]["days"]):
            raise ValueError("Reopened-limit yearly sample changed")


def _attach(raw: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    rows = raw.merge(pairs[["date", "code", "candidate", "pair_code"]],
                     on=["date", "code"], validate="many_to_one")
    if len(rows) != 4 * len(pairs):
        raise ValueError("Raw-minute reopened-limit leg is absent")
    rows["cash_return"] = np.where(rows.quality_clean_exit,
                                   rows.net_return, 0.0)
    rows["cash_stress10"] = 0.0
    clean = rows.quality_clean_exit
    rows.loc[clean, "cash_stress10"] = _stressed_returns(rows.loc[clean], 10)
    if (not np.isfinite(rows.cash_return).all()
            or not np.isfinite(rows.cash_stress10).all()):
        raise ValueError("Reopened-limit cash return is nonfinite")
    return rows


def _sections(rows: pd.DataFrame, year: str) -> dict:
    return {name: _summary(rows, year, name, EVENT)
            for name in ("full", "H1", "H2")}


def evaluate(source_dir: Path, raw_path: Path, outcome_dir: Path,
             issues_dir: Path, report_path: Path) -> dict:
    audit = json.loads((source_dir / "input_audit.json").read_text(
        encoding="utf-8"))
    if audit.get("note") != (
            "As-of-14:50 limit-reopen inputs only; no future outcomes opened"):
        raise ValueError("Reopened-limit input list was not frozen")
    main = pd.read_parquet(source_dir / "pairs.parquet")
    industry = pd.read_parquet(source_dir / "industry_pairs.parquet")
    _check(main, audit["main"])
    _check(industry, audit["same_industry"])
    unique = pd.concat([main, industry], ignore_index=True).drop_duplicates(
        ["date", "code"])
    raw = pd.read_parquet(raw_path)
    if (len(raw) != len(unique) * 4
            or set(raw.target_notional) != {20_000.0, 100_000.0}
            or set(raw.horizon) != {1, 5}
            or raw.duplicated(["date", "code", "target_notional",
                               "horizon"]).any()):
        raise ValueError("Original-minute reopened-limit list incomplete")
    # _score's announcement-date field is unused by _exact; supply the signal
    # date solely to run the same archived 100k execution/quality comparison.
    stored = pd.concat([
        _score(unique.assign(notice_date=unique.date), outcome_dir,
               issues_dir, horizon)
        for horizon in (1, 5)
    ], ignore_index=True)
    exact = _exact(stored, raw.loc[raw.target_notional.eq(100_000)])
    selected = _attach(raw, main)
    within = _attach(raw, industry)
    report = {"note": "Exploratory 2024/2025; 2025 not blind; 2026 not read",
              "exact_100k_rows": exact,
              "main": {}, "same_industry": {}}
    for size in (20_000, 100_000):
        report["main"][str(size)] = {}
        report["same_industry"][str(size)] = {}
        for horizon in (1, 5):
            part = selected.loc[selected.target_notional.eq(size)
                                & selected.horizon.eq(horizon)]
            subset = within.loc[within.target_notional.eq(size)
                                & within.horizon.eq(horizon)]
            report["main"][str(size)][str(horizon)] = {
                year: _sections(part, year) for year in ("2024", "2025")
            }
            report["same_industry"][str(size)][str(horizon)] = {
                year: _summary(subset, year, "full", EVENT)
                for year in ("2024", "2025")
            }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(
        "data/research/limit_reopen"))
    parser.add_argument("--raw", type=Path, default=Path(
        "data/research/limit_reopen/repriced.parquet"))
    parser.add_argument("--outcomes", type=Path, default=Path(
        "data/research/market_outcomes_ci"))
    parser.add_argument("--issues", type=Path, default=Path(
        "data/research/market_issues_ci"))
    parser.add_argument("--report", type=Path, default=Path(
        "data/research/limit_reopen/report.json"))
    args = parser.parse_args()
    report = evaluate(args.source, args.raw, args.outcomes,
                      args.issues, args.report)
    print({"exact_100k_rows": report["exact_100k_rows"],
           "main_20k_t1": {
               year: report["main"]["20000"]["1"][year]["full"]
               for year in ("2024", "2025")}})


if __name__ == "__main__":
    main()
