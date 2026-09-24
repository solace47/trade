"""Score the frozen first-actual-buyback pairs on 2024/2025 minute outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .buyback_eval import _score, _summary
from .buyback_inputs import CONTROL
from .first_buyback_inputs import EVENT


def _check_pairs(pairs: pd.DataFrame, expected: dict) -> None:
    if (pairs.empty or len(pairs) != 2 * expected["matched_pairs"]
            or pairs.duplicated(["date", "code"]).any()
            or set(pairs.candidate) != {EVENT, CONTROL}
            or not pairs.date.str[:4].isin(("2024", "2025")).all()):
        raise ValueError("First-buyback membership is not frozen or complete")
    groups = pairs.groupby(["date", "pair_code"]).candidate.agg(
        ["size", "nunique"])
    if (len(groups) != expected["matched_pairs"]
            or not groups["size"].eq(2).all()
            or not groups["nunique"].eq(2).all()
            or pairs.groupby(["date", "pair_code"]).recent_plan_10.nunique().ne(1).any()):
        raise ValueError("First-buyback event lacks one same-day peer")
    treated = pairs.loc[pairs.candidate.eq(EVENT)]
    if (treated.notice_date.isna().any()
            or not treated.date.gt(treated.notice_date).all()
            or treated.groupby("date").size().gt(5).any()
            or treated.date.nunique() != expected["matched_days"]):
        raise ValueError("First-buyback notice or capacity timing failed")
    for year in ("2024", "2025"):
        period = treated.loc[treated.date.str.startswith(year)]
        if (len(period) != expected["by_year"][year]["pairs"]
                or period.date.nunique() != expected["by_year"][year]["days"]):
            raise ValueError("First-buyback pair list changed by year")


def _attach_overlap(scored: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    tagged = scored.merge(
        pairs[["date", "code", "recent_plan_10"]],
        on=["date", "code"], validate="one_to_one")
    if len(tagged) != len(scored):
        raise ValueError("Scored first-buyback row lacks a prior-plan tag")
    return tagged


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


def evaluate(source_dir: Path, outcome_dir: Path, issues_dir: Path,
             report_path: Path, trades_path: Path) -> dict:
    audit = json.loads((source_dir / "input_audit.json").read_text(
        encoding="utf-8"))
    if audit.get("note") != "First actual buyback inputs only; no future returns opened":
        raise ValueError("First-buyback input list was not frozen")
    main = pd.read_parquet(source_dir / "pairs.parquet")
    industry = pd.read_parquet(source_dir / "industry_pairs.parquet")
    _check_pairs(main, audit["main"])
    _check_pairs(industry, audit["same_industry"])
    scored = _attach_overlap(_score(main, outcome_dir, issues_dir, 5), main)
    within = _attach_overlap(_score(industry, outcome_dir, issues_dir, 5),
                             industry)
    report = {
        "note": "Exploratory 2024/2025; 2025 not blind; 2026 not read",
        "main": {year: _sections(scored, year)
                 for year in ("2024", "2025")},
        "same_industry": {
            year: _summary(within, year, "full", EVENT)
            for year in ("2024", "2025")},
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    trades_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    scored.to_parquet(trades_path, index=False, compression="zstd")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("data/research/first_buyback"))
    parser.add_argument("--outcomes", type=Path,
                        default=Path("data/research/market_outcomes_ci"))
    parser.add_argument("--issues", type=Path,
                        default=Path("data/research/market_issues_ci"))
    parser.add_argument("--report", type=Path,
                        default=Path("data/research/first_buyback/report.json"))
    parser.add_argument("--trades", type=Path,
                        default=Path("data/research/first_buyback/trades.parquet"))
    args = parser.parse_args()
    report = evaluate(args.source, args.outcomes, args.issues,
                      args.report, args.trades)
    print({year: report["main"][year]["full"] for year in ("2024", "2025")})


if __name__ == "__main__":
    main()
