"""Score frozen insider-plan pairs on 2024/2025 minute outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .buyback_eval import _score, _summary
from .buyback_inputs import CONTROL
from .insider_plan_inputs import EVENT


def _check_pairs(pairs: pd.DataFrame, expected: dict) -> None:
    if (pairs.empty or len(pairs) != 2 * expected["matched_pairs"]
            or pairs.duplicated(["date", "code"]).any()
            or set(pairs.candidate) != {EVENT, CONTROL}
            or not pairs.date.str[:4].isin(("2024", "2025")).all()):
        raise ValueError("Insider-plan membership is not frozen or complete")
    groups = pairs.groupby(["date", "pair_code"]).candidate.agg(
        ["size", "nunique"])
    if (len(groups) != expected["matched_pairs"]
            or not groups["size"].eq(2).all()
            or not groups["nunique"].eq(2).all()):
        raise ValueError("Insider-plan event lacks one same-day peer")
    treated = pairs.loc[pairs.candidate.eq(EVENT)]
    if (treated.notice_date.isna().any()
            or not treated.date.gt(treated.notice_date).all()
            or treated.groupby("date").size().gt(5).any()
            or treated.date.nunique() != expected["matched_days"]):
        raise ValueError("Insider-plan notice or capacity timing failed")
    for year in ("2024", "2025"):
        period = treated.loc[treated.date.str.startswith(year)]
        if (len(period) != expected["by_year"][year]["pairs"]
                or period.date.nunique() != expected["by_year"][year]["days"]):
            raise ValueError("Insider-plan pair list changed by year")


def _loan_tags(pairs: pd.DataFrame, source_dir: Path) -> pd.DataFrame:
    titles = pd.concat([
        pd.read_parquet(source_dir / f"title_candidates_{year}.parquet")
        for year in (2024, 2025)
    ], ignore_index=True)[["pdf_url", "title"]]
    if titles.pdf_url.duplicated().any():
        raise ValueError("Insider-plan title index is not unique")
    events = pairs.loc[pairs.candidate.eq(EVENT),
                        ["date", "pair_code", "pdf_url"]]
    tagged = events.merge(titles, on="pdf_url", how="left",
                          validate="many_to_one")
    if tagged.title.isna().any():
        raise ValueError("Insider-plan PDF lacks title")
    tagged["loan_title"] = tagged.title.str.contains("贷款|金融机构")
    return tagged[["date", "pair_code", "loan_title"]]


def _tag_scored(scored: pd.DataFrame, tags: pd.DataFrame) -> pd.DataFrame:
    tagged = scored.merge(tags, on=["date", "pair_code"],
                          validate="many_to_one")
    if len(tagged) != len(scored):
        raise ValueError("Insider-plan scored leg lacks title tag")
    return tagged


def _sections(rows: pd.DataFrame, year: str) -> dict:
    result = {segment: _summary(rows, year, segment, EVENT)
              for segment in ("full", "H1", "H2", "without_peak_month")}
    result["without_loan_title"] = _summary(
        rows.loc[~rows.loan_title], year, "full", EVENT)
    return result


def evaluate(source_dir: Path, outcome_dir: Path, issues_dir: Path,
             report_path: Path, trades_path: Path) -> dict:
    audit = json.loads((source_dir / "input_audit.json").read_text(
        encoding="utf-8"))
    if audit.get("note") != (
            "Insider increase plan inputs only; no future returns opened"):
        raise ValueError("Insider-plan input list was not frozen")
    main = pd.read_parquet(source_dir / "pairs.parquet")
    industry = pd.read_parquet(source_dir / "industry_pairs.parquet")
    _check_pairs(main, audit["main"])
    _check_pairs(industry, audit["same_industry"])
    scored = _tag_scored(_score(main, outcome_dir, issues_dir, 5),
                         _loan_tags(main, source_dir))
    within = _tag_scored(_score(industry, outcome_dir, issues_dir, 5),
                         _loan_tags(industry, source_dir))
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
    parser.add_argument("--source", type=Path, default=Path(
        "data/research/insider_buy"))
    parser.add_argument("--outcomes", type=Path, default=Path(
        "data/research/market_outcomes_ci"))
    parser.add_argument("--issues", type=Path, default=Path(
        "data/research/market_issues_ci"))
    parser.add_argument("--report", type=Path, default=Path(
        "data/research/insider_buy/report.json"))
    parser.add_argument("--trades", type=Path, default=Path(
        "data/research/insider_buy/trades.parquet"))
    args = parser.parse_args()
    report = evaluate(args.source, args.outcomes, args.issues,
                      args.report, args.trades)
    print({year: report["main"][year]["full"]
           for year in ("2024", "2025")})


if __name__ == "__main__":
    main()
