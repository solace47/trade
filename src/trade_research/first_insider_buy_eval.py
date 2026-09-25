"""Score frozen first-insider-buy pairs from 2024/25 original minutes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .buyback_eval import _score, _summary
from .buyback_inputs import CONTROL
from .buyback_reprice import _exact
from .first_insider_buy_inputs import EVENT
from .strategy_scan import _stressed_returns


def _membership(pairs: pd.DataFrame, expected: dict) -> None:
    if (pairs.empty or len(pairs) != 2 * expected["matched_pairs"]
            or pairs.duplicated(["date", "code"]).any()
            or set(pairs.candidate) != {EVENT, CONTROL}
            or not pairs.date.str[:4].isin(("2024", "2025")).all()):
        raise ValueError("First-insider-buy membership changed after freezing")
    groups = pairs.groupby(["date", "pair_code"]).candidate.agg(
        ["size", "nunique"])
    if (len(groups) != expected["matched_pairs"]
            or not groups["size"].eq(2).all()
            or not groups["nunique"].eq(2).all()
            or pairs.groupby(["date", "pair_code"]).recent_plan_10
            .nunique().ne(1).any()):
        raise ValueError("First-insider-buy event lacks one same-day control")
    treated = pairs.loc[pairs.candidate.eq(EVENT)]
    if (treated.notice_date.isna().any()
            or not treated.date.gt(treated.notice_date).all()
            or treated.groupby("date").size().gt(5).any()
            or treated.date.nunique() != expected["matched_days"]):
        raise ValueError("First-insider-buy signal timing or capacity failed")
    for year in ("2024", "2025"):
        period = treated.loc[treated.date.str.startswith(year)]
        if (len(period) != expected["by_year"][year]["pairs"]
                or period.date.nunique() != expected["by_year"][year]["days"]):
            raise ValueError("First-insider-buy membership shifted by year")


def _raw_rows(pairs: pd.DataFrame, path: Path,
              outcome_dir: Path, issues_dir: Path) -> tuple[pd.DataFrame, int]:
    raw = pd.read_parquet(path)
    if (len(raw) != len(pairs) * 4
            or set(raw.target_notional) != {20_000.0, 100_000.0}
            or set(raw.horizon) != {1, 5}
            or raw.duplicated(["date", "code", "target_notional",
                               "horizon"]).any()):
        raise ValueError("Incomplete original-minute first-insider-buy repricing")
    stored = pd.concat([
        _score(pairs, outcome_dir, issues_dir, horizon)
        for horizon in (1, 5)
    ], ignore_index=True)
    exact = _exact(stored, raw.loc[raw.target_notional.eq(100_000)])
    tags = pairs[["date", "code", "candidate", "pair_code",
                  "related_actor_title", "prior_plan_notice",
                  "recent_plan_10", "buy_notice_lag_days"]]
    rows = raw.merge(tags, on=["date", "code"], validate="many_to_one")
    if len(rows) != len(raw) or set(rows.candidate) != {EVENT, CONTROL}:
        raise ValueError("Original-minute leg lacks its frozen event/control tag")
    rows["cash_return"] = np.where(rows.quality_clean_exit,
                                   rows.net_return, 0.0)
    rows["cash_stress10"] = 0.0
    clean = rows.quality_clean_exit
    rows.loc[clean, "cash_stress10"] = _stressed_returns(rows.loc[clean], 10)
    if (not np.isfinite(rows.cash_return).all()
            or not np.isfinite(rows.cash_stress10).all()):
        raise ValueError("Nonfinite first-insider-buy cash outcome")
    return rows, exact


def _sections(rows: pd.DataFrame, year: str) -> dict:
    result = {
        section: _summary(rows, year, section, EVENT)
        for section in ("full", "H1", "H2", "without_peak_month")
    }
    result["direct_actor_title_only"] = _summary(
        rows.loc[~rows.related_actor_title], year, "full", EVENT)
    result["without_recent_plan"] = _summary(
        rows.loc[~rows.recent_plan_10], year, "full", EVENT)
    result["known_prior_plan"] = _summary(
        rows.loc[rows.prior_plan_notice.notna()], year, "full", EVENT)
    result["without_long_notice_lag"] = _summary(
        rows.loc[rows.buy_notice_lag_days.le(7)], year, "full", EVENT)
    return result


def evaluate(source_dir: Path, outcome_dir: Path, issues_dir: Path,
             report_path: Path) -> dict:
    audit = json.loads((source_dir / "input_audit.json").read_text(
        encoding="utf-8"))
    if audit.get("note") != (
            "First actual insider buy inputs only; no future returns opened"):
        raise ValueError("First-insider-buy input list was not frozen")
    main = pd.read_parquet(source_dir / "pairs.parquet")
    industry = pd.read_parquet(source_dir / "industry_pairs.parquet")
    _membership(main, audit["main"])
    _membership(industry, audit["same_industry"])
    scored, exact_main = _raw_rows(
        main, source_dir / "repriced.parquet", outcome_dir, issues_dir)
    within, exact_industry = _raw_rows(
        industry, source_dir / "industry_repriced.parquet",
        outcome_dir, issues_dir)
    report = {"note": "2024/2025 exploratory; 2025 not blind; 2026 not read",
              "exact_100k_rows": exact_main + exact_industry,
              "main": {}, "same_industry": {}}
    for notional in (20_000, 100_000):
        report["main"][str(notional)] = {}
        report["same_industry"][str(notional)] = {}
        for horizon in (1, 5):
            sample = scored.loc[scored.target_notional.eq(notional)
                                & scored.horizon.eq(horizon)]
            peer = within.loc[within.target_notional.eq(notional)
                              & within.horizon.eq(horizon)]
            report["main"][str(notional)][str(horizon)] = {
                year: _sections(sample, year) for year in ("2024", "2025")}
            report["same_industry"][str(notional)][str(horizon)] = {
                year: _summary(peer, year, "full", EVENT)
                for year in ("2024", "2025")}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(
        "data/research/first_insider_buy"))
    parser.add_argument("--outcomes", type=Path, default=Path(
        "data/research/market_outcomes_ci"))
    parser.add_argument("--issues", type=Path, default=Path(
        "data/research/market_issues_ci"))
    parser.add_argument("--report", type=Path, default=Path(
        "data/research/first_insider_buy/report.json"))
    args = parser.parse_args()
    result = evaluate(args.source, args.outcomes, args.issues, args.report)
    print({"exact_100k_rows": result["exact_100k_rows"],
           "main_20k_t5": result["main"]["20000"]["5"]})


if __name__ == "__main__":
    main()
