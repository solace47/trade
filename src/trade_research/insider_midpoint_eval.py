"""Evaluate frozen midpoint zero-execution pairs with original minute trades."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .buyback_eval import _score, _summary
from .buyback_inputs import CONTROL
from .buyback_reprice import _exact
from .insider_midpoint_inputs import COOLDOWN, EVENT
from .strategy_scan import _stressed_returns


def _membership(pairs: pd.DataFrame, expected: dict) -> None:
    if (pairs.empty or len(pairs) != 2 * expected["matched_pairs"]
            or pairs.duplicated(["date", "code"]).any()
            or set(pairs.candidate) != {EVENT, CONTROL}
            or expected["cooldown_sessions"] != COOLDOWN
            or not pairs.date.str[:4].isin(("2024", "2025")).all()):
        raise ValueError("Midpoint membership changed after input freeze")
    groups = pairs.groupby(["date", "pair_code"]).candidate.agg(
        ["size", "nunique"])
    treated = pairs.loc[pairs.candidate.eq(EVENT)]
    if (len(groups) != expected["matched_pairs"]
            or not groups["size"].eq(2).all()
            or not groups["nunique"].eq(2).all()
            or not pairs.date.gt(pairs.event_notice_date).all()
            or treated.groupby("date").size().gt(5).any()):
        raise ValueError("Midpoint pair, disclosure timing or capacity failed")
    if treated.date.nunique() != expected["matched_days"]:
        raise ValueError("Midpoint signal days changed")
    for year in ("2024", "2025"):
        period = treated.loc[treated.date.str.startswith(year)]
        if (len(period) != expected["by_year"][year]["pairs"]
                or period.date.nunique() != expected["by_year"][year]["days"]):
            raise ValueError("Midpoint year membership shifted")


def _rows(pairs: pd.DataFrame, raw: pd.DataFrame,
          outcome_dir: Path, issues_dir: Path) -> tuple[pd.DataFrame, int]:
    membership = pairs[["date", "code", "candidate", "pair_code",
                        "event_exchange"]]
    selected = raw.merge(membership, on=["date", "code"],
                         validate="many_to_one")
    if len(selected) != len(pairs) * 4:
        raise ValueError("Frozen midpoint pair lacks raw-minute outcomes")
    stored = pd.concat([
        _score(pairs, outcome_dir, issues_dir, horizon)
        for horizon in (1, 5)
    ], ignore_index=True)
    exact = _exact(stored, selected.loc[
        selected.target_notional.eq(100_000)])
    selected["cash_return"] = np.where(selected.quality_clean_exit,
                                       selected.net_return, 0.0)
    selected["cash_stress10"] = 0.0
    clean = selected.quality_clean_exit
    selected.loc[clean, "cash_stress10"] = _stressed_returns(
        selected.loc[clean], 10)
    if (not np.isfinite(selected.cash_return).all()
            or not np.isfinite(selected.cash_stress10).all()):
        raise ValueError("Nonfinite midpoint cash return")
    return selected, exact


def _sections(rows: pd.DataFrame, year: str,
              peak_month: str) -> dict:
    result = {part: _summary(rows, year, part, EVENT)
              for part in ("full", "H1", "H2")}
    result["without_peak_month"] = _summary(
        rows.loc[~rows.date.str.startswith(peak_month)],
        year, "full", EVENT)
    result["by_exchange"] = {
        exchange: _summary(rows.loc[rows.event_exchange.eq(exchange)],
                           year, "full", EVENT)
        for exchange in ("sh", "sz")
    }
    return result


def evaluate(source_dir: Path, outcome_dir: Path, issues_dir: Path,
             report_path: Path) -> dict:
    input_audit = json.loads((source_dir / "input_audit.json").read_text(
        encoding="utf-8"))
    if input_audit.get("note") != (
            "Midpoint input membership only; no future returns opened"):
        raise ValueError("Midpoint input membership was not frozen")
    main = pd.read_parquet(source_dir / "pairs.parquet")
    industry = pd.read_parquet(source_dir / "industry_pairs.parquet")
    _membership(main, input_audit["main"])
    _membership(industry, input_audit["same_industry"])
    raw = pd.read_parquet(source_dir / "repriced.parquet")
    signals = pd.read_parquet(source_dir / "signals.parquet")
    if (len(raw) != len(signals) * 4
            or set(raw.target_notional) != {20_000.0, 100_000.0}
            or set(raw.horizon) != {1, 5}
            or set(raw.exit_window) != {"close"}
            or raw.duplicated(["date", "code", "target_notional",
                               "horizon"]).any()):
        raise ValueError("Incomplete raw-minute midpoint repricing")
    scored, exact_main = _rows(main, raw, outcome_dir, issues_dir)
    within, exact_industry = _rows(industry, raw, outcome_dir, issues_dir)
    report = {"note": "2024/2025 exploratory; 2025 not blind; 2026 not read",
              "raw_minute_stock_days": len(signals),
              "exact_100k_rows": exact_main + exact_industry,
              "peak_months": input_audit["peak_months"],
              "main": {}, "same_industry": {}}
    for size in (20_000, 100_000):
        report["main"][str(size)] = {}
        report["same_industry"][str(size)] = {}
        for horizon in (1, 5):
            sample = scored.loc[scored.target_notional.eq(size)
                                & scored.horizon.eq(horizon)]
            peer = within.loc[within.target_notional.eq(size)
                              & within.horizon.eq(horizon)]
            report["main"][str(size)][str(horizon)] = {
                year: _sections(sample, year,
                                input_audit["peak_months"][year])
                for year in ("2024", "2025")
            }
            report["same_industry"][str(size)][str(horizon)] = {
                year: _summary(peer, year, "full", EVENT)
                for year in ("2024", "2025")
            }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2)
                           + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(
        "data/research/insider_midpoint"))
    parser.add_argument("--outcomes", type=Path, default=Path(
        "data/research/market_outcomes_ci"))
    parser.add_argument("--issues", type=Path, default=Path(
        "data/research/market_issues_ci"))
    parser.add_argument("--report", type=Path, default=Path(
        "data/research/insider_midpoint/report.json"))
    args = parser.parse_args()
    report = evaluate(args.source, args.outcomes, args.issues, args.report)
    print({"exact_100k_rows": report["exact_100k_rows"],
           "main_20k_t5": {
               year: report["main"]["20000"]["5"][year]["full"]
               for year in ("2024", "2025")}})


if __name__ == "__main__":
    main()
