"""Evaluate frozen sell-completion pairs from raw 2024/2025 minute trades."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .buyback_eval import _score, _summary
from .buyback_inputs import CONTROL
from .buyback_reprice import _exact
from .hf_outcomes import Assumptions
from .insider_sell_completion_review import EVENT
from .strategy_scan import _stressed_returns


def _pairs(root: Path, stem: str, audit: dict) -> pd.DataFrame:
    frame = pd.concat([
        pd.read_parquet(root / f"{stem}_{year}.parquet")
        for year in (2024, 2025)
    ], ignore_index=True)
    expected = audit["main" if stem == "verified_pairs" else "same_industry"][
        "matched_pairs"]
    if (frame.empty or len(frame) != 2 * expected
            or frame.duplicated(["date", "code"]).any()
            or set(frame.candidate) != {EVENT, CONTROL}
            or not frame.date.str[:4].isin(("2024", "2025")).all()):
        raise ValueError("Frozen completion pairs changed after input audit")
    group = frame.groupby(["date", "pair_code"]).candidate.agg(
        ["size", "nunique"])
    if (len(group) != expected or not group["size"].eq(2).all()
            or not group["nunique"].eq(2).all()
            or frame.loc[frame.candidate.eq(EVENT)].groupby("date").size().gt(5).any()):
        raise ValueError("Malformed same-day completion/control pairs")
    return frame


def _scored(raw: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    rows = raw.merge(
        pairs[["date", "code", "candidate", "pair_code"]],
        on=["date", "code"], validate="many_to_one")
    if len(rows) != len(pairs) * 4:
        raise ValueError("Missing original-minute completion trade")
    rows["cash_return"] = np.where(
        rows.quality_clean_exit, rows.net_return, 0.0)
    rows["cash_stress10"] = 0.0
    clean = rows.quality_clean_exit
    # The base model already charges 5 bp slippage on each side.
    # The frozen stress adds 10 bp, so use 15 bp absolute slippage.
    rows.loc[clean, "cash_stress10"] = _stressed_returns(
        rows.loc[clean], Assumptions().slippage_bps_each_side + 10)
    if (not np.isfinite(rows.cash_return).all()
            or not np.isfinite(rows.cash_stress10).all()):
        raise ValueError("Nonfinite completion cash return")
    return rows


def _section(rows: pd.DataFrame, year: str, segment: str) -> dict:
    result = _summary(rows, year, segment, EVENT)
    result["extra_10bp_edge_mean"] = result.pop("stress10_edge_mean")
    frame = rows.loc[rows.date.str.startswith(year)]
    if segment == "H1":
        frame = frame.loc[frame.date.str[5:7].astype(int).le(6)]
    elif segment == "H2":
        frame = frame.loc[frame.date.str[5:7].astype(int).gt(6)]
    result["event_extra_10bp_cash_mean"] = float(
        frame.loc[frame.candidate.eq(EVENT)]
        .groupby("date").cash_stress10.mean().mean())
    return result


def _same_events(main: pd.DataFrame, industry: pd.DataFrame,
                 year: str) -> dict:
    def wide(rows: pd.DataFrame) -> pd.DataFrame:
        selected = rows.loc[
            rows.date.str.startswith(year)
            & rows.target_notional.eq(20_000)
            & rows.horizon.eq(5)]
        return selected.pivot(
            index=["date", "pair_code"], columns="candidate",
            values="cash_return")

    common = wide(main).join(wide(industry), how="inner",
                             lsuffix="_main", rsuffix="_industry")
    if common.empty or not np.allclose(
            common[f"{EVENT}_main"], common[f"{EVENT}_industry"]):
        raise ValueError("Industry comparison changed a frozen event outcome")
    common["main_edge"] = common[f"{EVENT}_main"] - common[f"{CONTROL}_main"]
    common["industry_edge"] = (
        common[f"{EVENT}_industry"] - common[f"{CONTROL}_industry"])
    daily = common.groupby(level="date")[["main_edge", "industry_edge"]].mean()
    return {"pairs": len(common), "days": len(daily),
            "main_edge_mean": float(daily.main_edge.mean()),
            "industry_edge_mean": float(daily.industry_edge.mean())}


def evaluate(root: Path, raw_path: Path, outcome_dir: Path,
             issues_dir: Path, report_path: Path) -> dict:
    audit = json.loads((root / "verified_input_audit.json").read_text(
        encoding="utf-8"))
    if (audit.get("outcomes_opened") is not False
            or not all(audit["review"][str(y)]["input_gate_passed"]
                       for y in (2024, 2025))):
        raise ValueError("Completion input gate was not frozen before outcomes")
    membership = {
        stem: _pairs(root, stem, audit)
        for stem in ("verified_pairs", "verified_industry_pairs")
    }
    raw = pd.read_parquet(raw_path)
    keys = pd.concat([
        frame[["date", "code"]] for frame in membership.values()
    ]).drop_duplicates()
    if (len(raw) != len(keys) * 4
            or set(raw.target_notional) != {20_000.0, 100_000.0}
            or set(raw.horizon) != {1, 5}
            or raw.duplicated(
                ["date", "code", "target_notional", "horizon"]).any()
            or not raw.date.str[:4].isin(("2024", "2025")).all()):
        raise ValueError("Incomplete frozen original-minute repricing")
    scored = {}
    exact = {}
    for stem, pairs in membership.items():
        stored = pd.concat([
            _score(pairs, outcome_dir, issues_dir, horizon)
            for horizon in (1, 5)
        ], ignore_index=True)
        subset = raw.loc[raw.target_notional.eq(100_000)].merge(
            pairs[["date", "code"]], on=["date", "code"],
            validate="many_to_one")
        exact[stem] = _exact(stored, subset)
        scored[stem] = _scored(raw, pairs)
    report = {
        "note": "2024/2025 exploratory; 2025 is not blind; 2026 unopened",
        "raw_100k_exact_rows": exact,
        "stress": "base 5 bp plus 10 bp per side; 15 bp absolute",
        "results": {}, "same_event_comparison": {},
    }
    for stem, rows in scored.items():
        report["results"][stem] = {}
        for size in (20_000, 100_000):
            report["results"][stem][str(size)] = {}
            for horizon in (1, 5):
                sample = rows.loc[
                    rows.target_notional.eq(size) & rows.horizon.eq(horizon)]
                report["results"][stem][str(size)][str(horizon)] = {
                    year: {
                        segment: _section(sample, year, segment)
                        for segment in ("full", "H1", "H2")
                    } for year in ("2024", "2025")
                }
    report["same_event_comparison"] = {
        year: _same_events(scored["verified_pairs"],
                           scored["verified_industry_pairs"], year)
        for year in ("2024", "2025")
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(
        "data/research/insider_sell_complete"))
    parser.add_argument("--raw", type=Path, default=Path(
        "data/research/insider_sell_complete/repriced.parquet"))
    parser.add_argument("--outcomes", type=Path, default=Path(
        "data/research/market_outcomes_ci"))
    parser.add_argument("--issues", type=Path, default=Path(
        "data/research/market_issues_ci"))
    parser.add_argument("--report", type=Path, default=Path(
        "data/research/insider_sell_complete/reprice_report.json"))
    args = parser.parse_args()
    report = evaluate(args.source, args.raw, args.outcomes, args.issues,
                      args.report)
    print({
        "raw_100k_exact_rows": report["raw_100k_exact_rows"],
        "main_20k_t5": {
            year: report["results"]["verified_pairs"]["20000"]["5"][
                year]["full"] for year in ("2024", "2025")},
        "same_event_comparison": report["same_event_comparison"],
    })


if __name__ == "__main__":
    main()
