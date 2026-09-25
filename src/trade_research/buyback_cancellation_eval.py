"""Evaluate the frozen pure-cancellation buyback hypothesis on raw minutes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.validate_buyback_cancellation_review import validate
from .annual_cash_direct_eval import _month_bootstrap
from .buyback_eval import _summary
from .buyback_inputs import CONTROL, EVENT
from .hf_outcomes import Assumptions
from .strategy_scan import _stressed_returns


def _tag_pairs(pairs: pd.DataFrame, review: pd.DataFrame,
               require_all: bool) -> pd.DataFrame:
    events = pairs.loc[pairs.candidate.eq(EVENT)]
    tags = review[["date", "code", "decision"]].rename(
        columns={"code": "pair_code"})
    if (pairs.duplicated(["date", "code"]).any()
            or events.duplicated(["date", "pair_code"]).any()
            or not events.code.eq(events.pair_code).all()):
        raise ValueError("Buyback pair membership is malformed")
    if require_all and (len(events) != len(review)
                        or set(events.pdf_url) != set(review.pdf_url)):
        raise ValueError("Buyback purpose ledger differs from frozen pairs")
    tagged = pairs.merge(tags, on=["date", "pair_code"],
                         validate="many_to_one")
    if len(tagged) != len(pairs):
        raise ValueError("Buyback pair has no reviewed purpose")
    counts = tagged.groupby(["date", "pair_code"]).candidate.agg(
        ["size", "nunique"])
    if (len(counts) != len(events) or not counts["size"].eq(2).all()
            or not counts["nunique"].eq(2).all()):
        raise ValueError("Buyback purpose lost a same-day control")
    return tagged


def prepare_industry(review_path: Path, source: Path, pair_dir: Path) -> dict:
    gate = validate(review_path, source, pair_dir / "pairs.parquet")
    if not gate["ready_for_matching"]:
        raise ValueError("Original-PDF purpose review is incomplete")
    review = pd.read_csv(review_path, dtype=str, keep_default_na=False)
    industry = pd.read_parquet(pair_dir / "industry_pairs.parquet")
    selected = _tag_pairs(industry, review, require_all=False)
    selected = selected.loc[selected.decision.eq("pure_cancel")].copy()
    if (selected.empty or selected.date.str[:4].isin(("2024", "2025"))
            .eq(False).any()):
        raise ValueError("No frozen pure-cancellation industry pairs")
    pair_path = pair_dir / "cancellation_industry_pairs.parquet"
    signals_path = pair_dir / "cancellation_industry_signals.parquet"
    selected.to_parquet(pair_path, index=False)
    selected[["date", "code", "isST", "reference_gap",
              "quote_outside_traded_range", "listing_age_sessions"]].to_parquet(
                  signals_path, index=False)
    return {"pairs": len(selected) // 2,
            "signals": len(selected), "signals_path": str(signals_path)}


def _score_raw(raw: pd.DataFrame, tagged: pd.DataFrame,
               expected_sizes: set[float], expected_horizons: set[int]) -> pd.DataFrame:
    sizes = len(expected_sizes) * len(expected_horizons)
    if (len(raw) != len(tagged) * sizes
            or set(raw.target_notional) != expected_sizes
            or set(raw.horizon) != expected_horizons
            or not raw.exit_window.eq("close").all()
            or raw.duplicated(["date", "code", "target_notional",
                               "horizon"]).any()
            or not raw.date.str[:4].isin(("2024", "2025")).all()):
        raise ValueError("Incomplete or out-of-period original-minute outcomes")
    rows = raw.merge(tagged[["date", "code", "candidate", "pair_code",
                             "decision"]], on=["date", "code"],
                     validate="many_to_one")
    if len(rows) != len(raw) or rows.quality_clean_exit.isna().any():
        raise ValueError("Original-minute record lacks a frozen reviewed pair")
    clean = rows.quality_clean_exit
    baseline_slip = Assumptions().slippage_bps_each_side
    if not np.allclose(_stressed_returns(rows.loc[clean], baseline_slip),
                       rows.loc[clean, "net_return"], atol=1e-12):
        raise ValueError("Original-minute net return disagrees with cost model")
    rows["cash_return"] = np.where(clean, rows.net_return, 0.0)
    rows["cash_stress10"] = 0.0
    rows.loc[clean, "cash_stress10"] = _stressed_returns(
        rows.loc[clean], baseline_slip + 10)
    if not np.isfinite(rows[["cash_return", "cash_stress10"]].to_numpy()).all():
        raise ValueError("Nonfinite original-minute cash return")
    return rows


def _daily_group_edges(rows: pd.DataFrame) -> pd.DataFrame:
    wide = rows.groupby(["date", "decision", "candidate"])[
        "cash_return"].mean().unstack("candidate")
    if wide.empty or wide[[EVENT, CONTROL]].isna().any().any():
        raise ValueError("Purpose event lacks its same-day control")
    return wide.assign(edge=wide[EVENT] - wide[CONTROL]).reset_index()[
        ["date", "decision", "edge"]]


def _same_day_purpose(rows: pd.DataFrame, year: str) -> dict:
    daily = _daily_group_edges(rows.loc[rows.date.str.startswith(year)
                                          & rows.decision.isin(
                                              ("pure_cancel", "employee"))])
    overlap = daily.pivot(index="date", columns="decision",
                          values="edge").dropna()
    if overlap.empty:
        return {"days": 0}
    gap = overlap.pure_cancel - overlap.employee
    return {"days": len(overlap),
            "pure_minus_employee_edge": float(gap.mean()),
            "month_ci": _month_bootstrap(
                gap.reset_index(drop=True),
                pd.Series(overlap.index.to_list()), 613)}


def _same_event_industry(main: pd.DataFrame, industry: pd.DataFrame,
                         year: str) -> dict:
    def wide(rows: pd.DataFrame) -> pd.DataFrame:
        sample = rows.loc[rows.date.str.startswith(year)
                          & rows.target_notional.eq(20_000)
                          & rows.horizon.eq(5)
                          & rows.decision.eq("pure_cancel")]
        return sample.pivot(index=["date", "pair_code"],
                            columns="candidate", values="cash_return")

    common = wide(main).join(wide(industry), how="inner",
                             lsuffix="_main", rsuffix="_industry")
    if (common.empty or common.isna().any().any()
            or not np.allclose(common[f"{EVENT}_main"],
                               common[f"{EVENT}_industry"], atol=1e-12)):
        raise ValueError("Industry control changed or lost a frozen event")
    common["main_edge"] = common[f"{EVENT}_main"] - common[f"{CONTROL}_main"]
    common["industry_edge"] = (common[f"{EVENT}_industry"]
                               - common[f"{CONTROL}_industry"])
    daily = common.groupby(level="date")[["main_edge", "industry_edge"]].mean()
    return {"pairs": len(common), "days": len(daily),
            "main_edge_mean": float(daily.main_edge.mean()),
            "industry_edge_mean": float(daily.industry_edge.mean()),
            "industry_edge_month_ci": _month_bootstrap(
                daily.industry_edge.reset_index(drop=True),
                pd.Series(daily.index.to_list()), 614)}


def _tail(rows: pd.DataFrame, year: str) -> dict:
    sample = rows.loc[rows.date.str.startswith(year)
                      & rows.target_notional.eq(20_000)
                      & rows.horizon.eq(5)
                      & rows.decision.eq("pure_cancel")]
    daily = _daily_group_edges(sample).set_index("date").edge
    months = pd.Series(daily.index.str[:7], index=daily.index)
    leave_one_out = {
        month: float(daily.loc[months.ne(month)].mean())
        for month in sorted(months.unique())
    }
    return {"day_median": float(daily.median()),
            "positive_day_share": float(daily.gt(0).mean()),
            "leave_one_month_out_edge_range": [min(leave_one_out.values()),
                                               max(leave_one_out.values())]}


def _complete_pairs(rows: pd.DataFrame, year: str) -> dict:
    sample = rows.loc[rows.date.str.startswith(year)
                      & rows.target_notional.eq(20_000)
                      & rows.horizon.eq(5)
                      & rows.decision.eq("pure_cancel")]
    complete = sample.groupby(["date", "pair_code"])
    eligible = complete.quality_clean_exit.transform("all")
    return _summary(sample.loc[eligible], year, "full")


def _matching_context(pairs: pd.DataFrame, year: str) -> dict:
    treated = pairs.loc[pairs.candidate.eq(EVENT)
                        & pairs.decision.eq("pure_cancel")
                        & pairs.date.str.startswith(year)]
    peers = pairs.loc[pairs.candidate.eq(CONTROL),
                      ["date", "pair_code", "industry", "float_mv",
                       "avg20_amount", "return20_prior_adjusted",
                       "return_1450"]]
    joined = treated.merge(peers, on=["date", "pair_code"],
                           suffixes=("_event", "_peer"),
                           validate="one_to_one")
    if len(joined) != len(treated):
        raise ValueError("Pure buyback pair lacks a frozen control")
    return {
        "pairs": len(joined),
        "same_industry_main_pairs": int(
            joined.industry_event.eq(joined.industry_peer).sum()),
        "float_mv_ratio_median": float((
            joined.float_mv_event / joined.float_mv_peer).median()),
        "avg20_amount_ratio_median": float((
            joined.avg20_amount_event / joined.avg20_amount_peer).median()),
        "prior20_return_gap_median": float((
            joined.return20_prior_adjusted_event
            - joined.return20_prior_adjusted_peer).median()),
        "return_1450_gap_median": float((
            joined.return_1450_event - joined.return_1450_peer).median()),
    }


def _industry_coverage_diagnostic(main: pd.DataFrame,
                                  industry_pairs: pd.DataFrame,
                                  year: str) -> dict:
    sample = main.loc[main.date.str.startswith(year)
                      & main.target_notional.eq(20_000)
                      & main.horizon.eq(5)
                      & main.decision.eq("pure_cancel")]
    wide = sample.pivot(index=["date", "pair_code"],
                        columns="candidate", values="cash_return")
    wide["edge"] = wide[EVENT] - wide[CONTROL]
    available = pd.MultiIndex.from_frame(industry_pairs.loc[
        industry_pairs.candidate.eq(EVENT), ["date", "pair_code"]])
    output = {}
    for label, mask in (("industry_available", wide.index.isin(available)),
                        ("industry_unavailable", ~wide.index.isin(available))):
        subset = wide.loc[mask]
        daily = subset.edge.groupby(level="date").mean()
        output[label] = {"pairs": len(subset), "days": len(daily),
                         "main_control_edge_mean": float(daily.mean())}
    return output


def evaluate(review_path: Path, source: Path, pair_dir: Path,
             raw_path: Path, industry_raw_path: Path,
             report_path: Path) -> dict:
    gate = validate(review_path, source, pair_dir / "pairs.parquet")
    if not gate["ready_for_matching"]:
        raise ValueError("Original-PDF purpose review is incomplete")
    review = pd.read_csv(review_path, dtype=str, keep_default_na=False)
    main_pairs = _tag_pairs(pd.read_parquet(pair_dir / "pairs.parquet"),
                            review, require_all=True)
    expected_industry = _tag_pairs(
        pd.read_parquet(pair_dir / "industry_pairs.parquet"),
        review, require_all=False)
    expected_industry = expected_industry.loc[
        expected_industry.decision.eq("pure_cancel")]
    industry_pairs = pd.read_parquet(
        pair_dir / "cancellation_industry_pairs.parquet")
    keys = ["date", "code", "candidate", "pair_code", "decision"]
    if (industry_pairs.empty or not industry_pairs[keys].sort_values(
            ["date", "code"]).reset_index(drop=True).equals(
                expected_industry[keys].sort_values(
                    ["date", "code"]).reset_index(drop=True))):
        raise ValueError("Pure industry subset differs from frozen membership")
    main = _score_raw(pd.read_parquet(raw_path), main_pairs,
                      {20_000.0, 100_000.0}, {1, 5})
    industry = _score_raw(pd.read_parquet(industry_raw_path), industry_pairs,
                          {20_000.0}, {5})
    report = {"note": "Exploratory 2024/2025; 2025 previously viewed; no 2026",
              "source_review": gate, "original_minute": {},
              "same_day_purpose": {}, "same_event_industry": {},
              "posthoc_tail": {}, "posthoc_complete_pairs": {},
              "matching_context": {}, "posthoc_industry_coverage": {}}
    for size in (20_000, 100_000):
        report["original_minute"][str(size)] = {}
        for horizon in (1, 5):
            sample = main.loc[main.target_notional.eq(size)
                              & main.horizon.eq(horizon)]
            report["original_minute"][str(size)][str(horizon)] = {}
            for year in ("2024", "2025"):
                report["original_minute"][str(size)][str(horizon)][year] = {
                    purpose: {segment: _summary(
                        sample.loc[sample.decision.eq(purpose)], year, segment)
                              for segment in ("full", "H1", "H2",
                                              "without_peak_month")}
                    for purpose in ("pure_cancel", "employee")
                }
    primary = main.loc[main.target_notional.eq(20_000)
                       & main.horizon.eq(5)]
    for year in ("2024", "2025"):
        report["same_day_purpose"][year] = _same_day_purpose(primary, year)
        report["same_event_industry"][year] = _same_event_industry(
            main, industry, year)
        report["posthoc_tail"][year] = _tail(main, year)
        report["posthoc_complete_pairs"][year] = _complete_pairs(main, year)
        report["matching_context"][year] = _matching_context(
            main_pairs, year)
        report["posthoc_industry_coverage"][year] = (
            _industry_coverage_diagnostic(main, industry_pairs, year))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2)
                           + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare-industry", "evaluate"))
    parser.add_argument("--review", type=Path, default=Path(
        "research/buyback_cancellation_review.csv"))
    parser.add_argument("--source", type=Path, default=Path(
        "data/research/buyback"))
    parser.add_argument("--pairs", type=Path, default=Path(
        "data/research/buyback"))
    parser.add_argument("--raw", type=Path, default=Path(
        "data/research/buyback/repriced.parquet"))
    parser.add_argument("--industry-raw", type=Path, default=Path(
        "data/research/buyback/cancellation_industry_repriced.parquet"))
    parser.add_argument("--report", type=Path, default=Path(
        "data/research/buyback/cancellation_report.json"))
    args = parser.parse_args()
    if args.command == "prepare-industry":
        print(prepare_industry(args.review, args.source, args.pairs))
    else:
        report = evaluate(args.review, args.source, args.pairs,
                          args.raw, args.industry_raw, args.report)
        print({year: report["original_minute"]["20000"]["5"][year]
               ["pure_cancel"]["full"] for year in ("2024", "2025")})


if __name__ == "__main__":
    main()
