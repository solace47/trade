"""Evaluate the previously frozen buyback funding-size hypothesis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .annual_cash_direct_eval import _month_bootstrap
from .buyback_eval import _score
from .buyback_inputs import CONTROL, EVENT
from .strategy_scan import _stressed_returns


def _month_difference_ci(daily: pd.DataFrame, seed: int) -> list[float]:
    """Resample the same calendar months for both funding groups."""
    counts = daily.groupby(["month", "funding_group"]).edge.agg(
        ["sum", "count"]).unstack("funding_group").fillna(0)
    if not {"high", "low"}.issubset(set(daily.funding_group)):
        raise ValueError("Both funding groups are required")
    months = sorted(daily.month.unique())
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(months), size=(2000, len(months)))
    high_sum = counts[("sum", "high")].reindex(months, fill_value=0).to_numpy()
    low_sum = counts[("sum", "low")].reindex(months, fill_value=0).to_numpy()
    high_n = counts[("count", "high")].reindex(months, fill_value=0).to_numpy()
    low_n = counts[("count", "low")].reindex(months, fill_value=0).to_numpy()
    h_den, l_den = high_n[draws].sum(axis=1), low_n[draws].sum(axis=1)
    valid = (h_den > 0) & (l_den > 0)
    spread = (high_sum[draws][valid].sum(axis=1) / h_den[valid]
              - low_sum[draws][valid].sum(axis=1) / l_den[valid])
    if len(spread) < 1900:
        raise ValueError("Insufficient joint month-block bootstrap draws")
    return [float(x) for x in np.quantile(spread, [.025, .975])]


def _attach_funding(rows: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    keys = funding[["date", "code", "floor_float_ratio"]].rename(
        columns={"code": "pair_code"})
    if keys.duplicated(["date", "pair_code"]).any():
        raise ValueError("Funding input duplicated a frozen event")
    scored = rows.merge(keys, on=["date", "pair_code"],
                        validate="many_to_one")
    if len(scored) != len(rows):
        raise ValueError("Frozen pair lacks a buyback funding input")
    scored = scored.loc[scored.floor_float_ratio.notna()].copy()
    scored["funding_group"] = np.where(
        scored.floor_float_ratio.ge(.01), "high", "low")
    if (scored.groupby(["date", "pair_code"]).candidate.nunique().ne(2).any()
            or scored.date.str[:4].isin(("2024", "2025")).eq(False).any()):
        raise ValueError("Funding subgroup has unpaired or later-year data")
    return scored


def _cash(rows: pd.DataFrame) -> pd.DataFrame:
    frame = rows.copy()
    frame["cash_return"] = np.where(frame.quality_clean_exit,
                                    frame.net_return, 0.0)
    frame["cash_stress10"] = 0.0
    clean = frame.quality_clean_exit
    frame.loc[clean, "cash_stress10"] = _stressed_returns(
        frame.loc[clean], 10)
    if (not np.isfinite(frame.cash_return).all()
            or not np.isfinite(frame.cash_stress10).all()):
        raise ValueError("Buyback cash return is not finite")
    return frame


def _daily(rows: pd.DataFrame) -> pd.DataFrame:
    wide = rows.groupby(["date", "funding_group", "candidate"])[
        ["cash_return", "cash_stress10"]].mean().unstack("candidate")
    if (wide.empty or wide.isna().any().any()
            or not {EVENT, CONTROL}.issubset(rows.candidate.unique())):
        raise ValueError("Funding event lacks its same-day peer")
    daily = pd.DataFrame({
        "date": [i[0] for i in wide.index],
        "funding_group": [i[1] for i in wide.index],
        "event": wide[("cash_return", EVENT)].to_numpy(),
        "control": wide[("cash_return", CONTROL)].to_numpy(),
        "stress_edge": (wide[("cash_stress10", EVENT)]
                        - wide[("cash_stress10", CONTROL)]).to_numpy(),
    })
    daily["edge"] = daily.event - daily.control
    daily["month"] = daily.date.str[:7]
    return daily


def _select_segment(rows: pd.DataFrame, year: str,
                    segment: str) -> pd.DataFrame:
    frame = rows.loc[rows.date.str.startswith(year)]
    if segment == "H1":
        frame = frame.loc[frame.date.str[5:7].astype(int).le(6)]
    elif segment == "H2":
        frame = frame.loc[frame.date.str[5:7].astype(int).gt(6)]
    elif segment == "without_peak_month":
        peak = "2024-02" if year == "2024" else "2025-04"
        frame = frame.loc[~frame.date.str.startswith(peak)]
    elif segment != "full":
        raise ValueError("Unknown frozen segment")
    return frame


def _summarize(rows: pd.DataFrame, year: str, segment: str) -> dict:
    frame = _select_segment(rows, year, segment)
    daily = _daily(frame)
    output: dict = {}
    for group in ("high", "low"):
        cells = daily.loc[daily.funding_group.eq(group)]
        included = frame.loc[frame.funding_group.eq(group)]
        treated = included.loc[included.candidate.eq(EVENT)]
        control = included.loc[included.candidate.eq(CONTROL)]
        if cells.empty or len(treated) != len(control):
            raise ValueError("Funding group has no matched pairs")
        output[group] = {
            "pairs": len(treated), "days": len(cells),
            "event_cash_mean": float(cells.event.mean()),
            "control_cash_mean": float(cells.control.mean()),
            "edge_mean": float(cells.edge.mean()),
            "edge_month_ci": _month_bootstrap(cells.edge, cells.date, 551),
            "stress10_edge_mean": float(cells.stress_edge.mean()),
            "event_entry_rate": float(treated.entry_status.eq("filled").mean()),
            "control_entry_rate": float(control.entry_status.eq("filled").mean()),
            "event_clean_exit_rate": float(treated.quality_clean_exit.mean()),
            "control_clean_exit_rate": float(control.quality_clean_exit.mean()),
        }
    output["high_minus_low_edge"] = float(
        output["high"]["edge_mean"] - output["low"]["edge_mean"])
    output["high_minus_low_month_ci"] = _month_difference_ci(daily, 552)
    overlap = daily.pivot(index="date", columns="funding_group",
                          values="edge").dropna()
    output["same_day_high_low"] = {
        "days": len(overlap),
        "edge_difference": float((overlap.high - overlap.low).mean()),
    }
    return output


def evaluate(source_dir: Path, raw_path: Path, outcome_dir: Path,
             issues_dir: Path, report_path: Path) -> dict:
    audit = json.loads((source_dir / "funding_input_audit.json").read_text(
        encoding="utf-8"))
    if audit.get("note") != "Input-only PDF funding floor; no funding-subgroup returns read":
        raise ValueError("Funding hypothesis input audit was not frozen")
    funding = pd.read_parquet(source_dir / "funding_inputs.parquet")
    if (len(funding) != 669 or funding.duplicated(["date", "code"]).any()
            or not funding.date.str[:4].isin(("2024", "2025")).all()):
        raise ValueError("Funding inputs differ from frozen pair list")
    membership = pd.read_parquet(source_dir / "pairs.parquet")
    raw = pd.read_parquet(raw_path)
    if (len(membership) != 1338 or len(raw) != 5352
            or raw.duplicated(["date", "code", "target_notional",
                               "horizon"]).any()
            or set(raw.target_notional) != {20_000.0, 100_000.0}
            or set(raw.horizon) != {1, 5}):
        raise ValueError("Original-minute frozen pair repricing is incomplete")
    joined = raw.merge(membership[["date", "code", "candidate", "pair_code"]],
                       on=["date", "code"], validate="many_to_one")
    if len(joined) != len(raw):
        raise ValueError("Raw minute record is outside frozen pair list")
    main = _cash(_attach_funding(joined, funding))
    report = {"note": "Exploratory funding mechanism; 2025 not blind; 2026 not read",
              "original_minute": {}, "same_industry_100k_t5": {}}
    for size in (20_000, 100_000):
        report["original_minute"][str(size)] = {}
        for horizon in (1, 5):
            rows = main.loc[main.target_notional.eq(size)
                            & main.horizon.eq(horizon)]
            report["original_minute"][str(size)][str(horizon)] = {
                year: {segment: _summarize(rows, year, segment)
                       for segment in ("full", "H1", "H2", "without_peak_month")}
                for year in ("2024", "2025")
            }
    industry_pairs = pd.read_parquet(source_dir / "industry_pairs.parquet")
    industry = _cash(_attach_funding(
        _score(industry_pairs, outcome_dir, issues_dir, 5), funding))
    report["same_industry_100k_t5"] = {
        year: _summarize(industry, year, "full")
        for year in ("2024", "2025")
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("data/research/buyback"))
    parser.add_argument("--raw", type=Path,
                        default=Path("data/research/buyback/repriced.parquet"))
    parser.add_argument("--outcomes", type=Path,
                        default=Path("data/research/market_outcomes_ci"))
    parser.add_argument("--issues", type=Path,
                        default=Path("data/research/market_issues_ci"))
    parser.add_argument("--report", type=Path,
                        default=Path("data/research/buyback/funding_report.json"))
    args = parser.parse_args()
    report = evaluate(args.source, args.raw, args.outcomes, args.issues,
                      args.report)
    print(report["original_minute"]["100000"]["5"])


if __name__ == "__main__":
    main()
