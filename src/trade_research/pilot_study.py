"""Predeclared signal diagnostics on the 100-stock pipeline pilot.

This sample was selected for data engineering, not market-wide inference.
The rule below is a frozen starting hypothesis from the original request;
no thresholds are optimized on these outcomes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


HOLDOUT_START = "2026-04-01"
STUDY_START = "2025-09-01"  # First date of the frozen historical-as-of cohort.


def predeclared_screen(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["isST"].eq(0)
        & ~frame["reference_gap"]
        & ~frame["quote_outside_traded_range"]
        & frame["return_1450"].between(0.015, 0.05)
        & frame["position_1450"].ge(0.7)
        & frame["volume_ratio_est"].ge(1.2)
        & frame["price_1450"].gt(frame["ma20_prior_adjusted"])
        & frame["amount_1450"].ge(30_000_000)
    )


def _metrics(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"trades": 0, "trading_dates": 0}
    value = frame["net_return"].astype(float)
    daily_mean = frame.groupby("date")["net_return"].mean().astype(float)
    rng = np.random.default_rng(20260923)
    sampled = rng.choice(daily_mean.to_numpy(), size=(2000, len(daily_mean)), replace=True)
    interval = np.quantile(sampled.mean(axis=1), [0.025, 0.975])
    return {
        "trades": len(frame),
        "trading_dates": int(frame["date"].nunique()),
        "mean_net_return": float(value.mean()),
        "median_net_return": float(value.median()),
        "win_rate": float((value > 0).mean()),
        "date_weighted_mean_net_return": float(daily_mean.mean()),
        "date_bootstrap_95pct_interval": [float(interval[0]), float(interval[1])],
        "worst_trade": float(value.min()),
        "best_trade": float(value.max()),
    }


def study(snapshot_file: Path, outcome_file: Path, output: Path) -> dict:
    signals = pd.read_parquet(snapshot_file)
    signals = signals.loc[signals["date"] >= STUDY_START].copy()
    outcomes = pd.read_parquet(outcome_file)
    one = outcomes.loc[outcomes["horizon"] == 1]
    frame = signals.merge(one, on=["date", "code"], how="inner", validate="one_to_one")
    if len(frame) != len(signals):
        raise ValueError("Some 14:50 signals have no T+1 outcome")
    frame["candidate"] = predeclared_screen(frame)
    frame["period"] = np.where(frame["date"] < HOLDOUT_START, "development", "holdout")
    filled = (frame["entry_status"] == "filled") & (frame["exit_status"] == "filled")
    eligible = frame.loc[filled].copy()
    rows = {}
    for period in ("development", "holdout"):
        all_cohort = frame.loc[frame["period"] == period]
        candidate_cohort = all_cohort.loc[all_cohort["candidate"]]
        cohort = eligible.loc[eligible["period"] == period]
        rows[period] = {
            "baseline": {
                "signals": len(all_cohort),
                "entry_fills": int((all_cohort["entry_status"] == "filled").sum()),
                "completed_exits": len(cohort),
                "conditional_filled_returns": _metrics(cohort),
            },
            "predeclared_candidate": {
                "signals": len(candidate_cohort),
                "entry_fills": int((candidate_cohort["entry_status"] == "filled").sum()),
                "completed_exits": int((candidate_cohort["exit_status"] == "filled").sum()),
                "conditional_filled_returns": _metrics(cohort.loc[cohort["candidate"]]),
            },
        }
    result = {
        "scope": "100-stock historical-as-of data-pipeline pilot only",
        "holdout_start": HOLDOUT_START,
        "study_start": STUDY_START,
        "horizon": "T+1 at 14:52--14:55 VWAP, delayed if exit unfillable",
        "rule": (
            "non-ST; no reference-price gap or out-of-range quote; 14:50 return 1.5%--5%; "
            "close position >=0.7; estimated volume ratio >=1.2; "
            "price above prior 20-day adjusted MA; 14:50 turnover >= CNY30m"
        ),
        "snapshots": len(signals),
        "candidates": int(frame["candidate"].sum()),
        "eligible_outcomes": len(eligible),
        "periods": rows,
        "limitation": (
            "A 100-stock pilot does not establish full-market performance; "
            "the archive's upstream quote vendor and corporate-action factors are unverified."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bao-root", type=Path, default=Path("data/baostock/pilot_2025_2026"))
    args = parser.parse_args()
    root = args.bao_root
    result = study(root / "hf_snapshots_1450.parquet", root / "hf_outcomes.parquet",
                   root / "metadata" / "pilot_study.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
