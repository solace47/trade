"""Evaluate the frozen open-close Amihud and margin-interest groups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .margin_heterogeneity_eval import write_reprice_signals
from .market_study import _week_bootstrap
from .oc_amihud_margin_study import CONTROL, TREATED
from .short_interest_eval import evaluate, summarize_repriced


PERIODS = {
    f"{group}_{year}": (f"{year}_H1_{group}", f"{year}_H2_{group}")
    for group in ("high", "low") for year in ("2024", "2025")
}
PERIODS.update({
    f"{group}_{year}_H{half}": (f"{year}_H{half}_{group}",)
    for group in ("high", "low")
    for year in ("2024", "2025") for half in (1, 2)
})


def _same_day_interaction(selected_path: Path, repriced_path: Path) -> dict:
    membership = pd.read_parquet(selected_path)[
        ["date", "code", "amihud_group", "candidate"]]
    raw = pd.read_parquet(repriced_path)
    raw = raw.loc[raw.target_notional.eq(20_000) & raw.horizon.eq(5)]
    joined = raw.merge(membership, on=["date", "code"],
                       validate="one_to_one")
    if (len(joined) != len(membership)
            or joined.duplicated(["date", "code"]).any()):
        raise ValueError("Incomplete 20k primary open-close Amihud repricing")
    valid = joined.exit_status.eq("filled") & joined.quality_clean_exit
    joined["cash"] = np.where(valid, joined.net_return, 0.0)
    daily = joined.groupby(["date", "amihud_group", "candidate"])[
        "cash"].mean().unstack("candidate")
    if daily[[TREATED, CONTROL]].isna().any().any():
        raise ValueError("A daily Amihud layer lacks a matched comparator")
    daily = (daily[TREATED] - daily[CONTROL]).rename("edge").reset_index()
    common = daily.pivot(index="date", columns="amihud_group", values="edge")
    common = common.dropna(subset=["high", "low"])
    result = {}
    for year in ("2024", "2025"):
        sample = common.loc[common.index.str.startswith(year)]
        delta = sample.high - sample.low
        result[year] = {
            "common_dates": len(sample),
            "high_edge": float(sample.high.mean()),
            "low_edge": float(sample.low.mean()),
            "high_minus_low_edge": float(delta.mean()),
            "week_ci": _week_bootstrap(delta.reset_index(drop=True),
                                       pd.Series(sample.index), 261),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", type=Path,
                        default=Path("data/research/oc_amihud_margin_selected.parquet"))
    parser.add_argument("--quintiles", type=Path,
                        default=Path("data/research/oc_amihud_margin_quintiles.parquet"))
    parser.add_argument("--snapshots", type=Path,
                        default=Path("data/research/market_snapshots_ci"))
    parser.add_argument("--outcomes", type=Path,
                        default=Path("data/research/market_outcomes_ci"))
    parser.add_argument("--issues", type=Path,
                        default=Path("data/research/market_issues_ci"))
    parser.add_argument("--report", type=Path,
                        default=Path("data/research/oc_amihud_margin_report.json"))
    parser.add_argument("--trades", type=Path,
                        default=Path("data/research/oc_amihud_margin_trades.parquet"))
    parser.add_argument("--reprice-signals", type=Path,
                        default=Path("data/research/oc_amihud_margin_reprice_signals.parquet"))
    parser.add_argument("--repriced", type=Path)
    parser.add_argument("--repriced-report", type=Path,
                        default=Path("data/research/oc_amihud_margin_reprice_report.json"))
    args = parser.parse_args()
    write_reprice_signals(args.selected, args.snapshots, args.reprice_signals)
    result = evaluate(args.selected, args.quintiles, args.outcomes,
                      args.issues, args.report, args.trades,
                      TREATED, CONTROL, PERIODS)
    print({"matched_pairs": len(pd.read_parquet(args.selected)) // 2,
           "full_market_stock_days": result["full_market_stock_days"],
           "matched_t5_edges": {key: value["5"]["edge_mean"]
                                for key, value in result["matched"].items()}})
    if args.repriced:
        reprice = summarize_repriced(args.selected, args.repriced,
                                     args.trades, args.repriced_report,
                                     TREATED, CONTROL, PERIODS)
        reprice["same_day_20k_interaction"] = _same_day_interaction(
            args.selected, args.repriced)
        args.repriced_report.write_text(
            json.dumps(reprice, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        print({"exact_100k_rows": reprice["exact_100k_rows"],
               "repriced_report": str(args.repriced_report)})


if __name__ == "__main__":
    main()
