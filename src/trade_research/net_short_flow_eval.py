"""Evaluate the previously frozen net short-position flow groups."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .margin_heterogeneity_eval import write_reprice_signals
from .net_short_flow_study import CONTROL, TREATED
from .short_interest_eval import evaluate, summarize_repriced


PERIODS = {
    "2024_postrunoff": ("2024_postrunoff",),
    "2025": ("2025_H1", "2025_H2"),
    "2025_H1": ("2025_H1",),
    "2025_H2": ("2025_H2",),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", type=Path,
                        default=Path("data/research/net_short_flow_selected.parquet"))
    parser.add_argument("--quintiles", type=Path,
                        default=Path("data/research/net_short_flow_quintiles.parquet"))
    parser.add_argument("--snapshots", type=Path,
                        default=Path("data/research/market_snapshots_ci"))
    parser.add_argument("--outcomes", type=Path,
                        default=Path("data/research/market_outcomes_ci"))
    parser.add_argument("--issues", type=Path,
                        default=Path("data/research/market_issues_ci"))
    parser.add_argument("--report", type=Path,
                        default=Path("data/research/net_short_flow_report.json"))
    parser.add_argument("--trades", type=Path,
                        default=Path("data/research/net_short_flow_trades.parquet"))
    parser.add_argument("--reprice-signals", type=Path,
                        default=Path("data/research/net_short_flow_reprice_signals.parquet"))
    parser.add_argument("--repriced", type=Path)
    parser.add_argument("--repriced-report", type=Path,
                        default=Path("data/research/net_short_flow_reprice_report.json"))
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
        print({"exact_100k_rows": reprice["exact_100k_rows"],
               "repriced_report": str(args.repriced_report)})


if __name__ == "__main__":
    main()
