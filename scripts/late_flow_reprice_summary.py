"""Summarize the fixed late-flow selections after raw-minute order-size repricing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trade_research.residual_liquidity import _period


def summarize(selected_path: Path, repriced_path: Path, output: Path) -> dict:
    original = pd.read_parquet(selected_path)
    original = original.loc[
        original.horizon.eq(1)
        & original.candidate.isin(("top_flow", "same_day_random"))
    ].copy()
    repriced = pd.read_parquet(repriced_path)
    if set(repriced.horizon) != {1} or set(repriced.exit_window) != {"close"}:
        raise ValueError("Repricing must use T+1 and the 14:52-14:55 exit")
    if set(repriced.target_notional) != {20_000.0, 50_000.0, 100_000.0}:
        raise ValueError("Required order sizes are missing")
    validation = original.merge(
        repriced.loc[repriced.target_notional.eq(100_000)],
        on=["date", "code", "horizon"], suffixes=("_original", "_repriced"),
        validate="many_to_one",
    )
    if len(validation) != len(original):
        raise ValueError("100,000 yuan repricing missed an original selection")
    for field in ("entry_status", "exit_status", "shares", "exit_date"):
        if not validation[f"{field}_original"].fillna("").equals(
            validation[f"{field}_repriced"].fillna("")
        ):
            raise ValueError(f"Stored and raw-minute repricing disagree on {field}")
    clean_repriced = validation.exit_status_repriced.eq("filled") \
        & validation.quality_clean_exit_repriced
    if not validation.quality_clean_exit_original.equals(clean_repriced):
        raise ValueError("Stored and raw-minute quality flags disagree")
    if not np.allclose(validation.net_return_original,
                       validation.net_return_repriced, equal_nan=True, atol=1e-12):
        raise ValueError("Stored and raw-minute net returns disagree")

    choices = original[["date", "code", "candidate"]]
    joined = choices.merge(repriced, on=["date", "code"], validate="many_to_many")
    report = {"verified_100k_matches_stored": True, "results": {}}
    for notional, block in joined.groupby("target_notional"):
        by_year = report["results"][str(int(notional))] = {}
        for year in ("2024", "2025"):
            annual = block.loc[block.date.str.startswith(year)]
            top = annual.loc[annual.candidate.eq("top_flow")]
            random = annual.loc[annual.candidate.eq("same_day_random")]
            by_year[year] = {}
            for period, rows in (
                ("H1", top.loc[top.date.str[5:7].astype(int).le(6)]),
                ("H2", top.loc[top.date.str[5:7].astype(int).gt(6)]),
                ("full", top),
            ):
                by_year[year][period] = _period(rows, random)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", type=Path,
                        default=Path("data/research/late_flow_trades.parquet"))
    parser.add_argument("--repriced", type=Path,
                        default=Path("data/research/late_flow_repriced.parquet"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/research/late_flow_reprice_report.json"))
    args = parser.parse_args()
    result = summarize(args.selected, args.repriced, args.output)
    print({"verified_100k": result["verified_100k_matches_stored"],
           "output": str(args.output)})


if __name__ == "__main__":
    main()
