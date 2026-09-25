"""Compare future-VWAP notional sizing with shares fixed at the 14:49 price."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("data/research")
SOURCE = ROOT / "late_reversal_1449"
OUTPUT = ROOT / "decision_sizing"
FULL_ENTRY = ROOT / "execution_path" / "full" / "entry_results.parquet"
KEY = ["date", "code", "target_notional", "entry_window", "exit_window",
       "horizon"]


def _pair_comparison(source_dir: Path, output_dir: Path) -> dict:
    original = pd.read_parquet(source_dir / "repriced.parquet")
    original = original.loc[
        original.horizon.eq(1) & original.exit_window.eq("close")
    ].copy()
    fixed = pd.read_parquet(output_dir / "repriced.parquet")
    if (len(original) != len(fixed) or len(fixed) != 4104
            or original.duplicated(KEY).any() or fixed.duplicated(KEY).any()):
        raise ValueError("The two fixed trade grids differ in size or keys")
    joined = original.merge(fixed, on=KEY, suffixes=("_future", "_known"),
                            validate="one_to_one")
    labels = pd.read_parquet(source_dir / "selections.parquet", columns=[
        "date", "code", "candidate", "pair_id",
    ])
    joined = joined.merge(labels, on=["date", "code"],
                          validate="many_to_one")
    if (len(joined) != 4104 or labels.duplicated(["date", "code"]).any()
            or set(joined.candidate) != {"late_decline", "late_rally_control"}):
        raise ValueError("An outcome lacks a frozen 14:49 pair label")
    joined["half"] = joined.date.str[:4] + "H" + joined.date.str[5:7].map(
        lambda month: "1" if int(month) <= 6 else "2")
    for suffix in ("future", "known"):
        joined[f"valid_{suffix}"] = (
            joined[f"exit_status_{suffix}"].eq("filled")
            & joined[f"quality_clean_exit_{suffix}"]
            & joined[f"exit_delay_sessions_{suffix}"].eq(0)
        )
        joined[f"cash_{suffix}"] = joined[f"net_return_{suffix}"].where(
            joined[f"valid_{suffix}"], 0.0)
    cells = {}
    for (amount, half, candidate), part in joined.groupby(
            ["target_notional", "half", "candidate"], sort=True):
        both_entry = (part.entry_status_future.eq("filled")
                      & part.entry_status_known.eq("filled"))
        both_exit = part.valid_future & part.valid_known
        daily = part.assign(
            difference=part.cash_known - part.cash_future
        ).groupby("date", sort=True).difference.mean()
        cells[f"{int(amount)}_{half}_{candidate}"] = {
            "signals": int(len(part)),
            "both_entry_filled": int(both_entry.sum()),
            "recorded_share_difference_rate": float((
                both_entry & part.shares_future.ne(part.shares_known)
            ).mean()),
            "entry_status_difference_rate": float(
                part.entry_status_future.ne(part.entry_status_known).mean()),
            "on_time_clean_exit_difference_rate": float(
                part.valid_future.ne(part.valid_known).mean()),
            "both_on_time_clean_exit": int(both_exit.sum()),
            "both_valid_net_difference_bps": float((
                part.loc[both_exit, "net_return_known"]
                - part.loc[both_exit, "net_return_future"]
            ).mean() * 10_000),
            "daily_cash_difference_bps": float(daily.mean() * 10_000),
        }
    share_plan_sensitive = any(
        value["recorded_share_difference_rate"] > .05
        for value in cells.values()
    )
    trade_outcome_sensitive = any(
        value["entry_status_difference_rate"] > .01
        or value["on_time_clean_exit_difference_rate"] > .01
        or abs(value["daily_cash_difference_bps"]) > 2
        for value in cells.values()
    )
    return {"compared_rows": len(joined),
            "share_plan_sensitive": share_plan_sensitive,
            "trade_outcome_sensitive": trade_outcome_sensitive,
            "model_sensitive": share_plan_sensitive or trade_outcome_sensitive,
            "by_size_half_candidate": cells}


def _full_market_share_check(path: Path) -> dict:
    """Input-only breadth: uses archived VWAP and shares, not holding returns."""
    rows = pd.read_parquet(path, columns=[
        "half", "board", "target_shares", "aggregate_status_5",
        "aggregate_price_5",
    ])
    rows = rows.loc[rows.aggregate_status_5.eq("filled")].copy()
    raw_vwap = rows.aggregate_price_5.to_numpy(dtype=float) / 1.0005
    if not np.isfinite(raw_vwap).all() or np.any(raw_vwap <= 0):
        raise ValueError("Invalid archived full-market entry prices")
    lot = np.where(rows.board.eq("star"), 200, 100)
    # The archived path deliberately rounds STAR targets to conservative
    # 200-share lots. Keep that assumption on both sides of this comparison.
    rows["future_shares"] = (
        np.floor(100_000 / raw_vwap / lot) * lot
    ).astype(int)
    rows["changed"] = rows.target_shares.ne(rows.future_shares)
    return {
        "filled_stock_days": int(len(rows)),
        "recorded_share_difference_rate": float(rows.changed.mean()),
        "by_half_board": {
            f"{half}_{board}": {
                "filled_stock_days": int(len(part)),
                "recorded_share_difference_rate": float(part.changed.mean()),
            }
            for (half, board), part in rows.groupby(["half", "board"])
        },
        "scope": "filled 14:49-qualified stock-days; share count only",
    }


def evaluate(source_dir: Path = SOURCE, output_dir: Path = OUTPUT,
             full_entry: Path = FULL_ENTRY) -> dict:
    report = {
        "strict_pairs": _pair_comparison(source_dir, output_dir),
        "full_market_input_only": _full_market_share_check(full_entry),
        "years": [2024, 2025],
        "cash_definition": "on-time quality-clean signal-day score, not portfolio P&L",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--full-entry", type=Path, default=FULL_ENTRY)
    args = parser.parse_args()
    result = evaluate(args.source, args.output, args.full_entry)
    print({"strict_pairs": result["strict_pairs"]["compared_rows"],
           "share_plan_sensitive": result["strict_pairs"][
               "share_plan_sensitive"],
           "trade_outcome_sensitive": result["strict_pairs"][
               "trade_outcome_sensitive"],
           "full_market_filled": result["full_market_input_only"][
               "filled_stock_days"]})


if __name__ == "__main__":
    main()
