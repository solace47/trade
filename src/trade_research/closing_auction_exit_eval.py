"""Evaluate a gated T+1 auction sale against the same-stock close-window sale."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .closing_auction_entry_eval import _increment
from .closing_auction_exit import OUTPUT, SOURCE
from .next_morning_exit_eval import _pair_policy, _sections, _summary
from .strategy_scan import _stressed_returns


def _verify_baseline(current: pd.DataFrame, source_dir: Path) -> int:
    original = pd.read_parquet(source_dir / "repriced.parquet")
    original = original.loc[
        original.horizon.eq(1) & original.exit_window.eq("close")
    ].copy()
    baseline = current.loc[current.exit_window.eq("close")].copy()
    columns = [
        "date", "code", "target_notional", "horizon", "entry_status",
        "entry_price", "shares", "target_exit_date", "exit_date",
        "exit_delay_sessions", "exit_status", "exit_price", "net_return",
        "quality_clean_exit",
    ]
    sort = ["date", "code", "target_notional"]
    if len(baseline) != len(original):
        raise ValueError("The baseline trade grid is incomplete")
    pd.testing.assert_frame_equal(
        baseline[columns].sort_values(sort).reset_index(drop=True),
        original[columns].sort_values(sort).reset_index(drop=True),
        check_dtype=False, check_exact=True,
    )
    return len(baseline)


def _price_difference(frame: pd.DataFrame) -> dict:
    auction = frame.loc[frame.exit_window.eq("auction"), [
        "date", "code", "candidate", "valid", "exit_price",
    ]]
    continuous = frame.loc[frame.exit_window.eq("close"), [
        "date", "code", "valid", "exit_price",
    ]]
    joined = auction.merge(continuous, on=["date", "code"],
                           suffixes=("_auction", "_close"),
                           validate="one_to_one")
    if len(joined) != len(auction):
        raise ValueError("Auction and continuous exits differ in stock-days")
    joined = joined.loc[joined.valid_auction & joined.valid_close].copy()
    joined["auction_minus_close_bps"] = (
        joined.exit_price_auction / joined.exit_price_close - 1
    ) * 10_000
    joined["half"] = joined.date.str[:4] + "H" + joined.date.str[5:7].map(
        lambda month: "1" if int(month) <= 6 else "2")
    return {
        f"{half}_{candidate}": {
            "dual_valid_stock_days": int(len(part)),
            "mean_auction_minus_close_bps": float(
                part.auction_minus_close_bps.mean()),
        }
        for (half, candidate), part in joined.groupby(["half", "candidate"])
    }


def evaluate(output_dir: Path = OUTPUT, source_dir: Path = SOURCE) -> dict:
    audit = json.loads((output_dir / "input_audit.json").read_text(
        encoding="utf-8"))
    if not audit["outcome_gate_passed"] or audit["pairs"] != 1026:
        raise ValueError("The fixed auction input gate did not pass")
    signals = pd.read_parquet(source_dir / "selections.parquet", columns=[
        "date", "code", "candidate", "pair_id", "price_1449",
    ])
    inputs = pd.read_parquet(output_dir / "auction_inputs.parquet")
    keys = ["date", "code", "candidate", "pair_id", "price_1449"]
    if (len(signals) != len(inputs)
            or len(signals) != audit["pairs"] * 2
            or len(signals.merge(inputs[keys], on=keys,
                                 validate="one_to_one")) != len(signals)):
        raise ValueError("Audited auction inputs differ from frozen pairs")
    repriced = pd.read_parquet(output_dir / "repriced.parquet")
    key = ["date", "code", "target_notional", "entry_window",
           "exit_window", "horizon"]
    if (len(repriced) != len(signals) * 2 * 2
            or repriced.duplicated(key).any()
            or set(repriced.target_notional) != {20000, 100000}
            or set(repriced.entry_window) != {"baseline"}
            or set(repriced.exit_window) != {"auction", "close"}
            or set(repriced.horizon) != {1}):
        raise ValueError("Incomplete or duplicate raw-minute T+1 trade grid")
    baseline_rows = _verify_baseline(repriced, source_dir)
    labelled = repriced.merge(signals[["date", "code", "candidate", "pair_id"]],
                              on=["date", "code"], validate="many_to_one")
    labelled["valid"] = (
        labelled.exit_status.eq("filled") & labelled.quality_clean_exit
        & labelled.exit_delay_sessions.eq(0)
    )
    labelled["cash"] = labelled.net_return.where(labelled.valid, 0.0)
    labelled["stress"] = 0.0
    labelled.loc[labelled.valid, "stress"] = _stressed_returns(
        labelled.loc[labelled.valid], 15)
    report = {
        "input_audit": audit,
        "baseline_repricing_exact_rows": baseline_rows,
        "cost_stress_bps_each_side": 15,
        "unfilled_or_delayed_cash_contribution": 0,
        "results": {},
    }
    for amount in (20000, 100000):
        part = labelled.loc[labelled.target_notional.eq(amount)]
        policies = {name: _pair_policy(part, name, audit["pairs"])
                    for name in ("close", "auction")}
        report["results"][str(amount)] = {
            "policies": {
                name: {period: _summary(section)
                       for period, section in _sections(pairs).items()}
                for name, pairs in policies.items()
            },
            "auction_minus_close": _increment(policies["auction"],
                                              policies["close"]),
            "same_stock_sale_price": _price_difference(part),
        }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = evaluate(args.output, args.source)
    print({"baseline_repricing_exact_rows":
           report["baseline_repricing_exact_rows"],
           "report": str(args.output / "report.json")})


if __name__ == "__main__":
    main()
