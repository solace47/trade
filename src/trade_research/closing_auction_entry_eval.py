"""Compare auction and continuous-session entry on frozen same-day pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .annual_cash_direct_eval import _month_bootstrap
from .closing_auction_entry import OUTPUT, SOURCE
from .late_variance_risk_eval import _archive_check
from .market_study import _week_bootstrap
from .next_morning_exit_eval import _intervals, _pair_policy, _sections, _summary
from .strategy_scan import _stressed_returns


def _verify_baseline(frame: pd.DataFrame, source_dir: Path) -> None:
    original = pd.read_parquet(source_dir / "repriced.parquet")
    original = original.loc[original.horizon.eq(1)].copy()
    current = frame.loc[frame.entry_window.eq("baseline")].copy()
    columns = [
        "date", "code", "target_notional", "entry_window", "exit_window",
        "horizon", "entry_status", "entry_price", "shares", "exit_date",
        "exit_status", "exit_price", "net_return", "quality_clean_exit",
    ]
    sort = ["date", "code", "target_notional", "exit_window"]
    if len(current) != len(original):
        raise ValueError("Baseline repricing is incomplete")
    pd.testing.assert_frame_equal(
        current[columns].sort_values(sort).reset_index(drop=True),
        original[columns].sort_values(sort).reset_index(drop=True),
        check_dtype=False,
        check_exact=True,
    )


def _increment(auction: pd.DataFrame, baseline: pd.DataFrame) -> dict:
    joined = auction[["date", "pair_id", "cash_down", "cash_up",
                      "valid_down", "valid_up"]].merge(
        baseline[["date", "pair_id", "cash_down", "cash_up",
                  "valid_down", "valid_up"]],
        on=["date", "pair_id"], suffixes=("_auction", "_baseline"),
        validate="one_to_one",
    )
    if len(joined) != len(auction):
        raise ValueError("Entry windows have different pair coverage")
    joined["down_gain"] = joined.cash_down_auction - joined.cash_down_baseline
    joined["edge_gain"] = (joined.cash_down_auction - joined.cash_up_auction
                           - joined.cash_down_baseline + joined.cash_up_baseline)
    joined["down_dual_valid"] = (
        joined.valid_down_auction & joined.valid_down_baseline
    )
    result = {}
    for period, section in _sections(joined).items():
        daily = section.groupby("date", sort=True)[
            ["down_gain", "edge_gain"]].mean().reset_index()
        dual = section.loc[section.down_dual_valid].groupby("date", sort=True)[
            "down_gain"].mean().reset_index()
        result[period] = {
            "pairs": int(len(section)),
            "days": int(len(daily)),
            "down_gain_cash": float(daily.down_gain.mean()),
            "edge_gain_cash": float(daily.edge_gain.mean()),
            "down_gain_intervals": _intervals(daily.down_gain, daily.date, 1101),
            "edge_gain_intervals": _intervals(daily.edge_gain, daily.date, 1103),
            "down_dual_valid_rate": float(section.down_dual_valid.mean()),
            "down_dual_valid_pairs": int(section.down_dual_valid.sum()),
            "down_dual_valid_gain": float(dual.down_gain.mean()),
            "down_dual_valid_gain_month_ci": _month_bootstrap(
                dual.down_gain, dual.date, 1105),
            "down_dual_valid_gain_week_ci": _week_bootstrap(
                dual.down_gain, dual.date, 1107),
        }
    return result


def evaluate(output_dir: Path = OUTPUT, source_dir: Path = SOURCE) -> dict:
    audit = json.loads((output_dir / "input_audit.json").read_text(
        encoding="utf-8"))
    if not audit["outcome_gate_passed"]:
        raise ValueError("Auction input gate failed; outcomes must remain closed")
    signals = pd.read_parquet(source_dir / "selections.parquet", columns=[
        "date", "code", "candidate", "pair_id",
    ])
    inputs = pd.read_parquet(output_dir / "auction_inputs.parquet")
    keys = ["date", "code", "candidate", "pair_id"]
    if (len(signals) != len(inputs) or len(signals) != audit["pairs"] * 2
            or len(signals.merge(inputs[keys], on=keys,
                                 validate="one_to_one")) != len(signals)):
        raise ValueError("Auction audit differs from frozen pairs")

    repriced = pd.read_parquet(output_dir / "repriced.parquet")
    key = ["date", "code", "target_notional", "entry_window",
           "exit_window", "horizon"]
    if (len(repriced) != len(signals) * 2 * 2 * 2
            or repriced.duplicated(key).any()
            or set(repriced.target_notional) != {20000, 100000}
            or set(repriced.entry_window) != {"baseline", "auction"}
            or set(repriced.exit_window) != {"morning", "close"}
            or set(repriced.horizon) != {1}):
        raise ValueError("Incomplete or duplicate T+1 raw-minute trade grid")
    _verify_baseline(repriced, source_dir)
    archive = _archive_check(repriced.loc[
        repriced.entry_window.eq("baseline")
        & repriced.exit_window.eq("close")
    ])
    labelled = repriced.merge(signals, on=["date", "code"],
                              validate="many_to_one")
    if len(labelled) != len(repriced):
        raise ValueError("Repriced trade lacks a frozen pair label")
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
        "baseline_repricing_exact_rows": len(repriced) // 2,
        "archive_check_100k_baseline_close_t1": archive,
        "cost_stress_bps_each_side": 15,
        "delayed_or_unfilled_exit_cash": 0,
        "results": {},
    }
    for amount in (20000, 100000):
        report["results"][str(amount)] = {}
        for exit_window in ("morning", "close"):
            frame = labelled.loc[labelled.target_notional.eq(amount)
                                 & labelled.exit_window.eq(exit_window)]
            pairs = {
                entry: _pair_policy(frame.loc[
                    frame.entry_window.eq(entry)], exit_window,
                    audit["pairs"])
                for entry in ("baseline", "auction")
            }
            report["results"][str(amount)][exit_window] = {
                "baseline": {period: _summary(section)
                             for period, section in _sections(pairs["baseline"]).items()},
                "auction": {period: _summary(section)
                            for period, section in _sections(pairs["auction"]).items()},
                "auction_minus_baseline": _increment(
                    pairs["auction"], pairs["baseline"]),
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
    print({"baseline_exact_rows": report["baseline_repricing_exact_rows"],
           "archive_check": report["archive_check_100k_baseline_close_t1"],
           "report": str(args.output / "report.json")})


if __name__ == "__main__":
    main()
