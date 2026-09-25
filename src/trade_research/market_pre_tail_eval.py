"""Evaluate frozen late-decline pairs by the market's pre-tail direction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .late_market_direction import (
    CONTROL, SOURCE, TREATMENT, _full_pool_diagnostic,
    _state_contrast, _summary,
)
from .late_variance_risk_eval import _archive_check
from .market_pre_tail_state import OUTPUT
from .strategy_scan import _stressed_returns


def _down_minus_up(pairs: pd.DataFrame, field: str, block: str) -> dict:
    original = _state_contrast(pairs, field, block)
    return {
        "down_minus_up": -original["up_minus_down"],
        "ci": [-original["ci"][1], -original["ci"][0]],
    }


def evaluate(output_dir: Path = OUTPUT, source_dir: Path = SOURCE,
             full_pool: bool = True) -> dict:
    audit = json.loads((output_dir / "input_audit.json").read_text(
        encoding="utf-8"))
    if not audit["outcome_gate_passed"]:
        raise ValueError("14:20 market-state input gate failed")
    states = pd.read_parquet(output_dir / "states.parquet", columns=[
        "date", "pre_tail_state",
    ]).rename(columns={"pre_tail_state": "market_state"})
    if (states.duplicated("date").any()
            or len(states) != audit["market_dates"]):
        raise ValueError("Market-state file differs from input audit")
    signals = pd.read_parquet(source_dir / "selections.parquet", columns=[
        "date", "code", "candidate", "pair_id",
    ])
    frozen = pd.read_parquet(output_dir / "paired_signals.parquet")
    label_keys = ["date", "code", "candidate", "pair_id"]
    if (len(signals) != audit["frozen_pairs"] * 2
            or len(frozen) != len(signals)
            or len(signals.merge(frozen[label_keys], on=label_keys,
                                 validate="one_to_one")) != len(signals)):
        raise ValueError("Frozen pair list differs from input audit")
    repriced = pd.read_parquet(source_dir / "repriced.parquet")
    repriced = repriced.loc[repriced.horizon.eq(1)].copy()
    key = ["date", "code", "target_notional", "entry_window",
           "exit_window", "horizon"]
    if (len(repriced) != len(signals) * 2 * 2
            or repriced.duplicated(key).any()
            or set(repriced.target_notional) != {20000, 100000}
            or set(repriced.entry_window) != {"baseline"}
            or set(repriced.exit_window) != {"morning", "close"}):
        raise ValueError("Incomplete T+1 raw-minute trade grid")
    archive = _archive_check(repriced.loc[repriced.exit_window.eq("close")])
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
        "archive_check_100k_close_t1": archive,
        "cost_stress_bps_each_side": 15,
        "delayed_or_unfilled_exit_cash": 0,
        "results": {},
    }
    for amount in (20000, 100000):
        report["results"][str(amount)] = {}
        for window in ("morning", "close"):
            rows = labelled.loc[
                labelled.target_notional.eq(amount)
                & labelled.exit_window.eq(window)]
            treatment = rows.loc[rows.candidate.eq(TREATMENT)]
            control = rows.loc[rows.candidate.eq(CONTROL)]
            pairs = treatment.merge(
                control, on=["date", "pair_id"],
                suffixes=("_treatment", "_control"),
                validate="one_to_one",
            ).merge(states, on="date", validate="many_to_one")
            if len(pairs) != audit["frozen_pairs"]:
                raise ValueError("A frozen pair lacks a market state or trade")
            pairs["edge"] = pairs.cash_treatment - pairs.cash_control
            detail = {}
            for year in ("2024", "2025"):
                annual = pairs.loc[pairs.date.str.startswith(year)]
                month = annual.date.str[5:7].astype(int)
                for period, section in (
                    (year, annual),
                    (f"{year}H1", annual.loc[month.le(6)]),
                    (f"{year}H2", annual.loc[month.gt(6)]),
                ):
                    detail[period] = {
                        state: _summary(section.loc[
                            section.market_state.eq(state)])
                        for state in ("down", "up", "flat")
                    }
                    detail[period]["down_minus_up_edge"] = {
                        block: _down_minus_up(section, "edge", block)
                        for block in ("week", "month")
                    }
                    detail[period]["down_minus_up_treatment"] = {
                        block: _down_minus_up(
                            section, "cash_treatment", block)
                        for block in ("week", "month")
                    }
            report["results"][str(amount)][window] = detail
    if full_pool:
        report["full_pool_post_hoc_100k_t1_close"] = _full_pool_diagnostic(states)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--skip-full-pool", action="store_true")
    args = parser.parse_args()
    report = evaluate(args.output, args.source, not args.skip_full_pool)
    print({"archive_check": report["archive_check_100k_close_t1"],
           "report": str(args.output / "report.json")})


if __name__ == "__main__":
    main()
