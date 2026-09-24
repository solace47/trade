"""Diagnose the frozen annual-cash pairs by public fund visibility.

The pairs and raw-minute executions were fixed before the fund interaction.
The additional subgroup definition is documented in fund-attention-plan.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .annual_cash_margin_eval import PERIODS
from .annual_cash_margin_study import CONTROL, TREATED
from .fund_attention_eval import _validate_input_audit
from .residual_liquidity import _period
from .short_interest_eval import summarize_repriced
from .strategy_scan import _stressed_returns


LEG_FIELDS = ("entry_status", "exit_status", "quality_clean_exit",
              "net_return", "exit_delay_sessions", "shares", "entry_price",
              "exit_price", "exit_date")


def _leg(pairs: pd.DataFrame, side: str) -> pd.DataFrame:
    return pd.DataFrame({"date": pairs.date.to_numpy(), **{
        field: pairs[f"{field}_{side}"].to_numpy()
        for field in LEG_FIELDS
    }})


def _segment(dates: pd.Series) -> pd.Series:
    return np.where(dates.str[5:7].astype(int).le(6), "early", "late")


def evaluate(inputs_path: Path, input_audit_path: Path,
             selected_path: Path, repriced_path: Path, trades_path: Path,
             report_path: Path) -> dict:
    inputs = pd.read_parquet(inputs_path, columns=["date", "code", "fund_visible"])
    _validate_input_audit(inputs, input_audit_path)
    selected = pd.read_parquet(selected_path)[
        ["date", "code", "candidate", "pair_code", "cash_group", "window"]]
    if (len(selected) != 1_012 or selected.duplicated(["date", "code"]).any()
            or set(selected.candidate) != {TREATED, CONTROL}
            or set(selected.date.str[:4]) != {"2024", "2025"}):
        raise ValueError("Frozen annual-cash pair list has changed")
    pair_sizes = selected.groupby(["date", "pair_code"]).candidate.agg(
        ["size", "nunique"])
    if not (pair_sizes["size"].eq(2).all()
            and pair_sizes["nunique"].eq(2).all()
            and len(pair_sizes) == 506):
        raise ValueError("Frozen high/low annual-cash pairs are incomplete")
    membership = selected.merge(inputs, on=["date", "code"],
                                how="left", validate="one_to_one")
    if membership.fund_visible.isna().any():
        raise ValueError("A frozen annual-cash stock lacks fund visibility")

    exact_path = report_path.with_name(report_path.stem + "_exact_check.json")
    exact = summarize_repriced(selected_path, repriced_path, trades_path,
                               exact_path, TREATED, CONTROL, PERIODS)
    if exact["exact_100k_rows"] != 2_024:
        raise ValueError("Stored 100k outcomes lack exact raw-minute matches")

    raw = pd.read_parquet(repriced_path)
    rows = raw.merge(membership, on=["date", "code"], how="inner",
                     validate="many_to_one")
    if len(rows) != 4_048 or rows.duplicated(
            ["date", "code", "target_notional", "horizon"]).any():
        raise ValueError("Raw-minute rows do not match the frozen pair list")
    valid = rows.loc[rows.exit_status.eq("filled")
                     & rows.quality_clean_exit]
    if (not np.isfinite(valid.net_return).all()
            or not np.allclose(_stressed_returns(valid, 5),
                               valid.net_return.to_numpy(), atol=1e-12)):
        raise ValueError("Raw-minute net returns disagree with execution costs")

    keys = ["date", "pair_code", "target_notional", "horizon"]
    high = rows.loc[rows.candidate.eq(TREATED)].drop(
        columns=["candidate", "code"])
    low = rows.loc[rows.candidate.eq(CONTROL)].drop(
        columns=["candidate", "code"])
    pairs = high.merge(low, on=keys, suffixes=("_high", "_low"),
                       validate="one_to_one")
    if (len(pairs) != 2_024
            or not pairs.cash_group_high.eq(pairs.cash_group_low).all()):
        raise ValueError("Raw-minute matched pairs are incomplete")
    pairs["year"] = pairs.date.str[:4]
    pairs["segment"] = _segment(pairs.date)
    pairs["same_fund_label"] = pairs.fund_visible_high.eq(
        pairs.fund_visible_low)

    report = {"exact_100k_rows": exact["exact_100k_rows"], "groups": {},
              "note": "Exploratory fixed-pair capacity diagnosis; 2025 is not blind"}
    for group in ("low_cash_conversion", "cash_supported"):
        report["groups"][group] = {}
        for year in ("2024", "2025"):
            report["groups"][group][year] = {}
            for segment in ("full", "early", "late"):
                sample = pairs.loc[pairs.cash_group_high.eq(group)
                                   & pairs.year.eq(year)]
                if segment != "full":
                    sample = sample.loc[sample.segment.eq(segment)]
                base = sample.loc[sample.target_notional.eq(20_000)
                                  & sample.horizon.eq(5)]
                section = {"pairs": len(base), "days": int(base.date.nunique()),
                           "label_disagreement_pairs": int(
                               (~base.same_fund_label).sum()),
                           "both_absent_pairs": int((
                               ~base.fund_visible_high
                               & ~base.fund_visible_low).sum()),
                           "both_visible_pairs": int((
                               base.fund_visible_high
                               & base.fund_visible_low).sum()),
                           "results": {}}
                for size in (20_000, 100_000):
                    section["results"][str(size)] = {}
                    for horizon in (1, 5):
                        by_execution = sample.loc[
                            sample.target_notional.eq(size)
                            & sample.horizon.eq(horizon)]
                        results = {}
                        for visible in (False, True):
                            subset = by_execution.loc[
                                by_execution.fund_visible_high.eq(visible)]
                            label = "visible" if visible else "absent"
                            if subset.empty:
                                results[label] = {"pairs": 0, "days": 0}
                                continue
                            high_leg, low_leg = _leg(subset, "high"), _leg(subset, "low")
                            scored = _period(high_leg, low_leg)
                            control = _period(low_leg, high_leg)
                            scored["same_day_control_mean"] = scored.pop(
                                "same_day_random_mean")
                            scored["stress10_control_mean"] = control[
                                "stress10_cash_mean"]
                            scored["stress10_edge_mean"] = (
                                scored["stress10_cash_mean"]
                                - scored["stress10_control_mean"])
                            scored["same_label_pairs"] = int(
                                subset.same_fund_label.sum())
                            results[label] = scored
                        section["results"][str(size)][str(horizon)] = results
                report["groups"][group][year][segment] = section
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path("data/research")
    parser.add_argument("--inputs", type=Path,
                        default=base / "fund_visibility_inputs.parquet")
    parser.add_argument("--input-audit", type=Path,
                        default=base / "fund_visibility_inputs.json")
    parser.add_argument("--selected", type=Path,
                        default=base / "annual_cash_margin_selected.parquet")
    parser.add_argument("--repriced", type=Path,
                        default=base / "annual_cash_margin_repriced.parquet")
    parser.add_argument("--trades", type=Path,
                        default=base / "annual_cash_margin_trades.parquet")
    parser.add_argument("--report", type=Path,
                        default=base / "fund_attention_capacity.json")
    args = parser.parse_args()
    result = evaluate(args.inputs, args.input_audit, args.selected,
                      args.repriced, args.trades, args.report)
    print({"exact_100k_rows": result["exact_100k_rows"],
           "report": str(args.report)})


if __name__ == "__main__":
    main()
