"""Evaluate a frozen absolute-ridge list using independently repriced minutes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .annual_cash_direct_eval import _month_bootstrap
from .late_variance_risk_eval import _archive_check
from .market_study import _week_bootstrap
from .strategy_scan import _stressed_returns


OUTPUT = Path("data/research/absolute_ridge_main")
PERIODS = ("2024H2", "2025H1", "2025H2", "2025")


def _period(date: str) -> str:
    return date[:4] + ("H1" if int(date[5:7]) <= 6 else "H2")


def _interval(values: pd.Series, seed: int) -> dict:
    return {
        "week_ci": _week_bootstrap(values, pd.Series(values.index), seed),
        "month_ci": _month_bootstrap(values, pd.Series(values.index), seed + 1),
    }


def summarize(model: pd.DataFrame, controls: pd.DataFrame,
              primary: bool) -> dict:
    """Include every selected stock in own cash; only paired rows in edge."""
    if model.empty:
        raise ValueError("A required period lacks absolute-model selections")
    pairs = model.merge(controls, on=["date", "pair_id"],
                        suffixes=("_model", "_control"),
                        validate="one_to_one")
    if len(pairs) != len(controls):
        raise ValueError("A frozen same-day control lacks a model stock")
    own_daily = model.groupby("date", sort=True).cash.mean()
    own_stress_daily = model.groupby("date", sort=True).stress15.mean()
    paired_daily = pairs.groupby("date", sort=True).agg(
        own=("cash_model", "mean"), control=("cash_control", "mean"),
        own_stress=("stress15_model", "mean"),
        control_stress=("stress15_control", "mean"),
    )
    paired_daily["edge"] = paired_daily.own - paired_daily.control
    paired_daily["stress_edge"] = (
        paired_daily.own_stress - paired_daily.control_stress)
    result = {
        "selected": len(model), "signal_days": model.date.nunique(),
        "matched_pairs": len(pairs), "matched_days": len(paired_daily),
        "own_all_cash": float(own_daily.mean()),
        "own_all_cash_stress15": float(own_stress_daily.mean()),
        "own_matched_cash": float(paired_daily.own.mean()),
        "control_matched_cash": float(paired_daily.control.mean()),
        "matched_edge_cash": float(paired_daily.edge.mean()),
        "matched_edge_stress15": float(paired_daily.stress_edge.mean()),
        "own_entry_rate": float(model.entry_status.eq("filled").mean()),
        "own_clean_ontime_exit_rate": float(model.valid.mean()),
        "control_entry_rate": float(pairs.entry_status_control.eq("filled").mean()),
        "control_clean_ontime_exit_rate": float(pairs.valid_control.mean()),
    }
    if primary:
        result["own_intervals"] = _interval(own_daily, 1941)
        result["edge_intervals"] = _interval(paired_daily.edge, 1943)
    return result


def evaluate(output_dir: Path = OUTPUT) -> dict:
    audit = json.loads((output_dir / "input_audit.json").read_text(
        encoding="utf-8"))
    if not audit["outcome_gate_passed"]:
        raise ValueError("Frozen input gate failed; outcomes must stay closed")
    selections = pd.read_parquet(output_dir / "selections.parquet")
    repriced = pd.read_parquet(output_dir / "repriced.parquet")
    keys = ["date", "code", "target_notional", "entry_window",
            "exit_window", "horizon"]
    if (len(repriced) != len(selections) * 8
            or repriced.duplicated(keys).any()
            or set(repriced.target_notional) != {20000, 100000}
            or set(repriced.entry_window) != {"baseline"}
            or set(repriced.exit_window) != {"morning", "close"}
            or set(repriced.horizon) != {1, 5}):
        raise ValueError("Incomplete raw-minute trade grid")
    archive = _archive_check(repriced.loc[repriced.exit_window.eq("close")])
    labelled = repriced.merge(
        selections[["date", "code", "candidate", "pair_id"]],
        on=["date", "code"], validate="many_to_one")
    if len(labelled) != len(repriced):
        raise ValueError("A raw-minute outcome lacks a frozen selection")
    labelled["valid"] = (
        labelled.exit_status.eq("filled") & labelled.quality_clean_exit
        & labelled.exit_delay_sessions.eq(0)
    )
    labelled["cash"] = labelled.net_return.where(labelled.valid, 0.0)
    labelled["stress15"] = 0.0
    labelled.loc[labelled.valid, "stress15"] = _stressed_returns(
        labelled.loc[labelled.valid], 15)
    labelled["period"] = labelled.date.map(_period)
    report = {"input_audit": audit, "archive_check_100k_close": archive,
              "additional_slippage_bps_each_side": 10, "results": {}}
    for notional in (20000, 100000):
        amount_report = {}
        for horizon in (1, 5):
            horizon_report = {}
            for window in ("morning", "close"):
                rows = labelled.loc[
                    labelled.target_notional.eq(notional)
                    & labelled.horizon.eq(horizon)
                    & labelled.exit_window.eq(window)]
                model = rows.loc[rows.candidate.eq("absolute_model")]
                controls = rows.loc[rows.candidate.eq("same_day_control")]
                if (len(model) != audit["signals"]
                        or len(controls) != audit["controls"]):
                    raise ValueError("Incomplete frozen treatment or controls")
                sections = {}
                for name in PERIODS:
                    use = (rows.date.str.startswith("2025") if name == "2025"
                           else rows.period.eq(name))
                    selected = model.loc[use.loc[model.index]]
                    matched = controls.loc[use.loc[controls.index]]
                    sections[name] = summarize(
                        selected, matched,
                        primary=notional == 100000 and horizon == 5
                        and window == "close")
                horizon_report[window] = sections
            amount_report[str(horizon)] = horizon_report
        report["results"][str(notional)] = amount_report
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = evaluate(args.output)
    print({"archive_check": report["archive_check_100k_close"],
           "report": str(args.output / "report.json")})


if __name__ == "__main__":
    main()
