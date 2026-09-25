"""Score frozen late-decline pairs after raw-minute entry and exit repricing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .annual_cash_direct_eval import _month_bootstrap
from .late_variance_risk_eval import _archive_check
from .market_study import _week_bootstrap
from .strategy_scan import _stressed_returns


OUTPUT = Path("data/research/late_to_open_reversal")
DECLINE = "late_decline"
CONTROL = "late_rally_control"


def _summary(pairs: pd.DataFrame) -> dict:
    if pairs.empty:
        raise ValueError("A required evaluation period has no pairs")
    daily = pairs.groupby("date", sort=True).agg(
        decline=("cash_down", "mean"),
        control=("cash_up", "mean"),
        decline_stress=("stress_down", "mean"),
        control_stress=("stress_up", "mean"),
    ).reset_index()
    daily["edge"] = daily.decline - daily.control
    daily["stress_edge"] = daily.decline_stress - daily.control_stress
    return {
        "pairs": int(len(pairs)),
        "days": int(len(daily)),
        "decline_entry_rate": float(pairs.entry_status_down.eq("filled").mean()),
        "control_entry_rate": float(pairs.entry_status_up.eq("filled").mean()),
        "decline_on_time_clean_exit_rate": float(pairs.valid_down.mean()),
        "control_on_time_clean_exit_rate": float(pairs.valid_up.mean()),
        "decline_delayed_exit_rate": float(
            pairs.exit_delay_sessions_down.fillna(0).gt(0).mean()),
        "control_delayed_exit_rate": float(
            pairs.exit_delay_sessions_up.fillna(0).gt(0).mean()),
        "decline_cash": float(daily.decline.mean()),
        "control_cash": float(daily.control.mean()),
        "decline_minus_control_cash": float(daily.edge.mean()),
        "decline_cash_stress15": float(daily.decline_stress.mean()),
        "decline_minus_control_stress15": float(daily.stress_edge.mean()),
        "edge_week_ci": _week_bootstrap(daily.edge, daily.date, 925),
        "edge_month_ci": _month_bootstrap(daily.edge, daily.date, 926),
    }


def evaluate(output_dir: Path = OUTPUT) -> dict:
    signals = pd.read_parquet(output_dir / "selections.parquet")
    input_audit = json.loads((output_dir / "input_audit.json").read_text(
        encoding="utf-8"))
    if not input_audit["outcome_gate_passed"]:
        raise ValueError("Frozen input gate failed; outcomes must stay closed")
    repriced = pd.read_parquet(output_dir / "repriced.parquet")
    key = ["date", "code", "target_notional", "entry_window",
           "exit_window", "horizon"]
    expected = len(signals) * 2 * 2 * 2
    if (len(repriced) != expected or repriced.duplicated(key).any()
            or set(repriced.target_notional) != {20000, 100000}
            or set(repriced.entry_window) != {"baseline"}
            or set(repriced.exit_window) != {"morning", "close"}
            or set(repriced.horizon) != {1, 5}):
        raise ValueError("Incomplete or duplicate repriced trade grid")
    archive = _archive_check(repriced.loc[repriced.exit_window.eq("close")])
    labelled = repriced.merge(
        signals[["date", "code", "candidate", "pair_id"]],
        on=["date", "code"], validate="many_to_one",
    )
    if len(labelled) != expected:
        raise ValueError("Repriced trade missing a frozen pair label")
    labelled["valid"] = (
        labelled.exit_status.eq("filled") & labelled.quality_clean_exit
        & labelled.exit_delay_sessions.eq(0)
    )
    labelled["cash"] = labelled.net_return.where(labelled.valid, 0.0)
    labelled["stress"] = 0.0
    labelled.loc[labelled.valid, "stress"] = _stressed_returns(
        labelled.loc[labelled.valid], 15)
    report = {
        "archive_check_100k_close": archive,
        "input_audit": input_audit,
        "cost_stress_bps_each_side": 15,
        "delayed_or_unfilled_exit_cash": 0,
        "results": {},
    }
    pair_count = len(signals) // 2
    for amount in (20000, 100000):
        report["results"][str(amount)] = {}
        for horizon in (1, 5):
            report["results"][str(amount)][str(horizon)] = {}
            for window in ("morning", "close"):
                rows = labelled.loc[
                    labelled.target_notional.eq(amount)
                    & labelled.horizon.eq(horizon)
                    & labelled.exit_window.eq(window)
                ]
                down = rows.loc[rows.candidate.eq(DECLINE)]
                up = rows.loc[rows.candidate.eq(CONTROL)]
                pairs = down.merge(up, on=["date", "pair_id"],
                                   suffixes=("_down", "_up"),
                                   validate="one_to_one")
                if len(pairs) != pair_count:
                    raise ValueError("A frozen pair lacks an outcome")
                sections = {}
                for year in ("2024", "2025"):
                    annual = pairs.loc[pairs.date.str.startswith(year)]
                    month = annual.date.str[5:7].astype(int)
                    sections[year] = _summary(annual)
                    sections[f"{year}H1"] = _summary(annual.loc[month.le(6)])
                    sections[f"{year}H2"] = _summary(annual.loc[month.gt(6)])
                report["results"][str(amount)][str(horizon)][window] = sections
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
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
