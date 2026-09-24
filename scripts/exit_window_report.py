"""Audit fixed-entry 2024 T+1 exit windows, including paired timing effects."""

import argparse
import json
from pathlib import Path

import pandas as pd

from trade_research.market_study import _week_bootstrap
from trade_research.strategy_scan import _stressed_returns

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--outcomes", type=Path,
                    default=Path("data/research/exit_reprice_2024.parquet"))
parser.add_argument("--membership", type=Path,
                    default=Path("data/research/exit_candidate_map_2024.parquet"))
parser.add_argument("--output", type=Path,
                    default=Path("data/research/exit_window_report_2024.json"))
args = parser.parse_args()
outcomes = pd.read_parquet(args.outcomes)
membership = pd.read_parquet(args.membership)
trades = outcomes.loc[outcomes.horizon.eq(1)].merge(
    membership, on=["date", "code"], validate="many_to_many"
)
rows = []
for (candidate, notional, window), frame in trades.groupby(
    ["candidate", "target_notional", "exit_window"]
):
    valid = frame.loc[frame.exit_status.eq("filled") & frame.quality_clean_exit].copy()
    dates = sorted(frame.date.unique())
    per_day_signals = frame.groupby("date").size().reindex(dates)
    daily = valid.groupby("date").net_return.mean()
    cash_aware = valid.groupby("date").net_return.sum().reindex(dates, fill_value=0) / per_day_signals
    stress = _stressed_returns(valid, 10)
    valid["stress10"] = stress
    stress_daily = valid.groupby("date").stress10.mean()
    interval = _week_bootstrap(daily.reset_index(drop=True), pd.Series(daily.index), 20260924)
    cash_interval = _week_bootstrap(cash_aware.reset_index(drop=True), pd.Series(dates), 20260925)
    filled = int(frame.entry_status.eq("filled").sum())
    ontime = int(valid.exit_delay_sessions.eq(0).sum())
    provisional_gate = (
        len(valid) >= 100
        and filled / len(frame) >= .8
        and len(valid) / max(filled, 1) >= .95
        and ontime / max(len(valid), 1) >= .95
        and valid.net_return.median() > 0
        and stress_daily.mean() > 0
        and cash_interval[0] > 0
    )
    rows.append({
        "candidate": candidate, "notional": int(notional), "window": window,
        "signals": len(frame), "signal_days": len(dates), "entry_fills": filled,
        "clean_exits": len(valid), "ontime_exits": ontime,
        "unresolved_or_quality_excluded": filled - len(valid),
        "median": float(valid.net_return.median()),
        "date_mean": float(daily.mean()),
        "date_week_ci": interval,
        "date_stress10": float(stress_daily.mean()),
        "cash_aware_day_mean": float(cash_aware.mean()),
        "cash_aware_week_ci": cash_interval,
        "trimmed_trade_mean": float(valid.net_return.clip(
            lower=valid.net_return.quantile(.05),
            upper=valid.net_return.quantile(.95),
        ).mean()),
        "provisional_gate_before_random_control": bool(provisional_gate),
    })
paired = []
base = trades.loc[trades.exit_window.eq("close"), [
    "date", "code", "candidate", "target_notional", "exit_status",
    "quality_clean_exit", "exit_delay_sessions", "net_return",
]].rename(columns={field: f"{field}_close" for field in (
    "exit_status", "quality_clean_exit", "exit_delay_sessions", "net_return"
)})
for window in ("morning", "late_morning"):
    other = trades.loc[trades.exit_window.eq(window), [
        "date", "code", "candidate", "target_notional", "exit_status",
        "quality_clean_exit", "exit_delay_sessions", "net_return",
    ]]
    joint = base.merge(other, on=["date", "code", "candidate", "target_notional"],
                       validate="one_to_one")
    joint = joint.loc[
        joint.exit_status_close.eq("filled")
        & joint.quality_clean_exit_close
        & joint.exit_delay_sessions_close.eq(0)
        & joint.exit_status.eq("filled")
        & joint.quality_clean_exit
        & joint.exit_delay_sessions.eq(0)
    ].copy()
    joint["difference"] = joint.net_return - joint.net_return_close
    for (candidate, notional), frame in joint.groupby(["candidate", "target_notional"]):
        daily = frame.groupby("date").difference.mean()
        paired.append({
            "candidate": candidate, "notional": int(notional),
            "window": window, "paired_ontime_trades": len(frame),
            "paired_days": len(daily),
            "difference_mean": float(daily.mean()),
            "difference_week_ci": _week_bootstrap(
                daily.reset_index(drop=True), pd.Series(daily.index), 20260926
            ),
        })
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps({"groups": rows, "paired": paired},
                                  ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
print({"groups": len(rows), "passed_provisional_gate": sum(
    row["provisional_gate_before_random_control"] for row in rows
), "paired": len(paired), "output": str(args.output)})
