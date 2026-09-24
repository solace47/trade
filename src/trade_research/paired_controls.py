"""Compare executable signal baskets with a fixed same-day random control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .market_study import _week_bootstrap
from .strategy_scan import _stressed_returns


def _daily_cash(frame: pd.DataFrame, dates: list[str]) -> tuple[pd.Series, pd.Series]:
    counts = frame.groupby("date").size().reindex(dates)
    if counts.isna().any():
        raise ValueError("Missing signal day in candidate or control")
    valid = frame.loc[frame.exit_status.eq("filled")
                      & frame.quality_clean_exit].copy()
    valid["stress10"] = _stressed_returns(valid, 10)
    normal = valid.groupby("date").net_return.sum().reindex(dates, fill_value=0) / counts
    stressed = valid.groupby("date").stress10.sum().reindex(dates, fill_value=0) / counts
    return normal, stressed


def compare(outcomes: Path, membership: Path, group_column: str,
            control_outcomes: Path, control_membership: Path,
            control_name: str, output: Path,
            target_notional: int = 20_000) -> list[dict]:
    candidate = pd.read_parquet(outcomes).merge(
        pd.read_parquet(membership), on=["date", "code"],
        validate="many_to_many"
    )
    control = pd.read_parquet(control_outcomes).merge(
        pd.read_parquet(control_membership), on=["date", "code"],
        validate="many_to_many"
    )
    candidate = candidate.loc[candidate.target_notional.eq(target_notional)
                              & candidate.exit_window.eq("close")].copy()
    control = control.loc[control.target_notional.eq(target_notional)
                          & control.exit_window.eq("close")
                          & control.candidate.eq(control_name)].copy()
    rows = []
    for (group, horizon), all_group in candidate.groupby([group_column, "horizon"]):
        for period, frame in (("full", all_group),
                              ("H1", all_group.loc[all_group.date.str[5:7].astype(int).le(6)]),
                              ("H2", all_group.loc[all_group.date.str[5:7].astype(int).gt(6)])):
            if frame.empty:
                continue
            dates = sorted(frame.date.unique())
            reference = control.loc[control.horizon.eq(horizon)
                                    & control.date.isin(dates)]
            normal, stressed = _daily_cash(frame, dates)
            random, _ = _daily_cash(reference, dates)
            valid = frame.loc[frame.exit_status.eq("filled")
                              & frame.quality_clean_exit]
            delta = normal - random
            rows.append({
                "group": group, "horizon": int(horizon), "period": period,
                "signals": len(frame), "days": len(dates),
                "entry_fills": int(frame.entry_status.eq("filled").sum()),
                "clean_exits": len(valid),
                "ontime_exits": int(valid.exit_delay_sessions.eq(0).sum()),
                "median_trade_net": float(valid.net_return.median()),
                "cash_mean": float(normal.mean()),
                "cash_stress10_mean": float(stressed.mean()),
                "cash_week_ci": _week_bootstrap(
                    normal.reset_index(drop=True), pd.Series(dates), 71
                ),
                "same_day_control_mean": float(random.mean()),
                "edge_mean": float(delta.mean()),
                "edge_week_ci": _week_bootstrap(
                    delta.reset_index(drop=True), pd.Series(dates), 72
                ),
            })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--group-column", choices=("candidate", "stratum"),
                        default="candidate")
    parser.add_argument("--control-outcomes", type=Path, required=True)
    parser.add_argument("--control-membership", type=Path, required=True)
    parser.add_argument("--control-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--notional", type=int, default=20_000)
    args = parser.parse_args()
    rows = compare(args.outcomes, args.membership, args.group_column,
                   args.control_outcomes, args.control_membership,
                   args.control_name, args.output, args.notional)
    print({"comparisons": len(rows), "output": str(args.output)})


if __name__ == "__main__":
    main()
