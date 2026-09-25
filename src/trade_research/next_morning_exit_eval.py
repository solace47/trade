"""Evaluate the registered 09:34 T+1 exit against fixed exit windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .annual_cash_direct_eval import _month_bootstrap
from .late_variance_risk_eval import _archive_check
from .market_study import _week_bootstrap
from .next_morning_exit import OUTPUT, SOURCE
from .strategy_scan import _stressed_returns


DOWN = "late_decline"
UP = "late_rally_control"


def _sections(pairs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    result = {}
    for year in ("2024", "2025"):
        annual = pairs.loc[pairs.date.str.startswith(year)].copy()
        month = annual.date.str[5:7].astype(int)
        result[year] = annual
        result[f"{year}H1"] = annual.loc[month.le(6)].copy()
        result[f"{year}H2"] = annual.loc[month.gt(6)].copy()
    if any(section.empty for section in result.values()):
        raise ValueError("A required evaluation period has no pairs")
    return result


def _intervals(values: pd.Series, dates: pd.Series, seed: int) -> dict:
    return {
        "week": _week_bootstrap(values, dates, seed),
        "month": _month_bootstrap(values, dates, seed + 1),
    }


def _summary(pairs: pd.DataFrame) -> dict:
    daily = pairs.groupby("date", sort=True).agg(
        down=("cash_down", "mean"), up=("cash_up", "mean"),
        down_stress=("stress_down", "mean"),
        up_stress=("stress_up", "mean"),
    ).reset_index()
    daily["edge"] = daily.down - daily.up
    daily["stress_edge"] = daily.down_stress - daily.up_stress
    return {
        "pairs": int(len(pairs)), "days": int(len(daily)),
        "down_entry_rate": float(pairs.entry_status_down.eq("filled").mean()),
        "up_entry_rate": float(pairs.entry_status_up.eq("filled").mean()),
        "down_on_time_clean_exit_rate": float(pairs.valid_down.mean()),
        "up_on_time_clean_exit_rate": float(pairs.valid_up.mean()),
        "down_cash": float(daily.down.mean()),
        "up_cash": float(daily.up.mean()),
        "edge_cash": float(daily.edge.mean()),
        "down_stress15": float(daily.down_stress.mean()),
        "edge_stress15": float(daily.stress_edge.mean()),
        "down_intervals": _intervals(daily.down, daily.date, 1071),
        "edge_intervals": _intervals(daily.edge, daily.date, 1073),
    }


def _increment(dynamic: pd.DataFrame, fixed: pd.DataFrame) -> dict:
    joined = dynamic[["date", "pair_id", "cash_down", "cash_up"]].merge(
        fixed[["date", "pair_id", "cash_down", "cash_up"]],
        on=["date", "pair_id"], suffixes=("_dynamic", "_fixed"),
        validate="one_to_one",
    )
    if len(joined) != len(dynamic):
        raise ValueError("Dynamic and fixed policies have different pairs")
    joined["down_gain"] = joined.cash_down_dynamic - joined.cash_down_fixed
    joined["edge_gain"] = (joined.cash_down_dynamic - joined.cash_up_dynamic
                           - joined.cash_down_fixed + joined.cash_up_fixed)
    report = {}
    for period, pairs in _sections(joined).items():
        daily = pairs.groupby("date", sort=True)[
            ["down_gain", "edge_gain"]].mean().reset_index()
        report[period] = {
            "down_gain": float(daily.down_gain.mean()),
            "edge_gain": float(daily.edge_gain.mean()),
            "down_gain_intervals": _intervals(daily.down_gain, daily.date, 1081),
            "edge_gain_intervals": _intervals(daily.edge_gain, daily.date, 1083),
        }
    return report


def _pair_policy(frame: pd.DataFrame, policy: str, pair_count: int) -> pd.DataFrame:
    window = frame.exit_choice if policy == "dynamic" else policy
    rows = frame.loc[frame.exit_window.eq(window)].copy()
    if len(rows) != pair_count * 2 or rows.duplicated(["date", "code"]).any():
        raise ValueError("Incomplete outcome grid for a registered policy")
    down = rows.loc[rows.candidate.eq(DOWN)]
    up = rows.loc[rows.candidate.eq(UP)]
    paired = down.merge(up, on=["date", "pair_id"],
                        suffixes=("_down", "_up"), validate="one_to_one")
    if len(paired) != pair_count:
        raise ValueError("A registered policy is missing a control leg")
    return paired


def evaluate(output_dir: Path = OUTPUT, source_dir: Path = SOURCE) -> dict:
    audit = json.loads((output_dir / "input_audit.json").read_text(
        encoding="utf-8"))
    if not audit["outcome_gate_passed"]:
        raise ValueError("09:34 input gate failed; outcomes must remain closed")
    decisions = pd.read_parquet(output_dir / "decisions.parquet")
    if (len(decisions) != audit["pairs"] * 2
            or decisions.duplicated(["date", "code"]).any()
            or set(decisions.exit_choice) != {"morning", "close"}):
        raise ValueError("Decisions do not match the audited input grid")
    label_columns = ["date", "code", "candidate", "pair_id", "price_1450"]
    source = pd.read_parquet(source_dir / "selections.parquet",
                             columns=label_columns)
    matched = source.merge(decisions[label_columns], on=label_columns,
                           validate="one_to_one")
    if len(source) != len(decisions) or len(matched) != len(source):
        raise ValueError("Audited decisions differ from the frozen pair list")

    repriced = pd.read_parquet(source_dir / "repriced.parquet")
    repriced = repriced.loc[repriced.horizon.eq(1)].copy()
    key = ["date", "code", "target_notional", "entry_window", "exit_window"]
    if (len(repriced) != len(decisions) * 2 * 2
            or repriced.duplicated(key).any()
            or set(repriced.target_notional) != {20000, 100000}
            or set(repriced.entry_window) != {"baseline"}
            or set(repriced.exit_window) != {"morning", "close"}):
        raise ValueError("Incomplete or duplicate T+1 raw-minute trade grid")
    archive = _archive_check(repriced.loc[repriced.exit_window.eq("close")])
    labelled = repriced.merge(
        decisions[["date", "code", "candidate", "pair_id", "exit_choice"]],
        on=["date", "code"], validate="many_to_one",
    )
    if len(labelled) != len(repriced):
        raise ValueError("An outcome lacks an audited 09:34 decision")
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
        frame = labelled.loc[labelled.target_notional.eq(amount)]
        policies = {name: _pair_policy(frame, name, audit["pairs"])
                    for name in ("dynamic", "morning", "close")}
        report["results"][str(amount)] = {
            "policies": {
                name: {period: _summary(section)
                       for period, section in _sections(pairs).items()}
                for name, pairs in policies.items()
            },
            "dynamic_minus_fixed": {
                name: _increment(policies["dynamic"], policies[name])
                for name in ("morning", "close")
            },
        }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--source", type=Path, default=SOURCE)
    args = parser.parse_args()
    report = evaluate(args.output, args.source)
    print({"archive_check": report["archive_check_100k_close_t1"],
           "report": str(args.output / "report.json")})


if __name__ == "__main__":
    main()
