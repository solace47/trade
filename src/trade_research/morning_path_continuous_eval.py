"""Score the frozen morning-path contrast after independent raw-minute repricing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .market_study import _quality_keys, _week_bootstrap
from .morning_path_continuous import HALVES, KEY, OUTPUT
from .quality_period import load_period_bad_symbols
from .strategy_scan import _stressed_returns


def _period_quality(trades: pd.DataFrame, issues_dir: Path,
                    period_report: Path,
                    bad_days: pd.DataFrame | None = None) -> pd.DataFrame:
    """Keep bad-day exclusions but limit whole-symbol flags to 2024–2025."""
    connection = duckdb.connect()
    try:
        connection.register("trades", trades)
        connection.register("bad_days", _quality_keys(issues_dir)
                            if bad_days is None else bad_days)
        connection.register("bad_symbols", load_period_bad_symbols(
            period_report, "2024-01-01", "2025-12-31"))
        evaluated = connection.execute("""
            SELECT r.*, r.exit_status = 'filled'
                AND NOT EXISTS (SELECT 1 FROM bad_symbols b
                                WHERE b.code = r.code)
                AND NOT EXISTS (SELECT 1 FROM bad_days q
                                WHERE q.code = r.code
                                  AND q.date >= r.date
                                  AND q.date <= r.exit_date)
                AS quality_clean_exit_period
            FROM trades r
        """).df()
    finally:
        connection.close()
    if len(evaluated) != len(trades):
        raise ValueError("Period quality join changed the trade count")
    evaluated = evaluated.drop(columns="quality_clean_exit").rename(
        columns={"quality_clean_exit_period": "quality_clean_exit"})
    return evaluated


def _adjusted_edge(strata: pd.DataFrame) -> dict:
    """Date-balanced linear sensitivity at zero pre-decision input imbalance."""
    names = ("gap_pp", "day_pp", "prior20_pp", "tail_pp",
             "max5", "log_price", "log_amount")
    x = np.column_stack((np.ones(len(strata)),
                         *(strata[f"delta_{name}"].to_numpy(dtype=float)
                           for name in names)))
    y = strata.edge_cash.to_numpy(dtype=float)
    date_count = strata.groupby("date").date.transform("size").to_numpy()
    weight = np.sqrt(1 / date_count)
    coefficient, _, rank, _ = np.linalg.lstsq(
        x * weight[:, None], y * weight, rcond=None)
    if rank != x.shape[1]:
        raise ValueError("Input-adjustment design is rank deficient")
    return {"adjusted_edge_at_zero_input_difference": float(coefficient[0]),
            "covariates": list(names), "strata": int(len(strata))}


def _one_grid(signals: pd.DataFrame, trades: pd.DataFrame) -> dict:
    joined = trades.merge(signals, on=["date", "code"],
                          how="inner", validate="one_to_one")
    if len(joined) != len(signals):
        raise ValueError("A frozen signal lacks a raw-minute outcome")
    joined["valid"] = (
        joined.entry_status.eq("filled")
        & joined.exit_status.eq("filled")
        & joined.quality_clean_exit
        & joined.exit_delay_sessions.eq(0)
    )
    joined["cash"] = joined.net_return.where(joined.valid, 0.0)
    for bps in (10, 15):
        joined[f"stress{bps}"] = 0.0
        joined.loc[joined.valid, f"stress{bps}"] = _stressed_returns(
            joined.loc[joined.valid], bps)
    if joined.valid.any() and not np.allclose(
            _stressed_returns(joined.loc[joined.valid], 5),
            joined.loc[joined.valid, "net_return"], atol=1e-12):
        raise ValueError("Raw-minute return disagrees with fee model")
    joined["log_price"] = np.log(joined.price_1449)
    joined["log_amount"] = np.log(joined.amount_1449)
    cols = ("cash", "stress10", "stress15", "valid", "gap_pp",
            "day_pp", "prior20_pp", "tail_pp", "max5",
            "log_price", "log_amount")
    strata = joined.groupby(list(KEY) + ["arm"], observed=True)[list(cols)].mean(
    ).unstack("arm")
    if (strata.isna().any().any() or set(strata.columns.get_level_values(1))
            != {"morning", "afternoon"}):
        raise ValueError("Frozen contrast lost an arm")
    strata.columns = [f"{name}_{arm}" for name, arm in strata.columns]
    strata = strata.reset_index()
    for name in cols:
        strata[f"delta_{name}"] = (strata[f"{name}_morning"]
                                   - strata[f"{name}_afternoon"])
    strata["edge_cash"] = strata.delta_cash
    daily = strata.groupby(["half", "date"], observed=True).agg(
        morning_cash=("cash_morning", "mean"),
        afternoon_cash=("cash_afternoon", "mean"),
        edge_cash=("edge_cash", "mean"),
        morning_stress10=("stress10_morning", "mean"),
        morning_stress15=("stress15_morning", "mean"),
        edge_stress10=("delta_stress10", "mean"),
        edge_stress15=("delta_stress15", "mean"),
        morning_completion=("valid_morning", "mean"),
        afternoon_completion=("valid_afternoon", "mean"),
        comparable_strata=("edge_cash", "size"),
    ).reset_index()
    report = {"by_half": {}, "by_year": {}, "raw_status": {}}
    for half in HALVES:
        period = daily.loc[daily.half.eq(half)]
        arm_trades = joined.loc[joined.half.eq(half)]
        section = {"days": int(len(period)), "strata": int(
            period.comparable_strata.sum()),
            "signals_each_arm": int(len(arm_trades) // 2)}
        for name in ("morning_cash", "afternoon_cash", "edge_cash",
                     "morning_stress10", "morning_stress15",
                     "edge_stress10", "edge_stress15",
                     "morning_completion", "afternoon_completion"):
            section[name] = float(period[name].mean())
        section["input_adjustment"] = _adjusted_edge(
            strata.loc[strata.half.eq(half)])
        report["by_half"][half] = section
    for year in ("2024", "2025"):
        period = daily.loc[daily.date.str.startswith(year)]
        report["by_year"][year] = {
            "days": int(len(period)),
            "edge_cash": float(period.edge_cash.mean()),
            "edge_week_95ci": _week_bootstrap(
                period.edge_cash, period.date, 1429 + int(year)),
            "input_adjustment": _adjusted_edge(
                strata.loc[strata.date.str.startswith(year)]),
        }
    for arm in ("morning", "afternoon"):
        arm_rows = joined.loc[joined.arm.eq(arm)]
        report["raw_status"][arm] = {
            "entry": arm_rows.entry_status.value_counts().to_dict(),
            "exit": arm_rows.exit_status.value_counts().to_dict(),
            "quality_bad": int((~arm_rows.quality_clean_exit).sum()),
            "delayed": int(arm_rows.exit_delay_sessions.fillna(0).gt(0).sum()),
        }
    return report


def evaluate(output: Path = OUTPUT,
             issues_dir: Path = Path("data/research/market_issues_ci"),
             period_report: Path = Path(
                 "data/research/quality_period_2024_2025.json")) -> dict:
    audit = json.loads((output / "input_audit.json").read_text(
        encoding="utf-8"))
    if not audit["outcome_gate_passed"]:
        raise ValueError("Input gate failed; outcomes must remain closed")
    signals = pd.read_parquet(output / "selections.parquet")
    trades = pd.read_parquet(output / "repriced.parquet")
    key = ["date", "code", "target_notional", "entry_window",
           "exit_window", "horizon"]
    if (len(trades) != len(signals) * 4 or trades.duplicated(key).any()
            or set(trades.target_notional) != {20000., 100000.}
            or set(trades.entry_window) != {"baseline"}
            or set(trades.exit_window) != {"morning", "close"}
            or set(trades.horizon) != {1}):
        raise ValueError("Incomplete frozen raw-minute outcome grid")
    trades = _period_quality(trades, issues_dir, period_report)
    result = {"input_audit": audit,
              "bad_symbol_policy": "2024–2025 severe fields only",
              "cash_for_unfilled_delayed_or_bad_quality": 0,
              "cost_stress_bps_each_side": [10, 15],
              "results": {}}
    for notional in (20000., 100000.):
        result["results"][str(int(notional))] = {}
        for window in ("morning", "close"):
            subset = trades.loc[trades.target_notional.eq(notional)
                                & trades.exit_window.eq(window)]
            result["results"][str(int(notional))][window] = _one_grid(
                signals, subset)
    primary = result["results"]["100000"]["morning"]
    result["provisional_strategy_gate_passed"] = bool(
        all(section["morning_completion"] >= .90
            and section["afternoon_completion"] >= .90
            and section["morning_stress15"] > 0
            and section["edge_cash"] > 0
            and section["input_adjustment"][
                "adjusted_edge_at_zero_input_difference"] > 0
            for section in primary["by_half"].values())
        and all(section["edge_week_95ci"][0] > 0
                for section in primary["by_year"].values())
    )
    (output / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = evaluate(args.output)
    print({"report": str(args.output / "report.json"),
           "provisional_strategy_gate_passed": report[
               "provisional_strategy_gate_passed"]})
