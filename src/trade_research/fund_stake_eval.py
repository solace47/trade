"""Evaluate the frozen disclosed-fund-stake exploratory interaction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .fund_attention_eval import _validate_input_audit
from .fund_stake_inputs import STRATUM
from .fund_visibility_inputs import QUARTERS
from .market_study import _quality_keys, _quality_symbols, _week_bootstrap


CELLS = [(1, 1), (1, 5), (3, 1), (3, 5)]


def _validate_stake_audit(inputs: pd.DataFrame, audit_path: Path) -> None:
    if not audit_path.exists():
        raise FileNotFoundError("Complete eight-quarter fund stake audit required")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    quarters = audit.get("quarter_source_reports")
    if (not isinstance(quarters, dict) or set(quarters) != set(QUARTERS)
            or any(not isinstance(n, int) or n <= 0 for n in quarters.values())
            or audit.get("stock_days") != len(inputs)
            or audit.get("visible_stock_days") != int(inputs.fund_visible.sum())
            or audit.get("stock_days_missing_denominator")
               != int(inputs.missing_denominator.gt(0).sum())):
        raise ValueError("Incomplete or mismatched fund stake audit")


def _summarize(scored: pd.DataFrame) -> dict:
    if scored.empty or not scored.quintile.isin((1, 5)).all():
        raise ValueError("Missing financing extreme quintiles")
    cells = scored.groupby(STRATUM + ["ownership_tercile", "quintile"],
                           observed=True).agg(
        stock_days=("cash_return", "size"),
        cash_mean=("cash_return", "mean"),
    ).reset_index()
    wide = cells.pivot(index=STRATUM,
                       columns=["ownership_tercile", "quintile"],
                       values="cash_mean").reindex(columns=CELLS)
    four = wide.dropna(subset=CELLS)
    complete = pd.DataFrame(index=four.index)
    complete["low_stake_edge"] = four[(1, 5)] - four[(1, 1)]
    complete["high_stake_edge"] = four[(3, 5)] - four[(3, 1)]
    complete["interaction"] = (complete.high_stake_edge
                               - complete.low_stake_edge)
    complete = complete.reset_index()
    included = scored.merge(complete[STRATUM], on=STRATUM, how="inner",
                            validate="many_to_one")
    result = {
        "total_extreme_stock_days": len(scored),
        "four_cell_stock_days": len(included),
        "total_strata": len(wide),
        "four_cell_strata": len(complete),
        "groups": {},
    }
    for group in ("low_cash_conversion", "cash_supported"):
        result["groups"][group] = {}
        for year in ("2024", "2025"):
            result["groups"][group][year] = {}
            for segment in ("full", "early", "late"):
                period = complete.loc[
                    complete.cash_group.eq(group)
                    & complete.date.str.startswith(year)]
                if segment == "early":
                    period = period.loc[period.date.str[5:7].astype(int).le(6)]
                elif segment == "late":
                    period = period.loc[period.date.str[5:7].astype(int).ge(7)]
                sample = included.merge(period[STRATUM], on=STRATUM,
                                        how="inner", validate="many_to_one")
                section = {"four_cell_strata": len(period),
                           "four_cell_stock_days": len(sample),
                           "days": int(period.date.nunique())}
                if not period.empty:
                    daily = period.groupby("date", as_index=False)[
                        ["low_stake_edge", "high_stake_edge",
                         "interaction"]].mean()
                    for field in ("low_stake_edge", "high_stake_edge",
                                  "interaction"):
                        section[field] = float(daily[field].mean())
                        section[f"{field}_week_ci"] = _week_bootstrap(
                            daily[field], daily.date, 431)
                    section["entry_rate"] = float(sample.entry_filled.mean())
                    section["clean_exit_rate"] = float(sample.clean_exit.mean())
                    section["cell_stock_days"] = {
                        f"stake{stake}_q{quintile}": int(n)
                        for (stake, quintile), n in sample.groupby(
                            ["ownership_tercile", "quintile"],
                            observed=True).size().items()
                    }
                    section["cell_execution"] = {
                        f"stake{stake}_q{quintile}": {
                            "entry_rate": float(cell.entry_filled.mean()),
                            "clean_exit_rate": float(cell.clean_exit.mean()),
                        }
                        for (stake, quintile), cell in sample.groupby(
                            ["ownership_tercile", "quintile"], observed=True)
                    }
                result["groups"][group][year][segment] = section
    return result


def _execution_placebo(scored: pd.DataFrame) -> pd.DataFrame:
    """Keep actual fills but give each clean exit its stratum's common return."""
    common = scored.loc[scored.clean_exit].groupby(STRATUM)[
        "raw_net_return"].mean().rename("common_return")
    placebo = scored.join(common, on=STRATUM).copy()
    placebo["cash_return"] = np.where(
        placebo.clean_exit, placebo.common_return, 0.0)
    if not np.isfinite(placebo.cash_return.to_numpy()).all():
        raise ValueError("Execution placebo lacks a common clean return")
    return placebo


def evaluate(inputs_path: Path, visibility_audit_path: Path,
             stake_audit_path: Path, outcome_dir: Path, issues_dir: Path,
             report_path: Path) -> dict:
    if not inputs_path.exists():
        raise FileNotFoundError("Complete fund stake inputs required")
    inputs = pd.read_parquet(inputs_path)
    if (inputs.empty or inputs.duplicated(["date", "code"]).any()
            or set(inputs.date.str[:4]) != {"2024", "2025"}
            or not inputs.fund_visible.eq(inputs.visible_funds.gt(0)).all()
            or inputs.ownership_tercile.dropna().isin((1, 2, 3)).eq(False).any()
            or inputs.loc[inputs.missing_denominator.gt(0),
                          "ownership_tercile"].notna().any()
            or inputs.loc[inputs.visible_funds.eq(0),
                          "ownership_tercile"].notna().any()):
        raise ValueError("Malformed fund stake inputs")
    _validate_input_audit(inputs, visibility_audit_path)
    _validate_stake_audit(inputs, stake_audit_path)
    selected = inputs.loc[inputs.quintile.isin((1, 5))
                          & inputs.ownership_tercile.isin((1, 3)),
                          ["date", "code", "board", "size_bucket",
                           "cash_group", "ownership_tercile", "quintile"]]
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.register("r", selected)
    connection.read_parquet(str(outcome_dir / "*.parquet")).create_view("o")
    connection.register("bad_days", _quality_keys(issues_dir))
    connection.register("bad_symbols", _quality_symbols(issues_dir))
    scored = connection.execute("""
        SELECT r.*, o.entry_status = 'filled' AS entry_filled,
               o.exit_status = 'filled'
                 AND NOT EXISTS (SELECT 1 FROM bad_symbols b
                                 WHERE b.code = r.code)
                 AND NOT EXISTS (SELECT 1 FROM bad_days b
                                 WHERE b.code = r.code
                                   AND b.date >= r.date
                                   AND b.date <= o.exit_date)
                 AS clean_exit,
               o.net_return AS raw_net_return,
               CASE WHEN o.exit_status = 'filled'
                 AND NOT EXISTS (SELECT 1 FROM bad_symbols b
                                 WHERE b.code = r.code)
                 AND NOT EXISTS (SELECT 1 FROM bad_days b
                                 WHERE b.code = r.code
                                   AND b.date >= r.date
                                   AND b.date <= o.exit_date)
                 THEN o.net_return ELSE 0 END AS cash_return
        FROM r JOIN o USING (date, code)
        WHERE o.horizon = 5
    """).df()
    if (len(scored) != len(selected)
            or scored[["entry_filled", "clean_exit", "cash_return"]]
               .isna().any().any()
            or not np.isfinite(scored.cash_return.to_numpy()).all()):
        raise ValueError("A fund stake stock-day lacks a clean minute outcome")
    report = _summarize(scored)
    placebo = _summarize(_execution_placebo(scored))
    report["execution_placebo_interaction"] = {
        year: placebo["groups"]["low_cash_conversion"][year]["full"]
        ["interaction"] for year in ("2024", "2025")
    }
    report["note"] = ("100k stored-minute exploratory association; "
                      "2025 is not blind; 2026 not read")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path(
        "data/research/fund_stake_inputs.parquet"))
    parser.add_argument("--visibility-audit", type=Path, default=Path(
        "data/research/fund_visibility_inputs.json"))
    parser.add_argument("--stake-audit", type=Path, default=Path(
        "data/research/fund_stake_inputs.json"))
    parser.add_argument("--outcomes", type=Path, default=Path(
        "data/research/market_outcomes_ci"))
    parser.add_argument("--issues", type=Path, default=Path(
        "data/research/market_issues_ci"))
    parser.add_argument("--report", type=Path, default=Path(
        "data/research/fund_stake_report.json"))
    args = parser.parse_args()
    report = evaluate(args.inputs, args.visibility_audit, args.stake_audit,
                      args.outcomes, args.issues, args.report)
    print({"total_extreme_stock_days": report["total_extreme_stock_days"],
           "four_cell_strata": report["four_cell_strata"],
           "report": str(args.report)})


if __name__ == "__main__":
    main()
