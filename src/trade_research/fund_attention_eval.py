"""Evaluate the frozen fund-visibility interaction after all inputs exist.

The source, hypothesis, and four-cell comparison are fixed in
docs/fund-attention-plan.md. The input builder requires every source quarter;
this module must not be run on a partial fund history.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .market_study import _quality_keys, _quality_symbols, _week_bootstrap


STRATUM = ["date", "board", "size_bucket", "cash_group"]
CELLS = [(False, 1), (False, 5), (True, 1), (True, 5)]


def _summarize(scored: pd.DataFrame) -> dict:
    """Compare high-minus-low financing within four-cell same-day strata."""
    if scored.empty or not scored.quintile.isin((1, 5)).all():
        raise ValueError("Missing frozen financing extreme quintiles")
    cells = scored.groupby(STRATUM + ["fund_visible", "quintile"],
                           observed=True).agg(
        stock_days=("cash_return", "size"),
        cash_mean=("cash_return", "mean"),
    ).reset_index()
    wide = cells.pivot(index=STRATUM, columns=["fund_visible", "quintile"],
                       values="cash_mean").reindex(columns=CELLS)
    four_cells = wide.dropna(subset=CELLS)
    complete = pd.DataFrame(index=four_cells.index)
    complete["absent_edge"] = (four_cells[(False, 5)]
                               - four_cells[(False, 1)])
    complete["visible_edge"] = (four_cells[(True, 5)]
                                - four_cells[(True, 1)])
    complete["interaction"] = (complete.absent_edge
                               - complete.visible_edge)
    complete = complete.reset_index()
    matched_keys = complete[STRATUM]
    included = scored.merge(matched_keys, on=STRATUM, how="inner",
                            validate="many_to_one")
    all_strata = wide.reset_index()[STRATUM]
    report = {"total_extreme_stock_days": len(scored),
              "four_cell_stock_days": len(included),
              "total_strata": len(wide), "four_cell_strata": len(complete),
              "groups": {}}
    for group in ("low_cash_conversion", "cash_supported"):
        report["groups"][group] = {}
        for year in ("2024", "2025"):
            report["groups"][group][year] = {}
            for segment in ("full", "early", "late"):
                if segment == "early":
                    dates = lambda series: series.str[5:7].astype(int).le(6)
                elif segment == "late":
                    dates = lambda series: series.str[5:7].astype(int).ge(7)
                else:
                    dates = lambda series: pd.Series(True, index=series.index)
                strata = complete.loc[
                    complete.cash_group.eq(group)
                    & complete.date.str.startswith(year)
                    & dates(complete.date)]
                source = all_strata.loc[
                    all_strata.cash_group.eq(group)
                    & all_strata.date.str.startswith(year)
                    & dates(all_strata.date)]
                sample = included.loc[
                    included.cash_group.eq(group)
                    & included.date.str.startswith(year)
                    & dates(included.date)]
                result = {"source_strata": len(source),
                          "four_cell_strata": len(strata),
                          "four_cell_stock_days": len(sample),
                          "days": int(strata.date.nunique())}
                if not strata.empty:
                    daily = strata.groupby("date", as_index=False)[
                        ["absent_edge", "visible_edge", "interaction"]].mean()
                    for field in ("absent_edge", "visible_edge", "interaction"):
                        result[field] = float(daily[field].mean())
                        result[f"{field}_week_ci"] = _week_bootstrap(
                            daily[field], daily.date, 431)
                    result["cell_stock_days"] = {
                        f"{'visible' if visible else 'absent'}_q{quintile}": int(count)
                        for (visible, quintile), count in sample.groupby(
                            ["fund_visible", "quintile"], observed=True).size().items()
                    }
                    result["entry_rate"] = float(sample.entry_filled.mean())
                    result["clean_exit_rate"] = float(sample.clean_exit.mean())
                report["groups"][group][year][segment] = result
    return report


def evaluate(inputs_path: Path, outcome_dir: Path, issues_dir: Path,
             report_path: Path) -> dict:
    if not inputs_path.exists():
        raise FileNotFoundError("Complete eight-quarter fund inputs are required")
    inputs = pd.read_parquet(inputs_path)
    if (inputs.empty or inputs.duplicated(["date", "code"]).any()
            or not inputs.date.str[:4].isin(("2024", "2025")).all()
            or inputs.fund_visible.isna().any()
            or not inputs.fund_visible.eq(inputs.visible_funds.gt(0)).all()):
        raise ValueError("Malformed or incomplete fund-visibility inputs")
    selected = inputs.loc[inputs.quintile.isin((1, 5)),
                          ["date", "code", "board", "size_bucket",
                           "cash_group", "fund_visible", "quintile"]]
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
            or scored[["entry_filled", "clean_exit", "cash_return"]].isna().any().any()
            or not np.isfinite(scored.cash_return.to_numpy()).all()):
        raise ValueError("A fund interaction stock-day lacks a clean minute outcome")
    report = _summarize(scored)
    report["note"] = ("100k stored-minute exploratory interaction; "
                      "2025 is not blind; 2026 remains untouched")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path(
        "data/research/fund_visibility_inputs.parquet"))
    parser.add_argument("--outcomes", type=Path, default=Path(
        "data/research/market_outcomes_ci"))
    parser.add_argument("--issues", type=Path, default=Path(
        "data/research/market_issues_ci"))
    parser.add_argument("--report", type=Path, default=Path(
        "data/research/fund_attention_report.json"))
    args = parser.parse_args()
    report = evaluate(args.inputs, args.outcomes, args.issues, args.report)
    print({"total_extreme_stock_days": report["total_extreme_stock_days"],
           "four_cell_strata": report["four_cell_strata"],
           "report": str(args.report)})


if __name__ == "__main__":
    main()
