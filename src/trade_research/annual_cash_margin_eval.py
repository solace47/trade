"""Evaluate frozen original-annual-cash and margin-interest selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .annual_cash_margin_study import CONTROL, TREATED
from .margin_heterogeneity_eval import write_reprice_signals
from .market_study import _quality_keys, _quality_symbols, _week_bootstrap
from .short_interest_eval import evaluate, summarize_repriced


PERIODS = {
    f"{group}_{year}": (f"{year}_early_{group}", f"{year}_late_{group}")
    for group in ("cash_supported", "low_cash_conversion")
    for year in ("2024", "2025")
}
PERIODS.update({
    f"{group}_{year}_{half}": (f"{year}_{half}_{group}",)
    for group in ("cash_supported", "low_cash_conversion")
    for year in ("2024", "2025") for half in ("early", "late")
})


def _extreme_q5_diagnostic(quintiles_path: Path, outcome_dir: Path,
                           issues_dir: Path) -> dict:
    """Post-hoc capacity diagnostic; it is not a frozen trading rule."""
    quintiles = pd.read_parquet(quintiles_path)[
        ["date", "code", "cash_group", "quintile", "margin_interest"]]
    top = quintiles.loc[quintiles.quintile.eq(5)].copy()
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.register("top", top)
    connection.read_parquet(str(outcome_dir / "*.parquet")).create_view("o")
    connection.register("bad_days", _quality_keys(issues_dir))
    connection.register("bad_symbols", _quality_symbols(issues_dir))
    daily = connection.execute("""
        WITH ranked AS (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY date, cash_group
                ORDER BY margin_interest DESC, code
            ) <= 5 AS extreme FROM top
        ), scored AS (
            SELECT r.date, r.cash_group, r.extreme,
                CASE WHEN o.exit_status = 'filled'
                    AND NOT EXISTS (
                        SELECT 1 FROM bad_symbols b WHERE b.code = r.code
                    )
                    AND NOT EXISTS (
                        SELECT 1 FROM bad_days d WHERE d.code = r.code
                          AND d.date >= r.date AND d.date <= o.exit_date
                    ) THEN o.net_return ELSE 0 END AS cash_return
            FROM ranked r JOIN o USING (date, code)
            WHERE o.horizon = 5
        )
        SELECT date, cash_group, extreme, COUNT(*) AS n,
               AVG(cash_return) AS cash_mean
        FROM scored GROUP BY 1, 2, 3
    """).df()
    if int(daily.n.sum()) != len(top):
        raise ValueError("Incomplete full-market extreme financing diagnostic")
    report = {"note": "Exploratory 100k stored-minute comparison; no size matching",
              "groups": {}}
    for group in ("cash_supported", "low_cash_conversion"):
        report["groups"][group] = {}
        for year in ("2024", "2025"):
            section = daily.loc[daily.cash_group.eq(group)
                                & daily.date.str.startswith(year)]
            wide = section.pivot(index="date", columns="extreme",
                                 values="cash_mean")
            if (wide.empty or set(wide.columns) != {True, False}
                    or wide.isna().any().any()):
                raise ValueError("Missing same-day rest-of-quintile comparator")
            difference = wide[True] - wide[False]
            report["groups"][group][year] = {
                "days": len(wide),
                "extreme_cash_mean": float(wide[True].mean()),
                "other_q5_cash_mean": float(wide[False].mean()),
                "difference": float(difference.mean()),
                "week_ci": _week_bootstrap(
                    difference.reset_index(drop=True),
                    pd.Series(difference.index), 223),
            }
    return report


def _same_day_interaction(selected_path: Path, repriced_path: Path) -> dict:
    membership = pd.read_parquet(selected_path)[
        ["date", "code", "cash_group", "candidate"]]
    raw = pd.read_parquet(repriced_path)
    raw = raw.loc[raw.target_notional.eq(20_000) & raw.horizon.eq(5)]
    joined = raw.merge(membership, on=["date", "code"],
                       validate="one_to_one")
    if (len(joined) != len(membership)
            or joined.duplicated(["date", "code"]).any()):
        raise ValueError("Incomplete 20k original-cash raw-minute outcomes")
    valid = joined.exit_status.eq("filled") & joined.quality_clean_exit
    joined["cash_return"] = np.where(valid, joined.net_return, 0.0)
    daily = joined.groupby(["date", "cash_group", "candidate"])[
        "cash_return"].mean().unstack("candidate")
    if daily[[TREATED, CONTROL]].isna().any().any():
        raise ValueError("A daily annual-cash layer lacks a comparator")
    daily = (daily[TREATED] - daily[CONTROL]).rename("edge").reset_index()
    common = daily.pivot(index="date", columns="cash_group", values="edge")
    common = common.dropna(subset=["cash_supported", "low_cash_conversion"])
    result = {}
    for year in ("2024", "2025"):
        sample = common.loc[common.index.str.startswith(year)]
        delta = sample.cash_supported - sample.low_cash_conversion
        result[year] = {
            "common_dates": len(sample),
            "cash_supported_edge": float(sample.cash_supported.mean()),
            "low_cash_conversion_edge": float(sample.low_cash_conversion.mean()),
            "cash_minus_low_conversion_edge": float(delta.mean()),
            "week_ci": _week_bootstrap(delta.reset_index(drop=True),
                                       pd.Series(sample.index), 319),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", type=Path, default=Path(
        "data/research/annual_cash_margin_selected.parquet"))
    parser.add_argument("--quintiles", type=Path, default=Path(
        "data/research/annual_cash_margin_quintiles.parquet"))
    parser.add_argument("--snapshots", type=Path, default=Path(
        "data/research/market_snapshots_ci"))
    parser.add_argument("--outcomes", type=Path, default=Path(
        "data/research/market_outcomes_ci"))
    parser.add_argument("--issues", type=Path, default=Path(
        "data/research/market_issues_ci"))
    parser.add_argument("--report", type=Path, default=Path(
        "data/research/annual_cash_margin_report.json"))
    parser.add_argument("--trades", type=Path, default=Path(
        "data/research/annual_cash_margin_trades.parquet"))
    parser.add_argument("--reprice-signals", type=Path, default=Path(
        "data/research/annual_cash_margin_reprice_signals.parquet"))
    parser.add_argument("--repriced", type=Path)
    parser.add_argument("--repriced-report", type=Path, default=Path(
        "data/research/annual_cash_margin_reprice_report.json"))
    args = parser.parse_args()
    write_reprice_signals(args.selected, args.snapshots, args.reprice_signals)
    result = evaluate(args.selected, args.quintiles, args.outcomes,
                      args.issues, args.report, args.trades,
                      TREATED, CONTROL, PERIODS)
    result["exploratory_extreme_q5"] = _extreme_q5_diagnostic(
        args.quintiles, args.outcomes, args.issues)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print({"matched_pairs": len(pd.read_parquet(args.selected)) // 2,
           "full_market_stock_days": result["full_market_stock_days"],
           "matched_t5_edges": {key: value["5"]["edge_mean"]
                                for key, value in result["matched"].items()}})
    if args.repriced:
        reprice = summarize_repriced(args.selected, args.repriced,
                                     args.trades, args.repriced_report,
                                     TREATED, CONTROL, PERIODS)
        reprice["same_day_20k_interaction"] = _same_day_interaction(
            args.selected, args.repriced)
        args.repriced_report.write_text(
            json.dumps(reprice, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        print({"exact_100k_rows": reprice["exact_100k_rows"],
               "repriced_report": str(args.repriced_report)})


if __name__ == "__main__":
    main()
