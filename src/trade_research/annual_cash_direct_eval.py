"""Test the frozen full-market annual-cash comparison at 14:50."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .annual_cash_direct_inputs import (
    EXPECTED_SUMMARIES, STRATUM, _two_cell_coverage,
)
from .market_study import _quality_keys, _quality_symbols, _week_bootstrap


def _month_bootstrap(values: pd.Series, dates: pd.Series,
                     seed: int) -> list[float]:
    """Resample whole calendar months, retaining repeated stock signals."""
    monthly = pd.DataFrame({
        "value": values.to_numpy(),
        "month": dates.str[:7].to_numpy(),
    }).groupby("month")["value"].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(monthly), size=(2000, len(monthly)))
    means = (monthly["sum"].to_numpy()[draws].sum(axis=1)
             / monthly["count"].to_numpy()[draws].sum(axis=1))
    return [float(x) for x in np.quantile(means, [.025, .975])]


def _summarize(scored: pd.DataFrame, exact_industry: bool = False) -> dict:
    keys = STRATUM + (["industry"] if exact_industry else [])
    cells = scored.groupby(keys + ["cash_group"], observed=True).agg(
        net=("cash_return", "mean"), n=("cash_return", "size")
    ).reset_index()
    wide = cells.pivot(index=keys, columns="cash_group", values="net")
    complete = wide.dropna(subset=["cash_supported", "low_cash_conversion"])
    spread = complete.assign(
        difference=complete.cash_supported - complete.low_cash_conversion
    ).reset_index()
    included = scored.merge(spread[keys], on=keys, how="inner",
                            validate="many_to_one")
    report = {"two_cell_strata": len(spread),
              "two_cell_stock_days": len(included),
              "by_year": {}}
    for year in ("2024", "2025"):
        report["by_year"][year] = {}
        for segment in ("full", "early", "late"):
            period = spread.loc[spread.date.str.startswith(year)]
            if segment == "early":
                period = period.loc[period.date.str[5:7].astype(int).le(6)]
            elif segment == "late":
                period = period.loc[period.date.str[5:7].astype(int).ge(7)]
            sample = included.merge(period[keys], on=keys, how="inner",
                                    validate="many_to_one")
            section = {
                "two_cell_strata": len(period),
                "two_cell_stock_days": len(sample),
                "signal_days": int(period.date.nunique()),
            }
            if not period.empty:
                daily = period.groupby("date", as_index=False)[
                    ["cash_supported", "low_cash_conversion",
                     "difference"]].mean()
                for field in ("cash_supported", "low_cash_conversion",
                              "difference"):
                    section[field] = float(daily[field].mean())
                    section[f"{field}_month_ci"] = _month_bootstrap(
                        daily[field], daily.date, 761)
                    section[f"{field}_week_ci"] = _week_bootstrap(
                        daily[field], daily.date, 761)
                section["entry_rate"] = {
                    group: float(frame.entry_filled.mean())
                    for group, frame in sample.groupby("cash_group")
                }
                section["clean_exit_rate"] = {
                    group: float(frame.clean_exit.mean())
                    for group, frame in sample.groupby("cash_group")
                }
                section["group_stock_days"] = sample.cash_group.value_counts(
                ).to_dict()
                section["unique_codes"] = int(sample.code.nunique())
            report["by_year"][year][segment] = section
    return report


def _every_fifth_dates(dates: pd.Series) -> set[str]:
    return {
        day
        for year in ("2024", "2025")
        for index, day in enumerate(sorted(
            set(dates.loc[dates.str.startswith(year)])))
        if index % 5 == 0
    }


def evaluate(inputs_path: Path, audit_path: Path, outcome_dir: Path,
             issues_dir: Path, report_path: Path) -> dict:
    if not inputs_path.exists() or not audit_path.exists():
        raise FileNotFoundError("Complete full-market annual-cash inputs required")
    inputs = pd.read_parquet(inputs_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (inputs.empty or inputs.duplicated(["date", "code"]).any()
            or set(inputs.date.str[:4]) != {"2024", "2025"}
            or not inputs.date.gt(inputs.notice_date).all()
            or audit.get("stock_days") != len(inputs)
            or audit.get("by_year") != _two_cell_coverage(inputs)
            or audit.get("exact_industry_by_year")
               != _two_cell_coverage(inputs, exact_industry=True)
            or audit.get("source_summary_index_by_year") != {
                str(year): n for year, n in EXPECTED_SUMMARIES.items()}
            or audit.get("source_indexed_summary_stock_years")
               != sum(EXPECTED_SUMMARIES.values())):
        raise ValueError("Malformed or incomplete original annual-cash inputs")
    cells = inputs.groupby(STRATUM + ["cash_group"], observed=True).size()
    complete = cells.groupby(level=STRATUM).size().eq(2)
    keys = complete.loc[complete].index.to_frame(index=False)
    selected = inputs.merge(keys, on=STRATUM, how="inner",
                            validate="many_to_one")
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
            or scored[["entry_filled", "clean_exit", "cash_return"]]
               .isna().any().any()
            or not np.isfinite(scored.cash_return.to_numpy()).all()):
        raise ValueError("A cash-quality stock-day lacks a clean minute outcome")
    report = _summarize(scored)
    report["exact_industry"] = _summarize(scored, exact_industry=True)
    report["every_fifth_session"] = _summarize(scored.loc[
        scored.date.isin(_every_fifth_dates(inputs.date))])
    report["note"] = ("100k stored-minute exploratory association; "
                      "2025 not blind; 2026 not read")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path(
        "data/research/annual_cash_direct_inputs.parquet"))
    parser.add_argument("--audit", type=Path, default=Path(
        "data/research/annual_cash_direct_inputs.json"))
    parser.add_argument("--outcomes", type=Path, default=Path(
        "data/research/market_outcomes_ci"))
    parser.add_argument("--issues", type=Path, default=Path(
        "data/research/market_issues_ci"))
    parser.add_argument("--report", type=Path, default=Path(
        "data/research/annual_cash_direct_report.json"))
    args = parser.parse_args()
    report = evaluate(args.inputs, args.audit, args.outcomes,
                      args.issues, args.report)
    print({"two_cell_strata": report["two_cell_strata"],
           "two_cell_stock_days": report["two_cell_stock_days"],
           "report": str(args.report)})


if __name__ == "__main__":
    main()
