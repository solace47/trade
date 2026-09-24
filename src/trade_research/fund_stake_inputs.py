"""Construct the predeclared lower bound on disclosed fund ownership.

Only fund reports and quarter-end daily turnover are read here. Outcome files
are not opened. See docs/fund-stake-plan.md for the frozen comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .fund_visibility_inputs import QUARTERS, _holding_intervals, _load_sources


QUARTER_ENDS = {
    "2023q4": "2023-12-31", "2024q1": "2024-03-31",
    "2024q2": "2024-06-30", "2024q3": "2024-09-30",
    "2024q4": "2024-12-31", "2025q1": "2025-03-31",
    "2025q2": "2025-06-30", "2025q3": "2025-09-30",
}
STRATUM = ["date", "board", "size_bucket", "cash_group"]


def _quarter_float_shares(daily_dir: Path) -> pd.DataFrame:
    """Use the latest valid trading day within eight days before quarter end."""
    if set(QUARTER_ENDS) != set(QUARTERS):
        raise ValueError("Fund quarters lack a quarter-end denominator date")
    if not list(daily_dir.glob("*.parquet")):
        raise FileNotFoundError(f"Missing daily turnover: {daily_dir}")
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    quarters = []
    for quarter, end in QUARTER_ENDS.items():
        start = (pd.Timestamp(end) - pd.Timedelta(days=8)).strftime("%Y-%m-%d")
        daily = connection.execute("""
            SELECT code, date AS denominator_date,
                   volume / (turn / 100.0) AS float_shares
            FROM read_parquet(?)
            WHERE date BETWEEN ? AND ? AND tradestatus = 1
              AND adjustflag = 3 AND volume > 0 AND turn > 0 AND close > 0
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY code ORDER BY date DESC) = 1
        """, [str(daily_dir / "*.parquet"), start, end]).df()
        daily["reportYear"] = int(quarter[:4])
        daily["report_quarter"] = int(quarter[-1])
        quarters.append(daily)
    result = pd.concat(quarters, ignore_index=True)
    if (result.duplicated(["reportYear", "report_quarter", "code"]).any()
            or not np.isfinite(result.float_shares).all()
            or result.float_shares.le(0).any()):
        raise ValueError("Invalid quarter-end floating shares")
    return result


def _active_stakes(metadata: pd.DataFrame, positions: pd.DataFrame,
                   denominator: pd.DataFrame) -> pd.DataFrame:
    intervals = _holding_intervals(metadata, positions)
    active = intervals.merge(
        positions[["uploadInfoId", "fundId", "code", "shares",
                   "reportYear", "report_quarter"]],
        on=["uploadInfoId", "fundId", "code"], how="left",
        validate="one_to_one")
    if active.shares.isna().any() or active.shares.lt(0).any():
        raise ValueError("Active report has missing or negative stock shares")
    active = active.merge(
        denominator[["reportYear", "report_quarter", "code", "float_shares"]],
        on=["reportYear", "report_quarter", "code"], how="left",
        validate="many_to_one")
    active["reported_float_fraction"] = active.shares / active.float_shares
    return active[["code", "fundId", "first_usable", "last_usable",
                   "reported_float_fraction"]]


def _stake_by_stock_day(visible: pd.DataFrame,
                        active: pd.DataFrame) -> pd.DataFrame:
    if (visible.duplicated(["date", "code"]).any()
            or visible.visible_funds.isna().any()):
        raise ValueError("Malformed fund visibility stock-days")
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.register("u", visible[["date", "code"]])
    connection.register("p", active)
    stakes = connection.execute("""
        SELECT u.date, u.code, COUNT(DISTINCT p.fundId) AS active_funds,
               COUNT(*) FILTER (WHERE p.code IS NOT NULL
                 AND p.reported_float_fraction IS NULL) AS missing_denominator,
               SUM(p.reported_float_fraction) AS reported_float_fraction
        FROM u LEFT JOIN p ON u.code = p.code
          AND u.date >= p.first_usable AND u.date <= p.last_usable
        GROUP BY u.date, u.code
    """).df()
    result = visible.merge(stakes, on=["date", "code"], how="left",
                           validate="one_to_one")
    if (len(result) != len(visible)
            or not result.active_funds.eq(result.visible_funds).all()):
        raise ValueError("Fund ownership disagrees with frozen visibility")
    result["reported_float_fraction"] = result.reported_float_fraction.fillna(0.0)
    result.loc[result.missing_denominator.gt(0), "reported_float_fraction"] = np.nan
    valid = result.reported_float_fraction.dropna()
    if (not np.isfinite(valid).all() or valid.lt(0).any()
            or valid.gt(1).any()
            or result.loc[result.visible_funds.eq(0),
                          "reported_float_fraction"].ne(0).any()
            or result.loc[result.visible_funds.gt(0)
                          & result.missing_denominator.eq(0),
                          "reported_float_fraction"].le(0).any()):
        raise ValueError("Invalid disclosed floating-share lower bound")
    eligible = result.loc[result.visible_funds.gt(0)
                          & result.missing_denominator.eq(0),
                          STRATUM + ["code", "reported_float_fraction"]]
    connection.register("eligible", eligible)
    ranked = connection.execute("""
        SELECT date, code, CAST(NTILE(3) OVER (
            PARTITION BY date, board, size_bucket, cash_group
            ORDER BY reported_float_fraction, code) AS TINYINT)
            AS ownership_tercile
        FROM eligible
    """).df()
    result = result.merge(ranked, on=["date", "code"], how="left",
                          validate="one_to_one")
    return result


def _coverage(result: pd.DataFrame) -> dict:
    extremes = result.loc[result.quintile.isin((1, 5))
                          & result.ownership_tercile.isin((1, 3))]
    cells = extremes.groupby(STRATUM + ["ownership_tercile", "quintile"],
                             observed=True).size().reset_index(name="n")
    complete = cells.groupby(STRATUM, observed=True).size().eq(4).rename(
        "complete").reset_index()
    included = extremes.merge(complete.loc[complete.complete, STRATUM],
                              on=STRATUM, how="inner", validate="many_to_one")
    summary = {}
    for cash_group in ("low_cash_conversion", "cash_supported"):
        summary[cash_group] = {}
        for year in ("2024", "2025"):
            source = extremes.loc[extremes.cash_group.eq(cash_group)
                                  & extremes.date.str.startswith(year)]
            strata = complete.loc[complete.cash_group.eq(cash_group)
                                  & complete.date.str.startswith(year)]
            sample = included.loc[included.cash_group.eq(cash_group)
                                   & included.date.str.startswith(year)]
            entry = {
                "source_extreme_stock_days": len(source),
                "four_cell_stock_days": len(sample),
                "source_strata": len(strata),
                "four_cell_strata": int(strata.complete.sum()),
                "four_cell_signal_days": int(sample.date.nunique()),
            }
            if not sample.empty:
                med = sample.groupby(
                    STRATUM + ["ownership_tercile", "quintile"], observed=True
                )[["float_mv", "avg20_amount"]].median().unstack(
                    ["ownership_tercile", "quintile"])
                entry["median_low_high_ratio"] = {
                    field: float(np.median(pd.concat([
                        med[(field, 1, quintile)]
                        / med[(field, 3, quintile)]
                        for quintile in (1, 5)], ignore_index=True)))
                    for field in ("float_mv", "avg20_amount")
                }
            summary[cash_group][year] = entry
    return summary


def build(visibility_path: Path, index_dir: Path, holdings_dir: Path,
          daily_dir: Path, output: Path, audit_path: Path) -> dict:
    metadata, positions, _ = _load_sources(index_dir, holdings_dir)
    end_dates = (metadata.reportYear.astype(str) + "q"
                 + metadata.report_quarter.astype(str)).map(QUARTER_ENDS)
    if end_dates.isna().any() or not metadata.available_after.gt(end_dates).all():
        raise ValueError("Fund report was available before quarter-end denominator")
    visible = pd.read_parquet(visibility_path)
    if (set(visible.date.str[:4]) != {"2024", "2025"}
            or visible.empty or visible.duplicated(["date", "code"]).any()
            or not visible.fund_visible.eq(visible.visible_funds.gt(0)).all()):
        raise ValueError("Incomplete frozen fund-visibility inputs")
    denominator = _quarter_float_shares(daily_dir)
    active = _active_stakes(metadata, positions, denominator)
    result = _stake_by_stock_day(visible, active)
    audit = {
        "quarter_source_reports": {
            q: int((metadata.reportYear.astype(int).eq(int(q[:4]))
                    & metadata.report_quarter.eq(int(q[-1]))).sum())
            for q in QUARTERS},
        "source_holding_rows": len(positions),
        "holding_rows_without_denominator": int(positions.merge(
            denominator[["reportYear", "report_quarter", "code"]],
            on=["reportYear", "report_quarter", "code"], how="left",
            indicator=True)._merge.eq("left_only").sum()),
        "stock_days": len(result),
        "stock_days_missing_denominator": int(result.missing_denominator.gt(0).sum()),
        "visible_stock_days": int(result.fund_visible.sum()),
        "four_cell_coverage": _coverage(result),
        "note": "Inputs only; no future returns were opened",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False, compression="zstd")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visibility", type=Path, default=Path(
        "data/research/fund_visibility_inputs.parquet"))
    parser.add_argument("--index-dir", type=Path, default=Path(
        "data/research/fund_xbrl_index"))
    parser.add_argument("--holdings-dir", type=Path, default=Path(
        "data/research/fund_xbrl_holdings"))
    parser.add_argument("--daily-dir", type=Path, default=Path(
        "data/baostock/market_2020_2026/daily"))
    parser.add_argument("--output", type=Path, default=Path(
        "data/research/fund_stake_inputs.parquet"))
    parser.add_argument("--audit", type=Path, default=Path(
        "data/research/fund_stake_inputs.json"))
    args = parser.parse_args()
    audit = build(args.visibility, args.index_dir, args.holdings_dir,
                  args.daily_dir, args.output, args.audit)
    print({"stock_days": audit["stock_days"],
           "stock_days_missing_denominator": audit["stock_days_missing_denominator"],
           "four_cell_coverage": audit["four_cell_coverage"]})


if __name__ == "__main__":
    main()
