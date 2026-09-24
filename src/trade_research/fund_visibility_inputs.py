"""Build point-in-time fund top-ten visibility without reading future returns.

The research rule is frozen in docs/fund-attention-plan.md. A newly available
report replaces a fund's previous holdings, even when the new report was
rejected by the parser. Fund reports expire after 180 calendar days.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


QUARTERS = ("2023q4", "2024q1", "2024q2", "2024q3", "2024q4",
            "2025q1", "2025q2", "2025q3")
MAX_AGE_DAYS = 180


def _load_sources(index_dir: Path, holdings_dir: Path
                  ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    indices, holdings, audits = [], [], []
    for quarter in QUARTERS:
        index_path = index_dir / f"{quarter}.parquet"
        holdings_path = holdings_dir / f"{quarter}.parquet"
        audit_path = holdings_dir / f"{quarter}_audit.parquet"
        for path in (index_path, holdings_path, audit_path):
            if not path.exists():
                raise FileNotFoundError(f"Missing full fund quarter: {path}")
        index = pd.read_parquet(index_path)
        holding = pd.read_parquet(holdings_path)
        audit = pd.read_parquet(audit_path)
        year, number = int(quarter[:4]), int(quarter[-1])
        if (index.empty or audit.empty or any(
                not frame.reportYear.astype(int).eq(year).all()
                or not frame.report_quarter.astype(int).eq(number).all()
                for frame in (index, holding, audit))):
            raise ValueError(f"Fund {quarter} source rows name another quarter")
        if (len(index) != len(audit)
                or index.uploadInfoId.duplicated().any()
                or audit.uploadInfoId.duplicated().any()
                or set(index.uploadInfoId) != set(audit.uploadInfoId)
                or not set(audit.status).issubset({"parsed", "rejected"})):
            raise ValueError(f"Fund {quarter} index and extraction audit differ")
        if (not set(holding.uploadInfoId).issubset(
                    set(audit.loc[audit.status.eq("parsed"), "uploadInfoId"]))
                or holding.duplicated(["uploadInfoId", "code"]).any()):
            raise ValueError(f"Fund {quarter} holdings are incomplete or duplicated")
        indices.append(index)
        holdings.append(holding)
        audits.append(audit)
    metadata = pd.concat(indices, ignore_index=True)
    positions = pd.concat(holdings, ignore_index=True)
    audit = pd.concat(audits, ignore_index=True)
    if metadata.uploadInfoId.duplicated().any():
        raise ValueError("Duplicate fund report ID across quarters")
    return metadata, positions, audit


def _holding_intervals(metadata: pd.DataFrame, positions: pd.DataFrame
                       ) -> pd.DataFrame:
    """Turn successive original reports into strict public-availability spans."""
    reports = metadata[["uploadInfoId", "fundId", "available_after",
                        "reportYear", "report_quarter"]].copy()
    reports["available"] = pd.to_datetime(reports.available_after,
                                           format="%Y-%m-%d", errors="raise")
    reports = reports.sort_values(["fundId", "available", "reportYear",
                                   "report_quarter", "uploadInfoId"])
    reports["period"] = (reports.reportYear.astype(int) * 4
                         + reports.report_quarter.astype(int))
    # A stale quarter uploaded after a newer one cannot replace the latest
    # available reporting period. A later revision of the same quarter can.
    reports["latest_period"] = reports.groupby("fundId").period.cummax()
    reports = reports.loc[reports.period.eq(reports.latest_period)].copy()
    reports["next_available"] = reports.groupby("fundId").available.shift(-1)
    reports["first_usable"] = reports.available + pd.Timedelta(days=1)
    reports["last_usable"] = reports.available + pd.Timedelta(days=MAX_AGE_DAYS)
    reports["last_usable"] = reports[["last_usable", "next_available"]].min(
        axis=1)
    # Every report, including an unparsed one with no positions, truncates the
    # preceding report before intervals are joined to extracted positions.
    active = reports.loc[reports.first_usable.le(reports.last_usable),
                         ["uploadInfoId", "fundId", "first_usable",
                          "last_usable"]]
    joined = positions[["uploadInfoId", "fundId", "code"]].merge(
        active, on="uploadInfoId", how="inner", validate="many_to_one",
        suffixes=("_position", ""))
    if not joined.fundId_position.eq(joined.fundId).all():
        raise ValueError("Fund holdings and report IDs have different fund owners")
    result = joined[["code", "fundId", "first_usable", "last_usable"]].copy()
    for column in ("first_usable", "last_usable"):
        result[column] = result[column].dt.strftime("%Y-%m-%d")
    return result.drop_duplicates()


def _visibility(universe: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    if universe.duplicated(["date", "code"]).any():
        raise ValueError("Duplicate stock-day in annual cash-quality universe")
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.register("universe", universe[["date", "code"]])
    connection.register("intervals", intervals)
    counts = connection.execute("""
        SELECT u.date, u.code, COUNT(DISTINCT i.fundId) AS visible_funds
        FROM universe u LEFT JOIN intervals i ON u.code = i.code
          AND u.date >= i.first_usable AND u.date <= i.last_usable
        GROUP BY u.date, u.code
    """).df()
    result = universe.merge(counts, on=["date", "code"],
                            how="left", validate="one_to_one")
    if result.visible_funds.isna().any() or len(result) != len(universe):
        raise ValueError("Incomplete fund visibility on eligible stock-days")
    result["visible_funds"] = result.visible_funds.astype("int32")
    result["fund_visible"] = result.visible_funds.gt(0)
    return result


def _four_cell_coverage(inputs: pd.DataFrame) -> dict:
    """Count feasible financing-by-visibility strata before opening outcomes."""
    for field in ("float_mv", "avg20_amount"):
        if field not in inputs or inputs[field].isna().any() or inputs[field].le(0).any():
            raise ValueError(f"Missing positive input balance field: {field}")
    extremes = inputs.loc[inputs.quintile.isin((1, 5))]
    keys = ["date", "board", "size_bucket", "cash_group"]
    cells = extremes.groupby(keys + ["fund_visible", "quintile"],
                             observed=True).size().reset_index(name="stock_days")
    complete = cells.groupby(keys, observed=True).size().eq(4).rename(
        "four_cell").reset_index()
    included = extremes.merge(complete.loc[complete.four_cell, keys],
                              on=keys, how="inner", validate="many_to_one")
    report = {}
    for cash_group in ("low_cash_conversion", "cash_supported"):
        report[cash_group] = {}
        for year in ("2024", "2025"):
            strata = complete.loc[complete.cash_group.eq(cash_group)
                                  & complete.date.str.startswith(year)]
            source = extremes.loc[extremes.cash_group.eq(cash_group)
                                  & extremes.date.str.startswith(year)]
            matched = included.loc[included.cash_group.eq(cash_group)
                                    & included.date.str.startswith(year)]
            section = {
                "source_strata": len(strata),
                "four_cell_strata": int(strata.four_cell.sum()),
                "source_signal_days": int(source.date.nunique()),
                "four_cell_signal_days": int(matched.date.nunique()),
                "source_extreme_stock_days": len(source),
                "four_cell_stock_days": len(matched),
            }
            if not matched.empty:
                medians = matched.groupby(
                    keys + ["fund_visible", "quintile"], observed=True
                )[["float_mv", "avg20_amount"]].median().unstack(
                    ["fund_visible", "quintile"])
                section["median_stratum_absent_visible_ratio"] = {
                    field: {
                        f"q{quintile}": float(np.median(
                            medians[(field, False, quintile)]
                            / medians[(field, True, quintile)]))
                        for quintile in (1, 5)
                    }
                    for field in ("float_mv", "avg20_amount")
                }
            report[cash_group][year] = section
    return report


def build(index_dir: Path, holdings_dir: Path, universe_path: Path,
          output: Path, report_path: Path) -> dict:
    metadata, positions, audit = _load_sources(index_dir, holdings_dir)
    intervals = _holding_intervals(metadata, positions)
    universe = pd.read_parquet(universe_path)
    if not universe.date.between("2024-01-01", "2025-12-31").all():
        raise ValueError("2023 and earlier cannot enter strategy stock-days")
    if set(universe.date.str[:4]) != {"2024", "2025"}:
        raise ValueError("Both exploratory signal years are required")
    result = _visibility(universe, intervals)
    quarter_reports = {
        quarter: int((metadata.reportYear.astype(int).eq(int(quarter[:4]))
                      & metadata.report_quarter.eq(int(quarter[-1]))).sum())
        for quarter in QUARTERS
    }
    report = {
        "source_reports": len(metadata),
        "quarter_source_reports": quarter_reports,
        "parsed_reports": int(audit.status.eq("parsed").sum()),
        "rejected_reports": int(audit.status.eq("rejected").sum()),
        "source_a_share_rows": len(positions),
        "visible_stock_days": int(result.fund_visible.sum()),
        "eligible_stock_days": len(result),
        "four_cell_coverage": _four_cell_coverage(result),
        "by_year": {},
        "note": "Inputs only; no future outcome was opened by this module",
    }
    for year, group in result.groupby(result.date.str[:4]):
        report["by_year"][year] = {
            "stock_days": len(group),
            "visible_share": float(group.fund_visible.mean()),
            "median_funds": float(group.visible_funds.median()),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False, compression="zstd")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path,
                        default=Path("data/research/fund_xbrl_index"))
    parser.add_argument("--holdings-dir", type=Path,
                        default=Path("data/research/fund_xbrl_holdings"))
    parser.add_argument("--universe", type=Path,
                        default=Path("data/research/annual_cash_margin_quintiles.parquet"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/research/fund_visibility_inputs.parquet"))
    parser.add_argument("--report", type=Path,
                        default=Path("data/research/fund_visibility_inputs.json"))
    args = parser.parse_args()
    print(build(args.index_dir, args.holdings_dir, args.universe,
                args.output, args.report))


if __name__ == "__main__":
    main()
