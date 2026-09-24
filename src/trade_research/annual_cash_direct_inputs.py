"""Build full-market original-annual-cash inputs before reading outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .annual_cash_margin_study import _original_reports
from .exchange_public_events import trading_dates


STRATUM = ["date", "board", "size_bucket", "momentum_bucket"]
EXPECTED_SUMMARIES = {2023: 4983, 2024: 5046}


def _two_cell_coverage(inputs: pd.DataFrame,
                       exact_industry: bool = False) -> dict:
    key_fields = STRATUM + (["industry"] if exact_industry else [])
    cells = inputs.groupby(key_fields + ["cash_group"], observed=True).size()
    complete = cells.groupby(level=key_fields).size().eq(2)
    complete_keys = complete.loc[complete].index.to_frame(index=False)
    included = inputs.merge(complete_keys, on=key_fields, how="inner",
                            validate="many_to_one")
    report = {}
    for year in ("2024", "2025"):
        source = inputs.loc[inputs.date.str.startswith(year)]
        sample = included.loc[included.date.str.startswith(year)]
        strata = complete.loc[complete.index.get_level_values("date")
                              .str.startswith(year)]
        entry = {
            "source_stock_days": len(source),
            "two_cell_stock_days": len(sample),
            "source_strata": len(strata),
            "two_cell_strata": int(strata.sum()),
            "two_cell_signal_days": int(sample.date.nunique()),
            "unique_codes": int(sample.code.nunique()),
            "cash_group_stock_days": sample.cash_group.value_counts().to_dict(),
        }
        if not sample.empty:
            code_days = sample.code.value_counts()
            entry["median_stock_days_per_code"] = float(code_days.median())
            entry["top10_code_stock_day_share"] = float(
                code_days.head(10).sum() / len(sample))
            med = sample.groupby(STRATUM + ["cash_group"], observed=True)[
                ["float_mv", "avg20_amount"]].median().unstack("cash_group")
            entry["median_supported_low_ratio"] = {
                field: float(np.median(med[(field, "cash_supported")]
                                       / med[(field, "low_cash_conversion")]))
                for field in ("float_mv", "avg20_amount")
            }
        report[year] = entry
    return report


def build(snapshot_dir: Path, daily_dir: Path, calendar_path: Path,
          industry_path: Path, annual_dir: Path, output: Path,
          audit_path: Path) -> dict:
    index_paths = tuple(annual_dir / f"cninfo_original_{year}.parquet"
                        for year in (2023, 2024))
    extracted_paths = tuple(annual_dir / f"original_cash_{year}.jsonl"
                            for year in (2023, 2024))
    indices = pd.concat([pd.read_parquet(
        path, columns=["report_year", "code", "kind"])
                         for path in index_paths], ignore_index=True)
    summary_counts = indices.loc[indices.kind.eq("summary")].groupby(
        "report_year").size().to_dict()
    if summary_counts != EXPECTED_SUMMARIES:
        raise ValueError("Full original annual-summary index is required")
    all_codes = set(indices.loc[indices.kind.eq("summary"), "code"])
    original, source = _original_reports(index_paths, extracted_paths, all_codes)
    if original.empty or set(original.report_year) != {2023, 2024}:
        raise ValueError("Both original annual-report years are required")
    days = trading_dates(calendar_path, "2024-01-01", "2025-12-31")
    next_days = pd.DataFrame({"trade_date": days[:-1], "date": days[1:]})
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.register("original", original)
    connection.register("next_days", next_days)
    connection.read_parquet(str(snapshot_dir / "*.parquet")).create_view(
        "snapshots")
    connection.read_parquet(str(industry_path)).create_view("industry")
    connection.execute("""
        CREATE TEMP TABLE liquidity AS
        SELECT date, code, amount, turn, avg20_amount, traded_count
        FROM (
            SELECT date, code, amount, turn,
                   AVG(amount) OVER (
                     PARTITION BY code ORDER BY date
                     ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                   ) AS avg20_amount,
                   COUNT(*) OVER (
                     PARTITION BY code ORDER BY date
                     ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                   ) AS traded_count
            FROM read_parquet(?)
            WHERE date BETWEEN '2023-10-01' AND '2025-12-31'
              AND tradestatus = 1 AND amount > 0 AND turn > 0
        )
    """, [str(daily_dir / "*.parquet")])
    inputs = connection.execute("""
        SELECT s.date, n.trade_date, s.code,
               CASE WHEN s.code LIKE 'sh.60%' THEN 'sh_main'
                    WHEN s.code LIKE 'sh.68%' THEN 'sh_star'
                    WHEN s.code LIKE 'sz.00%' THEN 'sz_main'
                    WHEN s.code LIKE 'sz.30%' THEN 'sz_gem'
                    ELSE 'other' END AS board,
               d.amount / (d.turn / 100.0) AS float_mv,
               d.avg20_amount, s.amount_1450, s.price_1450,
               s.return20_prior_adjusted, s.return_1450,
               s.open_1450 / s.preclose - 1 AS open_gap,
               a.report_year, a.notice_date, a.pdf_url,
               a.parent_profit_raw, a.operating_cash_raw,
               a.cash_to_parent_profit, i.industry
        FROM snapshots s
        JOIN next_days n ON n.date = s.date
        JOIN liquidity d ON d.date = n.trade_date AND d.code = s.code
        JOIN original a ON a.code = s.code
          AND a.report_year = CAST(LEFT(s.date, 4) AS INTEGER) - 1
        JOIN industry i ON i.code = s.code
          AND i.effective_date <= s.date
          AND (i.next_effective_date > s.date
               OR i.next_effective_date IS NULL)
        WHERE ((s.date BETWEEN '2024-05-15' AND '2024-12-17')
            OR (s.date BETWEEN '2025-05-15' AND '2025-12-17'))
          AND a.notice_date < s.date AND n.trade_date < s.date
          AND i.industry NOT LIKE 'J%'
          AND (s.code LIKE 'sh.60%' OR s.code LIKE 'sh.68%'
            OR s.code LIKE 'sz.00%' OR s.code LIKE 'sz.30%')
          AND d.traded_count = 20 AND d.avg20_amount >= 30000000
          AND s.amount_1450 >= 30000000
          AND s.tradestatus = 1 AND s.isST = 0
          AND s.listing_age_sessions >= 20
          AND NOT s.reference_gap AND NOT s.quote_outside_traded_range
          AND s.price_1450 >= 5 AND s.preclose > 0 AND s.open_1450 > 0
          AND s.open_1450 BETWEEN s.low_1450 - .005 AND s.high_1450 + .005
          AND s.return20_prior_adjusted IS NOT NULL
          AND d.amount > 0 AND d.turn > 0
        ORDER BY s.date, s.code
    """).df()
    if (inputs.empty or inputs.duplicated(["date", "code"]).any()
            or set(inputs.date.str[:4]) != {"2024", "2025"}
            or not inputs.date.gt(inputs.notice_date).all()
            or not inputs.date.gt(inputs.trade_date).all()
            or not np.isfinite(inputs.cash_to_parent_profit).all()
            or not np.isfinite(inputs.float_mv).all()
            or inputs.float_mv.le(0).any()):
        raise ValueError("Malformed full-market annual cash inputs")
    inputs["cash_group"] = np.where(inputs.cash_to_parent_profit.ge(1),
                                    "cash_supported", "low_cash_conversion")
    connection.register("eligible", inputs)
    ranked = connection.execute("""
        WITH sized AS (
            SELECT *, NTILE(5) OVER (
                PARTITION BY date, board ORDER BY float_mv, code
            ) AS size_bucket
            FROM eligible
        )
        SELECT *, NTILE(3) OVER (
            PARTITION BY date, board, size_bucket
            ORDER BY return20_prior_adjusted, code
        ) AS momentum_bucket
        FROM sized ORDER BY date, code
    """).df()
    audit = {
        "source_summary_index_by_year": summary_counts,
        "source_indexed_summary_stock_years": source["indexed_margin_stock_years"],
        "source_extracted_stock_years": source["extracted_stock_years"],
        "source_rejected_stock_years": (source["indexed_margin_stock_years"]
                                         - source["extracted_stock_years"]),
        "source_profitable_timely_stock_years": source[
            "profitable_timely_stock_years"],
        "stock_days": len(ranked),
        "by_year": _two_cell_coverage(ranked),
        "exact_industry_by_year": _two_cell_coverage(
            ranked, exact_industry=True),
        "note": "Input-only full A-share screen; no future returns read",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_parquet(output, index=False, compression="zstd")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, default=Path(
        "data/research/market_snapshots_ci"))
    parser.add_argument("--daily", type=Path, default=Path(
        "data/baostock/market_2020_2026/daily"))
    parser.add_argument("--calendar", type=Path, default=Path(
        "data/baostock/market_2020_2026/metadata/calendar.parquet"))
    parser.add_argument("--industry", type=Path, default=Path(
        "data/research/industry_intervals.parquet"))
    parser.add_argument("--annual", type=Path, default=Path(
        "data/research/annual_cash_quality"))
    parser.add_argument("--output", type=Path, default=Path(
        "data/research/annual_cash_direct_inputs.parquet"))
    parser.add_argument("--audit", type=Path, default=Path(
        "data/research/annual_cash_direct_inputs.json"))
    args = parser.parse_args()
    audit = build(args.snapshots, args.daily, args.calendar, args.industry,
                  args.annual, args.output, args.audit)
    print({"stock_days": audit["stock_days"], "by_year": audit["by_year"]})


if __name__ == "__main__":
    main()
