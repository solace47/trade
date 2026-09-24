"""Build past-only stock price non-synchronicity for margin-signal dates.

Each 120-observation regression ends on the completed margin record date t.
The market and historical industry returns exclude the stock itself.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


def r_squared_from_moments(frame: pd.DataFrame) -> np.ndarray:
    """Intercept OLS R-squared for two regressors from rolling raw moments."""
    n = frame.n.to_numpy(dtype=float)
    y = frame.sum_y.to_numpy(dtype=float)
    x1 = frame.sum_x1.to_numpy(dtype=float)
    x2 = frame.sum_x2.to_numpy(dtype=float)
    s11 = frame.sum_x1_sq.to_numpy(dtype=float) - x1 * x1 / n
    s22 = frame.sum_x2_sq.to_numpy(dtype=float) - x2 * x2 / n
    s12 = frame.sum_x1_x2.to_numpy(dtype=float) - x1 * x2 / n
    s1y = frame.sum_x1_y.to_numpy(dtype=float) - x1 * y / n
    s2y = frame.sum_x2_y.to_numpy(dtype=float) - x2 * y / n
    syy = frame.sum_y_sq.to_numpy(dtype=float) - y * y / n
    determinant = s11 * s22 - s12 * s12
    valid = ((n == 120) & (frame.span.to_numpy() <= 125)
             & (s11 > 1e-8) & (s22 > 1e-8) & (syy > 1e-8)
             & (determinant > 1e-6 * s11 * s22))
    result = np.full(len(frame), np.nan)
    explained = np.zeros(len(frame))
    explained[valid] = ((s22[valid] * s1y[valid] ** 2
                         - 2 * s12[valid] * s1y[valid] * s2y[valid]
                         + s11[valid] * s2y[valid] ** 2)
                        / determinant[valid])
    result[valid] = explained[valid] / syy[valid]
    if np.any((result[valid] < -1e-6) | (result[valid] > 1 + 1e-6)):
        raise ValueError("Invalid stock return regression R-squared")
    return np.clip(result, 0, 1)


def build(daily_dir: Path, industry_intervals: Path, calendar_path: Path,
          universe_path: Path, output_path: Path) -> pd.DataFrame:
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.execute(f"""
        CREATE TEMP TABLE returns AS
        SELECT d.date, d.code, i.industry,
               d.close / d.preclose - 1 AS y, c.session_index
        FROM read_parquet('{daily_dir}/*.parquet') d
        JOIN read_parquet('{industry_intervals}') i ON i.code = d.code
          AND i.effective_date <= d.date
          AND (i.next_effective_date IS NULL
               OR d.date < i.next_effective_date)
        JOIN (
            SELECT calendar_date AS date,
                   ROW_NUMBER() OVER (ORDER BY calendar_date) AS session_index
            FROM read_parquet('{calendar_path}')
            WHERE is_trading_day = '1'
              AND calendar_date BETWEEN '2023-07-03' AND '2025-12-31'
        ) c ON c.date = d.date
        WHERE d.date BETWEEN '2023-07-03' AND '2025-12-31'
          AND d.tradestatus = 1 AND d.preclose > 0 AND d.close > 0
          AND d.close / d.preclose - 1 BETWEEN -.30 AND .30
          AND (d.code LIKE 'sh.60%' OR d.code LIKE 'sh.68%'
            OR d.code LIKE 'sz.00%' OR d.code LIKE 'sz.30%')
    """)
    if connection.execute("""
        SELECT COUNT(*) - COUNT(DISTINCT date || code) FROM returns
    """).fetchone()[0]:
        raise ValueError("Duplicated point-in-time industry/price key")
    connection.execute("""
        CREATE TEMP TABLE aligned AS
        WITH market AS (
            SELECT date, SUM(y) AS total, COUNT(*) AS members
            FROM returns GROUP BY date
        ), peers AS (
            SELECT date, industry, SUM(y) AS total, COUNT(*) AS members
            FROM returns GROUP BY date, industry
        )
        SELECT r.date, r.code, r.y, r.session_index,
               (m.total - r.y) / (m.members - 1) AS x1,
               (p.total - r.y) / (p.members - 1) AS x2
        FROM returns r
        JOIN market m USING(date)
        JOIN peers p USING(date, industry)
        WHERE m.members > 1 AND p.members >= 10
    """)
    moments = connection.execute(f"""
        WITH rolling AS (
            SELECT date AS trade_date, code, session_index,
                   COUNT(*) OVER w AS n,
                   session_index - MIN(session_index) OVER w AS span,
                   SUM(y) OVER w AS sum_y,
                   SUM(x1) OVER w AS sum_x1,
                   SUM(x2) OVER w AS sum_x2,
                   SUM(y * y) OVER w AS sum_y_sq,
                   SUM(x1 * x1) OVER w AS sum_x1_sq,
                   SUM(x2 * x2) OVER w AS sum_x2_sq,
                   SUM(x1 * x2) OVER w AS sum_x1_x2,
                   SUM(x1 * y) OVER w AS sum_x1_y,
                   SUM(x2 * y) OVER w AS sum_x2_y
            FROM aligned
            WINDOW w AS (PARTITION BY code ORDER BY session_index
                         ROWS BETWEEN 119 PRECEDING AND CURRENT ROW)
        )
        SELECT r.* EXCLUDE(session_index)
        FROM rolling r
        JOIN (SELECT DISTINCT trade_date, code
              FROM read_parquet('{universe_path}')) u USING(trade_date, code)
        WHERE r.trade_date BETWEEN '2024-01-01' AND '2025-12-31'
    """).df()
    if moments.empty or moments.duplicated(["trade_date", "code"]).any():
        raise ValueError("Empty or duplicated non-synchronicity source")
    moments["r_squared"] = r_squared_from_moments(moments)
    result = moments.loc[moments.r_squared.notna(),
                         ["trade_date", "code", "n", "span", "r_squared"]].copy()
    result["nonsynch"] = 1 - result.r_squared
    if (result.empty or not result.nonsynch.between(0, 1).all()
            or not result.trade_date.str[:4].isin(("2024", "2025")).all()):
        raise ValueError("Malformed historical non-synchronicity output")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False, compression="zstd")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily", type=Path,
                        default=Path("data/baostock/market_2020_2026/daily"))
    parser.add_argument("--industry", type=Path,
                        default=Path("data/research/industry_intervals_warmup.parquet"))
    parser.add_argument("--calendar", type=Path, default=Path(
        "data/baostock/market_2020_2026/metadata/calendar.parquet"))
    parser.add_argument("--universe", type=Path,
                        default=Path("data/research/margin_universe_quintiles.parquet"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/research/price_nonsynch_2024_2025.parquet"))
    args = parser.parse_args()
    result = build(args.daily, args.industry, args.calendar,
                   args.universe, args.output)
    print({"stock_days": len(result),
           "first": result.trade_date.min(), "last": result.trade_date.max(),
           "median_nonsynch": float(result.nonsynch.median()),
           "high_half": int(result.nonsynch.ge(.5).sum())})


if __name__ == "__main__":
    main()
