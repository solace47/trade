"""Past-only open-to-close Amihud proxy for next-day margin signals."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


def build(daily_dir: Path, calendar_path: Path, universe_path: Path,
          output_path: Path) -> pd.DataFrame:
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.execute(f"""
        CREATE TEMP TABLE valid_days AS
        SELECT d.date AS trade_date, d.code, c.session_index,
               abs(d.close / d.open - 1) * 100000000 / d.amount
                   AS oc_amihud_day
        FROM read_parquet('{daily_dir}/*.parquet') d
        JOIN (
            SELECT calendar_date AS date,
                   ROW_NUMBER() OVER (ORDER BY calendar_date) AS session_index
            FROM read_parquet('{calendar_path}')
            WHERE is_trading_day = '1'
              AND calendar_date BETWEEN '2023-07-03' AND '2025-12-31'
        ) c ON c.date = d.date
        WHERE d.date BETWEEN '2023-07-03' AND '2025-12-31'
          AND d.tradestatus = 1 AND d.open > 0 AND d.close > 0
          AND d.amount > 0 AND abs(d.close / d.open - 1) <= .30
          AND (d.code LIKE 'sh.60%' OR d.code LIKE 'sh.68%'
            OR d.code LIKE 'sz.00%' OR d.code LIKE 'sz.30%')
    """)
    if connection.execute("""
        SELECT COUNT(*) - COUNT(DISTINCT trade_date || code)
        FROM valid_days
    """).fetchone()[0]:
        raise ValueError("Duplicate daily price key in Amihud history")
    frame = connection.execute(f"""
        WITH rolling AS (
            SELECT trade_date, code, session_index,
                   COUNT(*) OVER w AS n,
                   session_index - MIN(session_index) OVER w AS span,
                   AVG(oc_amihud_day) OVER w AS oc_amihud60
            FROM valid_days
            WINDOW w AS (PARTITION BY code ORDER BY session_index
                         ROWS BETWEEN 59 PRECEDING AND CURRENT ROW)
        )
        SELECT r.trade_date, r.code, r.n, r.span, r.oc_amihud60
        FROM rolling r
        JOIN (SELECT DISTINCT trade_date, code
              FROM read_parquet('{universe_path}')) u
          USING(trade_date, code)
        WHERE r.trade_date BETWEEN '2024-01-01' AND '2025-12-31'
          AND r.n = 60 AND r.span <= 65 AND r.oc_amihud60 > 0
        ORDER BY r.trade_date, r.code
    """).df()
    if (frame.empty or frame.duplicated(["trade_date", "code"]).any()
            or not frame.trade_date.str[:4].isin(("2024", "2025")).all()
            or not np.isfinite(frame.oc_amihud60).all()
            or not frame.n.eq(60).all() or not frame.span.le(65).all()):
        raise ValueError("Malformed past-only open-close Amihud archive")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False, compression="zstd")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily", type=Path,
                        default=Path("data/baostock/market_2020_2026/daily"))
    parser.add_argument("--calendar", type=Path, default=Path(
        "data/baostock/market_2020_2026/metadata/calendar.parquet"))
    parser.add_argument("--universe", type=Path,
                        default=Path("data/research/margin_universe_quintiles.parquet"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/research/open_close_amihud60_2024_2025.parquet"))
    args = parser.parse_args()
    result = build(args.daily, args.calendar, args.universe, args.output)
    print({"stock_days": len(result), "first": result.trade_date.min(),
           "last": result.trade_date.max(),
           "median_oc_amihud60": float(result.oc_amihud60.median())})


if __name__ == "__main__":
    main()
