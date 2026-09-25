"""Extract intraday features through the archive's 14:50 label.

This assumes labels denote bar ends; the publisher has not certified that
convention. Later labels are reserved for execution and outcomes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from .study_periods import DEVELOPMENT_YEAR, VALIDATION_YEAR


def build_year(minute_paths: list[str], year: int, output: Path,
               threads: int = 4) -> None:
    if year not in (DEVELOPMENT_YEAR, VALIDATION_YEAR):
        raise ValueError("Intraday strategy features are restricted to 2024-2025")
    if not minute_paths or threads < 1:
        raise ValueError("Minute sources and a positive thread count are required")
    output.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute(f"SET threads = {threads}")
    connection.from_parquet(minute_paths).create_view("source_minutes")
    path_literal = str(output).replace("'", "''")
    connection.execute(f"""
        COPY (
            WITH selected AS (
                SELECT lower(exchange) || '.' || symbol AS code,
                       strftime(timestamp, '%Y-%m-%d') AS date,
                       strftime(timestamp, '%H%M') AS label,
                       close, volume, turnover
                FROM source_minutes
                WHERE timestamp >= TIMESTAMP '{year}-01-01'
                  AND timestamp < TIMESTAMP '{year + 1}-01-01'
                  AND strftime(timestamp, '%H%M') <= '1450'
            ), aggregated AS (
                SELECT date, code, COUNT(*) AS cutoff_bars,
                       MAX(close) FILTER (WHERE label = '1000') AS price_1000,
                       MAX(close) FILTER (WHERE label = '1301') AS price_1301,
                       MAX(close) FILTER (WHERE label = '1420') AS price_1420,
                       MAX(close) FILTER (WHERE label = '1435') AS price_1435,
                       MAX(close) FILTER (WHERE label = '1450') AS price_1450,
                       SUM(volume) AS volume_1450,
                       SUM(turnover) AS amount_1450,
                       SUM(volume) FILTER (WHERE label BETWEEN '1421' AND '1450')
                           AS volume_last30,
                       SUM(turnover) FILTER (WHERE label BETWEEN '1421' AND '1450')
                           AS amount_last30,
                       SUM(volume) FILTER (WHERE label BETWEEN '1436' AND '1450')
                           AS volume_last15,
                       COUNT(*) FILTER (WHERE label = '1000') AS count_1000,
                       COUNT(*) FILTER (WHERE label = '1301') AS count_1301,
                       COUNT(*) FILTER (WHERE label = '1420') AS count_1420,
                       COUNT(*) FILTER (WHERE label = '1435') AS count_1435,
                       COUNT(*) FILTER (WHERE label = '1450') AS count_1450
                FROM selected GROUP BY date, code
            )
            SELECT date, code, price_1000, price_1301, price_1420,
                   price_1435, price_1450,
                   price_1450 / NULLIF(price_1420, 0) - 1 AS return_last30,
                   price_1450 / NULLIF(price_1435, 0) - 1 AS return_last15,
                   price_1450 / NULLIF(price_1301, 0) - 1 AS return_afternoon,
                   volume_last30 / NULLIF(volume_1450, 0) AS volume_share_last30,
                   volume_last15 / NULLIF(volume_1450, 0) AS volume_share_last15,
                   amount_last30 / NULLIF(amount_1450, 0) AS amount_share_last30,
                   price_1450 / NULLIF(amount_last30 / NULLIF(volume_last30, 0), 0)
                       - 1 AS premium_to_last30_vwap
            FROM aggregated
            WHERE cutoff_bars = 231 AND count_1000 = 1 AND count_1301 = 1
              AND count_1420 = 1 AND count_1435 = 1 AND count_1450 = 1
              AND price_1000 > 0 AND price_1301 > 0 AND price_1420 > 0
              AND price_1435 > 0 AND price_1450 > 0 AND volume_1450 > 0
        ) TO '{path_literal}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minute-root", type=Path,
                        default=Path("data/hf/pilot/data/stock_1m"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/research/intraday_features"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    paths = [str(args.minute_root / exchange / "*.parquet")
             for exchange in ("SH", "SZ")]
    for year in (DEVELOPMENT_YEAR, VALIDATION_YEAR):
        destination = args.output_dir / f"{year}.parquet"
        build_year(paths, year, destination, args.threads)
        print(f"Wrote {destination}", flush=True)


if __name__ == "__main__":
    main()
