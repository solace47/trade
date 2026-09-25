"""Extract total and signed 14:20–14:50 variance from completed minute closes."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from .study_periods import DEVELOPMENT_YEAR, VALIDATION_YEAR


def build_year(minute_paths: list[str], year: int, output: Path,
               threads: int = 4) -> None:
    if year not in (DEVELOPMENT_YEAR, VALIDATION_YEAR):
        raise ValueError("Late variance extraction is restricted to 2024-2025")
    if not minute_paths or threads < 1:
        raise ValueError("Minute sources and a positive thread count are required")
    output.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute(f"SET threads = {threads}")
    connection.execute("SET memory_limit = '8GB'")
    temp_literal = str(output.parent / "duckdb_tmp").replace("'", "''")
    connection.execute(f"SET temp_directory = '{temp_literal}'")
    connection.from_parquet(minute_paths).create_view("source_minutes")
    path_literal = str(output).replace("'", "''")
    connection.execute(f"""
        COPY (
            WITH selected AS (
                SELECT lower(exchange) || '.' || symbol AS code,
                       strftime(timestamp, '%Y-%m-%d') AS date,
                       strftime(timestamp, '%H%M') AS label,
                       timestamp, close
                FROM source_minutes
                WHERE timestamp >= TIMESTAMP '{year}-01-01'
                  AND timestamp < TIMESTAMP '{year + 1}-01-01'
                  AND strftime(timestamp, '%H%M') BETWEEN '1420' AND '1450'
            ), lagged AS (
                SELECT *, LAG(close) OVER (
                    PARTITION BY date, code ORDER BY timestamp
                ) AS previous_close
                FROM selected
            ), returns AS (
                SELECT *, LN(close / NULLIF(previous_close, 0)) AS minute_return
                FROM lagged
            ), aggregated AS (
                SELECT date, code, COUNT(*) AS bars,
                       COUNT(DISTINCT label) AS distinct_labels,
                       MAX(close) FILTER (WHERE label = '1450') AS price_1450,
                       COUNT(*) FILTER (
                           WHERE label >= '1421' AND close > 0
                             AND previous_close > 0
                       ) AS valid_returns,
                       SUM(POWER(minute_return, 2))
                           FILTER (WHERE label >= '1421'
                                   AND close > 0 AND previous_close > 0)
                           AS realized_variance_last30,
                       SUM(POWER(minute_return, 2))
                           FILTER (WHERE label >= '1421' AND minute_return < 0)
                           AS downside_variance_last30,
                       SUM(POWER(minute_return, 2))
                           FILTER (WHERE label >= '1421' AND minute_return > 0)
                           AS upside_variance_last30,
                       COUNT(*) FILTER (
                           WHERE label >= '1421' AND minute_return < 0
                       ) AS down_moves_last30,
                       COUNT(*) FILTER (
                           WHERE label >= '1421' AND minute_return > 0
                       ) AS up_moves_last30
                FROM returns GROUP BY date, code
            )
            SELECT date, code, price_1450, realized_variance_last30,
                   coalesce(downside_variance_last30, 0)
                       AS downside_variance_last30,
                   coalesce(upside_variance_last30, 0)
                       AS upside_variance_last30,
                   coalesce(downside_variance_last30, 0)
                       / NULLIF(realized_variance_last30, 0)
                       AS downside_share_last30,
                   down_moves_last30, up_moves_last30
            FROM aggregated
            WHERE bars = 31 AND distinct_labels = 31
              AND valid_returns = 30 AND price_1450 > 0
        ) TO '{path_literal}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minute-root", type=Path,
                        default=Path("data/hf/pilot/data/stock_1m"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/research/late_variance"))
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
