"""Extract 14:50-safe market-adjusted realized variance for all SH/SZ stocks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

from .study_periods import DEVELOPMENT_YEAR, VALIDATION_YEAR


ROOT = Path("data/research")
OUTPUT = ROOT / "late_residual_variance"


def _literal(path: Path) -> str:
    return str(path).replace("'", "''")


def build_year(minute_paths: list[str], year: int, output_dir: Path = OUTPUT,
               threads: int = 4, min_market_stocks: int = 1000) -> dict:
    if year not in (DEVELOPMENT_YEAR, VALIDATION_YEAR):
        raise ValueError("Only 2024-2025 research years are permitted")
    if not minute_paths or threads < 1 or min_market_stocks < 2:
        raise ValueError("Minute files, threads and market breadth are required")
    output_dir.mkdir(parents=True, exist_ok=True)
    minute_output = output_dir / f"{year}_returns.parquet"
    market_output = output_dir / f"{year}_market.parquet"
    variance_output = output_dir / f"{year}_variance.parquet"
    connection = duckdb.connect()
    connection.execute(f"SET threads = {threads}")
    connection.execute("SET memory_limit = '8GB'")
    connection.execute(f"SET temp_directory = '{_literal(output_dir / 'duckdb_tmp')}'")
    connection.from_parquet(minute_paths).create_view("source_minutes")
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
            ), complete AS (
                SELECT date, code FROM selected
                GROUP BY date, code
                HAVING COUNT(*) = 31 AND COUNT(close) = 31
                   AND COUNT(DISTINCT label) = 31
                   AND MIN(close) > 0
            ), lagged AS (
                SELECT s.date, s.code, s.label, s.close,
                       LAG(s.close) OVER (
                           PARTITION BY s.date, s.code ORDER BY s.timestamp
                       ) AS previous_close
                FROM selected s JOIN complete c USING (date, code)
            )
            SELECT date, code, label, close,
                   LN(close / previous_close) AS minute_return
            FROM lagged WHERE label BETWEEN '1421' AND '1450'
        ) TO '{_literal(minute_output)}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    connection.read_parquet(str(minute_output)).create_view("returns")
    connection.execute(f"""
        COPY (
            SELECT date, label, COUNT(*) AS stocks,
                   MEDIAN(minute_return) AS market_return
            FROM returns GROUP BY date, label
            HAVING COUNT(*) >= {min_market_stocks}
        ) TO '{_literal(market_output)}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    connection.read_parquet(str(market_output)).create_view("market")
    connection.execute(f"""
        COPY (
            WITH valid_dates AS (
                SELECT date FROM market GROUP BY date
                HAVING COUNT(*) = 30
            )
            SELECT r.date, r.code,
                   MAX(r.close) FILTER (WHERE r.label = '1450') AS price_1450,
                   SUM(POWER(r.minute_return, 2))
                       AS realized_variance_last30,
                   SUM(POWER(r.minute_return - m.market_return, 2))
                       AS residual_variance_last30
            FROM returns r JOIN market m USING (date, label)
            JOIN valid_dates d USING (date)
            GROUP BY r.date, r.code HAVING COUNT(*) = 30
        ) TO '{_literal(variance_output)}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    original = ROOT / "late_variance" / f"{year}.parquet"
    connection.read_parquet(str(original)).create_view("original")
    connection.read_parquet(str(variance_output)).create_view("residual")
    audit = connection.execute("""
        SELECT COUNT(*) AS days, COUNT(DISTINCT r.date) AS market_days,
               COUNT(DISTINCT r.code) AS symbols,
               COUNT(*) FILTER (WHERE o.code IS NULL) AS absent_original,
               COUNT(*) FILTER (WHERE o.code IS NOT NULL AND (
                   ABS(r.price_1450 - o.price_1450) > .005
                   OR ABS(r.realized_variance_last30
                          - o.realized_variance_last30) > 1e-10
               )) AS mismatch_original
        FROM residual r LEFT JOIN original o USING (date, code)
    """).df().iloc[0].to_dict()
    audit["year"] = year
    audit["min_market_stocks"] = int(connection.execute(
        "SELECT MIN(stocks) FROM market").fetchone()[0])
    audit["original_days_missing_residual"] = int(connection.execute("""
        SELECT COUNT(*) FROM original o
        LEFT JOIN residual r USING (date, code)
        WHERE r.code IS NULL
    """).fetchone()[0])
    audit = {key: int(value) for key, value in audit.items()}
    audit["market_zero_share"] = float(connection.execute(
        "SELECT AVG(CASE WHEN market_return = 0 THEN 1.0 ELSE 0 END) "
        "FROM market").fetchone()[0])
    if (audit["days"] == 0
            or audit["min_market_stocks"] < min_market_stocks
            or audit["absent_original"] or audit["mismatch_original"]
            or audit["original_days_missing_residual"]):
        raise ValueError(f"Residual variance extraction failed audit: {audit}")
    (output_dir / f"{year}_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return audit


def build_mean_year(year: int, output_dir: Path = OUTPUT,
                    threads: int = 4, min_market_stocks: int = 1000) -> dict:
    """Use clipped equal-weight minute returns when the median is degenerate."""
    if year not in (DEVELOPMENT_YEAR, VALIDATION_YEAR):
        raise ValueError("Only 2024-2025 research years are permitted")
    if threads < 1 or min_market_stocks < 2:
        raise ValueError("Positive threads and market breadth are required")
    returns_path = output_dir / f"{year}_returns.parquet"
    if not returns_path.exists():
        raise FileNotFoundError(returns_path)
    market_output = output_dir / f"{year}_market_mean.parquet"
    variance_output = output_dir / f"{year}_variance_mean.parquet"
    connection = duckdb.connect()
    connection.execute(f"SET threads = {threads}")
    connection.execute("SET memory_limit = '8GB'")
    connection.execute(f"SET temp_directory = '{_literal(output_dir / 'duckdb_tmp')}'")
    connection.read_parquet(str(returns_path)).create_view("returns")
    connection.execute(f"""
        COPY (
            SELECT date, label, COUNT(*) AS stocks,
                   AVG(GREATEST(-.02, LEAST(.02, minute_return)))
                       AS market_return
            FROM returns GROUP BY date, label
            HAVING COUNT(*) >= {min_market_stocks}
        ) TO '{_literal(market_output)}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    connection.read_parquet(str(market_output)).create_view("market")
    connection.execute(f"""
        COPY (
            WITH valid_dates AS (
                SELECT date FROM market GROUP BY date HAVING COUNT(*) = 30
            )
            SELECT r.date, r.code,
                   MAX(r.close) FILTER (WHERE r.label = '1450') AS price_1450,
                   SUM(POWER(r.minute_return, 2))
                       AS realized_variance_last30,
                   SUM(POWER(r.minute_return - m.market_return, 2))
                       AS residual_variance_last30
            FROM returns r JOIN market m USING (date, label)
            JOIN valid_dates d USING (date)
            GROUP BY r.date, r.code HAVING COUNT(*) = 30
        ) TO '{_literal(variance_output)}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    connection.read_parquet(str(ROOT / "late_variance" / f"{year}.parquet")
                            ).create_view("original")
    connection.read_parquet(str(variance_output)).create_view("residual")
    comparison = connection.execute("""
        SELECT COUNT(*) AS days,
               COUNT(*) FILTER (WHERE o.code IS NULL) AS absent_original,
               COUNT(*) FILTER (WHERE o.code IS NOT NULL AND (
                   ABS(r.price_1450 - o.price_1450) > .005
                   OR ABS(r.realized_variance_last30
                          - o.realized_variance_last30) > 1e-10
               )) AS mismatch_original
        FROM residual r LEFT JOIN original o USING (date, code)
    """).df().iloc[0].to_dict()
    audit = {key: int(value) for key, value in comparison.items()}
    audit.update(year=year)
    audit["market_zero_share"] = float(connection.execute(
        "SELECT AVG(CASE WHEN market_return = 0 THEN 1.0 ELSE 0 END) "
        "FROM market").fetchone()[0])
    audit["min_market_stocks"] = int(connection.execute(
        "SELECT MIN(stocks) FROM market").fetchone()[0])
    audit["original_days_missing_residual"] = int(connection.execute("""
        SELECT COUNT(*) FROM original o
        LEFT JOIN residual r USING (date, code)
        WHERE r.code IS NULL
    """).fetchone()[0])
    if (not audit["days"] or audit["absent_original"]
            or audit["mismatch_original"]
            or audit["original_days_missing_residual"]
            or audit["min_market_stocks"] < min_market_stocks):
        raise ValueError(f"Clipped-mean extraction failed audit: {audit}")
    (output_dir / f"{year}_mean_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minute-root", type=Path,
                        default=Path("data/hf/pilot/data/stock_1m"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--year", type=int, choices=(2024, 2025))
    parser.add_argument("--market-proxy", choices=("median", "clipped_mean"),
                        default="median")
    args = parser.parse_args()
    paths = [str(args.minute_root / exchange / "*.parquet")
             for exchange in ("SH", "SZ")]
    for year in ((args.year,) if args.year else (2024, 2025)):
        if args.market_proxy == "median":
            result = build_year(paths, year, args.output_dir, args.threads)
        else:
            result = build_mean_year(year, args.output_dir, args.threads)
        print(result, flush=True)


if __name__ == "__main__":
    main()
