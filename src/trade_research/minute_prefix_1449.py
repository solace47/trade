"""Build 14:49-labeled inputs without using later source bars.

The archive does not document whether labels denote minute starts or ends.
Using labels through 14:49 keeps the measured bars before a 14:50 decision
under either interpretation, apart from unobserved feed latency.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

from .study_periods import DEVELOPMENT_YEAR, VALIDATION_YEAR


MIN_SNAPSHOT_COVERAGE = 0.999


def build_year(minute_paths: list[str], year: int, output: Path,
               threads: int = 4, *, historical_training: bool = False) -> None:
    if year not in (DEVELOPMENT_YEAR, VALIDATION_YEAR) and not (
            historical_training and year in (2022, 2023)):
        raise ValueError("Prefix inputs are restricted to 2024–2025")
    if not minute_paths or threads < 1:
        raise ValueError("Minute sources and a positive thread count are required")
    output.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    try:
        connection.execute(f"SET threads = {threads}")
        connection.execute("SET memory_limit = '8GB'")
        connection.execute("SET preserve_insertion_order = false")
        temp_literal = str(output.parent / "duckdb_tmp").replace("'", "''")
        connection.execute(f"SET temp_directory = '{temp_literal}'")
        connection.from_parquet(minute_paths).create_view("source_minutes")
        path_literal = str(output).replace("'", "''")
        connection.execute(f"""
            COPY (
                WITH labeled AS (
                    SELECT lower(exchange) || '.' || symbol AS code,
                           strftime(timestamp, '%Y-%m-%d') AS date,
                           strftime(timestamp, '%H%M') AS label,
                           open, high, low, close, volume, turnover
                    FROM source_minutes
                    WHERE timestamp >= TIMESTAMP '{year}-01-01'
                      AND timestamp < TIMESTAMP '{year + 1}-01-01'
                ), prefix AS (
                    SELECT *,
                           open IS NOT NULL AND high IS NOT NULL
                           AND low IS NOT NULL AND close IS NOT NULL
                           AND volume IS NOT NULL AND turnover IS NOT NULL
                           AND low > 0 AND high >= GREATEST(open, low, close)
                           AND low <= LEAST(open, high, close)
                           AND volume >= 0 AND turnover >= 0 AS valid_bar
                    FROM labeled
                    WHERE label BETWEEN '0930' AND '1130'
                       OR label BETWEEN '1301' AND '1449'
                ), aggregated AS (
                    SELECT date, code, COUNT(*) AS bar_count,
                           COUNT(DISTINCT label) AS unique_labels,
                           COUNT(*) FILTER (WHERE valid_bar) AS valid_bars,
                           MIN(label) AS first_label, MAX(label) AS last_label,
                           MAX(close) FILTER (WHERE label = '1420') AS price_1420,
                           MAX(close) FILTER (WHERE label = '1435') AS price_1435,
                           MAX(close) FILTER (WHERE label = '1449') AS price_1449,
                           SUM(volume) FILTER (WHERE label BETWEEN '1446' AND '1449')
                               AS volume_1446_1449,
                           SUM(turnover) FILTER (WHERE label BETWEEN '1446' AND '1449')
                               AS amount_1446_1449,
                           SUM(volume) AS volume_1449,
                           SUM(turnover) AS amount_1449,
                           SUM(volume) FILTER (WHERE label BETWEEN '1421' AND '1449')
                               AS volume_last29,
                           SUM(turnover) FILTER (WHERE label BETWEEN '1421' AND '1449')
                               AS amount_last29,
                           MAX(high) FILTER (WHERE volume > 0) AS high_1449,
                           MIN(low) FILTER (WHERE volume > 0) AS low_1449
                    FROM prefix GROUP BY date, code
                )
                SELECT date, code, price_1420, price_1435, price_1449,
                       amount_1446_1449 / NULLIF(volume_1446_1449, 0)
                           AS vwap_1446_1449,
                       volume_1449, amount_1449, volume_last29, amount_last29,
                       high_1449, low_1449,
                       price_1449 / price_1420 - 1 AS return_last29,
                       price_1449 / price_1435 - 1 AS return_last14,
                       price_1449 > high_1449 + 0.005
                           OR price_1449 < low_1449 - 0.005
                           AS quote_outside_traded_range
                FROM aggregated
                WHERE bar_count = 230 AND unique_labels = 230
                  AND valid_bars = 230 AND first_label = '0930'
                  AND last_label = '1449' AND price_1420 > 0
                  AND price_1435 > 0 AND price_1449 > 0
                  AND volume_1449 > 0 AND amount_1449 > 0
                  AND high_1449 >= low_1449
            ) TO '{path_literal}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
    finally:
        connection.close()


def audit(prefix_dir: Path, snapshot_dir: Path) -> dict:
    """Check prefix keys and coverage; 14:50 snapshots are audit-only inputs."""
    connection = duckdb.connect()
    try:
        connection.execute("SET memory_limit = '8GB'")
        connection.from_parquet(str(prefix_dir / "*" / "part_*.parquet")).create_view("prefix")
        connection.from_parquet(str(snapshot_dir / "*.parquet")).create_view("snapshots")
        keys = connection.execute("""
            SELECT COUNT(*) AS rows, COUNT(DISTINCT (date, code)) AS keys
            FROM prefix
        """).fetchone()
        result = connection.execute("""
            WITH matched AS (
                SELECT COALESCE(p.date, s.date) AS date,
                       p.code AS prefix_code, s.code AS snapshot_code,
                       p.price_1449, p.amount_1449,
                       p.low_1449, p.high_1449,
                       s.price_1450, s.amount_1450
                FROM prefix p FULL OUTER JOIN snapshots s
                  ON p.date = s.date AND p.code = s.code
                WHERE COALESCE(p.date, s.date) BETWEEN '2024-01-01' AND '2025-12-31'
            )
            SELECT LEFT(date, 4) AS year,
                   COUNT(prefix_code) AS prefix_rows,
                   COUNT(snapshot_code) AS snapshot_rows,
                   COUNT(*) FILTER (WHERE prefix_code IS NOT NULL
                                      AND snapshot_code IS NOT NULL) AS overlap_rows,
                   COUNT(*) FILTER (WHERE prefix_code IS NOT NULL
                                      AND snapshot_code IS NULL) AS prefix_only_rows,
                   COUNT(*) FILTER (WHERE prefix_code IS NULL
                                      AND snapshot_code IS NOT NULL) AS snapshot_only_rows,
                   COUNT(*) FILTER (WHERE prefix_code IS NOT NULL
                                      AND snapshot_code IS NOT NULL
                                      AND ABS(price_1449 / price_1450 - 1) > 1e-8)
                       AS price_changed_rows,
                   COUNT(*) FILTER (WHERE prefix_code IS NOT NULL
                                      AND snapshot_code IS NOT NULL
                                      AND amount_1449 > amount_1450 + 0.01)
                       AS amount_after_cutoff_disagreement,
                   COUNT(*) FILTER (WHERE price_1449 < low_1449 - 0.005
                                      OR price_1449 > high_1449 + 0.005)
                       AS price_outside_prefix_range
            FROM matched GROUP BY year ORDER BY year
        """)
        names = [item[0] for item in result.description]
        rows = result.fetchall()
    finally:
        connection.close()
    by_year = {row[0]: dict(zip(names[1:], row[1:], strict=True)) for row in rows}
    valid_years = set(by_year) == {"2024", "2025"}
    passes = valid_years and keys[0] == keys[1]
    for row in by_year.values():
        row["snapshot_coverage"] = row["overlap_rows"] / row["snapshot_rows"]
        passes &= (
            row["snapshot_coverage"] >= MIN_SNAPSHOT_COVERAGE
            and row["amount_after_cutoff_disagreement"] == 0
        )
    return {
        "prefix_rows": keys[0], "prefix_unique_keys": keys[1],
        "minimum_snapshot_coverage": MIN_SNAPSHOT_COVERAGE,
        "input_gate_passed": bool(passes), "by_year": by_year,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minute-root", type=Path,
                        default=Path("data/hf/pilot/data/stock_1m"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/research/minute_prefix_1449"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Number of security files processed per output part")
    parser.add_argument("--snapshot-dir", type=Path,
                        default=Path("data/research/market_snapshots_ci"))
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    paths = sorted(str(path) for exchange in ("SH", "SZ")
                   for path in (args.minute_root / exchange).glob("*.parquet"))
    if not paths:
        parser.error("No SH/SZ minute files found")
    for year in (DEVELOPMENT_YEAR, VALIDATION_YEAR):
        for offset in range(0, len(paths), args.batch_size):
            part = offset // args.batch_size
            destination = args.output_dir / str(year) / f"part_{part:03d}.parquet"
            build_year(paths[offset:offset + args.batch_size],
                       year, destination, args.threads)
            print(f"Wrote {destination}", flush=True)
    report = audit(args.output_dir, args.snapshot_dir)
    (args.output_dir / "input_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
