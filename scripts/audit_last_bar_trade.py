"""Audit whether a 14:49 decision price has a trade in its labeled minute.

This is an input-only check on 2024–2025. It never reads execution or returns.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path("data/research")
MINUTES = Path("data/hf/pilot/data/stock_1m")
STUDIES = ("absolute_ridge_1449", "late_reversal_1449")
OUTPUT = ROOT / "last_bar_trade" / "input_audit.json"


def records(connection: duckdb.DuckDBPyConnection, query: str) -> list[dict]:
    result = connection.execute(query)
    columns = [column[0] for column in result.description]
    return [dict(zip(columns, row, strict=True)) for row in result.fetchall()]


def audit() -> dict:
    paths = sorted(str(path) for exchange in ("SH", "SZ")
                   for path in (MINUTES / exchange).glob("*.parquet"))
    if not paths:
        raise FileNotFoundError("No original SH/SZ minute files found")
    connection = duckdb.connect()
    try:
        connection.execute("SET threads = 4")
        connection.execute("SET memory_limit = '8GB'")
        connection.execute("SET preserve_insertion_order = false")
        connection.from_parquet(paths).create_view("raw_minutes")
        connection.from_parquet(str(ROOT / "minute_prefix_1449" / "*"
                                         / "part_*.parquet")).create_view("prefix")
        connection.execute("""
            CREATE TEMP TABLE audited AS
            WITH last_bar AS (
                SELECT strftime(timestamp, '%Y-%m-%d') AS date,
                       lower(exchange) || '.' || symbol AS code,
                       COUNT(*) AS raw_rows, MAX(close) AS close,
                       MAX(volume) AS volume, MAX(turnover) AS turnover
                FROM raw_minutes
                WHERE timestamp >= TIMESTAMP '2024-01-01'
                  AND timestamp < TIMESTAMP '2026-01-01'
                  AND strftime(timestamp, '%H%M') = '1449'
                GROUP BY date, code
            )
            SELECT p.date, p.code, p.price_1449, p.amount_1449,
                   b.raw_rows, b.close, b.volume, b.turnover,
                   CASE WHEN p.date < '2024-07-01' THEN '2024H1'
                        WHEN p.date < '2025-01-01' THEN '2024H2'
                        WHEN p.date < '2025-07-01' THEN '2025H1'
                        ELSE '2025H2' END AS period,
                   CASE WHEN p.amount_1449 < 30000000 THEN '<30m'
                        WHEN p.amount_1449 < 100000000 THEN '30m-100m'
                        WHEN p.amount_1449 <= 1000000000 THEN '100m-1b'
                        ELSE '>1b' END AS amount_band
            FROM prefix p LEFT JOIN last_bar b USING (date, code)
        """)
        integrity = records(connection, """
            SELECT COUNT(*) AS rows, COUNT(DISTINCT (date, code)) AS unique_keys,
                   COUNT(*) FILTER (WHERE raw_rows IS NULL) AS missing_bars,
                   COUNT(*) FILTER (WHERE raw_rows <> 1) AS duplicate_bars,
                   COUNT(*) FILTER (WHERE volume > 0
                       AND abs(price_1449 - close) > .005) AS price_mismatches
            FROM audited
        """)[0]
        if (integrity["rows"] != integrity["unique_keys"]
                or integrity["missing_bars"]
                or integrity["duplicate_bars"]
                or integrity["price_mismatches"]):
            raise ValueError(f"14:49 original-bar join failed: {integrity}")
        by_period_band = records(connection, """
            SELECT period, amount_band, COUNT(*) AS stock_days,
                   COUNT(*) FILTER (WHERE volume = 0) AS zero_volume_days,
                   ROUND(100.0 * COUNT(*) FILTER (WHERE volume = 0)
                         / COUNT(*), 3) AS zero_volume_percent
            FROM audited GROUP BY period, amount_band
            ORDER BY period, amount_band
        """)
        frames = []
        for study in STUDIES:
            frame = pd.read_parquet(ROOT / study / "selections.parquet",
                                    columns=["date", "code", "candidate"])
            frame["study"] = study
            frames.append(frame)
        connection.register("selected", pd.concat(frames, ignore_index=True))
        by_study = records(connection, """
            SELECT s.study, COUNT(*) AS stock_days,
                   COUNT(*) FILTER (WHERE a.date IS NULL) AS missing_prefix,
                   COUNT(*) FILTER (WHERE a.volume = 0) AS zero_volume_days,
                   ROUND(100.0 * COUNT(*) FILTER (WHERE a.volume = 0)
                         / COUNT(*), 3) AS zero_volume_percent
            FROM selected s LEFT JOIN audited a USING (date, code)
            GROUP BY s.study ORDER BY s.study
        """)
        if any(row["missing_prefix"] for row in by_study):
            raise ValueError("A frozen selection lacks a 14:49 prefix")
        by_candidate_half = records(connection, """
            SELECT s.study, s.candidate, a.period,
                   COUNT(*) AS stock_days,
                   COUNT(*) FILTER (WHERE a.volume = 0) AS zero_volume_days
            FROM selected s JOIN audited a USING (date, code)
            GROUP BY s.study, s.candidate, a.period
            ORDER BY s.study, s.candidate, a.period
        """)
        return {"years": [2024, 2025], "integrity": integrity,
                "by_period_band": by_period_band, "by_study": by_study,
                "by_candidate_half": by_candidate_half,
                "outcomes_read": False}
    finally:
        connection.close()


def main() -> None:
    report = audit()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
