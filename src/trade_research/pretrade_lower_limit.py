"""Freeze an input-only 14:49 lower-limit risk screen before exit reads."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb


ROOT = Path("data/research")
SOURCE = ROOT / "pretrade_upper_limit" / "inputs.parquet"
OUTPUT = ROOT / "pretrade_lower_limit"
HALVES = ("2024H1", "2024H2", "2025H1", "2025H2")


def freeze(source: Path = SOURCE, output: Path = OUTPUT) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "inputs.parquet"
    connection = duckdb.connect()
    try:
        connection.read_parquet(str(source)).create_view("source")
        escaped = str(destination).replace("'", "''")
        connection.execute(f"""
            COPY (
                WITH cents AS (
                    SELECT date, code, half, price_1449, preclose,
                           CAST(ROUND(preclose * 100, 0) AS BIGINT)
                               AS preclose_cents
                    FROM source
                ), priced AS (
                    SELECT *, CAST(FLOOR((preclose_cents * 90 + 50) / 100)
                                   AS BIGINT) AS lower_cents
                    FROM cents
                )
                SELECT date, code, half, price_1449, preclose,
                       lower_cents / 100.0 AS lower_limit,
                       price_1449 <= lower_cents / 100.0 + 0.005
                           AS at_lower
                FROM priced
            ) TO '{escaped}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        connection.read_parquet(str(destination)).create_view("inputs")
        cursor = connection.execute("""
            SELECT half, COUNT(*) AS stock_days,
                   COUNT(DISTINCT (date, code)) AS unique_keys,
                   COUNT(DISTINCT date) AS days,
                   COUNT(*) FILTER (WHERE at_lower) AS lower_stock_days,
                   COUNT(DISTINCT date) FILTER (WHERE at_lower) AS lower_days
            FROM inputs GROUP BY half ORDER BY half
        """)
        names = [field[0] for field in cursor.description]
        by_half = [dict(zip(names, values, strict=True))
                   for values in cursor.fetchall()]
    finally:
        connection.close()
    gate = ([row["half"] for row in by_half] == list(HALVES)
            and all(row["stock_days"] == row["unique_keys"]
                    and row["lower_stock_days"] >= 100
                    and row["lower_days"] >= 30
                    for row in by_half))
    report = {"source": "frozen 14:49 main-board input pool",
              "by_half": by_half, "input_gate_passed": bool(gate),
              "post_decision_data_read": False}
    (output / "input_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return report


def freeze_sample(output: Path = OUTPUT) -> dict:
    audit = json.loads((output / "input_audit.json").read_text(encoding="utf-8"))
    if not audit["input_gate_passed"]:
        raise ValueError("Input gate failed; do not read exit status")
    connection = duckdb.connect()
    try:
        connection.read_parquet(str(output / "inputs.parquet")
                                ).create_view("inputs")
        destination = output / "sample.parquet"
        escaped = str(destination).replace("'", "''")
        connection.execute(f"""
            COPY (
                WITH ranked AS (
                    SELECT date, code, half, at_lower,
                           COUNT(*) FILTER (WHERE at_lower) OVER
                               (PARTITION BY date) AS lower_count,
                           ROW_NUMBER() OVER (
                               PARTITION BY date, at_lower
                               ORDER BY md5(date || code), code
                           ) AS rank_in_day
                    FROM inputs
                )
                SELECT date, code, half, at_lower FROM ranked
                WHERE at_lower OR rank_in_day <= lower_count
            ) TO '{escaped}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        cursor = connection.execute(f"""
            SELECT half, at_lower, COUNT(*) AS rows,
                   COUNT(DISTINCT (date, code)) AS unique_keys,
                   COUNT(DISTINCT date) AS dates
            FROM read_parquet('{escaped}')
            GROUP BY half, at_lower ORDER BY half, at_lower
        """)
        names = [field[0] for field in cursor.description]
        by_half = [dict(zip(names, values, strict=True))
                   for values in cursor.fetchall()]
    finally:
        connection.close()
    cells = {(row["half"], row["at_lower"]): row for row in by_half}
    valid = len(cells) == 8 and all(
        cells[(half, False)]["rows"] == cells[(half, True)]["rows"]
        and cells[(half, False)]["dates"] == cells[(half, True)]["dates"]
        and cells[(half, False)]["rows"] == cells[(half, False)]["unique_keys"]
        and cells[(half, True)]["rows"] == cells[(half, True)]["unique_keys"]
        for half in HALVES)
    report = {"by_half": by_half, "sample_gate_passed": valid}
    (output / "sample_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "sample"))
    stage = parser.parse_args().stage
    result = {"freeze": freeze, "sample": freeze_sample}[stage]()
    print(json.dumps(result, ensure_ascii=False, indent=2))
