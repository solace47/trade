"""Audit the old ridge list's 14:49/14:50 input change without reading returns."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import duckdb
import pyarrow.parquet as pq


def audit(root: Path, minute_root: Path, sample_per_cell: int = 5) -> dict:
    selected = root / "absolute_ridge_main"
    prior = json.loads((selected / "input_audit.json").read_text(encoding="utf-8"))
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.read_parquet(str(selected / "selections.parquet")).create_view("members")
    connection.read_parquet(str(root / "minute_prefix_1449" / "*" / "*.parquet")
                            ).create_view("prefix")
    connection.read_parquet(str(root / "market_snapshots_ci" / "*.parquet")
                            ).create_view("snapshots")
    membership = connection.execute("""
        SELECT COUNT(*) AS n_rows, COUNT(DISTINCT (date, code)) AS n_keys,
               COUNT(*) FILTER (WHERE candidate = 'absolute_model') models,
               COUNT(*) FILTER (WHERE candidate = 'same_day_control') controls
        FROM members
    """).fetchone()
    if (membership[0] != membership[1]
            or membership[2] != prior["signals"]
            or membership[3] != prior["controls"]
            or membership[0] != membership[2] + membership[3]):
        raise ValueError("The frozen ridge membership is incomplete or duplicated")
    connection.execute("""
        CREATE VIEW joined AS
        SELECT m.date, m.code, m.candidate, p.price_1449, s.price_1450,
               p.amount_1449, s.amount_1450,
               CASE WHEN m.date < '2025-01-01' THEN '2024H2'
                    WHEN m.date < '2025-07-01' THEN '2025H1'
                    ELSE '2025H2' END AS half,
               10000 * (s.price_1450 / p.price_1449 - 1) AS move_bps
        FROM members m
        JOIN prefix p USING (date, code)
        JOIN snapshots s USING (date, code)
        WHERE m.date BETWEEN '2024-07-01' AND '2025-12-17'
          AND p.price_1449 > 0 AND s.price_1450 > 0
    """)
    joined = connection.execute("SELECT COUNT(*) FROM joined").fetchone()[0]
    if joined != membership[0]:
        raise ValueError("A frozen stock-day lacks a positive 14:49 or 14:50 price")
    table = connection.execute("""
        SELECT half, candidate, COUNT(*) AS stock_days,
               ROUND(AVG(move_bps), 2) AS mean_move_bps,
               ROUND(QUANTILE_CONT(ABS(move_bps), .5), 2) AS median_abs_move_bps,
               ROUND(QUANTILE_CONT(ABS(move_bps), .95), 2) AS p95_abs_move_bps,
               ROUND(100 * AVG((ABS(price_1450 - price_1449) > .005)::INT), 1)
                   AS price_changed_pct,
               ROUND(100 * AVG(((amount_1449 BETWEEN 1e8 AND 1e9)
                    <> (amount_1450 BETWEEN 1e8 AND 1e9))::INT), 1)
                   AS amount_boundary_changed_pct
        FROM joined GROUP BY half, candidate ORDER BY half, candidate
    """).df().to_dict("records")
    if len(table) != 6 or sample_per_cell < 1:
        raise ValueError("Expected three half-years and both fixed membership arms")
    sample = connection.execute(f"""
        SELECT date, code, half, candidate, price_1449, price_1450
        FROM joined QUALIFY ROW_NUMBER() OVER (
            PARTITION BY half, candidate
            ORDER BY MD5('ridge-cutoff-v1' || date || code)) <= {sample_per_cell}
    """).df()
    for row in sample.itertuples(index=False):
        exchange, number = row.code.split(".")
        source = minute_root / exchange.upper() / f"{number}.parquet"
        first = datetime.fromisoformat(f"{row.date} 14:49:00")
        last = datetime.fromisoformat(f"{row.date} 14:50:00")
        raw = pq.read_table(source, columns=["timestamp", "close"], filters=[
            ("timestamp", ">=", first), ("timestamp", "<=", last),
        ]).to_pandas()
        prices = dict(zip(raw.timestamp.dt.strftime("%H:%M"), raw.close))
        if (len(raw) != 2 or set(prices) != {"14:49", "14:50"}
                or abs(prices["14:49"] - row.price_1449) > .0001
                or abs(prices["14:50"] - row.price_1450) > .0001):
            raise ValueError(f"Raw-minute mismatch for {row.date} {row.code}")
    result = {
        "scope": "post hoc input audit of the old frozen ridge list; no outcomes read",
        "stock_days": joined, "raw_minute_sample": len(sample),
        "raw_minute_mismatches": 0, "by_half_and_arm": table,
    }
    (selected / "cutoff_input_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/research"))
    parser.add_argument("--minute-root", type=Path,
                        default=Path("data/hf/pilot/data/stock_1m"))
    parser.add_argument("--sample-per-cell", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.minute_root, args.sample_per_cell),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
