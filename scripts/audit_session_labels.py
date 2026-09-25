"""Audit opening and closing label contents in 2024–2025 raw SH/SZ minutes.

The audit can reject an auction-only interpretation of a bar. It cannot
establish whether other timestamps denote minute starts or ends.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb


LABELS = {"09:30", "09:31", "13:01", "15:00"}


def audit(minute_root: Path) -> dict:
    paths = sorted(str(path) for exchange in ("SH", "SZ")
                   for path in (minute_root / exchange).glob("*.parquet"))
    if not paths:
        raise ValueError("No SH/SZ minute source files")
    connection = duckdb.connect()
    try:
        connection.execute("SET threads = 4")
        connection.execute("SET memory_limit = '8GB'")
        connection.from_parquet(paths).create_view("minutes")
        frame = connection.execute("""
            SELECT YEAR(timestamp) AS year, exchange,
                   STRFTIME(timestamp, '%H:%M') AS label,
                   COUNT(*) AS bars,
                   COUNT(*) FILTER (WHERE volume > 0) AS positive_volume,
                   COUNT(*) FILTER (WHERE volume = 0) AS zero_volume,
                   COUNT(*) FILTER (WHERE volume IS NULL) AS null_volume,
                   COUNT(*) FILTER (
                       WHERE volume > 0 AND high - low > .005
                   ) AS multiple_price_positive,
                   COUNT(*) FILTER (
                       WHERE volume > 0 AND high - low <= .005
                   ) AS single_price_positive,
                   COUNT(*) FILTER (
                       WHERE volume > 0 AND high - low > .005
                         AND ABS(turnover / volume - open) > .005
                   ) AS multiple_price_vwap_differs_from_open
            FROM minutes
            WHERE timestamp >= '2024-01-01'
              AND timestamp < '2026-01-01'
              AND STRFTIME(timestamp, '%H:%M') IN (
                  '09:30', '09:31', '13:00', '13:01', '15:00'
              )
            GROUP BY 1, 2, 3 ORDER BY 1, 2, 3
        """).df()
    finally:
        connection.close()
    if (len(frame) != 16 or set(frame.year) != {2024, 2025}
            or set(frame.exchange) != {"SH", "SZ"}
            or set(frame.label) != LABELS
            or not frame.bars.eq(frame.positive_volume + frame.zero_volume
                                 + frame.null_volume).all()
            or not frame.positive_volume.eq(frame.multiple_price_positive
                                            + frame.single_price_positive).all()):
        raise ValueError("Incomplete or inconsistent session labels")
    frame["multiple_price_rate_of_positive"] = (
        frame.multiple_price_positive / frame.positive_volume
    )
    return {
        "scope": "2024–2025 SH/SZ raw minute bars; timestamp meaning unconfirmed",
        "source_files": len(paths),
        "multiple_price_definition": "positive volume and high-low > CNY 0.005",
        "by_year_exchange_label": frame.to_dict("records"),
    }


def audit_morning_anchor(morning_dir: Path, snapshot_dir: Path) -> list[dict]:
    """Compare the archived 09:30 close with the known daily open; no returns."""
    connection = duckdb.connect()
    try:
        connection.execute("SET threads = 4")
        connection.read_parquet(str(morning_dir / "parts" / "*.parquet")
                                ).create_view("morning")
        connection.read_parquet(str(snapshot_dir / "*.parquet")
                                ).create_view("snapshots")
        frame = connection.execute("""
            WITH anchors AS (
                SELECT YEAR(CAST(m.date AS DATE)) AS year,
                       UPPER(LEFT(m.code, 2)) AS exchange,
                       m.price_0930 / s.open_1450 - 1 AS anchor_gap,
                       m.price_1130 / m.price_0930 - 1 AS old_morning,
                       m.price_1130 / s.open_1450 - 1 AS open_morning
                FROM morning m JOIN snapshots s USING (date, code)
                WHERE m.date BETWEEN '2024-01-01' AND '2025-12-17'
                  AND s.tradestatus = 1 AND s.open_1450 > 0
                  AND m.price_0930 > 0
            )
            SELECT year, exchange, COUNT(*) AS stock_days,
                   COUNT(*) FILTER (
                       WHERE ABS(anchor_gap) > .001
                   ) AS anchor_gap_over_ten_bps,
                   COUNT(*) FILTER (
                       WHERE FLOOR(old_morning * 200)
                          != FLOOR(open_morning * 200)
                   ) AS half_percentage_point_bin_changed,
                   COUNT(*) FILTER (
                       WHERE (old_morning BETWEEN 0 AND .03)
                          != (open_morning BETWEEN 0 AND .03)
                   ) AS morning_0_to_3pct_filter_changed
            FROM anchors GROUP BY 1, 2 ORDER BY 1, 2
        """).df()
    finally:
        connection.close()
    if (len(frame) != 4 or set(frame.year) != {2024, 2025}
            or set(frame.exchange) != {"SH", "SZ"}):
        raise ValueError("Incomplete daily-open anchor comparison")
    return frame.to_dict("records")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minute-root", type=Path,
                        default=Path("data/hf/pilot/data/stock_1m"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/research/session_label_audit.json"))
    parser.add_argument("--morning-dir", type=Path,
                        default=Path("data/research/morning_burst"))
    parser.add_argument("--snapshot-dir", type=Path,
                        default=Path("data/research/market_snapshots_ci"))
    args = parser.parse_args()
    report = audit(args.minute_root)
    report["morning_anchor_vs_daily_open"] = audit_morning_anchor(
        args.morning_dir, args.snapshot_dir
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)
                           + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
