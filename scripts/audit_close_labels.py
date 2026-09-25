"""Count 2024–2025 closing-minute volume labels across the raw SH/SZ archive.

The counts describe the source format; they do not certify execution time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb


def audit(minute_root: Path) -> dict:
    paths = sorted(str(path) for market in ("SH", "SZ")
                   for path in (minute_root / market).glob("*.parquet"))
    if not paths:
        raise ValueError("No SH/SZ minute source files")
    connection = duckdb.connect()
    try:
        connection.execute("SET threads = 4")
        connection.execute("SET memory_limit = '8GB'")
        connection.from_parquet(paths).create_view("minutes")
        frame = connection.execute("""
            SELECT YEAR(timestamp) AS year, exchange,
                   STRFTIME(timestamp, '%H:%M') AS bar_label,
                   COUNT(*) AS bars,
                   COUNT(*) FILTER (WHERE volume > 0) AS positive_volume_bars,
                   COUNT(*) FILTER (WHERE volume = 0) AS zero_volume_bars,
                   COUNT(*) FILTER (WHERE volume IS NULL) AS null_volume_bars,
                   SUM(volume) AS shares,
                   COUNT(*) FILTER (
                       WHERE volume > 0 AND (
                           open != high OR open != low OR open != close
                       )
                   ) AS nonflat_positive_bars
            FROM minutes
            WHERE timestamp >= '2024-01-01'
              AND timestamp < '2026-01-01'
              AND STRFTIME(timestamp, '%H:%M') BETWEEN '14:57' AND '15:00'
            GROUP BY 1, 2, 3
            ORDER BY 1, 2, 3
        """).df()
    finally:
        connection.close()
    if (len(frame) != 16 or set(frame.year) != {2024, 2025}
            or set(frame.exchange) != {"SH", "SZ"}
            or set(frame.bar_label) != {"14:57", "14:58", "14:59", "15:00"}
            or not frame.bars.eq(frame.positive_volume_bars
                                 + frame.zero_volume_bars
                                 + frame.null_volume_bars).all()):
        raise ValueError("Incomplete or inconsistent closing labels")
    return {
        "scope": "2024–2025 SH/SZ raw minute bars; source timing unverified",
        "source_files": len(paths),
        "by_year_exchange_label": frame.to_dict("records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minute-root", type=Path,
                        default=Path("data/hf/pilot/data/stock_1m"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/research/close_label_audit.json"))
    args = parser.parse_args()
    report = audit(args.minute_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)
                           + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
