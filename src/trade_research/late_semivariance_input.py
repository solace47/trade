"""Audit pre-outcome pairability of 14:50 downside semivariance.

The match count is an upper bound over all high stocks, before ranking,
cooldown and control reuse.
Never read an outcome table when the predeclared gate fails.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb


ROOT = Path("data/research")
BASE = """
    s.isST = 0 AND s.listing_age_sessions >= 20
    AND NOT s.reference_gap AND NOT s.quote_outside_traded_range
    AND s.price_1450 >= 5
    AND s.amount_1450 BETWEEN 100000000 AND 1000000000
    AND s.return20_prior_adjusted BETWEEN -.10 AND .10
    AND s.return_1450 BETWEEN 0 AND .03
    AND s.position_1450 >= .7
    AND i.return_last30 BETWEEN -.003 AND .003
    AND abs(s.price_1450 - i.price_1450) <= .005
    AND abs(s.price_1450 - v.price_1450) <= .005
    AND v.realized_variance_last30 > 0
    AND v.down_moves_last30 >= 3 AND v.up_moves_last30 >= 3
"""
MATCH = """
    l.date = h.date
    AND abs(l.return20_prior_adjusted - h.return20_prior_adjusted) <= .03
    AND abs(l.return_1450 - h.return_1450) <= .005
    AND l.amount_1450 / h.amount_1450 BETWEEN .5 AND 2
    AND abs(l.position_1450 - h.position_1450) <= .15
    AND l.price_1450 / h.price_1450 BETWEEN .5 AND 2
    AND abs(l.volume_share_last30 - h.volume_share_last30) <= .07
"""


def audit(output: Path = ROOT / "late_semivariance" / "input_audit.json") -> dict:
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    for name, path in (("snapshots", "market_snapshots_ci"),
                       ("intraday", "intraday_features"),
                       ("variance", "late_semivariance")):
        connection.read_parquet(str(ROOT / path / "*.parquet")
                                ).create_view(name)
    quality = connection.execute("""
        SELECT COUNT(*) AS rows,
               COUNT(*) FILTER (
                   WHERE abs(realized_variance_last30
                       - downside_variance_last30
                       - upside_variance_last30) > 1e-14
                   OR downside_share_last30 < 0
                   OR downside_share_last30 > 1
               ) AS invalid
        FROM variance
    """).df().iloc[0].to_dict()
    if quality["invalid"] or quality["rows"] == 0:
        raise ValueError("Invalid signed-variance partition")
    connection.execute(f"""
        CREATE TEMP TABLE base AS
        SELECT s.date, s.code, s.return20_prior_adjusted,
               s.return_1450, s.amount_1450, s.position_1450,
               s.price_1450, i.return_last30, i.volume_share_last30,
               v.realized_variance_last30, v.downside_share_last30
        FROM snapshots s JOIN intraday i USING (date, code)
        JOIN variance v USING (date, code)
        WHERE ((s.date BETWEEN '2024-01-01' AND '2024-12-17')
            OR (s.date BETWEEN '2025-01-01' AND '2025-12-17'))
          AND {BASE}
    """)
    connection.execute("""
        CREATE TEMP VIEW high AS
        SELECT * FROM base WHERE downside_share_last30 >= .65
    """)
    connection.execute("""
        CREATE TEMP VIEW low AS
        SELECT * FROM base WHERE downside_share_last30 <= .35
    """)
    rows = connection.execute(f"""
        WITH possible AS (
            SELECT h.date, h.code,
                   EXISTS (SELECT 1 FROM low l WHERE {MATCH}) AS common,
                   EXISTS (SELECT 1 FROM low l WHERE {MATCH}
                       AND abs(l.return_last30-h.return_last30) <= .002
                   ) AS with_tail,
                   EXISTS (SELECT 1 FROM low l WHERE {MATCH}
                       AND abs(l.return_last30-h.return_last30) <= .002
                       AND l.realized_variance_last30
                           / h.realized_variance_last30 BETWEEN .5 AND 2
                   ) AS full_match
            FROM high h
        )
        SELECT substr(date, 1, 4) || 'H'
               || CASE WHEN cast(substr(date, 6, 2) AS INTEGER) <= 6
                       THEN '1' ELSE '2' END AS period,
               COUNT(*) AS all_high,
               COUNT(*) FILTER (WHERE common) AS common_match,
               COUNT(*) FILTER (WHERE with_tail) AS tail_match,
               COUNT(*) FILTER (WHERE full_match) AS full_match,
               COUNT(DISTINCT date) AS signal_days
        FROM possible GROUP BY 1 ORDER BY 1
    """).df().to_dict("records")
    pools = connection.execute("""
        SELECT COUNT(*) AS base,
               COUNT(*) FILTER (WHERE downside_share_last30 >= .65)
                   AS high_pool,
               COUNT(*) FILTER (WHERE downside_share_last30 <= .35)
                   AS low_pool,
               CORR(downside_share_last30, return_last30)
                   AS share_tail_return_corr,
               MEDIAN(return_last30) FILTER (
                   WHERE downside_share_last30 >= .65
               ) AS high_median_tail_return,
               MEDIAN(return_last30) FILTER (
                   WHERE downside_share_last30 <= .35
               ) AS low_median_tail_return
        FROM base
    """).df().iloc[0].to_dict()
    if len(rows) != 4 or any(row["full_match"] > row["all_high"]
                             for row in rows):
        raise ValueError("Incomplete semivariance input audit")
    # All high candidates with a possible control upper-bound every ranking.
    # A selected subset could have a higher match fraction, so only the
    # predeclared 50-pair-per-half gate can be rejected from this audit.
    gate_possible = all(row["full_match"] >= 50 for row in rows)
    report = {
        "feature_rows": int(quality["rows"]),
        "base_pool": {
            key: int(value) if key in ("base", "high_pool", "low_pool")
            else float(value) for key, value in pools.items()
        },
        "by_half_upper_bound": [
            {key: int(value) if key != "period" else value
             for key, value in row.items()} for row in rows
        ],
        "outcome_gate_possible": gate_possible,
        "note": "All-high upper bound excludes ranking, cooldown and control reuse",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "late_semivariance" / "input_audit.json")
    args = parser.parse_args()
    print(audit(args.output))


if __name__ == "__main__":
    main()
