"""Test a fixed two-phase intraday climb against the same calm up-day pool.

Selection sees labels through 14:50 under the bar-end assumption. Prior negative results
motivated this exploratory test; 2025 is not a blind holdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

from .market_study import _quality_keys, _quality_symbols
from .residual_liquidity import _period


ROOT = Path("data/research")
CAPACITY = 5
HORIZONS = (1, 5)
STEADY = "price_1000 / open_1450 - 1 >= .001 AND price_1420 / price_1301 - 1 >= .001"
BASE = """
    s.isST = 0 AND s.listing_age_sessions >= 20
    AND NOT s.reference_gap AND NOT s.quote_outside_traded_range
    AND s.amount_1450 BETWEEN 100000000 AND 1000000000
    AND s.price_1450 >= 5
    AND s.return20_prior_adjusted BETWEEN -.10 AND .10
    AND s.return_1450 BETWEEN .005 AND .03
    AND s.position_1450 >= .7
    AND i.return_last30 BETWEEN -.002 AND .002
    AND i.volume_share_last30 <= .12
    AND abs(s.price_1450 - i.price_1450) <= .005
"""


def select(connection: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, dict]:
    connection.execute(f"""
        CREATE TEMP TABLE eligible AS
        SELECT s.date, s.code, s.return_1450, s.amount_1450,
               s.return20_prior_adjusted, s.position_1450,
               s.open_1450, i.price_1000, i.price_1301,
               i.price_1420, i.return_last30, i.volume_share_last30,
               ({STEADY}) AS steady
        FROM snapshots s JOIN intraday i USING (date, code)
        WHERE ((s.date BETWEEN '2024-01-01' AND '2024-12-17')
            OR (s.date BETWEEN '2025-01-01' AND '2025-12-17'))
          AND {BASE}
    """)
    connection.execute("""
        CREATE TEMP TABLE comparable_dates AS
        SELECT date FROM eligible GROUP BY date
        HAVING COUNT(*) FILTER (WHERE steady) > 0
           AND COUNT(*) FILTER (WHERE NOT steady) > 0
    """)
    conditions = {
        "two_phase_climb": "steady",
        "calm_up_control": "NOT steady",
        "same_pool_random": "TRUE",
    }
    parts = []
    for name, condition in conditions.items():
        picked = connection.execute(f"""
            SELECT date, code, return_1450, amount_1450,
                   return20_prior_adjusted, position_1450,
                   ROW_NUMBER() OVER (
                       PARTITION BY date
                       ORDER BY md5('steady-path-v1' || date || code), code
                   ) AS daily_rank
            FROM eligible JOIN comparable_dates USING (date)
            WHERE {condition}
            QUALIFY daily_rank <= {CAPACITY}
        """).df()
        picked["candidate"] = name
        parts.append(picked)
    selected = pd.concat(parts, ignore_index=True)
    if (selected.empty or selected.duplicated(
            ["candidate", "date", "code"]).any()
            or not selected.date.str[:4].isin(("2024", "2025")).all()):
        raise ValueError("Invalid 2024–2025 steady-path selection")
    counts = connection.execute("""
        SELECT substr(date, 1, 4) AS period, COUNT(*) AS pool,
               COUNT(*) FILTER (WHERE steady) AS steady_pool,
               COUNT(DISTINCT date) AS days,
               COUNT(DISTINCT date) FILTER (WHERE steady) AS steady_days,
               COUNT(DISTINCT c.date) AS comparable_days
        FROM eligible e LEFT JOIN comparable_dates c USING (date)
        GROUP BY 1 ORDER BY 1
    """).df().to_dict("records")
    return selected, {"pool_counts": counts, "conditions": conditions}


def study(output_dir: Path = ROOT / "steady_path") -> dict:
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.read_parquet(str(ROOT / "market_snapshots_ci" / "*.parquet")
                            ).create_view("snapshots")
    connection.read_parquet(str(ROOT / "intraday_features" / "*.parquet")
                            ).create_view("intraday")
    selected, input_audit = select(connection)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(output_dir / "selections.parquet", index=False,
                        compression="zstd")
    connection.read_parquet(str(ROOT / "market_outcomes_ci" / "*.parquet")
                            ).create_view("outcomes")
    connection.register("selected", selected)
    connection.register("bad_days", _quality_keys(ROOT / "market_issues_ci"))
    connection.register("bad_symbols", _quality_symbols(ROOT / "market_issues_ci"))
    trades = connection.execute("""
        SELECT r.*, o.horizon, o.entry_status, o.entry_price, o.shares,
               o.exit_status, o.exit_date, o.exit_delay_sessions,
               o.exit_price, o.net_return,
               o.exit_status = 'filled'
               AND NOT EXISTS (SELECT 1 FROM bad_symbols b
                               WHERE b.code = r.code)
               AND NOT EXISTS (
                   SELECT 1 FROM bad_days q
                   WHERE q.code = r.code AND q.date >= r.date
                     AND q.date <= o.exit_date
               ) AS quality_clean_exit
        FROM selected r JOIN outcomes o USING (date, code)
        WHERE o.horizon IN (1, 5)
    """).df()
    if len(trades) != len(selected) * len(HORIZONS):
        raise ValueError("A selected stock-day lacks an archived outcome")
    trades.to_parquet(output_dir / "trades.parquet", index=False,
                      compression="zstd")
    report = {
        "hypothesis": "A two-phase climb preserves more return than other calm up-days",
        "pool": BASE.strip(), "treatment": STEADY,
        "selection": "deterministic hash, at most five per date and basket",
        "input_audit": input_audit,
        "note": "2025 was previously viewed; no 2026 outcomes used",
        "results": {},
    }
    for candidate in ("two_phase_climb", "calm_up_control"):
        report["results"][candidate] = {}
        for year in ("2024", "2025"):
            report["results"][candidate][year] = {}
            for horizon in HORIZONS:
                rows = trades.loc[trades.candidate.eq(candidate)
                                  & trades.date.str.startswith(year)
                                  & trades.horizon.eq(horizon)]
                reference = trades.loc[
                    trades.candidate.eq("calm_up_control"
                                        if candidate == "two_phase_climb"
                                        else "same_pool_random")
                    & trades.date.str.startswith(year)
                    & trades.horizon.eq(horizon)
                ]
                result = {}
                for period, section in (
                    ("H1", rows.loc[rows.date.str[5:7].astype(int).le(6)]),
                    ("H2", rows.loc[rows.date.str[5:7].astype(int).gt(6)]),
                    ("full", rows),
                ):
                    if not section.empty:
                        result[period] = _period(section, reference)
                report["results"][candidate][year][str(horizon)] = result
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "steady_path")
    args = parser.parse_args()
    report = study(args.output)
    print({"input_audit": report["input_audit"]["pool_counts"],
           "output": str(args.output / "report.json")})


if __name__ == "__main__":
    main()
