"""Describe where filled T+1 tail trades gain or lose before seeking a rule.

This is a market baseline, not a stock selector. Only 2024–2025 outcomes enter
the report; the next open is used for attribution after the signal, never for
selection. The result is conditional on a clean, on-time modeled fill.
"""

import json
from pathlib import Path

import duckdb
import pandas as pd

from trade_research.market_study import (
    _quality_keys, _quality_symbols, _week_bootstrap,
)
from trade_research.hf_outcomes import Assumptions


ROOT = Path("data/research")
PERIODS = (("2024", "2024-12-17"), ("2025", "2025-12-17"))


def main() -> None:
    shards = {
        path.name.split("_")[1]
        for path in (ROOT / "market_snapshots_ci").glob("shard_*_part_*.parquet")
    }
    if len(shards) != 20:
        raise ValueError("Expected 20 complete market snapshot shards")
    slip = Assumptions().slippage_bps_each_side / 10_000
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.read_parquet(str(ROOT / "market_snapshots_ci" / "*.parquet")
                            ).create_view("snapshots")
    connection.read_parquet(str(ROOT / "market_outcomes_ci" / "*.parquet")
                            ).create_view("outcomes")
    connection.register("bad_days", _quality_keys(ROOT / "market_issues_ci"))
    connection.register("bad_symbols", _quality_symbols(ROOT / "market_issues_ci"))
    ranges = " OR ".join(
        f"(s.date BETWEEN '{year}-01-01' AND '{last}')"
        for year, last in PERIODS
    )
    daily = connection.execute(f"""
        WITH eligible AS (
            SELECT s.date, s.code, o.entry_status, o.entry_price,
                   o.exit_status, o.exit_date, o.exit_delay_sessions,
                   o.exit_price, o.net_return,
                   n.open_1450 AS next_open,
                   NOT EXISTS (SELECT 1 FROM bad_symbols b
                               WHERE b.code = s.code)
                   AND NOT EXISTS (
                       SELECT 1 FROM bad_days q
                       WHERE q.code = s.code AND q.date >= s.date
                         AND q.date <= o.exit_date
                   ) AS quality_clean_exit
            FROM snapshots s JOIN outcomes o USING (date, code)
            LEFT JOIN snapshots n
              ON n.date = o.target_exit_date AND n.code = s.code
            WHERE o.horizon = 1 AND ({ranges})
              AND s.isST = 0 AND s.listing_age_sessions >= 20
              AND NOT s.reference_gap AND NOT s.quote_outside_traded_range
              AND s.amount_1450 BETWEEN 100000000 AND 1000000000
        ), attributed AS (
            SELECT *, entry_status = 'filled' AND exit_status = 'filled'
                        AND exit_delay_sessions = 0 AND quality_clean_exit
                        AND entry_price > 0 AND exit_price > 0
                        AND next_open > 0 AS usable
            FROM eligible
        )
        SELECT date, COUNT(*) AS eligible_signals,
               COUNT(*) FILTER (WHERE usable) AS clean_ontime,
               AVG(next_open / (entry_price / (1 + {slip})) - 1)
                   FILTER (WHERE usable) AS overnight_gross,
               AVG((exit_price / (1 - {slip})) / next_open - 1)
                   FILTER (WHERE usable) AS next_day_gross,
               AVG(net_return) FILTER (WHERE usable) AS net_return
        FROM attributed GROUP BY date ORDER BY date
    """).df()
    if daily.empty or daily[["overnight_gross", "next_day_gross",
                                  "net_return"]].isna().any().any():
        raise ValueError("Missing clean T+1 fills on a research day")
    report = []
    half = daily.date.str[5:7].astype(int).le(6).map(
        {True: "H1", False: "H2"}
    )
    for (year, section), frame in daily.groupby([daily.date.str[:4], half]):
        row = {
            "period": f"{year}-{section}",
            "days": len(frame),
            "eligible_signals": int(frame.eligible_signals.sum()),
            "clean_ontime": int(frame.clean_ontime.sum()),
            "completion_rate": float(
                frame.clean_ontime.sum() / frame.eligible_signals.sum()
            ),
        }
        for name in ("overnight_gross", "next_day_gross", "net_return"):
            row[name] = float(frame[name].mean())
            row[f"{name}_week_ci"] = _week_bootstrap(
                frame[name], frame.date, 20260925
            )
        report.append(row)
    destination = ROOT / "tail_return_decomposition.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
