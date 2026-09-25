"""Audit a 14:20 market trend independent of the last 30-minute move."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path("data/research")
SOURCE = ROOT / "late_to_open_reversal"
OUTPUT = ROOT / "market_pre_tail_state"
HALVES = ("2024H1", "2024H2", "2025H1", "2025H2")


def select_inputs(connection: duckdb.DuckDBPyConnection,
                  signals: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    states = connection.execute("""
        WITH eligible AS (
            SELECT i.date,
                   GREATEST(-.05, LEAST(.05,
                       (1 + s.return_1450) / (1 + i.return_last30) - 1
                   )) AS pre_tail_return,
                   GREATEST(-.02, LEAST(.02, i.return_last30))
                       AS tail_return
            FROM intraday i JOIN snapshots s USING (date, code)
            WHERE i.date BETWEEN '2024-01-01' AND '2025-12-17'
              AND s.isST = 0 AND s.listing_age_sessions >= 20
              AND NOT s.reference_gap AND NOT s.quote_outside_traded_range
              AND s.amount_1450 >= 30000000
              AND ABS(s.price_1450 - i.price_1450) <= .005
              AND s.return_1450 IS NOT NULL
              AND i.return_last30 IS NOT NULL
              AND 1 + i.return_last30 > 0
        ), market AS (
            SELECT date, COUNT(*) AS stocks,
                   AVG(pre_tail_return) AS market_pre_tail_return,
                   AVG(tail_return) AS market_tail_return
            FROM eligible GROUP BY date HAVING COUNT(*) >= 1000
        )
        SELECT *, CASE
            WHEN market_pre_tail_return <= -.005 THEN 'down'
            WHEN market_pre_tail_return >= .005 THEN 'up'
            ELSE 'flat' END AS pre_tail_state
        FROM market ORDER BY date
    """).df()
    if (states.empty or states.duplicated("date").any()
            or states.stocks.lt(1000).any()
            or not states.date.str[:4].isin(("2024", "2025")).all()):
        raise ValueError("Invalid 14:20 market states")
    correlation = float(states.market_pre_tail_return.corr(
        states.market_tail_return))
    if not np.isfinite(correlation):
        raise ValueError("Market-state correlation is undefined")
    treatment = signals.loc[signals.candidate.eq("late_decline"),
                            ["date", "pair_id"]]
    assigned = treatment.merge(states[["date", "pre_tail_state"]],
                               on="date", validate="many_to_one")
    if len(assigned) != len(treatment):
        raise ValueError("A frozen pair lacks a 14:20 market state")
    assigned["half"] = assigned.date.str[:4] + "H" + np.where(
        assigned.date.str[5:7].astype(int).le(6), "1", "2")
    by_half = assigned.groupby(["half", "pre_tail_state"]).agg(
        pairs=("pair_id", "size"), days=("date", "nunique"),
    ).reset_index().to_dict("records")
    counts = {(row["half"], row["pre_tail_state"]): row for row in by_half}
    gate = abs(correlation) < .7 and all(
        (half, state) in counts
        and counts[(half, state)]["pairs"] >= 30
        and counts[(half, state)]["days"] >= 15
        for half in HALVES for state in ("down", "up")
    )
    audit = {
        "market_dates": len(states), "frozen_pairs": len(treatment),
        "pre_tail_vs_tail_correlation": correlation,
        "by_half": by_half, "outcome_gate_passed": gate,
    }
    return states, audit


def build_inputs(output_dir: Path = OUTPUT) -> dict:
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.read_parquet(str(ROOT / "market_snapshots_ci" / "*.parquet")
                            ).create_view("snapshots")
    connection.read_parquet(str(ROOT / "intraday_features" / "*.parquet")
                            ).create_view("intraday")
    signals = pd.read_parquet(SOURCE / "selections.parquet", columns=[
        "date", "code", "candidate", "pair_id",
    ])
    if (signals.empty or signals.duplicated(["date", "code"]).any()
            or set(signals.candidate) != {"late_decline", "late_rally_control"}
            or not signals.date.str[:4].isin(("2024", "2025")).all()):
        raise ValueError("Invalid frozen all-market pair list")
    states, audit = select_inputs(connection, signals)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").unlink(missing_ok=True)
    states.to_parquet(output_dir / "states.parquet", index=False,
                      compression="zstd")
    signals.to_parquet(output_dir / "paired_signals.parquet", index=False,
                       compression="zstd")
    (output_dir / "input_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(build_inputs(args.output))


if __name__ == "__main__":
    main()
