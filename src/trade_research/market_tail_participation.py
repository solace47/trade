"""Audit 14:50 market-wide tail turnover before reading stratified outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path("data/research")
OUTPUT = ROOT / "market_tail_participation"
HALVES = ("2024H1", "2024H2", "2025H1", "2025H2")


def _correlation(left: pd.Series, right: pd.Series) -> float | None:
    if len(left) < 2:
        return None
    value = float(left.corr(right))
    return value if np.isfinite(value) else None


def select_inputs(connection: duckdb.DuckDBPyConnection,
                  signals: pd.DataFrame,
                  variance: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    states = connection.execute("""
        WITH valid AS (
            SELECT s.date, s.code,
                   s.amount_1450 * i.amount_share_last30 AS tail_amount,
                   i.return_last30
            FROM snapshots s JOIN intraday i USING (date, code)
            WHERE s.date BETWEEN '2024-01-01' AND '2025-12-17'
              AND s.isST = 0 AND s.listing_age_sessions >= 20
              AND s.price_1450 >= 5 AND s.amount_1450 >= 30000000
              AND NOT s.reference_gap AND NOT s.quote_outside_traded_range
              AND ABS(s.price_1450 - i.price_1450) <= .005
              AND s.amount_1450 * i.amount_share_last30 > 0
              AND i.return_last30 IS NOT NULL
        ), historical AS (
            SELECT date, code, tail_amount, return_last30,
                   MEDIAN(tail_amount) OVER (
                       PARTITION BY code ORDER BY date
                       ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
                   ) AS prior_tail_amount,
                   LAG(date, 20) OVER (PARTITION BY code ORDER BY date)
                       AS oldest_prior_date
            FROM valid
        ), market AS (
            SELECT date, COUNT(*) AS stocks,
                   MEDIAN(tail_amount / prior_tail_amount)
                       AS market_participation,
                   AVG(return_last30) AS market_tail_return
            FROM historical
            WHERE oldest_prior_date IS NOT NULL AND prior_tail_amount > 0
              AND date_diff('day', CAST(oldest_prior_date AS DATE),
                                   CAST(date AS DATE)) <= 45
            GROUP BY date HAVING COUNT(*) >= 1000
        )
        SELECT *, CASE WHEN market_participation >= 1.2 THEN 'high'
                       WHEN market_participation <= .8 THEN 'low'
                       ELSE 'middle' END AS participation_state
        FROM market ORDER BY date
    """).df()
    if (states.empty or states.duplicated("date").any()
            or states.stocks.lt(1000).any()
            or not states.date.str[:4].isin(("2024", "2025")).all()
            or not np.isfinite(states.market_participation).all()):
        raise ValueError("Invalid market participation state")
    if (signals.empty or signals.duplicated(["date", "code"]).any()
            or set(signals.candidate) != {"late_decline", "late_rally_control"}
            or signals.groupby(["date", "pair_id"]).candidate.nunique().ne(2).any()):
        raise ValueError("Invalid frozen same-day pair list")
    treatment = signals.loc[signals.candidate.eq("late_decline"),
                            ["date", "pair_id"]]
    assigned = treatment.merge(states[["date", "participation_state"]],
                               on="date", how="left", validate="many_to_one")
    usable = assigned.dropna(subset=["participation_state"]).copy()
    usable["half"] = usable.date.str[:4] + "H" + np.where(
        usable.date.str[5:7].astype(int).le(6), "1", "2")
    by_half = usable.groupby(["half", "participation_state"]).agg(
        pairs=("pair_id", "size"), days=("date", "nunique")
    ).reset_index().to_dict("records")
    counts = {(row["half"], row["participation_state"]): row
              for row in by_half}
    variance_dates = variance[["date", "market_surprise"]].drop_duplicates("date")
    overlap = states.merge(variance_dates, on="date", validate="one_to_one")
    direction_correlation = _correlation(
        states.market_participation, states.market_tail_return)
    variance_correlation = _correlation(
        overlap.market_participation, overlap.market_surprise)
    correlations_valid = (len(overlap) >= 300
                          and direction_correlation is not None
                          and variance_correlation is not None)
    gate = (correlations_valid and len(usable) >= .9 * len(treatment)
            and abs(direction_correlation) < .7
            and abs(variance_correlation) < .7
            and all((half, state) in counts
                    and counts[(half, state)]["pairs"] >= 30
                    and counts[(half, state)]["days"] >= 15
                    for half in HALVES for state in ("high", "low")))
    audit = {
        "market_dates": len(states), "variance_overlap_dates": len(overlap),
        "frozen_pairs": len(treatment), "classified_pairs": len(usable),
        "warmup_or_missing_state_pairs": len(treatment) - len(usable),
        "participation_vs_direction_correlation": direction_correlation,
        "participation_vs_variance_correlation": variance_correlation,
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
    signals = pd.read_parquet(
        ROOT / "late_to_open_reversal" / "selections.parquet",
        columns=["date", "code", "candidate", "pair_id"],
    )
    variance = pd.read_parquet(
        ROOT / "market_variance_state" / "states.parquet",
        columns=["date", "market_surprise"],
    )
    states, audit = select_inputs(connection, signals, variance)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").unlink(missing_ok=True)
    states.to_parquet(output_dir / "states.parquet", index=False,
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
    print(json.dumps(build_inputs(args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
