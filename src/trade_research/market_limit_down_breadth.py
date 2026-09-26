"""Audit the frozen 14:49 market-wide limit-down state without outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path("data/research")
OUTPUT = ROOT / "market_limit_down_breadth"
HALVES = ("2024H1", "2024H2", "2025H1", "2025H2")
COMPARATORS = ("up_share", "pre_1420_return", "late_29_return")


def audit_inputs(root: Path = ROOT, output: Path = OUTPUT) -> dict:
    connection = duckdb.connect()
    try:
        connection.execute("SET threads = 4")
        connection.read_parquet(str(root / "minute_prefix_1449" / "*" /
                                    "part_*.parquet")).create_view("prefix")
        connection.read_parquet(str(root / "market_snapshots_ci" /
                                    "*.parquet")).create_view("snapshots")
        daily = connection.execute("""
            WITH eligible AS (
                SELECT p.date, p.code, p.price_1449, s.preclose,
                       p.price_1420,
                       CASE WHEN p.code LIKE 'sh.68%'
                                  OR p.code LIKE 'sz.30%'
                            THEN CAST(.80 AS DECIMAL(4, 2))
                            ELSE CAST(.90 AS DECIMAL(4, 2))
                       END AS down_factor
                FROM prefix p JOIN snapshots s USING (date, code)
                WHERE p.date BETWEEN '2024-01-01' AND '2025-12-31'
                  AND (p.code LIKE 'sh.60%' OR p.code LIKE 'sh.68%'
                    OR p.code LIKE 'sz.00%' OR p.code LIKE 'sz.30%')
                  AND s.isST = 0 AND s.tradestatus = 1
                  AND s.listing_age_sessions >= 20
                  AND s.reference_gap IS FALSE
                  AND p.quote_outside_traded_range IS FALSE
                  AND s.preclose > 0 AND p.price_1449 > 0
                  AND p.amount_1449 >= 5000000
            ), priced AS (
                SELECT *, ROUND(CAST(preclose AS DECIMAL(18, 2))
                                * down_factor, 2) AS down_limit
                FROM eligible
            )
            SELECT date, COUNT(*) AS stocks,
                   COUNT(*) FILTER (
                       WHERE ABS(price_1449 - down_limit) <= .005
                   ) AS down_count,
                   AVG(CASE WHEN price_1449 > preclose
                            THEN 1.0 ELSE 0.0 END) AS up_share,
                   AVG(GREATEST(-.05, LEAST(.05,
                       price_1420 / preclose - 1
                   ))) AS pre_1420_return,
                   AVG(GREATEST(-.02, LEAST(.02,
                       price_1449 / price_1420 - 1
                   ))) AS late_29_return
            FROM priced GROUP BY date ORDER BY date
        """).df()
    finally:
        connection.close()

    if daily.empty or daily.duplicated("date").any():
        raise ValueError("Market dates are absent or duplicated")
    if not daily.date.str[:4].isin(("2024", "2025")).all():
        raise ValueError("Only 2024–2025 inputs are allowed")
    daily["half"] = daily.date.str[:4] + "H" + np.where(
        daily.date.str[5:7].astype(int).le(6), "1", "2")
    daily["down_share"] = daily.down_count / daily.stocks
    daily["state"] = np.select(
        [daily.down_share.ge(.002), daily.down_count.eq(0)],
        ["stress", "calm"], default="middle")
    usable = daily.loc[daily.stocks.ge(1000)].copy()
    correlations = {
        comparator: float(usable.down_share.corr(usable[comparator]))
        for comparator in COMPARATORS
    }

    signals = pd.read_parquet(root / "late_reversal_1449" /
                              "selections.parquet", columns=[
                                  "date", "code", "candidate", "pair_id",
                              ])
    if (len(signals) != 2052 or signals.duplicated(["date", "code"]).any()
            or set(signals.candidate) != {
                "late_decline", "late_rally_control"
            } or not signals.date.str[:4].isin(("2024", "2025")).all()):
        raise ValueError("The frozen 14:49 pair list has changed")
    treatment = signals.loc[signals.candidate.eq("late_decline"),
                            ["date", "pair_id"]]
    assigned = treatment.merge(usable[["date", "half", "state"]],
                               on="date", validate="many_to_one")
    market_days = usable.groupby("half").agg(
        days=("date", "size"), median_stocks=("stocks", "median"))
    counts = assigned.groupby(["half", "state"]).agg(
        pairs=("pair_id", "size"), signal_days=("date", "nunique"))
    states_by_half = {
        half: {
            state: {
                "pairs": int(counts.loc[(half, state), "pairs"])
                if (half, state) in counts.index else 0,
                "signal_days": int(counts.loc[(half, state), "signal_days"])
                if (half, state) in counts.index else 0,
            }
            for state in ("stress", "middle", "calm")
        }
        for half in HALVES
    }
    coverage_ok = (len(assigned) == len(treatment)
                   and all(half in market_days.index
                           and market_days.loc[half, "days"] >= 100
                           for half in HALVES))
    orthogonal_ok = all(np.isfinite(value) and abs(value) < .7
                        for value in correlations.values())
    states_ok = all(
        states_by_half[half][state]["pairs"] >= 30
        and states_by_half[half][state]["signal_days"] >= 15
        for half in HALVES for state in ("stress", "calm")
    )
    report = {
        "years": [2024, 2025],
        "market_days": int(len(daily)),
        "usable_market_days": int(len(usable)),
        "minimum_usable_stocks": int(usable.stocks.min()) if len(usable) else 0,
        "frozen_pairs": int(len(treatment)),
        "pairs_with_state": int(len(assigned)),
        "correlations_with_down_share": correlations,
        "market_by_half": {
            half: {
                "days": int(market_days.loc[half, "days"]),
                "median_stocks": float(market_days.loc[half, "median_stocks"]),
            }
            for half in HALVES if half in market_days.index
        },
        "states_by_half": states_by_half,
        "coverage_gate_passed": bool(coverage_ok),
        "orthogonality_gate_passed": bool(orthogonal_ok),
        "state_count_gate_passed": bool(states_ok),
        "outcome_gate_passed": bool(coverage_ok and orthogonal_ok and states_ok),
    }
    output.mkdir(parents=True, exist_ok=True)
    usable.to_parquet(output / "states.parquet", index=False,
                      compression="zstd")
    (output / "input_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(audit_inputs(args.root, args.output),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
