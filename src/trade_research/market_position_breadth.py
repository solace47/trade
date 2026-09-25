"""Input-only audit of 14:49 market-wide intraday range position."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path("data/research")
OUTPUT = ROOT / "market_position_breadth"
HALVES = ("2024H1", "2024H2", "2025H1", "2025H2")
COMPARATORS = ("up_share", "pre_1420_return", "late_29_return")


def _half(dates: pd.Series) -> pd.Series:
    return dates.str[:4] + "H" + np.where(
        dates.str[5:7].astype(int).le(6), "1", "2"
    )


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
                SELECT p.date,
                       (p.price_1449 - p.low_1449)
                           / (p.high_1449 - p.low_1449) AS position,
                       p.price_1449 > s.preclose AS up,
                       GREATEST(-.05, LEAST(.05,
                           p.price_1420 / s.preclose - 1
                       )) AS pre_1420_return,
                       GREATEST(-.02, LEAST(.02,
                           p.price_1449 / p.price_1420 - 1
                       )) AS late_29_return
                FROM prefix p JOIN snapshots s USING (date, code)
                WHERE p.date BETWEEN '2024-01-01' AND '2025-12-31'
                  AND s.isST = 0 AND s.tradestatus = 1
                  AND s.listing_age_sessions >= 20
                  AND s.reference_gap IS FALSE
                  AND p.quote_outside_traded_range IS FALSE
                  AND s.preclose > 0 AND p.amount_1449 >= 30000000
                  AND p.high_1449 - p.low_1449 >= .01 * s.preclose
            )
            SELECT date, COUNT(*) AS stocks,
                   AVG(CASE WHEN position <= .2 THEN 1.0 ELSE 0.0 END)
                       AS low_share,
                   AVG(CASE WHEN position >= .8 THEN 1.0 ELSE 0.0 END)
                       AS high_share,
                   AVG(CASE WHEN up THEN 1.0 ELSE 0.0 END) AS up_share,
                   AVG(pre_1420_return) AS pre_1420_return,
                   AVG(late_29_return) AS late_29_return
            FROM eligible GROUP BY date ORDER BY date
        """).df()
    finally:
        connection.close()

    if daily.empty or daily.duplicated("date").any():
        raise ValueError("Market dates are absent or duplicated")
    if not daily.date.str[:4].isin(("2024", "2025")).all():
        raise ValueError("Only 2024–2025 inputs are allowed")
    daily["half"] = _half(daily.date)
    daily["low_minus_high"] = daily.low_share - daily.high_share
    daily["state"] = np.select(
        [daily.low_minus_high.ge(.20), daily.low_minus_high.le(-.20)],
        ["low", "high"], default="middle",
    )
    usable = daily.loc[daily.stocks.ge(1000)].copy()
    correlations = {
        comparator: float(usable.low_minus_high.corr(usable[comparator]))
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
    by_half = usable.groupby("half", sort=True).agg(
        market_days=("date", "size"),
        median_stocks=("stocks", "median"),
    ).to_dict("index")
    assigned_counts = assigned.groupby(["half", "state"], sort=True).agg(
        pairs=("pair_id", "size"), signal_days=("date", "nunique"),
    ).to_dict("index")
    states_by_half = {
        half: {
            state: {
                "pairs": int(assigned_counts.get((half, state), {}).get(
                    "pairs", 0)),
                "signal_days": int(assigned_counts.get((half, state), {}).get(
                    "signal_days", 0)),
            }
            for state in ("low", "middle", "high")
        }
        for half in HALVES
    }
    coverage_ok = all(
        half in by_half and by_half[half]["market_days"] >= 100
        for half in HALVES
    ) and len(assigned) == len(treatment)
    orthogonal_ok = all(
        np.isfinite(value) and abs(value) < .7
        for value in correlations.values()
    )
    state_counts_ok = all(
        states_by_half[half][state]["pairs"] >= 30
        and states_by_half[half][state]["signal_days"] >= 15
        for half in HALVES for state in ("low", "high")
    )
    report = {
        "years": [2024, 2025],
        "all_market_days": int(len(daily)),
        "usable_market_days": int(len(usable)),
        "excluded_market_days_under_1000_stocks": int(
            daily.stocks.lt(1000).sum()),
        "frozen_pairs": int(len(treatment)),
        "pairs_with_market_state": int(len(assigned)),
        "correlation_with_low_minus_high": correlations,
        "market_by_half": {
            half: {
                "market_days": int(row["market_days"]),
                "median_stocks": float(row["median_stocks"]),
            }
            for half, row in by_half.items()
        },
        "states_by_half": states_by_half,
        "coverage_gate_passed": bool(coverage_ok),
        "orthogonality_gate_passed": bool(orthogonal_ok),
        "state_count_gate_passed": bool(state_counts_ok),
        "outcome_gate_passed": bool(
            coverage_ok and orthogonal_ok and state_counts_ok
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").unlink(missing_ok=True)
    usable.to_parquet(output / "states.parquet", index=False,
                      compression="zstd")
    (output / "input_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
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
