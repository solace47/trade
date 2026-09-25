"""Test a fixed quiet-trend x 14:50 market-breadth interaction.

Inputs to the selector are completed 14:50 snapshots and intraday bars.
Only 2024-2025 may enter outcomes; 2025 is an explored replication year.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .intraday_scan import SCREENS
from .market_study import _quality_keys, _quality_symbols
from .residual_liquidity import _period


ROOT = Path("data/research")
CAPACITY = 5
HORIZONS = (1, 5)
QUIET = SCREENS["quiet_trend"][0]
UP_POOL = (
    "return_1450 BETWEEN 0 AND .02 AND position_1450 >= .7 "
    "AND return20_prior_adjusted > 0"
)


def select_inputs(connection: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, dict]:
    """Select treatment and same-day nonquiet controls without outcomes."""
    connection.execute("""
        CREATE TEMP TABLE breadth AS
        SELECT date, AVG((return_1450 > 0)::INT) AS up_share,
               COUNT(*) AS market_stocks
        FROM snapshots
        WHERE ((date BETWEEN '2024-01-01' AND '2024-12-17')
            OR (date BETWEEN '2025-01-01' AND '2025-12-17'))
          AND isST = 0 AND listing_age_sessions >= 20
          AND NOT reference_gap AND NOT quote_outside_traded_range
          AND return_1450 IS NOT NULL
        GROUP BY date
    """)
    connection.execute("""
        CREATE TEMP TABLE eligible AS
        SELECT s.date, s.code, s.return_1450, s.position_1450,
               s.return20_prior_adjusted, s.amount_1450,
               i.return_last30, i.volume_share_last30,
               b.up_share, b.market_stocks,
               CASE WHEN b.up_share >= .6 THEN 'advance'
                    WHEN b.up_share <= .4 THEN 'decline'
                    ELSE 'mixed' END AS regime
        FROM snapshots s JOIN intraday i USING (date, code)
        JOIN breadth b USING (date)
        WHERE s.isST = 0 AND s.listing_age_sessions >= 20
          AND NOT s.reference_gap AND NOT s.quote_outside_traded_range
          AND s.amount_1450 BETWEEN 100000000 AND 1000000000
          AND abs(s.price_1450 - i.price_1450) <= .005
    """)
    connection.execute(f"""
        CREATE TEMP TABLE candidates AS
        SELECT *, ({QUIET}) AS quiet
        FROM eligible WHERE {UP_POOL}
    """)
    connection.execute("""
        CREATE TEMP TABLE comparable_dates AS
        SELECT date FROM candidates GROUP BY date
        HAVING COUNT(*) FILTER (WHERE quiet) > 0
           AND COUNT(*) FILTER (WHERE NOT quiet) > 0
    """)
    treatment = connection.execute(f"""
        SELECT date, code, return_1450, position_1450,
               return20_prior_adjusted, amount_1450,
               return_last30, volume_share_last30,
               up_share, market_stocks, regime,
               ROW_NUMBER() OVER (
                   PARTITION BY date ORDER BY return20_prior_adjusted DESC, code
               ) AS daily_rank
        FROM candidates JOIN comparable_dates USING (date)
        WHERE quiet
        QUALIFY daily_rank <= {CAPACITY}
    """).df()
    control_pool = connection.execute("""
        SELECT date, code, return_1450, position_1450,
               return20_prior_adjusted, amount_1450,
               return_last30, volume_share_last30,
               up_share, market_stocks, regime
        FROM candidates JOIN comparable_dates USING (date)
        WHERE NOT quiet
    """).df()
    controls_by_date = {date: rows.set_index("code", drop=False)
                        for date, rows in control_pool.groupby("date")}
    pairs = []
    for date, ranked in treatment.groupby("date", sort=True):
        available = controls_by_date[date].copy()
        for row in ranked.sort_values("daily_rank").itertuples(index=False):
            prior_gap = (available.return20_prior_adjusted
                         - row.return20_prior_adjusted).abs()
            current_gap = (available.return_1450 - row.return_1450).abs()
            amount_ratio = available.amount_1450 / row.amount_1450
            position_gap = (available.position_1450 - row.position_1450).abs()
            eligible = available.loc[
                prior_gap.le(.03) & current_gap.le(.005)
                & amount_ratio.between(.5, 2) & position_gap.le(.15)
            ].copy()
            if eligible.empty:
                continue
            distance = (prior_gap.loc[eligible.index] / .03
                        + current_gap.loc[eligible.index] / .005
                        + np.abs(np.log(amount_ratio.loc[eligible.index]))
                          / np.log(2)
                        + position_gap.loc[eligible.index] / .15)
            eligible["distance"] = distance
            matched = eligible.reset_index(drop=True).sort_values(
                ["distance", "code"]
            ).iloc[0]
            treatment_row = row._asdict()
            treatment_row.update(candidate="quiet_trend", pair_id=row.code)
            control_row = matched.drop(labels="distance").to_dict()
            control_row.update(candidate="nonquiet_control",
                               pair_id=row.code, daily_rank=row.daily_rank)
            pairs.extend((treatment_row, control_row))
            available = available.drop(index=matched.code)
    signals = pd.DataFrame(pairs)
    if (signals.empty or signals.duplicated(["candidate", "date", "code"]).any()
            or not signals.date.str[:4].isin(("2024", "2025")).all()):
        raise ValueError("Invalid quiet-breadth selection")
    if signals.groupby(["date", "pair_id"]).candidate.nunique().ne(2).any():
        raise ValueError("A quiet signal lacks its same-day control")
    coverage = signals.groupby(["candidate", signals.date.str[:4], "regime"])
    counts = coverage.agg(signals=("code", "size"), days=("date", "nunique"))
    paired = signals.loc[signals.candidate.eq("quiet_trend")]
    balance = paired.merge(
        signals.loc[signals.candidate.eq("nonquiet_control")],
        on=["date", "pair_id"], suffixes=("_quiet", "_control"),
        validate="one_to_one",
    )
    audit = {
        "coverage": counts.reset_index().to_dict("records"),
        "quiet_candidates": len(treatment),
        "paired_quiet": len(paired),
        "prior20_median_abs_gap": float((
            balance.return20_prior_adjusted_quiet
            - balance.return20_prior_adjusted_control).abs().median()),
        "current_median_abs_gap": float((
            balance.return_1450_quiet
            - balance.return_1450_control).abs().median()),
        "pool": UP_POOL, "quiet": QUIET,
    }
    return signals, audit


def study(output_dir: Path = ROOT / "quiet_breadth") -> dict:
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.read_parquet(str(ROOT / "market_snapshots_ci" / "*.parquet")
                            ).create_view("snapshots")
    connection.read_parquet(str(ROOT / "intraday_features" / "*.parquet")
                            ).create_view("intraday")
    signals, audit = select_inputs(connection)
    output_dir.mkdir(parents=True, exist_ok=True)
    signals.to_parquet(output_dir / "selections.parquet", index=False,
                       compression="zstd")
    connection.register("selected", signals)
    connection.read_parquet(str(ROOT / "market_outcomes_ci" / "*.parquet")
                            ).create_view("outcomes")
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
    if len(trades) != len(signals) * len(HORIZONS):
        raise ValueError("Missing archived outcome for a selected stock-day")
    trades.to_parquet(output_dir / "trades.parquet", index=False,
                      compression="zstd")
    report = {"input_audit": audit, "order_size": 100000,
              "notes": "2025 previously viewed; 2026 not used", "results": []}
    for year in ("2024", "2025"):
        for half, month_rule in (
            ("H1", lambda month: month <= 6),
            ("H2", lambda month: month > 6),
            ("full", lambda month: month > 0),
        ):
            for regime in ("advance", "mixed", "decline"):
                for horizon in HORIZONS:
                    rows = trades.loc[
                        trades.candidate.eq("quiet_trend")
                        & trades.date.str.startswith(year)
                        & month_rule(trades.date.str[5:7].astype(int))
                        & trades.regime.eq(regime)
                        & trades.horizon.eq(horizon)
                    ]
                    if rows.empty:
                        continue
                    controls = trades.loc[
                        trades.candidate.eq("nonquiet_control")
                        & trades.horizon.eq(horizon)
                    ]
                    result = _period(rows, controls)
                    report["results"].append({"year": year, "half": half,
                                              "regime": regime,
                                              "horizon": horizon, **result})
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "quiet_breadth")
    args = parser.parse_args()
    report = study(args.output)
    print({"groups": len(report["results"]),
           "output": str(args.output / "report.json")})


if __name__ == "__main__":
    main()
