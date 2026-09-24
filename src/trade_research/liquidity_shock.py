"""Test a 14:50 adaptation of price-adjusted Amihud liquidity shocks.

The source paper uses complete weeks in 2000-2019. This experiment instead
uses the current 14:50 partial day versus the previous 60 traded days, with
2023 prices only warming up the prior average. It is a new hypothesis, not a
replication of the paper or a live-tradable rule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

from .market_study import _quality_keys, _quality_symbols
from .residual_liquidity import CAPACITY, HORIZONS, _period
from .strategy_scan import _last_safe_entry


RULES = {
    "adjusted_absolute": "adjusted_absolute DESC",
    "adjusted_relative": "adjusted_relative DESC",
    "unadjusted_absolute": "absolute_shock DESC",
    "adjusted_relative_low": "adjusted_relative ASC",
}


def study(daily_dir: Path, snapshot_dir: Path, outcome_dir: Path,
          issues_dir: Path, report_path: Path, trades_path: Path) -> dict:
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.read_parquet(str(daily_dir / "*.parquet")).create_view("d")
    connection.read_parquet(str(snapshot_dir / "*.parquet")).create_view("s")
    connection.read_parquet(str(outcome_dir / "*.parquet")).create_view("o")
    connection.register("bad_days", _quality_keys(issues_dir))
    connection.register("bad_symbols", _quality_symbols(issues_dir))
    connection.execute("""
        CREATE TEMP TABLE market_dates AS
        SELECT DISTINCT date FROM s
        WHERE date >= '2024-01-01' AND date < '2026-01-01'
    """)
    connection.execute("CREATE TEMP VIEW snapshots AS SELECT date FROM market_dates")
    last_entry = {
        year: _last_safe_entry(connection, f"{year}-01-01", f"{year + 1}-01-01")
        for year in (2024, 2025)
    }
    # The window ends at t-1. Today's full-day price and amount are never
    # selected from this table; the signal's current return and amount come
    # only from the 14:50 snapshot.
    connection.execute("""
        CREATE TEMP TABLE prior_illiquidity AS
        WITH traded AS (
            SELECT date, code,
                   abs(close / preclose - 1) * 100000000 / amount AS illiquidity
            FROM d
            WHERE date >= '2023-01-01' AND date < '2026-01-01'
              AND tradestatus = 1 AND preclose > 0 AND amount > 0
        ), historical AS (
            SELECT date, code,
                   avg(illiquidity) OVER (
                       PARTITION BY code ORDER BY date
                       ROWS BETWEEN 60 PRECEDING AND 1 PRECEDING
                   ) AS prior60_illiquidity,
                   count(*) OVER (
                       PARTITION BY code ORDER BY date
                       ROWS BETWEEN 60 PRECEDING AND 1 PRECEDING
                   ) AS prior_observations
            FROM traded
        )
        SELECT date, code, prior60_illiquidity FROM historical
        WHERE date >= '2024-01-01' AND date < '2026-01-01'
          AND prior_observations >= 30 AND prior60_illiquidity > 0
    """)
    connection.execute(f"""
        CREATE TEMP TABLE eligible AS
        WITH point_in_time AS (
            SELECT s.date, s.code, s.return_1450, s.amount_1450,
                   p.prior60_illiquidity,
                   abs(s.return_1450) * 100000000
                     / (s.amount_1450 * 241.0 / 231.0) AS current_illiquidity
            FROM s JOIN prior_illiquidity p USING (date, code)
            WHERE ((s.date BETWEEN '2024-01-01' AND '{last_entry[2024]}')
                OR (s.date BETWEEN '2025-01-01' AND '{last_entry[2025]}'))
              AND s.isST = 0 AND s.listing_age_sessions >= 60
              AND NOT s.reference_gap AND NOT s.quote_outside_traded_range
              AND s.amount_1450 BETWEEN 100000000 AND 1000000000
              AND s.price_1450 >= 5
        ), shocks AS (
            SELECT *, prior60_illiquidity - current_illiquidity
                         AS absolute_shock,
                   greatest(-5.0, least(1.0,
                       1 - current_illiquidity / prior60_illiquidity
                   )) AS relative_shock
            FROM point_in_time
        ), slopes AS (
            SELECT date, avg(absolute_shock) AS mean_absolute,
                   avg(relative_shock) AS mean_relative,
                   avg(return_1450) AS mean_return,
                   covar_pop(absolute_shock, return_1450)
                       / NULLIF(var_pop(return_1450), 0) AS beta_absolute,
                   covar_pop(relative_shock, return_1450)
                       / NULLIF(var_pop(return_1450), 0) AS beta_relative
            FROM shocks GROUP BY date
        )
        SELECT h.date, h.code, h.amount_1450, h.return_1450,
               h.prior60_illiquidity, h.current_illiquidity,
               h.absolute_shock, h.relative_shock,
               h.absolute_shock - m.mean_absolute
                 - m.beta_absolute * (h.return_1450 - m.mean_return)
                   AS adjusted_absolute,
               h.relative_shock - m.mean_relative
                 - m.beta_relative * (h.return_1450 - m.mean_return)
                   AS adjusted_relative
        FROM shocks h JOIN slopes m USING (date)
        WHERE m.beta_absolute IS NOT NULL AND m.beta_relative IS NOT NULL
    """)
    selections = []
    for name, order in RULES.items():
        picked = connection.execute(f"""
            SELECT date, code, daily_rank FROM (
                SELECT date, code, ROW_NUMBER() OVER (
                    PARTITION BY date ORDER BY {order}, code
                ) AS daily_rank FROM eligible
            ) WHERE daily_rank <= {CAPACITY}
        """).df()
        picked["candidate"] = name
        selections.append(picked)
    control = connection.execute(f"""
        SELECT date, code, daily_rank FROM (
            SELECT date, code, ROW_NUMBER() OVER (
                PARTITION BY date ORDER BY md5(date || code), code
            ) AS daily_rank FROM eligible
        ) WHERE daily_rank <= {CAPACITY}
    """).df()
    control["candidate"] = "random_liquid"
    selections.append(control)
    selected = pd.concat(selections, ignore_index=True)
    if selected.duplicated(["candidate", "date", "code"]).any():
        raise ValueError("Duplicate ranked stock-day")
    connection.register("selected", selected)
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
        FROM selected r JOIN o USING (date, code)
        WHERE o.horizon IN (1, 2, 3, 5)
    """).df()
    if len(trades) != len(selected) * len(HORIZONS):
        raise ValueError("A ranked stock-day lacks an outcome")
    report = {
        "hypotheses": list(RULES), "capacity": CAPACITY,
        "last_entry": last_entry,
        "control": "deterministic same-day random eligible stocks",
        "source": "https://www.zgglkx.com/CN/Y2023/V31/I7/68",
        "difference_from_paper": "14:50 partial day versus prior 60 traded days; "
                                 "paper used weekly Amihud shocks versus 12 prior weeks",
        "note": "2025 was previously viewed and is not a blind holdout",
        "results": {},
    }
    for name in RULES:
        report["results"][name] = {}
        for year in ("2024", "2025"):
            annual = trades.loc[trades.candidate.eq(name)
                                & trades.date.str.startswith(year)]
            control_rows = trades.loc[trades.candidate.eq("random_liquid")
                                      & trades.date.str.startswith(year)]
            for horizon in HORIZONS:
                subset = annual.loc[annual.horizon.eq(horizon)]
                matching = control_rows.loc[control_rows.horizon.eq(horizon)]
                report["results"][name].setdefault(year, {})[str(horizon)] = {}
                for period, rows in (
                    ("H1", subset.loc[subset.date.str[5:7].astype(int).le(6)]),
                    ("H2", subset.loc[subset.date.str[5:7].astype(int).gt(6)]),
                    ("full", subset),
                ):
                    if not rows.empty:
                        report["results"][name][year][str(horizon)][period] = \
                            _period(rows, matching)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    trades_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    trades.to_parquet(trades_path, index=False, compression="zstd")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily", type=Path,
                        default=Path("data/baostock/market_2020_2026/daily"))
    parser.add_argument("--snapshots", type=Path,
                        default=Path("data/research/market_snapshots_ci"))
    parser.add_argument("--outcomes", type=Path,
                        default=Path("data/research/market_outcomes_ci"))
    parser.add_argument("--issues", type=Path,
                        default=Path("data/research/market_issues_ci"))
    parser.add_argument("--report", type=Path,
                        default=Path("data/research/liquidity_shock_report.json"))
    parser.add_argument("--trades", type=Path,
                        default=Path("data/research/liquidity_shock_trades.parquet"))
    args = parser.parse_args()
    report = study(args.daily, args.snapshots, args.outcomes, args.issues,
                   args.report, args.trades)
    print({"hypotheses": list(report["results"]), "report": str(args.report)})


if __name__ == "__main__":
    main()
