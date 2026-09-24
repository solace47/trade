"""Test a weekly liquidity shock using Friday 14:50 information only.

The prior benchmark is the previous 12 valid weeks. The current week combines
completed earlier days with the partial signal day through 14:50. This is
closer to the cited weekly study than a single-day shock, but still differs
from its complete-week signal and historical sample.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

from .market_study import _quality_keys, _quality_symbols
from .residual_liquidity import CAPACITY, _period
from .strategy_scan import _last_safe_entry


HORIZON = 5
RANKS = {
    "weekly_adjusted_absolute": ("all", "adjusted_absolute DESC"),
    "weekly_adjusted_relative": ("all", "adjusted_relative DESC"),
    "weekly_relative_conditioned": ("conditioned", "adjusted_relative DESC"),
}


def study(daily_dir: Path, snapshot_dir: Path, outcome_dir: Path,
          issues_dir: Path, report_path: Path, trades_path: Path) -> dict:
    c = duckdb.connect()
    c.execute("SET threads = 4")
    c.read_parquet(str(daily_dir / "*.parquet")).create_view("d")
    c.read_parquet(str(snapshot_dir / "*.parquet")).create_view("s")
    c.read_parquet(str(outcome_dir / "*.parquet")).create_view("o")
    c.register("bad_days", _quality_keys(issues_dir))
    c.register("bad_symbols", _quality_symbols(issues_dir))
    c.execute("""
        CREATE TEMP TABLE market_weeks AS
        SELECT date_trunc('week', CAST(date AS DATE)) AS week_start,
               max(date) AS signal_date
        FROM (SELECT DISTINCT date FROM s
              WHERE date >= '2024-01-01' AND date < '2026-01-01')
        GROUP BY 1
    """)
    # The year-end buffer is computed on market trading days, not signal
    # Fridays: ten sessions cover T+5 and a further five-session exit delay.
    c.execute("CREATE TEMP VIEW snapshots AS SELECT DISTINCT date FROM s")
    last_entry = {
        year: _last_safe_entry(c, f"{year}-01-01", f"{year + 1}-01-01")
        for year in (2024, 2025)
    }
    c.execute("""
        CREATE TEMP TABLE daily_traded AS
        SELECT date, code, date_trunc('week', CAST(date AS DATE)) AS week_start,
               abs(close / preclose - 1) * 100000000 / amount AS illiquidity,
               ln(close / preclose) AS log_growth
        FROM d
        WHERE date >= '2023-01-01' AND date < '2026-01-01'
          AND tradestatus = 1 AND close > 0 AND preclose > 0 AND amount > 0
    """)
    c.execute("""
        CREATE TEMP TABLE historical_weeks AS
        WITH valid_week AS (
            SELECT code, week_start, avg(illiquidity) AS weekly_illiquidity
            FROM daily_traded GROUP BY code, week_start
            HAVING count(*) >= 3
        ), rolling AS (
            SELECT code, week_start,
                   avg(weekly_illiquidity) OVER (
                       PARTITION BY code ORDER BY week_start
                       ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
                   ) AS prior12_illiquidity,
                   count(*) OVER (
                       PARTITION BY code ORDER BY week_start
                       ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
                   ) AS prior_weeks
            FROM valid_week
        )
        SELECT * FROM rolling WHERE prior_weeks >= 6
    """)
    c.execute("""
        CREATE TEMP TABLE known_week AS
        SELECT d.code, m.week_start, m.signal_date,
               count(*) AS prior_days_in_week,
               sum(d.illiquidity) AS prior_illiquidity_sum,
               sum(d.log_growth) AS prior_log_growth
        FROM daily_traded d JOIN market_weeks m USING (week_start)
        WHERE d.date < m.signal_date
        GROUP BY d.code, m.week_start, m.signal_date
        HAVING count(*) >= 2
    """)
    c.execute(f"""
        CREATE TEMP TABLE eligible AS
        WITH point_in_time AS (
            SELECT s.date, s.code, s.return_1450, s.return20_prior_adjusted,
                   s.amount_1450, p.prior12_illiquidity,
                   (k.prior_illiquidity_sum
                     + abs(s.return_1450) * 100000000
                       / (s.amount_1450 * 241.0 / 231.0))
                       / (k.prior_days_in_week + 1) AS current_week_illiquidity,
                   exp(k.prior_log_growth) * (1 + s.return_1450) - 1
                       AS current_week_return
            FROM s JOIN market_weeks m ON s.date = m.signal_date
            JOIN known_week k ON s.date = k.signal_date AND s.code = k.code
            ASOF JOIN historical_weeks p
              ON s.code = p.code AND m.week_start > p.week_start
            WHERE ((s.date BETWEEN '2024-01-01' AND '{last_entry[2024]}')
                OR (s.date BETWEEN '2025-01-01' AND '{last_entry[2025]}'))
              AND s.isST = 0 AND s.listing_age_sessions >= 60
              AND NOT s.reference_gap AND NOT s.quote_outside_traded_range
              AND s.amount_1450 BETWEEN 100000000 AND 1000000000
              AND s.price_1450 >= 5 AND p.prior12_illiquidity > 0
        ), shocks AS (
            SELECT *, prior12_illiquidity - current_week_illiquidity
                         AS absolute_shock,
                   greatest(-5.0, least(1.0,
                       1 - current_week_illiquidity / prior12_illiquidity
                   )) AS relative_shock
            FROM point_in_time
        ), slopes AS (
            SELECT date, avg(absolute_shock) AS mean_absolute,
                   avg(relative_shock) AS mean_relative,
                   avg(current_week_return) AS mean_return,
                   covar_pop(absolute_shock, current_week_return)
                     / NULLIF(var_pop(current_week_return), 0) AS beta_absolute,
                   covar_pop(relative_shock, current_week_return)
                     / NULLIF(var_pop(current_week_return), 0) AS beta_relative
            FROM shocks GROUP BY date
        )
        SELECT h.date, h.code, h.return20_prior_adjusted,
               h.absolute_shock - m.mean_absolute
                 - m.beta_absolute * (h.current_week_return - m.mean_return)
                   AS adjusted_absolute,
               h.relative_shock - m.mean_relative
                 - m.beta_relative * (h.current_week_return - m.mean_return)
                   AS adjusted_relative
        FROM shocks h JOIN slopes m USING (date)
        WHERE m.beta_absolute IS NOT NULL AND m.beta_relative IS NOT NULL
    """)
    c.execute("""
        CREATE TEMP VIEW conditioned AS SELECT * FROM eligible
        WHERE (code LIKE 'sh.60%' OR code LIKE 'sz.00%')
          AND return20_prior_adjusted BETWEEN -.10 AND .10
    """)
    selected = []
    for name, (pool, order) in RANKS.items():
        picked = c.execute(f"""
            SELECT date, code, daily_rank FROM (
                SELECT date, code, ROW_NUMBER() OVER (
                    PARTITION BY date ORDER BY {order}, code
                ) AS daily_rank FROM {pool if pool == 'conditioned' else 'eligible'}
            ) WHERE daily_rank <= {CAPACITY}
        """).df()
        picked["candidate"] = name
        selected.append(picked)
    for pool, label in (("eligible", "random_all"),
                        ("conditioned", "random_conditioned")):
        random = c.execute(f"""
            SELECT date, code, daily_rank FROM (
                SELECT date, code, ROW_NUMBER() OVER (
                    PARTITION BY date ORDER BY md5(date || code), code
                ) AS daily_rank FROM {pool}
            ) WHERE daily_rank <= {CAPACITY}
        """).df()
        random["candidate"] = label
        selected.append(random)
    membership = pd.concat(selected, ignore_index=True)
    if membership.duplicated(["candidate", "date", "code"]).any():
        raise ValueError("Duplicate ranked stock-day")
    c.register("membership", membership)
    trades = c.execute(f"""
        SELECT r.*, o.entry_status, o.entry_price, o.shares,
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
        FROM membership r JOIN o USING (date, code)
        WHERE o.horizon = {HORIZON}
    """).df()
    if len(trades) != len(membership):
        raise ValueError("A ranked stock-day lacks a five-session outcome")
    if not trades.date.str[:4].isin(("2024", "2025")).all():
        raise ValueError("Weekly study escaped the 2024-2025 window")
    report = {
        "hypotheses": list(RANKS), "horizon": HORIZON,
        "capacity": CAPACITY, "last_entry": last_entry,
        "control": "same signal day and same eligibility pool",
        "source": "https://www.zgglkx.com/CN/Y2023/V31/I7/68",
        "note": "2025 was previously viewed; weekly adaptation is exploratory",
        "results": {},
    }
    for name, (pool, _) in RANKS.items():
        reference_name = "random_conditioned" if pool == "conditioned" \
            else "random_all"
        report["results"][name] = {}
        for year in ("2024", "2025"):
            annual = trades.loc[trades.candidate.eq(name)
                                & trades.date.str.startswith(year)]
            reference = trades.loc[trades.candidate.eq(reference_name)
                                   & trades.date.str.startswith(year)]
            report["results"][name][year] = {}
            for period, rows in (
                ("H1", annual.loc[annual.date.str[5:7].astype(int).le(6)]),
                ("H2", annual.loc[annual.date.str[5:7].astype(int).gt(6)]),
                ("full", annual),
            ):
                if not rows.empty:
                    report["results"][name][year][period] = _period(rows, reference)
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
                        default=Path("data/research/weekly_liquidity_report.json"))
    parser.add_argument("--trades", type=Path,
                        default=Path("data/research/weekly_liquidity_trades.parquet"))
    args = parser.parse_args()
    report = study(args.daily, args.snapshots, args.outcomes, args.issues,
                   args.report, args.trades)
    print({"hypotheses": list(report["results"]), "report": str(args.report)})


if __name__ == "__main__":
    main()
