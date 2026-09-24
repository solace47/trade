"""Evaluate one registered late directional-turnover proxy hypothesis.

Motivation: Narayan, Narayan & Westerlund (2015),
https://doi.org/10.1016/j.pacfin.2015.07.003, found predictive content in
true intraday order imbalances. This source lacks aggressor-side trades, so
the minute close-sign-weighted turnover below is a weaker proxy, not a
replication of their variable or result.

The rule is fixed before reading outcomes: on the main board, require a
liquid, seasoned, calm 14:50 stock with prior-20-day adjusted return within
/-10%; rank each day's eligible stocks by the proxy, draw five names from
the top quintile by deterministic hash, and compare to five random eligible
names and five bottom-quintile names. T+1 is primary; T+2 and T+5 diagnose
persistence. Check each 2024 and 2025 half separately. 2025 has been seen
before and is not blind. No 2026 outcomes enter this study.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

from .market_study import _quality_keys, _quality_symbols
from .residual_liquidity import _period
from .strategy_scan import _last_safe_entry


CAPACITY = 5
HORIZONS = (1, 2, 5)
ELIGIBILITY = """
    s.isST = 0 AND s.listing_age_sessions >= 20
    AND NOT s.reference_gap AND NOT s.quote_outside_traded_range
    AND (s.code LIKE 'sh.60%' OR s.code LIKE 'sz.00%')
    AND s.amount_1450 BETWEEN 100000000 AND 1000000000
    AND s.price_1450 >= 5
    AND s.return20_prior_adjusted BETWEEN -.10 AND .10
    AND s.volume_ratio_est BETWEEN .5 AND 2
    AND abs(s.return_1450) <= .03
    AND abs(i.return_last30) <= .003
    AND abs(s.price_1450 - i.price_1450) <= .005
    AND abs(s.price_1450 - f.price_1450) <= .005
"""


def study(snapshot_dir: Path, outcome_dir: Path, issues_dir: Path,
          intraday_dir: Path, proxy_dir: Path, report_path: Path,
          trades_path: Path) -> dict:
    c = duckdb.connect()
    c.execute("SET threads = 4")
    c.read_parquet(str(snapshot_dir / "*.parquet")).create_view("s")
    c.read_parquet(str(outcome_dir / "*.parquet")).create_view("o")
    c.read_parquet(str(intraday_dir / "*.parquet")).create_view("i")
    c.read_parquet(str(proxy_dir / "*.parquet")).create_view("f")
    c.register("bad_days", _quality_keys(issues_dir))
    c.register("bad_symbols", _quality_symbols(issues_dir))
    c.execute("CREATE TEMP VIEW snapshots AS SELECT date FROM s")
    last_entry = {
        year: _last_safe_entry(c, f"{year}-01-01", f"{year + 1}-01-01")
        for year in (2024, 2025)
    }
    c.execute(f"""
        CREATE TEMP TABLE eligible AS
        SELECT s.date, s.code, s.return_1450, s.amount_1450,
               i.return_last30,
               f.signed_turnover_proxy_last30 AS flow_proxy,
               f.moving_turnover_share_last30 AS moving_share
        FROM s JOIN i USING (date, code) JOIN f USING (date, code)
        WHERE ((s.date BETWEEN '2024-01-01' AND '{last_entry[2024]}')
            OR (s.date BETWEEN '2025-01-01' AND '{last_entry[2025]}'))
          AND {ELIGIBILITY}
    """)
    c.execute("""
        CREATE TEMP TABLE bucketed AS
        SELECT *, NTILE(5) OVER (
            PARTITION BY date ORDER BY flow_proxy, code
        ) AS flow_bucket
        FROM eligible
        QUALIFY COUNT(*) OVER (PARTITION BY date) >= 25
    """)
    selections = []
    for name, condition in (
        ("top_flow", "flow_bucket = 5"),
        ("bottom_flow", "flow_bucket = 1"),
        ("same_day_random", "TRUE"),
    ):
        selected = c.execute(f"""
            SELECT date, code, daily_rank FROM (
                SELECT date, code, ROW_NUMBER() OVER (
                    PARTITION BY date
                    ORDER BY md5('flow-v1' || date || code), code
                ) AS daily_rank
                FROM bucketed WHERE {condition}
            ) WHERE daily_rank <= {CAPACITY}
        """).df()
        selected["candidate"] = name
        selections.append(selected)
    chosen = pd.concat(selections, ignore_index=True)
    if chosen.duplicated(["candidate", "date", "code"]).any():
        raise ValueError("Duplicate ranked stock-day")
    c.register("chosen", chosen)
    trades = c.execute(f"""
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
        FROM chosen r JOIN o USING (date, code)
        WHERE o.horizon IN (1, 2, 5)
    """).df()
    if len(trades) != len(chosen) * len(HORIZONS):
        raise ValueError("A ranked stock-day lacks an outcome")
    if not trades.date.str[:4].isin(("2024", "2025")).all():
        raise ValueError("Research window escaped 2024-2025")
    report = {
        "hypothesis": "Higher late directional-turnover proxy predicts higher T+1 return",
        "proxy_is_not_true_order_imbalance": True,
        "capacity": CAPACITY, "horizons": HORIZONS,
        "eligibility": ELIGIBILITY.strip(), "last_entry": last_entry,
        "control": "five same-day random names from the identical eligible pool",
        "note": "2025 was previously viewed and is not a blind holdout",
        "eligible_count": int(c.execute("SELECT COUNT(*) FROM bucketed").fetchone()[0]),
        "results": {},
    }
    for name in ("top_flow", "bottom_flow"):
        report["results"][name] = {}
        for year in ("2024", "2025"):
            annual = trades.loc[trades.candidate.eq(name)
                                & trades.date.str.startswith(year)]
            control = trades.loc[trades.candidate.eq("same_day_random")
                                 & trades.date.str.startswith(year)]
            for horizon in HORIZONS:
                sample = annual.loc[annual.horizon.eq(horizon)]
                reference = control.loc[control.horizon.eq(horizon)]
                report["results"][name].setdefault(year, {})[str(horizon)] = {}
                for period, rows in (
                    ("H1", sample.loc[sample.date.str[5:7].astype(int).le(6)]),
                    ("H2", sample.loc[sample.date.str[5:7].astype(int).gt(6)]),
                    ("full", sample),
                ):
                    if not rows.empty:
                        report["results"][name][year][str(horizon)][period] = \
                            _period(rows, reference)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    trades_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    trades.to_parquet(trades_path, index=False, compression="zstd")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path,
                        default=Path("data/research/market_snapshots_ci"))
    parser.add_argument("--outcomes", type=Path,
                        default=Path("data/research/market_outcomes_ci"))
    parser.add_argument("--issues", type=Path,
                        default=Path("data/research/market_issues_ci"))
    parser.add_argument("--features", type=Path,
                        default=Path("data/research/intraday_features"))
    parser.add_argument("--proxy", type=Path,
                        default=Path("data/research/late_flow_proxy"))
    parser.add_argument("--report", type=Path,
                        default=Path("data/research/late_flow_report.json"))
    parser.add_argument("--trades", type=Path,
                        default=Path("data/research/late_flow_trades.parquet"))
    args = parser.parse_args()
    result = study(args.snapshots, args.outcomes, args.issues, args.features,
                   args.proxy, args.report, args.trades)
    print({"eligible_count": result["eligible_count"],
           "report": str(args.report)})


if __name__ == "__main__":
    main()
