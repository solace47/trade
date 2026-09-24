"""Test fixed risk exclusions without optimizing the stock rank.

Each policy picks the first five names under the same deterministic hash.
The 2024 factor bins motivated these exclusions; 2025 has already been viewed
in earlier work and is not an untouched holdout.
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
FILTERS = {
    "avoid_overheated": "return20_prior_adjusted <= .10",
    "avoid_late_extreme": "abs(return_last30) <= .003",
    "combined_calm": (
        "return20_prior_adjusted <= .10 "
        "AND abs(return_last30) <= .003 "
        "AND abs(return_1450) <= .03 "
        "AND volume_ratio_est BETWEEN .5 AND 2"
    ),
}


def study(snapshot_dir: Path, outcome_dir: Path, issues_dir: Path,
          feature_dir: Path, report_path: Path, trades_path: Path) -> dict:
    c = duckdb.connect()
    c.execute("SET threads = 4")
    c.read_parquet(str(snapshot_dir / "*.parquet")).create_view("s")
    c.read_parquet(str(outcome_dir / "*.parquet")).create_view("o")
    c.read_parquet(str(feature_dir / "*.parquet")).create_view("i")
    c.register("bad_days", _quality_keys(issues_dir))
    c.register("bad_symbols", _quality_symbols(issues_dir))
    c.execute("CREATE TEMP VIEW snapshots AS SELECT date FROM s")
    last_entry = {
        year: _last_safe_entry(c, f"{year}-01-01", f"{year + 1}-01-01")
        for year in (2024, 2025)
    }
    c.execute(f"""
        CREATE TEMP TABLE eligible AS
        SELECT s.date, s.code, s.return_1450,
               s.return20_prior_adjusted, s.volume_ratio_est,
               i.return_last30
        FROM s JOIN i USING (date, code)
        WHERE ((s.date BETWEEN '2024-01-01' AND '{last_entry[2024]}')
            OR (s.date BETWEEN '2025-01-01' AND '{last_entry[2025]}'))
          AND s.isST = 0 AND s.listing_age_sessions >= 20
          AND NOT s.reference_gap AND NOT s.quote_outside_traded_range
          AND s.amount_1450 BETWEEN 100000000 AND 1000000000
          AND s.price_1450 >= 5
          AND s.return20_prior_adjusted IS NOT NULL
          AND s.volume_ratio_est IS NOT NULL
          AND abs(s.price_1450 - i.price_1450) <= .005
    """)
    picks = []
    for name, condition in {"same_day_random": "TRUE", **FILTERS}.items():
        chosen = c.execute(f"""
            SELECT date, code, daily_rank FROM (
                SELECT date, code, ROW_NUMBER() OVER (
                    PARTITION BY date
                    ORDER BY md5('risk-v1' || date || code), code
                ) AS daily_rank
                FROM eligible WHERE {condition}
            ) WHERE daily_rank <= {CAPACITY}
        """).df()
        chosen["candidate"] = name
        picks.append(chosen)
    selected = pd.concat(picks, ignore_index=True)
    if selected.duplicated(["candidate", "date", "code"]).any():
        raise ValueError("Duplicate ranked stock-day")
    c.register("selected", selected)
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
        FROM selected r JOIN o USING (date, code)
        WHERE o.horizon = {HORIZON}
    """).df()
    if len(trades) != len(selected):
        raise ValueError("A ranked stock-day lacks a five-session outcome")
    report = {"filters": FILTERS, "horizon": HORIZON,
              "capacity": CAPACITY, "last_entry": last_entry,
              "control": "same-day eligible random using identical hash rank",
              "note": "Exploratory: motivated by 2024 bins; 2025 already viewed",
              "results": {}}
    for name in FILTERS:
        report["results"][name] = {}
        for year in ("2024", "2025"):
            annual = trades.loc[trades.candidate.eq(name)
                                & trades.date.str.startswith(year)]
            control = trades.loc[trades.candidate.eq("same_day_random")
                                 & trades.date.str.startswith(year)]
            report["results"][name][year] = {}
            for period, rows in (
                ("H1", annual.loc[annual.date.str[5:7].astype(int).le(6)]),
                ("H2", annual.loc[annual.date.str[5:7].astype(int).gt(6)]),
                ("full", annual),
            ):
                if not rows.empty:
                    report["results"][name][year][period] = _period(rows, control)
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
    parser.add_argument("--report", type=Path,
                        default=Path("data/research/risk_filter_report.json"))
    parser.add_argument("--trades", type=Path,
                        default=Path("data/research/risk_filter_trades.parquet"))
    args = parser.parse_args()
    report = study(args.snapshots, args.outcomes, args.issues, args.features,
                   args.report, args.trades)
    print({"filters": list(report["results"]), "report": str(args.report)})


if __name__ == "__main__":
    main()
