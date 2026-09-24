"""Test fixed market-relative return and liquidity hypotheses at 14:50.

The four rules below are registered before reading their outcomes. 2024 is
development and 2025 has already been inspected in earlier work, so neither
year is an untouched holdout. The 2026 outcomes are deliberately not read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

from .market_study import _quality_keys, _quality_symbols, _week_bootstrap
from .strategy_scan import _last_safe_entry, _stressed_returns


HORIZONS = (1, 2, 3, 5)
CAPACITY = 5
RULES = {
    # Volume is a liquidity-shock proxy, not signed order flow or analyst news.
    "shock_reversal": (
        "residual_return BETWEEN -.08 AND -.02 "
        "AND volume_ratio_est BETWEEN 1.5 AND 5 "
        "AND return_last30 BETWEEN -.002 AND .003 "
        "AND return20_prior_adjusted >= -.15",
        "residual_return ASC",
    ),
    "quiet_reversal": (
        "residual_return BETWEEN -.08 AND -.02 "
        "AND volume_ratio_est BETWEEN .5 AND 1.2 "
        "AND return_last30 BETWEEN -.002 AND .003 "
        "AND return20_prior_adjusted >= -.15",
        "residual_return ASC",
    ),
    "prior_residual_loser": (
        "prior20_residual <= -.08 "
        "AND residual_return BETWEEN -.01 AND .01 "
        "AND return_last30 BETWEEN 0 AND .003 "
        "AND volume_ratio_est BETWEEN .5 AND 1.5",
        "prior20_residual ASC",
    ),
    "orderly_strength": (
        "residual_return BETWEEN .01 AND .04 "
        "AND volume_ratio_est BETWEEN .6 AND 1.5 "
        "AND return_last30 BETWEEN -.002 AND .003 "
        "AND return20_prior_adjusted > 0",
        "residual_return DESC",
    ),
}


def _period(frame: pd.DataFrame, reference: pd.DataFrame) -> dict:
    dates = sorted(frame.date.unique())
    control = reference.loc[reference.date.isin(dates)].copy()
    if not dates or control.date.nunique() != len(dates):
        raise ValueError("Every candidate day needs a same-day control")

    def daily_cash(rows: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
        counts = rows.groupby("date").size().reindex(dates)
        valid = rows.loc[rows.exit_status.eq("filled")
                         & rows.quality_clean_exit].copy()
        valid["stress10"] = _stressed_returns(valid, 10)
        cash = valid.groupby("date").net_return.sum().reindex(dates, fill_value=0) / counts
        stress = valid.groupby("date").stress10.sum().reindex(dates, fill_value=0) / counts
        return cash, stress, valid

    cash, stress, valid = daily_cash(frame)
    random_cash, _, _ = daily_cash(control)
    delta = cash - random_cash
    return {
        "signals": len(frame), "days": len(dates),
        "entry_fills": int(frame.entry_status.eq("filled").sum()),
        "clean_exits": len(valid),
        "ontime_exits": int(valid.exit_delay_sessions.eq(0).sum()),
        "cash_mean": float(cash.mean()),
        "stress10_cash_mean": float(stress.mean()),
        "cash_week_ci": _week_bootstrap(
            cash.reset_index(drop=True), pd.Series(dates), 241
        ),
        "same_day_random_mean": float(random_cash.mean()),
        "edge_mean": float(delta.mean()),
        "edge_week_ci": _week_bootstrap(
            delta.reset_index(drop=True), pd.Series(dates), 242
        ),
    }


def study(snapshot_dir: Path, outcome_dir: Path, issues_dir: Path,
          feature_dir: Path, report_path: Path,
          selections_path: Path) -> dict:
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.read_parquet(str(snapshot_dir / "*.parquet")).create_view("s")
    connection.read_parquet(str(outcome_dir / "*.parquet")).create_view("o")
    connection.read_parquet(str(feature_dir / "*.parquet")).create_view("i")
    connection.register("bad_days", _quality_keys(issues_dir))
    connection.register("bad_symbols", _quality_symbols(issues_dir))
    connection.execute("""
        CREATE TEMP TABLE market AS
        SELECT date, median(return_1450) AS market_return,
               median(return20_prior_adjusted) AS market_prior20
        FROM s
        WHERE date >= '2024-01-01' AND date < '2026-01-01'
          AND isST = 0 AND listing_age_sessions >= 20
          AND NOT reference_gap AND NOT quote_outside_traded_range
          AND amount_1450 >= 30000000
        GROUP BY date
    """)
    connection.execute("CREATE TEMP VIEW snapshots AS SELECT date FROM market")
    last_entry = {
        year: _last_safe_entry(connection, f"{year}-01-01", f"{year + 1}-01-01")
        for year in (2024, 2025)
    }
    connection.execute(f"""
        CREATE TEMP TABLE eligible AS
        SELECT s.date, s.code, s.amount_1450, s.return_1450,
               s.volume_ratio_est, s.return20_prior_adjusted,
               i.return_last30,
               s.return_1450 - m.market_return AS residual_return,
               s.return20_prior_adjusted - m.market_prior20 AS prior20_residual
        FROM s JOIN i USING (date, code) JOIN market m USING (date)
        WHERE ((s.date BETWEEN '2024-01-01' AND '{last_entry[2024]}')
            OR (s.date BETWEEN '2025-01-01' AND '{last_entry[2025]}'))
          AND s.isST = 0 AND s.listing_age_sessions >= 20
          AND NOT s.reference_gap AND NOT s.quote_outside_traded_range
          AND s.amount_1450 BETWEEN 100000000 AND 1000000000
          AND abs(s.price_1450 - i.price_1450) <= .005
          AND s.price_1450 >= 5
          AND s.volume_ratio_est IS NOT NULL
          AND s.return20_prior_adjusted IS NOT NULL
    """)
    selections = []
    for name, (condition, order) in RULES.items():
        picked = connection.execute(f"""
            SELECT date, code, daily_rank FROM (
                SELECT date, code, ROW_NUMBER() OVER (
                    PARTITION BY date ORDER BY {order}, code
                ) AS daily_rank
                FROM eligible WHERE {condition}
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
    if trades.date.str[:4].isin(("2022", "2023", "2026")).any():
        raise ValueError("Research window escaped 2024-2025")
    report = {"hypotheses": list(RULES), "capacity": CAPACITY,
              "last_entry": last_entry,
              "control": "deterministic same-day random eligible stocks",
              "note": "2025 was previously viewed and is not a blind holdout",
              "results": {}}
    for name in RULES:
        report["results"][name] = {}
        for year in ("2024", "2025"):
            annual = trades.loc[trades.candidate.eq(name)
                                & trades.date.str.startswith(year)]
            reference = trades.loc[trades.candidate.eq("random_liquid")
                                   & trades.date.str.startswith(year)]
            for horizon in HORIZONS:
                subset = annual.loc[annual.horizon.eq(horizon)]
                matching = reference.loc[reference.horizon.eq(horizon)]
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
    selections_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    trades.to_parquet(selections_path, index=False, compression="zstd")
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
                        default=Path("data/research/residual_liquidity_report.json"))
    parser.add_argument("--selections", type=Path,
                        default=Path("data/research/residual_liquidity_trades.parquet"))
    args = parser.parse_args()
    report = study(args.snapshots, args.outcomes, args.issues, args.features,
                   args.report, args.selections)
    print({"hypotheses": list(report["results"]), "report": str(args.report)})


if __name__ == "__main__":
    main()
