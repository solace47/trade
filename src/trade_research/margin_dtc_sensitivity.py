"""Post-outcome diagnostics for the frozen margin-DTC experiment.

These checks explain confounding and repeated positions.  They are post hoc,
not an independent validation or a source of publishable trading thresholds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .exchange_public_events import trading_dates
from .margin_dtc_study import CONTROL, TREATED
from .market_study import _quality_keys, _quality_symbols, _week_bootstrap


def coarsened_balance(universe_path: Path, outcome_dir: Path,
                      issues_dir: Path) -> dict:
    """Recompare all quintile stocks within liquidity/momentum terciles."""
    universe = pd.read_parquet(universe_path)
    c = duckdb.connect()
    c.execute("SET threads = 4")
    c.register("u", universe)
    c.read_parquet(str(outcome_dir / "*.parquet")).create_view("o")
    c.register("bad_days", _quality_keys(issues_dir))
    c.register("bad_symbols", _quality_symbols(issues_dir))
    c.execute("""
        CREATE TEMP TABLE coarsened AS
        SELECT date, code, board, size_bucket, quintile,
               NTILE(3) OVER (
                   PARTITION BY date, board, size_bucket
                   ORDER BY avg20_amount, code
               ) AS liquidity_bucket,
               NTILE(3) OVER (
                   PARTITION BY date, board, size_bucket
                   ORDER BY return20_prior_adjusted, code
               ) AS momentum_bucket
        FROM u
    """)
    cells = c.execute("""
        SELECT r.date, r.board, r.size_bucket, r.liquidity_bucket,
               r.momentum_bucket, r.quintile,
               COUNT(*) AS stock_days,
               AVG(CASE WHEN o.exit_status = 'filled'
                         AND NOT EXISTS (
                           SELECT 1 FROM bad_symbols b WHERE b.code = r.code
                         )
                         AND NOT EXISTS (
                           SELECT 1 FROM bad_days q
                           WHERE q.code = r.code AND q.date >= r.date
                             AND q.date <= o.exit_date
                         ) THEN o.net_return ELSE 0 END) AS cash_mean
        FROM coarsened r JOIN o USING (date, code)
        WHERE r.quintile IN (1, 5) AND o.horizon = 5
        GROUP BY r.date, r.board, r.size_bucket, r.liquidity_bucket,
                 r.momentum_bucket, r.quintile
    """).df()
    keys = ["date", "board", "size_bucket", "liquidity_bucket",
            "momentum_bucket"]
    means = cells.pivot(index=keys, columns="quintile", values="cash_mean")
    counts = cells.pivot(index=keys, columns="quintile", values="stock_days")
    paired = means.dropna(subset=[1, 5]).copy()
    paired["edge"] = paired[5] - paired[1]
    paired = paired.reset_index()
    result = {"method": "post hoc: same date, board, float-cap fifth, trailing-amount third, prior-return third",
              "results": {}}
    for year in ("2024", "2025"):
        section = paired.loc[paired.date.str.startswith(year)]
        daily = section.groupby("date", as_index=False)[[1, 5, "edge"]].mean()
        original = universe.loc[
            universe.date.str.startswith(year) & universe.quintile.isin((1, 5))
        ].quintile.value_counts()
        retained = counts.loc[
            counts.index.get_level_values("date").str.startswith(year)
            & counts.index.isin(means.dropna(subset=[1, 5]).index)
        ].sum()
        result["results"][year] = {
            "days": len(daily), "paired_cells": len(section),
            "retained_high_stock_days": int(retained.get(5, 0)),
            "retained_low_stock_days": int(retained.get(1, 0)),
            "original_high_stock_days": int(original.get(5, 0)),
            "original_low_stock_days": int(original.get(1, 0)),
            "high_cash_mean": float(daily[5].mean()),
            "low_cash_mean": float(daily[1].mean()),
            "edge_mean": float(daily.edge.mean()),
            "edge_week_ci": _week_bootstrap(daily.edge, daily.date, 244),
        }
    return result


def concentration(selected_path: Path, trades_path: Path,
                  calendar: Path) -> dict:
    """Quantify repeated stock exposure and a 10-session entry cooldown."""
    selected = pd.read_parquet(selected_path)
    trades = pd.read_parquet(trades_path)
    rows = trades.loc[trades.horizon.eq(5)].copy()
    rows["cash"] = np.where(
        rows.exit_status.eq("filled") & rows.quality_clean_exit,
        rows.net_return, 0.0,
    )
    high = rows.loc[rows.candidate.eq(TREATED),
                    ["date", "pair_code", "code", "cash"]].rename(
                        columns={"cash": "high_cash"})
    low = rows.loc[rows.candidate.eq(CONTROL),
                   ["date", "pair_code", "cash"]].rename(
                       columns={"cash": "low_cash"})
    pairs = high.merge(low, on=["date", "pair_code"], validate="one_to_one")
    if len(pairs) * 2 != len(selected):
        raise ValueError("Repeated-position check lost a selected pair")
    pairs["edge"] = pairs.high_cash - pairs.low_cash
    sessions = trading_dates(calendar, "2024-01-01", "2025-12-31")
    session_index = {day: index for index, day in enumerate(sessions)}
    last_entry: dict[str, int] = {}
    kept = []
    for row in pairs.sort_values(["date", "code"]).itertuples():
        index = session_index[row.date]
        prior = last_entry.get(row.code, -1000)
        valid = index - prior > 10
        kept.append(valid)
        if valid:
            last_entry[row.code] = index
    cooldown = pairs.sort_values(["date", "code"]).loc[kept]
    result = {"method": "post hoc: no new position in the same stock for ten market sessions",
              "results": {}}
    for year in ("2024", "2025"):
        annual = pairs.loc[pairs.date.str.startswith(year)]
        reduced = cooldown.loc[cooldown.date.str.startswith(year)]
        daily = annual.groupby("date", as_index=False).edge.mean()
        reduced_daily = reduced.groupby("date", as_index=False)[
            ["high_cash", "low_cash", "edge"]
        ].mean()
        leave_one = []
        for code in annual.code.unique():
            other = annual.loc[annual.code.ne(code)]
            leave_one.append(float(other.groupby("date").edge.mean().mean()))
        symbols = sorted(annual.code.unique())
        matrix = annual.pivot(index="date", columns="code", values="edge")
        matrix = matrix.reindex(index=sorted(annual.date.unique()), columns=symbols)
        values = np.nan_to_num(matrix.to_numpy())
        counts = matrix.notna().to_numpy(dtype=int)
        random = np.random.default_rng(1024)
        boot = []
        for _ in range(2000):
            weights = random.multinomial(len(symbols),
                                         [1 / len(symbols)] * len(symbols))
            n = counts @ weights
            summed = values @ weights
            boot.append(float((summed[n > 0] / n[n > 0]).mean()))
        result["results"][year] = {
            "pairs": len(annual), "unique_high_symbols": len(symbols),
            "top_ten_symbol_share": float(
                annual.code.value_counts().head(10).sum() / len(annual)
            ),
            "original_edge_mean": float(daily.edge.mean()),
            "symbol_resample_ci": np.quantile(boot, [.025, .975]).tolist(),
            "leave_one_symbol_edge_range": [min(leave_one), max(leave_one)],
            "cooldown_pairs": len(reduced), "cooldown_days": len(reduced_daily),
            "cooldown_high_cash_mean": float(reduced_daily.high_cash.mean()),
            "cooldown_low_cash_mean": float(reduced_daily.low_cash.mean()),
            "cooldown_edge_mean": float(reduced_daily.edge.mean()),
            "cooldown_edge_week_ci": _week_bootstrap(
                reduced_daily.edge, reduced_daily.date, 245
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", type=Path,
                        default=Path("data/research/margin_selected.parquet"))
    parser.add_argument("--universe", type=Path,
                        default=Path("data/research/margin_universe_quintiles.parquet"))
    parser.add_argument("--outcomes", type=Path,
                        default=Path("data/research/market_outcomes_ci"))
    parser.add_argument("--issues", type=Path,
                        default=Path("data/research/market_issues_ci"))
    parser.add_argument("--trades", type=Path,
                        default=Path("data/research/margin_dtc_trades.parquet"))
    parser.add_argument("--calendar", type=Path, default=Path(
        "data/baostock/market_2020_2026/metadata/calendar.parquet"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/research/margin_dtc_sensitivity.json"))
    args = parser.parse_args()
    result = {"coarsened_balance": coarsened_balance(
        args.universe, args.outcomes, args.issues),
        "concentration": concentration(args.selected, args.trades, args.calendar)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
