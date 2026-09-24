"""Frozen input selection for official margin days-to-cover at 14:50.

See docs/margin-days-to-cover-plan.md.  Selection reads no forward returns.
The margin balance for trade day t is assigned to the following session only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .exchange_public_events import trading_dates
from .strategy_scan import _last_safe_entry


CAPACITY = 5
MAX_DISTANCE = 5.0
TREATED = "high_margin_dtc"
CONTROL = "same_day_low_margin_dtc"


def _distance(choices: pd.DataFrame, signal: pd.Series) -> pd.Series:
    return (
        np.abs(np.log(choices.avg20_amount / signal.avg20_amount)) / .5
        + np.abs(np.log(choices.amount_1450 / signal.amount_1450)) / .5
        + np.abs(np.log(choices.float_mv / signal.float_mv)) / .5
        + (choices.return20_prior_adjusted
           - signal.return20_prior_adjusted).abs() / .10
        + (choices.return_1450 - signal.return_1450).abs() / .02
        + (choices.open_gap - signal.open_gap).abs() / .02
        + np.abs(np.log(choices.price_1450 / signal.price_1450)) / .7
    )


def match_controls(treated: pd.DataFrame,
                   low: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Match only on signal-time covariates, same day and board."""
    choices_by_key = {
        key: group.sort_values("code").reset_index(drop=True)
        for key, group in low.groupby(["date", "board", "size_bucket"], sort=True)
    }
    high_rows = []
    control_rows = []
    unmatched = 0
    for _, signal in treated.sort_values(
            ["date", "margin_dtc", "code"], ascending=[True, False, True]
    ).iterrows():
        key = signal.date, signal.board, signal.size_bucket
        choices = choices_by_key.get(key, pd.DataFrame())
        if choices.empty:
            unmatched += 1
            continue
        eligible = choices.loc[
            (choices.avg20_amount / signal.avg20_amount).between(.5, 2)
            & (choices.float_mv / signal.float_mv).between(.5, 2)
            & (choices.return20_prior_adjusted
               - signal.return20_prior_adjusted).abs().le(.15)
            & (choices.return_1450 - signal.return_1450).abs().le(.05)
        ]
        if eligible.empty:
            unmatched += 1
            continue
        distances = _distance(eligible, signal)
        nearest_index = int(np.argmin(distances.to_numpy()))
        best = float(distances.iloc[nearest_index])
        if not np.isfinite(best) or best > MAX_DISTANCE:
            unmatched += 1
            continue
        reference = eligible.iloc[nearest_index].copy()
        reference["pair_code"] = signal.code
        reference["match_distance"] = best
        reference["candidate"] = CONTROL
        control_rows.append(reference)
        item = signal.copy()
        item["pair_code"] = signal.code
        item["match_distance"] = best
        item["candidate"] = TREATED
        high_rows.append(item)
        choices_by_key[key] = choices.drop(eligible.index[nearest_index])
    if not high_rows:
        raise ValueError("No high-margin-DTC signal has a comparable low-DTC control")
    return pd.DataFrame(high_rows), pd.DataFrame(control_rows), unmatched


def select(snapshot_dir: Path, daily_dir: Path, calendar: Path,
           margin_file: Path, selected_path: Path,
           universe_path: Path) -> tuple[pd.DataFrame, dict]:
    days = trading_dates(calendar, "2024-01-01", "2025-12-31")
    next_days = pd.DataFrame({"trade_date": days[:-1], "date": days[1:]})
    if not next_days.date.gt(next_days.trade_date).all():
        raise ValueError("Margin data joined to its own or an earlier session")
    c = duckdb.connect()
    c.execute("SET threads = 4")
    c.read_parquet(str(snapshot_dir / "*.parquet")).create_view("snapshots")
    last_entry = {year: _last_safe_entry(c, f"{year}-01-01", f"{year+1}-01-01")
                  for year in (2024, 2025)}
    c.register("next_days", next_days)
    c.execute(f"CREATE TEMP VIEW margin AS SELECT * FROM read_parquet('{margin_file}')")
    source = c.execute("""
        SELECT COUNT(*) AS rows, COUNT(DISTINCT trade_date) AS dates,
               COUNT(DISTINCT trade_date || code) AS keys,
               MIN(trade_date) AS first_date, MAX(trade_date) AS last_date
        FROM margin
    """).fetchone()
    if (source[1] != len(days) or source[0] != source[2]
            or source[3] != days[0] or source[4] != days[-1]):
        raise ValueError("Incomplete or duplicate margin source before selection")
    c.execute(f"""
        CREATE TEMP TABLE t_liquidity AS
        SELECT date, code, amount, turn, avg20_amount, traded_count
        FROM (
            SELECT date, code, amount, turn,
                   AVG(amount) OVER (
                       PARTITION BY code ORDER BY date
                       ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                   ) AS avg20_amount,
                   COUNT(*) OVER (
                       PARTITION BY code ORDER BY date
                       ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                   ) AS traded_count
            FROM read_parquet('{daily_dir}/*.parquet')
            WHERE date BETWEEN '2023-10-01' AND '2025-12-31'
              AND tradestatus = 1 AND amount > 0 AND turn > 0
        ) WHERE date BETWEEN '2024-01-01' AND '2025-12-31'
    """)
    c.execute(f"""
        CREATE TEMP TABLE eligible AS
        SELECT s.date, m.trade_date, s.code,
               CASE WHEN s.code LIKE 'sh.60%' THEN 'sh_main'
                    WHEN s.code LIKE 'sh.68%' THEN 'sh_star'
                    WHEN s.code LIKE 'sz.00%' THEN 'sz_main'
                    WHEN s.code LIKE 'sz.30%' THEN 'sz_gem'
                    ELSE 'other' END AS board,
               m.balance_yuan / d.avg20_amount AS margin_dtc,
               m.balance_yuan, m.buy_yuan, d.avg20_amount,
               d.amount / (d.turn / 100) AS float_mv,
               s.price_1450, s.amount_1450, s.return_1450,
               s.return20_prior_adjusted,
               s.open_1450 / s.preclose - 1 AS open_gap
        FROM margin m
        JOIN t_liquidity d ON d.date = m.trade_date AND d.code = m.code
        JOIN next_days n ON n.trade_date = m.trade_date
        JOIN snapshots s ON s.date = n.date AND s.code = m.code
        WHERE ((s.date BETWEEN '2024-01-01' AND '{last_entry[2024]}')
            OR (s.date BETWEEN '2025-01-01' AND '{last_entry[2025]}'))
          AND (s.code LIKE 'sh.60%' OR s.code LIKE 'sh.68%'
            OR s.code LIKE 'sz.00%' OR s.code LIKE 'sz.30%')
          AND d.traded_count = 20 AND d.avg20_amount >= 30000000
          AND s.tradestatus = 1 AND s.isST = 0
          AND s.listing_age_sessions >= 20
          AND NOT s.reference_gap AND NOT s.quote_outside_traded_range
          AND s.price_1450 >= 5 AND s.amount_1450 >= 30000000
          AND s.preclose > 0 AND s.open_1450 > 0
          AND s.open_1450 BETWEEN s.low_1450 - .005 AND s.high_1450 + .005
          AND s.return20_prior_adjusted IS NOT NULL
    """)
    c.execute("""
        CREATE TEMP TABLE size_groups AS
        SELECT *, NTILE(5) OVER (
            PARTITION BY date, board ORDER BY float_mv, code
        ) AS size_bucket
        FROM eligible
    """)
    c.execute("""
        CREATE TEMP TABLE quintiles AS
        SELECT *, NTILE(5) OVER (
            PARTITION BY date, board, size_bucket ORDER BY margin_dtc, code
        ) AS quintile
        FROM size_groups
    """)
    universe = c.execute("""
        SELECT date, trade_date, code, board, size_bucket,
               margin_dtc, float_mv, quintile
        FROM quintiles ORDER BY date, code
    """).df()
    if universe.empty or universe.duplicated(["date", "code"]).any():
        raise ValueError("Empty or duplicated eligible margin stock-days")
    treated = c.execute(f"""
        SELECT * FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY date ORDER BY margin_dtc DESC, code
            ) AS daily_rank
            FROM quintiles WHERE quintile = 5
        ) WHERE daily_rank <= {CAPACITY}
        ORDER BY date, daily_rank
    """).df()
    low = c.execute("SELECT * FROM quintiles WHERE quintile = 1").df()
    picked, controls, unmatched = match_controls(treated, low)
    selected = pd.concat([picked, controls], ignore_index=True)
    pairs = selected.groupby(["date", "pair_code"]).candidate.nunique()
    if not pairs.eq(2).all() or len(selected) != 2 * len(pairs):
        raise ValueError("Incomplete matched margin pairs")
    if not selected.date.gt(selected.trade_date).all():
        raise ValueError("Margin selection has a future-information join")
    if selected.duplicated(["date", "code", "candidate"]).any():
        raise ValueError("Stock-day selected twice within a margin basket")
    report = {
        "source_rows": source[0], "source_dates": source[1],
        "last_entry": last_entry, "eligible_stock_days": len(universe),
        "eligible_by_board": universe.board.value_counts().to_dict(),
        "capacity_selected": len(treated), "matched_pairs": len(picked),
        "unmatched": unmatched, "matched_days": picked.date.nunique(),
        "median_distance": float(controls.match_distance.median()),
        "p90_distance": float(controls.match_distance.quantile(.9)),
        "note": "Selection-only: no forward outcomes were loaded",
    }
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    universe_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(selected_path, index=False, compression="zstd")
    universe.to_parquet(universe_path, index=False, compression="zstd")
    return selected, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path,
                        default=Path("data/research/market_snapshots_ci"))
    parser.add_argument("--daily", type=Path,
                        default=Path("data/baostock/market_2020_2026/daily"))
    parser.add_argument("--calendar", type=Path, default=Path(
        "data/baostock/market_2020_2026/metadata/calendar.parquet"))
    parser.add_argument("--margin", type=Path,
                        default=Path("data/research/margin/margin_2024_2025.parquet"))
    parser.add_argument("--selected", type=Path,
                        default=Path("data/research/margin_selected.parquet"))
    parser.add_argument("--universe", type=Path,
                        default=Path("data/research/margin_universe_quintiles.parquet"))
    args = parser.parse_args()
    _, report = select(args.snapshots, args.daily, args.calendar,
                       args.margin, args.selected, args.universe)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
