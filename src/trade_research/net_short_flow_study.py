"""Input-only selection of five-session net short-position changes.

The frozen plan remains in Git history. The exchange balance at t is available
for the next session's 14:50 signal, never for the same session.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .exchange_public_events import trading_dates
from .short_interest_study import _quintiles, select_pairs


WINDOWS = (("2024_postrunoff", "2024-10-15", "2024-12-17"),
           ("2025_H1", "2025-01-02", "2025-06-30"),
           ("2025_H2", "2025-07-01", "2025-12-17"))
TREATED = "high_positive_net_short_flow"
CONTROL = "same_day_negative_net_short_flow"


def _window(day: str) -> str:
    for name, start, end in WINDOWS:
        if start <= day <= end:
            return name
    raise ValueError(f"Net short-flow signal outside frozen windows: {day}")


def _prepare_inputs(universe_path: Path, margin_path: Path,
                    daily_dir: Path, calendar_path: Path) -> pd.DataFrame:
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.execute(f"CREATE TEMP VIEW margin AS SELECT * FROM read_parquet('{margin_path}')")
    source = connection.execute("""
        SELECT COUNT(*), COUNT(DISTINCT trade_date),
               COUNT(DISTINCT trade_date || code), MIN(trade_date), MAX(trade_date)
        FROM margin
    """).fetchone()
    if (source[1] != 485 or source[0] != source[2]
            or source[3] != "2024-01-02" or source[4] != "2025-12-31"):
        raise ValueError("Incomplete official short-position source")
    connection.execute(f"""
        CREATE TEMP TABLE calendar AS
        SELECT calendar_date AS trade_date,
               ROW_NUMBER() OVER (ORDER BY calendar_date) AS session_index
        FROM read_parquet('{calendar_path}')
        WHERE is_trading_day = '1'
          AND calendar_date BETWEEN '2024-01-01' AND '2025-12-31'
    """)
    connection.execute("""
        CREATE TEMP TABLE balances AS
        SELECT m.trade_date, m.code, m.short_balance_shares,
               m.short_sell_shares, c.session_index,
               LAG(m.short_balance_shares, 5) OVER (
                   PARTITION BY m.code ORDER BY c.session_index
               ) AS prev5_shares,
               LAG(c.session_index, 5) OVER (
                   PARTITION BY m.code ORDER BY c.session_index
               ) AS prev5_index,
               SUM(m.short_sell_shares) OVER (
                   PARTITION BY m.code ORDER BY c.session_index
                   ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
               ) AS sold5_shares
        FROM margin m JOIN calendar c USING (trade_date)
    """)
    connection.execute(f"""
        CREATE TEMP TABLE weekly_volume AS
        SELECT date, code, close,
               SUM(volume) OVER (
                   PARTITION BY code ORDER BY CAST(date AS DATE)
                   RANGE BETWEEN INTERVAL 364 DAYS PRECEDING AND CURRENT ROW
               ) AS trailing_volume,
               COUNT(DISTINCT DATE_TRUNC('week', CAST(date AS DATE))) OVER (
                   PARTITION BY code ORDER BY CAST(date AS DATE)
                   RANGE BETWEEN INTERVAL 364 DAYS PRECEDING AND CURRENT ROW
               ) AS observed_weeks
        FROM read_parquet('{daily_dir}/*.parquet')
        WHERE date BETWEEN '2023-01-01' AND '2025-12-31'
          AND tradestatus = 1 AND volume > 0
    """)
    connection.execute(f"""
        CREATE TEMP VIEW joined AS
        SELECT u.* EXCLUDE (quintile), b.short_balance_shares,
               b.prev5_shares, b.sold5_shares,
               b.short_balance_shares - b.prev5_shares AS net_short_shares,
               b.prev5_shares * v.close / u.float_mv AS prior_short_interest,
               v.trailing_volume, v.observed_weeks,
               (b.short_balance_shares - b.prev5_shares)
                 / (v.trailing_volume / v.observed_weeks) AS net_short_flow
        FROM read_parquet('{universe_path}') u
        JOIN balances b ON b.trade_date = u.trade_date AND b.code = u.code
        JOIN weekly_volume v ON v.date = u.trade_date AND v.code = u.code
        WHERE ((u.date BETWEEN '2024-10-15' AND '2024-12-17')
            OR (u.date BETWEEN '2025-01-02' AND '2025-12-17'))
          AND u.date > u.trade_date
          AND u.avg20_amount >= 100000000
          AND u.amount_1450 >= 100000000
          AND b.session_index - b.prev5_index = 5
          AND b.short_balance_shares > 0 AND b.prev5_shares > 0
          AND b.sold5_shares >= b.short_balance_shares - b.prev5_shares
          AND v.observed_weeks >= 40 AND v.trailing_volume > 0
          AND v.close > 0 AND u.float_mv > 0
    """)
    frame = connection.execute("SELECT * FROM joined ORDER BY date, code").df()
    if (frame.empty or frame.duplicated(["date", "code"]).any()
            or not frame.date.gt(frame.trade_date).all()
            or not np.isfinite(frame.net_short_flow).all()):
        raise ValueError("Malformed five-day short-position flow inputs")
    frame["window"] = frame.date.map(_window)
    return frame


def select(universe_path: Path, margin_path: Path, daily_dir: Path,
           calendar_path: Path, selected_path: Path,
           quintiles_path: Path, report_path: Path) -> dict:
    inputs = _prepare_inputs(universe_path, margin_path, daily_dir,
                             calendar_path)
    quintiles = _quintiles(inputs, "net_short_flow")
    days = trading_dates(calendar_path, "2024-01-01", "2025-12-31")
    selected, summary = select_pairs(
        quintiles, {day: index for index, day in enumerate(days)},
        score_field="net_short_flow", treated=TREATED, control=CONTROL,
        signed_flows=True, prior_level_caliper=(.5, 2))
    high = selected.loc[selected.candidate.eq(TREATED)]
    low = selected.loc[selected.candidate.eq(CONTROL)]
    if not high.net_short_flow.gt(0).all() or not low.net_short_flow.lt(0).all():
        raise ValueError("Net-short-flow pairing mixed signs")
    report = {"note": "Frozen input-only flow selection; no returns loaded",
              "eligible_stock_days": len(inputs),
              "quintile_stock_days": len(quintiles),
              "eligible_by_window": inputs.window.value_counts().to_dict(),
              "matched_by_window": high.window.value_counts().to_dict(),
              "matched_by_board": high.board.value_counts().to_dict(),
              "top10_share_by_window": {
                  name: float(group.code.value_counts().head(10).sum() / len(group))
                  for name, group in high.groupby("window")},
              "median_flow_by_window": {
                  name: {"high": float(group.net_short_flow.median()),
                         "low": float(low.loc[low.window.eq(name),
                                              "net_short_flow"].median())}
                  for name, group in high.groupby("window")},
              **summary}
    for path in (selected_path, quintiles_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(selected_path, index=False, compression="zstd")
    quintiles.to_parquet(quintiles_path, index=False, compression="zstd")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-universe", type=Path,
                        default=Path("data/research/margin_universe_quintiles.parquet"))
    parser.add_argument("--margin", type=Path,
                        default=Path("data/research/margin/margin_2024_2025.parquet"))
    parser.add_argument("--daily", type=Path,
                        default=Path("data/baostock/market_2020_2026/daily"))
    parser.add_argument("--calendar", type=Path, default=Path(
        "data/baostock/market_2020_2026/metadata/calendar.parquet"))
    parser.add_argument("--selected", type=Path,
                        default=Path("data/research/net_short_flow_selected.parquet"))
    parser.add_argument("--quintiles", type=Path,
                        default=Path("data/research/net_short_flow_quintiles.parquet"))
    parser.add_argument("--report", type=Path,
                        default=Path("data/research/net_short_flow_selection.json"))
    args = parser.parse_args()
    result = select(args.base_universe, args.margin, args.daily,
                    args.calendar, args.selected, args.quintiles, args.report)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
