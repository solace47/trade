"""Input-only selection for five-session margin-purchase participation.

The hypothesis and screens were frozen before reading their outcomes; the
original plan remains in Git history. Day t purchases enter t+1 only.
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


WINDOWS = (("2024_H1", "2024-01-10", "2024-06-30"),
           ("2024_H2", "2024-07-01", "2024-12-17"),
           ("2025_H1", "2025-01-02", "2025-06-30"),
           ("2025_H2", "2025-07-01", "2025-12-17"))
TREATED = "high_margin_buy_share"
CONTROL = "same_day_low_margin_buy_share"


def _window(day: str) -> str:
    for name, start, end in WINDOWS:
        if start <= day <= end:
            return name
    raise ValueError(f"Margin-buy signal outside frozen windows: {day}")


def _prepare_inputs(universe_path: Path, margin_path: Path,
                    daily_dir: Path, snapshot_dir: Path,
                    calendar_path: Path) -> pd.DataFrame:
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
        raise ValueError("Incomplete official margin-purchase source")
    connection.execute(f"""
        CREATE TEMP TABLE calendar AS
        SELECT calendar_date AS trade_date,
               ROW_NUMBER() OVER(ORDER BY calendar_date) AS session_index
        FROM read_parquet('{calendar_path}')
        WHERE is_trading_day = '1'
          AND calendar_date BETWEEN '2024-01-01' AND '2025-12-31'
    """)
    connection.execute(f"""
        CREATE TEMP TABLE activity AS
        SELECT m.trade_date, m.code, m.buy_yuan, m.balance_yuan,
               c.session_index,
               SUM(m.buy_yuan) OVER (
                   PARTITION BY m.code ORDER BY c.session_index
                   ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
               ) AS buy5_yuan,
               SUM(d.amount) OVER (
                   PARTITION BY m.code ORDER BY c.session_index
                   ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
               ) AS turnover5_yuan,
               LAG(m.balance_yuan, 5) OVER (
                   PARTITION BY m.code ORDER BY c.session_index
               ) AS prev5_balance_yuan,
               LAG(c.session_index, 5) OVER (
                   PARTITION BY m.code ORDER BY c.session_index
               ) AS prev5_index
        FROM margin m
        JOIN read_parquet('{daily_dir}/*.parquet') d
          ON d.date = m.trade_date AND d.code = m.code
        JOIN calendar c USING(trade_date)
        WHERE d.tradestatus = 1 AND d.amount > 0
          AND m.buy_yuan BETWEEN 0 AND d.amount
    """)
    connection.execute(f"""
        CREATE TEMP VIEW joined AS
        SELECT u.* EXCLUDE (quintile),
               s.return5_prior_adjusted,
               a.buy5_yuan, a.turnover5_yuan, a.prev5_balance_yuan,
               a.prev5_balance_yuan / u.float_mv AS prior_financing_interest,
               a.buy5_yuan / a.turnover5_yuan AS buy_activity_score
        FROM read_parquet('{universe_path}') u
        JOIN activity a ON a.trade_date = u.trade_date AND a.code = u.code
        JOIN read_parquet('{snapshot_dir}/*.parquet') s
          ON s.date = u.date AND s.code = u.code
        WHERE ((u.date BETWEEN '2024-01-10' AND '2024-12-17')
            OR (u.date BETWEEN '2025-01-02' AND '2025-12-17'))
          AND u.date > u.trade_date
          AND u.avg20_amount >= 100000000
          AND u.amount_1450 >= 100000000
          AND u.float_mv > 0
          AND s.return5_prior_adjusted > 0
          AND u.return20_prior_adjusted > 0
          AND a.session_index - a.prev5_index = 5
          AND a.prev5_balance_yuan > 0
          AND a.turnover5_yuan > 0
    """)
    frame = connection.execute("SELECT * FROM joined ORDER BY date, code").df()
    if (frame.empty or frame.duplicated(["date", "code"]).any()
            or not frame.date.gt(frame.trade_date).all()
            or not np.isfinite(frame.buy_activity_score).all()
            or not frame.buy_activity_score.between(0, 1).all()):
        raise ValueError("Malformed margin-purchase inputs or time alignment")
    frame["window"] = frame.date.map(_window)
    return frame


def select(universe_path: Path, margin_path: Path, daily_dir: Path,
           snapshot_dir: Path, calendar_path: Path,
           selected_path: Path, quintiles_path: Path,
           report_path: Path) -> dict:
    inputs = _prepare_inputs(universe_path, margin_path, daily_dir,
                             snapshot_dir, calendar_path)
    quintiles = _quintiles(inputs, "buy_activity_score")
    days = trading_dates(calendar_path, "2024-01-01", "2025-12-31")
    selected, summary = select_pairs(
        quintiles, {day: index for index, day in enumerate(days)},
        score_field="buy_activity_score", treated=TREATED, control=CONTROL,
        prior_level_caliper=(.5, 2),
        prior_level_field="prior_financing_interest",
        prior5_return_caliper=.05)
    high = selected.loc[selected.candidate.eq(TREATED)]
    low = selected.loc[selected.candidate.eq(CONTROL)]
    paired = high[["date", "pair_code", "buy_activity_score"]].merge(
        low[["date", "pair_code", "buy_activity_score"]],
        on=["date", "pair_code"], suffixes=("_high", "_low"),
        validate="one_to_one")
    if (len(paired) != len(high)
            or not paired.buy_activity_score_high.gt(
                paired.buy_activity_score_low).all()):
        raise ValueError("Margin-purchase group ordering failed")
    report = {"note": "Frozen input-only selection; no forward returns loaded",
              "eligible_stock_days": len(inputs),
              "quintile_stock_days": len(quintiles),
              "eligible_by_window": inputs.window.value_counts().to_dict(),
              "matched_by_window": high.window.value_counts().to_dict(),
              "matched_by_board": high.board.value_counts().to_dict(),
              "top10_share_by_window": {
                  name: float(group.code.value_counts().head(10).sum() / len(group))
                  for name, group in high.groupby("window")},
              "median_buy_share_by_window": {
                  name: {"high": float(group.buy_activity_score.median()),
                         "low": float(low.loc[low.window.eq(name),
                                          "buy_activity_score"].median())}
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
    parser.add_argument("--snapshots", type=Path,
                        default=Path("data/research/market_snapshots_ci"))
    parser.add_argument("--calendar", type=Path, default=Path(
        "data/baostock/market_2020_2026/metadata/calendar.parquet"))
    parser.add_argument("--selected", type=Path,
                        default=Path("data/research/margin_buy_activity_selected.parquet"))
    parser.add_argument("--quintiles", type=Path,
                        default=Path("data/research/margin_buy_activity_quintiles.parquet"))
    parser.add_argument("--report", type=Path,
                        default=Path("data/research/margin_buy_activity_selection.json"))
    args = parser.parse_args()
    result = select(args.base_universe, args.margin, args.daily, args.snapshots,
                    args.calendar, args.selected, args.quintiles, args.report)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
