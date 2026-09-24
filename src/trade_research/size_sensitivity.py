"""Reprice a small, fixed signal list at several order sizes using raw minutes."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb
import pandas as pd

from .hf_outcomes import Assumptions, outcomes_for_symbol
from .market_study import _quality_keys


def _attach_quality(outcomes: pd.DataFrame, issues_dir: Path) -> pd.DataFrame:
    connection = duckdb.connect()
    connection.register("repriced", outcomes)
    connection.register("bad_days", _quality_keys(issues_dir))
    return connection.execute("""
        SELECT r.*, r.exit_date IS NOT NULL AND NOT EXISTS (
            SELECT 1 FROM bad_days AS q
            WHERE q.code = r.code AND q.date >= r.date
              AND q.date <= r.exit_date
        ) AS quality_clean_exit
        FROM repriced AS r
    """).df()


def run(signals_file: Path, minute_root: Path, daily_root: Path,
        calendar_file: Path, output: Path, notionals: tuple[float, ...],
        issues_dir: Path, workers: int = 4) -> pd.DataFrame:
    if not notionals or any(value <= 0 for value in notionals):
        raise ValueError("Order sizes must be positive")
    if workers < 1:
        raise ValueError("Workers must be positive")
    signals = pd.read_parquet(signals_file)
    if signals.empty or signals.duplicated(["date", "code"]).any():
        raise ValueError("Signals must be nonempty and unique by stock-day")
    trading = pd.read_parquet(calendar_file)
    calendar = sorted(trading.loc[
        trading["is_trading_day"].eq("1")
        & trading["calendar_date"].between("2024-01-01", "2025-12-31"),
        "calendar_date",
    ].tolist())
    index = {date: offset for offset, date in enumerate(calendar)}
    if not set(signals["date"]).issubset(index):
        raise ValueError("A signal date is outside the execution calendar")

    def one(item: tuple[str, pd.DataFrame]) -> list[pd.DataFrame]:
        code, stock_signals = item
        exchange, symbol = code.split(".")
        minute_file = minute_root / exchange.upper() / f"{symbol}.parquet"
        daily_file = daily_root / f"{exchange}_{symbol}.parquet"
        relevant_dates = {
            date for signal_date in stock_signals["date"]
            for date in calendar[index[signal_date]:index[signal_date] + 11]
        }
        minute = pd.read_parquet(minute_file,
                                 columns=["timestamp", "volume", "turnover"],
                                 filters=[
                                     ("timestamp", ">=", pd.Timestamp(min(relevant_dates))),
                                     ("timestamp", "<", pd.Timestamp(max(relevant_dates))
                                      + pd.Timedelta(days=1)),
                                 ])
        times = minute["timestamp"]
        minute = minute.loc[
            times.dt.hour.eq(14) & times.dt.minute.between(52, 55)
        ].copy()
        minute["date"] = minute["timestamp"].dt.strftime("%Y-%m-%d")
        minute["label"] = minute["timestamp"].dt.strftime("%H%M")
        minute = minute.loc[minute["date"].isin(relevant_dates)].sort_values(
            "timestamp"
        )
        daily = pd.read_parquet(daily_file)
        frames = []
        for notional in notionals:
            frame = outcomes_for_symbol(
                stock_signals, minute, daily, calendar,
                Assumptions(target_notional=notional),
            )
            frame["target_notional"] = notional
            frames.append(frame)
        return frames

    grouped = list(signals.groupby("code", sort=True))
    frames = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for count, group_frames in enumerate(pool.map(one, grouped), start=1):
            frames.extend(group_frames)
            if count % 100 == 0 or count == len(grouped):
                print(f"Repriced {count}/{len(grouped)} stocks", flush=True)
    result = pd.concat(frames, ignore_index=True)
    expected = len(signals) * len(notionals) * 4
    if len(result) != expected:
        raise ValueError(f"Expected {expected} outcomes, found {len(result)}")
    result = _attach_quality(result, issues_dir)
    result = result.sort_values(["target_notional", "date", "code", "horizon"])
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False, compression="zstd")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--minute-root", type=Path,
                        default=Path("data/hf/pilot/data/stock_1m"))
    parser.add_argument("--daily-root", type=Path,
                        default=Path("data/baostock/market_2020_2026/daily"))
    parser.add_argument("--calendar", type=Path, default=Path(
        "data/baostock/market_2020_2026/metadata/calendar.parquet"
    ))
    parser.add_argument("--issues", type=Path,
                        default=Path("data/research/market_issues_ci"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--notionals", type=float, nargs="+", default=[20_000, 50_000, 100_000])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    result = run(args.signals, args.minute_root, args.daily_root,
                 args.calendar, args.output, tuple(args.notionals),
                 args.issues, args.workers)
    print({"signals": len(result) // (len(args.notionals) * 4),
           "outcomes": len(result), "order_sizes": args.notionals})


if __name__ == "__main__":
    main()
