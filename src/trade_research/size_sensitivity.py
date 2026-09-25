"""Reprice a small, fixed signal list at several order sizes using raw minutes."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb
import pandas as pd

from .hf_outcomes import (
    Assumptions, ENTRY_WINDOWS, EXIT_WINDOWS, HORIZONS,
    RESEARCH_HORIZONS, outcomes_for_symbol,
)
from .market_study import _quality_keys, _quality_symbols


def _attach_quality(outcomes: pd.DataFrame, issues_dir: Path) -> pd.DataFrame:
    connection = duckdb.connect()
    connection.register("repriced", outcomes)
    connection.register("bad_days", _quality_keys(issues_dir))
    connection.register("bad_symbols", _quality_symbols(issues_dir))
    return connection.execute("""
        SELECT r.*, r.exit_status = 'filled'
        AND NOT EXISTS (SELECT 1 FROM bad_symbols b WHERE b.code = r.code)
        AND NOT EXISTS (
            SELECT 1 FROM bad_days AS q
            WHERE q.code = r.code AND q.date >= r.date
              AND q.date <= r.exit_date
        ) AS quality_clean_exit
        FROM repriced AS r
    """).df()


def run(signals_file: Path, minute_root: Path, daily_root: Path,
        calendar_file: Path, output: Path, notionals: tuple[float, ...],
        issues_dir: Path, exit_windows: tuple[str, ...] = ("close",),
        horizons: tuple[int, ...] = HORIZONS, workers: int = 4,
        entry_windows: tuple[str, ...] = ("baseline",),
        sizing_price_column: str | None = None) -> pd.DataFrame:
    if not notionals or any(value <= 0 for value in notionals):
        raise ValueError("Order sizes must be positive")
    if workers < 1:
        raise ValueError("Workers must be positive")
    if not exit_windows or any(window not in EXIT_WINDOWS for window in exit_windows):
        raise ValueError("Unknown or missing exit windows")
    if not entry_windows or any(window not in ENTRY_WINDOWS for window in entry_windows):
        raise ValueError("Unknown or missing entry windows")
    if not horizons or any(horizon not in RESEARCH_HORIZONS for horizon in horizons):
        raise ValueError("Unsupported holding periods")
    signals = pd.read_parquet(signals_file)
    if signals.empty or signals.duplicated(["date", "code"]).any():
        raise ValueError("Signals must be nonempty and unique by stock-day")
    if sizing_price_column is not None and sizing_price_column not in signals.columns:
        raise ValueError("The decision-time sizing price is missing")
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
            for date in calendar[
                index[signal_date]:index[signal_date] + max(horizons)
                + Assumptions().maximum_exit_delay_sessions + 1
            ]
        }
        minute = pd.read_parquet(minute_file,
                                 columns=["timestamp", "volume", "turnover"],
                                 filters=[
                                     ("timestamp", ">=", pd.Timestamp(min(relevant_dates))),
                                     ("timestamp", "<", pd.Timestamp(max(relevant_dates))
                                      + pd.Timedelta(days=1)),
                                 ])
        needed_labels = set().union(
            *(ENTRY_WINDOWS[window] for window in entry_windows),
            *(EXIT_WINDOWS[window] for window in exit_windows),
        )
        labels = minute["timestamp"].dt.strftime("%H%M")
        minute = minute.loc[labels.isin(needed_labels)].copy()
        minute["date"] = minute["timestamp"].dt.strftime("%Y-%m-%d")
        minute["label"] = minute["timestamp"].dt.strftime("%H%M")
        minute = minute.loc[minute["date"].isin(relevant_dates)].sort_values(
            "timestamp"
        )
        daily = pd.read_parquet(daily_file)
        first_needed, last_needed = min(relevant_dates), max(relevant_dates)
        prior_traded = daily.loc[
            daily["date"].lt(first_needed) & daily["tradestatus"].eq(1), "date"
        ].max()
        daily_start = prior_traded if pd.notna(prior_traded) else first_needed
        daily = daily.loc[daily["date"].between(daily_start, last_needed)].copy()
        frames = []
        for notional in notionals:
            for entry_window in entry_windows:
                for exit_window in exit_windows:
                    frame = outcomes_for_symbol(
                        stock_signals, minute, daily, calendar,
                        Assumptions(target_notional=notional),
                        exit_labels=EXIT_WINDOWS[exit_window],
                        horizons=horizons,
                        entry_labels=ENTRY_WINDOWS[entry_window],
                        sizing_price_column=sizing_price_column,
                    )
                    frame["target_notional"] = notional
                    frame["entry_window"] = entry_window
                    frame["exit_window"] = exit_window
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
    expected = (len(signals) * len(notionals) * len(entry_windows)
                * len(exit_windows) * len(horizons))
    if len(result) != expected:
        raise ValueError(f"Expected {expected} outcomes, found {len(result)}")
    result = _attach_quality(result, issues_dir)
    result = result.sort_values(
        ["target_notional", "entry_window", "exit_window", "date", "code",
         "horizon"]
    )
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
    parser.add_argument("--exit-windows", choices=EXIT_WINDOWS, nargs="+",
                        default=["close"])
    parser.add_argument("--entry-windows", choices=ENTRY_WINDOWS, nargs="+",
                        default=["baseline"])
    parser.add_argument("--horizons", type=int, choices=RESEARCH_HORIZONS, nargs="+",
                        default=list(HORIZONS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sizing-price-column", type=str)
    args = parser.parse_args()
    result = run(args.signals, args.minute_root, args.daily_root,
                 args.calendar, args.output, tuple(args.notionals),
                 args.issues, tuple(args.exit_windows), tuple(args.horizons),
                 args.workers, tuple(args.entry_windows),
                 args.sizing_price_column)
    print({"signals": len(result) // (len(args.notionals) * len(args.entry_windows)
                                  * len(args.exit_windows) * len(args.horizons)),
           "outcomes": len(result), "order_sizes": args.notionals,
           "entry_windows": args.entry_windows,
           "exit_windows": args.exit_windows, "horizons": args.horizons,
           "sizing_price_column": args.sizing_price_column})


if __name__ == "__main__":
    main()
