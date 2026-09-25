"""Audit 09:34 inputs before testing a conditional T+1 exit."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("data/research")
SOURCE = ROOT / "late_to_open_reversal"
OUTPUT = ROOT / "next_morning_exit"
MINUTES = Path("data/hf/pilot/data/stock_1m")
CALENDAR = Path("data/baostock/market_2020_2026/metadata/calendar.parquet")


def _half(dates: pd.Series) -> pd.Series:
    return dates.str[:4] + "H" + np.where(
        dates.str[5:7].astype(int).le(6), "1", "2")


def _next_trading_dates(signals: pd.DataFrame, calendar_file: Path) -> dict[str, str]:
    calendar = pd.read_parquet(calendar_file)
    trading = sorted(calendar.loc[
        calendar.is_trading_day.astype(str).eq("1")
        & calendar.calendar_date.between("2024-01-01", "2025-12-31"),
        "calendar_date"].tolist())
    successors = dict(zip(trading[:-1], trading[1:]))
    missing = set(signals.date) - successors.keys()
    if missing:
        raise ValueError("A frozen signal lacks a T+1 trading date")
    return successors


def build_inputs(output_dir: Path = OUTPUT, minute_root: Path = MINUTES,
                 calendar_file: Path = CALENDAR, workers: int = 4) -> dict:
    if workers < 1:
        raise ValueError("Workers must be positive")
    signals = pd.read_parquet(SOURCE / "selections.parquet", columns=[
        "date", "code", "candidate", "pair_id", "price_1450",
    ])
    if (signals.empty or signals.duplicated(["date", "code"]).any()
            or not signals.date.str[:4].isin(("2024", "2025")).all()):
        raise ValueError("Invalid frozen 14:50 pair list")
    successors = _next_trading_dates(signals, calendar_file)
    signals["target_date"] = signals.date.map(successors)

    def one(item: tuple[str, pd.DataFrame]) -> pd.DataFrame:
        code, stock_signals = item
        exchange, symbol = code.split(".")
        minute_file = minute_root / exchange.upper() / f"{symbol}.parquet"
        dates = set(stock_signals.target_date)
        minute = pd.read_parquet(minute_file,
                                 columns=["timestamp", "close", "volume"],
                                 filters=[
                                     ("timestamp", ">=", pd.Timestamp(min(dates))),
                                     ("timestamp", "<", pd.Timestamp(max(dates))
                                      + pd.Timedelta(days=1)),
                                 ])
        minute = minute.loc[minute.timestamp.dt.strftime("%H%M").eq("0934")
                            ].copy()
        minute["target_date"] = minute.timestamp.dt.strftime("%Y-%m-%d")
        minute = minute.loc[minute.target_date.isin(dates)]
        grouped = minute.groupby("target_date", sort=False).agg(
            bar_count=("close", "size"), price_0934=("close", "first"),
            volume_0934=("volume", "first"),
        ).reset_index()
        grouped["code"] = code
        return grouped

    grouped = list(signals.groupby("code", sort=True))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        frames = list(pool.map(one, grouped))
    observations = pd.concat(frames, ignore_index=True)
    decisions = signals.merge(observations, on=["code", "target_date"],
                              how="left", validate="one_to_one")
    decisions["observed_0934"] = (
        decisions.bar_count.eq(1) & decisions.price_0934.gt(0)
        & decisions.volume_0934.gt(0)
    )
    decisions["exit_choice"] = np.where(
        decisions.observed_0934
        & decisions.price_0934.lt(decisions.price_1450),
        "close", "morning",
    )
    decisions["half"] = _half(decisions.date)
    branch = decisions.groupby(["half", "candidate", "exit_choice"]).agg(
        signals=("code", "size"), days=("date", "nunique"),
    ).reset_index().to_dict("records")
    coverage = decisions.groupby("candidate").observed_0934.mean().to_dict()
    counts = {(x["half"], x["candidate"], x["exit_choice"]): x
              for x in branch}
    gate = (
        all(value >= .95 for value in coverage.values())
        and all((half, candidate, choice) in counts
                and counts[(half, candidate, choice)]["signals"] >= 30
                and counts[(half, candidate, choice)]["days"] >= 15
                for half in ("2024H1", "2024H2", "2025H1", "2025H2")
                for candidate in ("late_decline", "late_rally_control")
                for choice in ("morning", "close"))
    )
    audit = {"pairs": len(signals) // 2,
             "observed_0934_rate": coverage,
             "missing_or_inactive_0934": int((~decisions.observed_0934).sum()),
             "branch_by_half": branch, "outcome_gate_passed": gate}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").unlink(missing_ok=True)
    decisions.to_parquet(output_dir / "decisions.parquet", index=False,
                         compression="zstd")
    (output_dir / "input_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--minute-root", type=Path, default=MINUTES)
    parser.add_argument("--calendar", type=Path, default=CALENDAR)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(build_inputs(args.output, args.minute_root, args.calendar, args.workers))


if __name__ == "__main__":
    main()
