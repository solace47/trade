"""Freeze same-year T+20 sale-plan pairs before long-horizon outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .exchange_public_events import trading_dates
from .insider_sell_eval import _check_pairs


HORIZON = 20
MAX_EXIT_DELAY = 5


def _subset(pairs: pd.DataFrame, calendar: list[str]) -> pd.DataFrame:
    position = {day: i for i, day in enumerate(calendar)}
    keep = pairs.date.map(
        lambda day: (position[day] + HORIZON + MAX_EXIT_DELAY < len(calendar)
                     and calendar[position[day] + HORIZON + MAX_EXIT_DELAY][:4]
                     == day[:4]))
    result = pairs.loc[keep].copy()
    groups = result.groupby(["date", "pair_code"]).candidate.agg(
        ["size", "nunique"])
    if (result.empty or not groups["size"].eq(2).all()
            or not groups["nunique"].eq(2).all()):
        raise ValueError("Long-horizon sale-plan pairs are incomplete")
    return result


def build(source_dir: Path, calendar_path: Path) -> dict:
    audit = json.loads((source_dir / "input_audit.json").read_text(
        encoding="utf-8"))
    if audit.get("note") != (
            "Major holder sale plan inputs only; no future returns opened"):
        raise ValueError("Long-horizon source was not previously frozen")
    main = pd.read_parquet(source_dir / "pairs.parquet")
    industry = pd.read_parquet(source_dir / "industry_pairs.parquet")
    _check_pairs(main, audit["main"])
    _check_pairs(industry, audit["same_industry"])
    calendar = trading_dates(calendar_path, "2024-01-01", "2025-12-31")
    long_main = _subset(main, calendar)
    long_industry = _subset(industry, calendar)
    unique = pd.concat([long_main, long_industry],
                       ignore_index=True).drop_duplicates(["date", "code"])
    report = {"note": "Post-T+5 mechanism follow-up; inputs only",
              "horizon": HORIZON, "max_exit_delay": MAX_EXIT_DELAY,
              "main": {year: len(long_main.loc[
                  long_main.date.str.startswith(year)]) // 2
                  for year in ("2024", "2025")},
              "same_industry": {year: len(long_industry.loc[
                  long_industry.date.str.startswith(year)]) // 2
                  for year in ("2024", "2025")},
              "unique_stock_days": len(unique)}
    long_main.to_parquet(source_dir / "long_pairs.parquet", index=False,
                         compression="zstd")
    long_industry.to_parquet(source_dir / "long_industry_pairs.parquet",
                             index=False, compression="zstd")
    unique.to_parquet(source_dir / "long_signals.parquet", index=False,
                      compression="zstd")
    (source_dir / "long_input_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(
        "data/research/insider_sell"))
    parser.add_argument("--calendar", type=Path, default=Path(
        "data/baostock/market_2020_2026/metadata/calendar.parquet"))
    args = parser.parse_args()
    print(build(args.source, args.calendar))


if __name__ == "__main__":
    main()
