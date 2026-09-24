"""Freeze large, already-disclosed IPO share unlocks at 14:50."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .buyback_inputs import _match, _universe
from .exchange_public_events import trading_dates
from .ipo_unlock_source import strict_title


EVENT = "large_ipo_unlock"
MIN_UNLOCK_PRIOR_FLOAT = .20
MAX_CURRENT_RETURN_GAP = .01


def _confirmed_audits(base: Path) -> tuple[pd.DataFrame, dict]:
    frames, counts = [], {}
    for year in (2024, 2025):
        index = pd.read_parquet(base / f"search_{year}.parquet")
        audit = pd.read_json(base / f"pdf_audit_{year}.jsonl", lines=True)
        selected = index.loc[index.title.map(strict_title)]
        if (index.empty or index.pdf_url.duplicated().any()
                or audit.pdf_url.duplicated().any()
                or set(selected.pdf_url) != set(audit.pdf_url)
                or not audit.notice_date.str.startswith(str(year)).all()):
            raise ValueError("Incomplete original IPO unlock PDF audit")
        ok = audit.loc[audit.status.eq("ok")].copy()
        if (ok.unlock_date.isna().any() or ok.unlock_shares.isna().any()
                or not ok.unlock_date.gt(ok.notice_date).all()):
            raise ValueError("Malformed original IPO unlock date or shares")
        counts[str(year)] = {"search_pdf": len(index),
                             "title_candidates": len(selected),
                             "original_confirmed": len(ok),
                             "pdf_status": audit.status.value_counts().to_dict()}
        frames.append(ok)
    return pd.concat(frames, ignore_index=True), counts


def _announcements(base: Path, calendar: list[str]) -> tuple[pd.DataFrame, dict]:
    all_ok, counts = _confirmed_audits(base)
    events = all_ok.loc[
        all_ok.unlock_date.str[:4].isin(("2024", "2025"))
        & all_ok.unlock_date.str[5:].le("12-17")
        & all_ok.unlock_date.isin(calendar)
    ].copy()
    if events.empty or events.duplicated(["unlock_date", "code"]).any():
        raise ValueError("Duplicate or empty IPO unlock stock-date")
    events = events.rename(columns={"unlock_date": "date"})
    return events, counts


def _prior_float(events: pd.DataFrame, daily_dir: Path,
                 calendar: list[str]) -> pd.DataFrame:
    previous = dict(zip(calendar[1:], calendar[:-1]))
    source = events.copy()
    source["trade_date"] = source.date.map(previous)
    if source.trade_date.isna().any():
        raise ValueError("IPO unlock lacks a prior trading date")
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.register("events", source)
    joined = connection.execute("""
        SELECT e.*, d.volume AS prior_volume, d.turn AS prior_turn,
               d.tradestatus AS prior_tradestatus
        FROM events e LEFT JOIN read_parquet(?) d
          ON d.date = e.trade_date AND d.code = e.code
    """, [str(daily_dir / "*.parquet")]).df()
    if len(joined) != len(source) or joined.duplicated(["date", "code"]).any():
        raise ValueError("IPO unlock prior daily join changed membership")
    joined = joined.loc[
        joined.prior_tradestatus.eq(1)
        & joined.prior_volume.gt(0) & joined.prior_turn.gt(0)
    ].copy()
    joined["prior_float_shares"] = (
        joined.prior_volume / (joined.prior_turn / 100))
    joined["unlock_prior_float_ratio"] = (
        joined.unlock_shares / joined.prior_float_shares)
    if (joined.empty or not np.isfinite(joined.unlock_prior_float_ratio).all()
            or joined.unlock_prior_float_ratio.le(0).any()):
        raise ValueError("IPO unlock share-to-prior-float ratio invalid")
    return joined


def build(base: Path, snapshot_dir: Path, daily_dir: Path,
          calendar_path: Path, industry_path: Path, output_dir: Path) -> dict:
    calendar = trading_dates(calendar_path, "2024-01-01", "2026-01-15")
    events, source = _announcements(base, calendar)
    available = _prior_float(events, daily_dir, calendar)
    large = available.loc[available.unlock_prior_float_ratio.ge(
        MIN_UNLOCK_PRIOR_FLOAT)].copy()
    if large.empty:
        raise ValueError("No large original IPO unlock remains")
    universe = _universe(snapshot_dir, daily_dir, industry_path,
                         large, calendar)
    # Small confirmed unlocks must not silently serve as untreated controls.
    low_keys = pd.MultiIndex.from_frame(available.loc[
        available.unlock_prior_float_ratio.lt(MIN_UNLOCK_PRIOR_FLOAT),
        ["date", "code"]])
    universe = universe.loc[~pd.MultiIndex.from_frame(
        universe[["date", "code"]]).isin(low_keys)].copy()
    main, main_report = _match(universe, large, calendar, False, EVENT,
                               MAX_CURRENT_RETURN_GAP)
    industry, industry_report = _match(universe, large, calendar, True, EVENT,
                                       MAX_CURRENT_RETURN_GAP)
    audit = {"source": source, "valid_year_and_calendar": len(events),
             "valid_prior_float": len(available),
             "large_prior_float": len(large),
             "large_by_year": {year: len(large.loc[
                 large.date.str.startswith(year)])
                 for year in ("2024", "2025")},
             "universe_stock_days": len(universe),
             "main": main_report, "same_industry": industry_report,
             "note": "Original IPO unlock inputs only; no future returns opened"}
    output_dir.mkdir(parents=True, exist_ok=True)
    large.to_parquet(output_dir / "events.parquet", index=False,
                     compression="zstd")
    main.to_parquet(output_dir / "pairs.parquet", index=False,
                    compression="zstd")
    industry.to_parquet(output_dir / "industry_pairs.parquet", index=False,
                        compression="zstd")
    (output_dir / "input_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path(
        "data/research/unlock"))
    parser.add_argument("--snapshots", type=Path, default=Path(
        "data/research/market_snapshots_ci"))
    parser.add_argument("--daily", type=Path, default=Path(
        "data/baostock/market_2020_2026/daily"))
    parser.add_argument("--calendar", type=Path, default=Path(
        "data/baostock/market_2020_2026/metadata/calendar.parquet"))
    parser.add_argument("--industry", type=Path, default=Path(
        "data/research/industry_intervals.parquet"))
    parser.add_argument("--output", type=Path, default=Path(
        "data/research/unlock"))
    args = parser.parse_args()
    print(build(args.base, args.snapshots, args.daily, args.calendar,
                args.industry, args.output))


if __name__ == "__main__":
    main()
