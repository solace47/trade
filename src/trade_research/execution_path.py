"""Audit aggregate VWAP against a fixed minute participation entry model.

Freeze selects stock-days using only the audited 14:49 prefix and fields known
before the decision. Evaluate reads only their 14:52--14:55 raw minute bars;
neither stage reads an exit price or holding return.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from math import floor
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .hf_outcomes import Assumptions, _board_limit_rate, _fill, _limit_price, _order_shares
from .market_study import _quality_keys


ROOT = Path("data/research")
OUTPUT = ROOT / "execution_path"
LABELS = ("1452", "1453", "1454", "1455")
HALVES = ("2024H1", "2024H2", "2025H1", "2025H2")
BOARDS = ("main", "chinext", "star")


def freeze(prefix_dir: Path = ROOT / "minute_prefix_1449",
           snapshot_dir: Path = ROOT / "market_snapshots_ci",
           output_dir: Path = OUTPUT, all_eligible: bool = False) -> dict:
    """Freeze either the fixed pilot sample or its unchanged eligible universe."""
    connection = duckdb.connect()
    try:
        connection.execute("SET threads = 4")
        connection.from_parquet(str(prefix_dir / "*" / "*.parquet")).create_view("prefix")
        connection.from_parquet(str(snapshot_dir / "*.parquet")).create_view("snapshots")
        query = """
            WITH eligible AS (
                SELECT p.date, p.code, p.price_1449, p.amount_1449,
                       s.preclose,
                       CASE WHEN p.code LIKE 'sh.68%' THEN 'star'
                            WHEN p.code LIKE 'sz.30%' THEN 'chinext'
                            ELSE 'main' END AS board,
                       left(p.date, 4) || 'H' ||
                         CASE WHEN substr(p.date, 6, 2) <= '06'
                              THEN '1' ELSE '2' END AS half
                FROM prefix p JOIN snapshots s USING (date, code)
                WHERE p.date BETWEEN '2024-01-01' AND '2025-12-31'
                  AND s.tradestatus = 1 AND s.isST = 0
                  AND s.listing_age_sessions >= 20
                  AND NOT s.reference_gap
                  AND NOT p.quote_outside_traded_range
                  AND p.price_1449 >= 5
                  AND p.amount_1449 BETWEEN 100000000 AND 10000000000
                  AND s.return20_prior_adjusted BETWEEN -0.10 AND 0.10
                  AND p.price_1449 / s.preclose - 1 BETWEEN -0.03 AND 0.03
                  AND p.return_last29 BETWEEN -0.01 AND 0.01
            ), ranked AS (
                SELECT *, row_number() OVER (
                    PARTITION BY date, board ORDER BY md5(date || code), code
                ) AS choice FROM eligible
            )
            SELECT date, code, board, half, price_1449, amount_1449,
                   preclose FROM ranked
        """ + ("" if all_eligible else " WHERE choice = 1") + \
            " ORDER BY date, board, code"
        selected = connection.execute(query).df()
    finally:
        connection.close()
    if (selected.empty or selected.duplicated(["date", "code"]).any()
            or set(selected.half) != set(HALVES)
            or set(selected.board) != set(BOARDS)
            or not np.isfinite(selected[["price_1449", "preclose"]].to_numpy()).all()):
        raise ValueError("Incomplete or invalid pre-decision sample")
    cells = selected.groupby(["half", "board"]).agg(
        stock_days=("code", "size"), dates=("date", "nunique")
    )
    if len(cells) != 12 or cells.stock_days.lt(100).any():
        raise ValueError("Fewer than 100 stock-days in a half-year board cell")
    report = {
        "cutoff": "14:49", "years": [2024, 2025],
        "sampling": ("all eligible stock-days, pilot filters unchanged"
                     if all_eligible else
                     "one per date and board, md5(date || code) ascending"),
        "selected_stock_days": len(selected),
        "cells": {f"{half}_{board}": {
            "stock_days": int(row.stock_days), "dates": int(row.dates),
        } for (half, board), row in cells.iterrows()},
        "post_decision_data_read": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(output_dir / "frozen_signals.parquet", index=False,
                        compression="zstd")
    (output_dir / "freeze_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def minute_participation(bars: pd.DataFrame, code: str, date: str,
                         preclose: float, shares: int,
                         slippage_bps: float,
                         lot_override: int | None = None) -> tuple[int, float | None]:
    """Take at most 10% of each minute's volume, earliest minute first.

    Child orders use 100-share lots, or conservative 200-share lots on STAR.
    This is still a bar-VWAP estimate, not verified queue execution.
    """
    lot = lot_override or (200 if code.startswith("sh.68") else 100)
    upper = _limit_price(preclose, _board_limit_rate(code, 0, date), True)
    remaining, filled, value = shares, 0, 0.0
    for row in bars.itertuples(index=False):
        if remaining == 0:
            break
        if row.volume <= 0 or row.turnover <= 0:
            continue
        price = row.turnover / row.volume * (1 + slippage_bps / 10_000)
        if price >= upper - .005:
            continue
        capacity = floor(row.volume * .1 / lot) * lot
        taken = min(remaining, capacity)
        remaining -= taken
        filled += taken
        value += taken * price
    return filled, value / filled if filled else None


def _one_stock(item: tuple[str, pd.DataFrame], minute_root: Path) -> list[dict]:
    code, signals = item
    exchange, symbol = code.split(".")
    path = minute_root / exchange.upper() / f"{symbol}.parquet"
    start = pd.Timestamp(signals.date.min())
    end = pd.Timestamp(signals.date.max()) + pd.Timedelta(days=1)
    minute = pd.read_parquet(path, columns=["timestamp", "volume", "turnover"],
                             filters=[("timestamp", ">=", start),
                                      ("timestamp", "<", end)])
    # Filter the four clock minutes before formatting calendar dates; a full
    # two-year file has about 120,000 rows but only a few thousand entry bars.
    clock_minute = minute.timestamp.dt.hour * 60 + minute.timestamp.dt.minute
    minute = minute.loc[clock_minute.between(14 * 60 + 52,
                                             14 * 60 + 55)].copy()
    minute["date"] = minute.timestamp.dt.strftime("%Y-%m-%d")
    minute["label"] = minute.timestamp.dt.strftime("%H%M")
    minute = minute.loc[minute.date.isin(signals.date)
                        & minute.label.isin(LABELS)]
    by_date = {date: bars.sort_values("label")
               for date, bars in minute.groupby("date")}
    rows = []
    for signal in signals.itertuples(index=False):
        bars = by_date.get(signal.date)
        result = {"date": signal.date, "code": code, "board": signal.board,
                  "half": signal.half, "price_1449": signal.price_1449,
                  "preclose": signal.preclose}
        shares = _order_shares(code, signal.price_1449, 100_000)
        if signal.board == "star":
            shares = shares // 200 * 200
        result["target_shares"] = shares
        if (bars is None or bars.label.tolist() != list(LABELS)
                or not np.isfinite(bars[["volume", "turnover"]].to_numpy()).all()
                or bars.volume.lt(0).any() or bars.turnover.lt(0).any()
                or shares == 0):
            result["bars_complete"] = False
            rows.append(result)
            continue
        result["bars_complete"] = True
        total_volume = float(bars.volume.sum())
        total_turnover = float(bars.turnover.sum())
        quote = pd.Series({"vwap": total_turnover / total_volume
                           if total_volume > 0 else 0.,
                           "volume": total_volume})
        daily = pd.Series({"date": signal.date, "preclose": signal.preclose,
                           "tradestatus": 1, "isST": 0})
        for slip in (5, 10):
            aggregate_price, aggregate_status = _fill(
                quote, daily, code, "buy", shares,
                Assumptions(slippage_bps_each_side=slip))
            taken, path_price = minute_participation(
                bars, code, signal.date, signal.preclose, shares, slip)
            result[f"aggregate_status_{slip}"] = aggregate_status
            result[f"aggregate_price_{slip}"] = aggregate_price
            result[f"participation_shares_{slip}"] = taken
            result[f"participation_price_{slip}"] = path_price
            result[f"participation_filled_{slip}"] = taken == shares
            result[f"extra_entry_bps_{slip}"] = (
                10_000 * (path_price / aggregate_price - 1)
                if aggregate_price is not None and taken == shares else None)
            if signal.board == "star":
                # A submitted STAR order can be partially matched in shares;
                # this is an optimistic bound beside the frozen 200-share
                # child-order stress model, not a revised primary result.
                partial, partial_price = minute_participation(
                    bars, code, signal.date, signal.preclose, shares, slip,
                    lot_override=1)
                result[f"share_partial_filled_{slip}"] = partial == shares
                result[f"share_partial_price_{slip}"] = partial_price
        rows.append(result)
    return rows


def evaluate(signals_file: Path = OUTPUT / "frozen_signals.parquet",
             minute_root: Path = Path("data/hf/pilot/data/stock_1m"),
             issues_dir: Path = ROOT / "market_issues_ci",
             output_dir: Path = OUTPUT, workers: int = 4) -> dict:
    """Read only the frozen stock-days' four raw entry minutes."""
    signals = pd.read_parquet(signals_file)
    if (signals.empty or signals.duplicated(["date", "code"]).any()
            or not signals.date.between("2024-01-01", "2025-12-31").all()):
        raise ValueError("Invalid frozen entry sample")
    grouped = list(signals.groupby("code", sort=True))
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for count, group_rows in enumerate(
                pool.map(lambda item: _one_stock(item, minute_root), grouped), start=1):
            rows.extend(group_rows)
            if count % 500 == 0 or count == len(grouped):
                print(f"Read raw entry minutes for {count}/{len(grouped)} stocks",
                      flush=True)
    result = pd.DataFrame(rows)
    if len(result) != len(signals):
        raise ValueError("Entry result count differs from frozen sample")
    bad = _quality_keys(issues_dir)[["date", "code"]].drop_duplicates()
    result = result.merge(bad.assign(quality_issue=True), on=["date", "code"],
                          how="left", validate="one_to_one")
    result["quality_issue"] = result.quality_issue.fillna(False).astype(bool)
    clean = result.loc[~result.quality_issue & result.bars_complete].copy()
    by_half = {}
    for half in HALVES:
        part = clean.loc[clean.half.eq(half)]
        both = part.loc[part.aggregate_status_5.eq("filled")
                        & part.participation_filled_5]
        by_half[half] = {
            "clean_complete_stock_days": len(part),
            "aggregate_fill_rate_5": float(part.aggregate_status_5.eq("filled").mean()),
            "participation_fill_rate_5": float(part.participation_filled_5.mean()),
            "aggregate_fill_rate_10": float(part.aggregate_status_10.eq("filled").mean()),
            "participation_fill_rate_10": float(part.participation_filled_10.mean()),
            "both_filled_5": len(both),
            "mean_extra_entry_bps_5": float(both.extra_entry_bps_5.mean()),
            "median_extra_entry_bps_5": float(both.extra_entry_bps_5.median()),
        }
    by_half_board = {}
    for (half, board), part in clean.groupby(["half", "board"]):
        both = part.loc[part.aggregate_status_5.eq("filled")
                        & part.participation_filled_5]
        by_half_board[f"{half}_{board}"] = {
            "clean_complete_stock_days": len(part),
            "aggregate_fill_rate_5": float(part.aggregate_status_5.eq("filled").mean()),
            "participation_fill_rate_5": float(part.participation_filled_5.mean()),
            "participation_fill_rate_10": float(part.participation_filled_10.mean()),
            "both_filled_5": len(both),
            "mean_extra_entry_bps_5": float(both.extra_entry_bps_5.mean()),
        }
    both_all = clean.loc[clean.aggregate_status_5.eq("filled")
                         & clean.participation_filled_5]
    report = {
        "frozen_stock_days": len(signals),
        "incomplete_bars": int((~result.bars_complete).sum()),
        "quality_issue_stock_days": int(result.quality_issue.sum()),
        "model": "10% per-minute volume participation, earliest first",
        "same_stock_day_control": True,
        "holding_returns_read": False,
        "by_half": by_half,
        "by_half_board": by_half_board,
        "both_filled_abs_price_diff_over_5bps_rate": float(
            both_all.extra_entry_bps_5.abs().gt(5).mean()),
        "both_filled_price_diff_p10_p90_bps": [
            float(both_all.extra_entry_bps_5.quantile(.1)),
            float(both_all.extra_entry_bps_5.quantile(.9)),
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    result.sort_values(["date", "code"]).to_parquet(
        output_dir / "entry_results.parquet", index=False, compression="zstd")
    (output_dir / "entry_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "evaluate",
                                           "freeze-full", "evaluate-full"))
    args = parser.parse_args()
    if args.stage == "freeze":
        report = freeze()
    elif args.stage == "evaluate":
        report = evaluate()
    elif args.stage == "freeze-full":
        report = freeze(output_dir=OUTPUT / "full", all_eligible=True)
    else:
        report = evaluate(signals_file=OUTPUT / "full" / "frozen_signals.parquet",
                          output_dir=OUTPUT / "full")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
