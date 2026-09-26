"""Freeze A-share limit-up touches that reopened by the 14:50 cutoff.

Only 14:50 snapshots, prior daily bars, historical industry, and the trading
calendar are read. No entry window or later return enters the pair list.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .buyback_inputs import _capacity_events, _universe, CONTROL
from .exchange_public_events import trading_dates
from .margin_heterogeneity import _match_one


EVENT = "limit_up_reopened"
MAX_CURRENT_RETURN_GAP = .01


def _touches(snapshot_dir: Path) -> pd.DataFrame:
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    return connection.execute("""
        WITH eligible AS (
          SELECT date, code, price_1450, high_1450, preclose,
                 CASE WHEN code LIKE 'sh.68%' OR code LIKE 'sz.30%'
                      THEN CAST(1.20 AS DECIMAL(4, 2))
                      ELSE CAST(1.10 AS DECIMAL(4, 2)) END AS up_factor
          FROM read_parquet(?)
          WHERE date BETWEEN '2024-01-01' AND '2025-12-17'
            AND SUBSTR(date, 6) <= '12-17'
            AND (code LIKE 'sh.60%' OR code LIKE 'sh.68%'
              OR code LIKE 'sz.00%' OR code LIKE 'sz.30%')
            AND isST = 0 AND tradestatus = 1 AND listing_age_sessions >= 20
            AND NOT reference_gap AND NOT quote_outside_traded_range
            AND amount_1450 >= 30000000 AND price_1450 >= 5
            AND preclose > 0 AND open_1450 > 0
        ), limited AS (
          SELECT *, ROUND(CAST(preclose AS DECIMAL(18, 2))
                          * up_factor, 2) AS limit_price
          FROM eligible
        )
        SELECT date, code, limit_price, high_1450, price_1450,
               price_1450 / preclose - 1 AS current_return,
               price_1450 <= limit_price * .99
                 AND price_1450 / preclose - 1 >= .03 AS reopened
        FROM limited
        WHERE high_1450 >= limit_price - .005
        ORDER BY date, code
    """, [str(snapshot_dir / "*.parquet")]).df()


def _match(universe: pd.DataFrame, events: pd.DataFrame,
           touched: pd.DataFrame, calendar: list[str],
           same_industry: bool) -> tuple[pd.DataFrame, dict]:
    signal = universe.merge(
        events[["date", "code", "limit_price", "high_1450"]],
        on=["date", "code"], how="inner",
                            validate="one_to_one")
    if signal.empty:
        raise ValueError("No reopened limit-up stock has eligible 14:50 inputs")
    selected = _capacity_events(signal, calendar)
    touched_keys = pd.MultiIndex.from_frame(touched[["date", "code"]])
    control_mask = ~pd.MultiIndex.from_frame(
        universe[["date", "code"]]).isin(touched_keys)
    controls_by_day = {
        day: group.copy()
        for day, group in universe.loc[control_mask].groupby("date", sort=False)
    }
    event_rows, peer_rows = [], []
    unmatched = 0
    for day, candidates in selected.groupby("date", sort=True):
        controls = controls_by_day[day].copy()
        for _, item in candidates.iterrows():
            choices = controls.loc[
                controls.board.eq(item.board)
                & controls.size_bucket.eq(item.size_bucket)
                & controls.momentum_bucket.eq(item.momentum_bucket)
                & (controls.return_1450 - item.return_1450).abs().le(
                    MAX_CURRENT_RETURN_GAP)
            ]
            if same_industry:
                choices = choices.loc[choices.industry.eq(item.industry)]
            matched = _match_one(item, choices)
            if matched is None:
                unmatched += 1
                continue
            peer, distance = matched
            left = item.copy()
            left["candidate"] = EVENT
            left["pair_code"] = item.code
            left["match_distance"] = distance
            right = peer.copy()
            right["candidate"] = CONTROL
            right["pair_code"] = item.code
            right["match_distance"] = distance
            event_rows.append(left)
            peer_rows.append(right)
            controls = controls.loc[controls.code.ne(peer.code)]
    if not event_rows:
        raise ValueError("Reopened limit-up stock has no same-day untouched peer")
    pairs = pd.concat([pd.DataFrame(event_rows), pd.DataFrame(peer_rows)],
                      ignore_index=True)
    if (pairs.duplicated(["date", "code"]).any()
            or len(pairs) != 2 * len(event_rows)
            or pairs.groupby(["date", "pair_code"]).candidate.nunique().ne(2).any()):
        raise ValueError("Reopened limit-up pairs are incomplete")
    event = pairs.loc[pairs.candidate.eq(EVENT)]
    control = pairs.loc[pairs.candidate.eq(CONTROL)]
    balance = event.merge(control, on=["date", "pair_code"],
                          suffixes=("_event", "_peer"), validate="one_to_one")
    report = {"eligible_event_quotes": len(signal),
              "capacity_selected": len(selected),
              "matched_pairs": len(event), "unmatched": unmatched,
              "matched_days": int(event.date.nunique()), "by_year": {},
              "balance": {}}
    for year in ("2024", "2025"):
        period = event.loc[event.date.str.startswith(year)]
        report["by_year"][year] = {
            "pairs": len(period), "days": int(period.date.nunique()),
            "unique_codes": int(period.code.nunique()),
        }
    for field in ("float_mv", "avg20_amount", "price_1450"):
        report["balance"][field + "_median_event_peer_ratio"] = float(
            (balance[field + "_event"] / balance[field + "_peer"]).median())
    for field in ("return20_prior_adjusted", "return_1450"):
        report["balance"][field + "_median_event_minus_peer"] = float(
            (balance[field + "_event"] - balance[field + "_peer"]).median())
    if (balance.return_1450_event - balance.return_1450_peer).abs().gt(
            MAX_CURRENT_RETURN_GAP + 1e-12).any():
        raise ValueError("Reopened limit-up current-return match drifted")
    return pairs, report


def build(snapshot_dir: Path, daily_dir: Path, calendar_path: Path,
          industry_path: Path, output_dir: Path) -> dict:
    calendar = trading_dates(calendar_path, "2024-01-01", "2026-01-15")
    touched = _touches(snapshot_dir)
    events = touched.loc[touched.reopened].copy()
    if (events.empty or touched.duplicated(["date", "code"]).any()
            or events.duplicated(["date", "code"]).any()
            or not events.date.str[:4].isin(("2024", "2025")).all()
            or not np.isfinite(events.limit_price).all()):
        raise ValueError("Malformed as-of-14:50 price-limit events")
    universe = _universe(snapshot_dir, daily_dir, industry_path,
                         events, calendar)
    main, main_report = _match(universe, events, touched, calendar, False)
    industry, industry_report = _match(
        universe, events, touched, calendar, True)
    audit = {"source": {
        year: {"touched": len(touched.loc[touched.date.str.startswith(year)]),
               "reopened": len(events.loc[events.date.str.startswith(year)]),
               "days": int(events.loc[events.date.str.startswith(year)]
                           .date.nunique())}
        for year in ("2024", "2025")},
        "universe_stock_days": len(universe),
        "main": main_report, "same_industry": industry_report,
        "note": "As-of-14:50 limit-reopen inputs only; no future outcomes opened"}
    output_dir.mkdir(parents=True, exist_ok=True)
    events.to_parquet(output_dir / "events.parquet", index=False,
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
    parser.add_argument("--snapshots", type=Path, default=Path(
        "data/research/market_snapshots_ci"))
    parser.add_argument("--daily", type=Path, default=Path(
        "data/baostock/market_2020_2026/daily"))
    parser.add_argument("--calendar", type=Path, default=Path(
        "data/baostock/market_2020_2026/metadata/calendar.parquet"))
    parser.add_argument("--industry", type=Path, default=Path(
        "data/research/industry_intervals.parquet"))
    parser.add_argument("--output", type=Path, default=Path(
        "data/research/limit_reopen"))
    args = parser.parse_args()
    print(build(args.snapshots, args.daily, args.calendar,
                args.industry, args.output))


if __name__ == "__main__":
    main()
