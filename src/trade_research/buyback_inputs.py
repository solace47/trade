"""Pair next-session buyback-plan events to observable 14:50 peers.

This module reads original announcement metadata, prior daily bars and 14:50
snapshots only. It does not open future trade outcomes.
"""

from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .buyback_source import strict_plan_title
from .exchange_public_events import trading_dates
from .margin_heterogeneity import _match_one


CAPACITY = 5
COOLDOWN = 10
EVENT = "buyback_plan"
CONTROL = "same_day_nonannouncer"


def _events(source_dir: Path, calendar: list[str]) -> tuple[pd.DataFrame, dict]:
    frames = []
    raw_counts = {}
    for year in (2024, 2025):
        index = pd.read_parquet(source_dir / f"title_candidates_{year}.parquet")
        extraction = pd.read_json(source_dir / f"pdf_audit_{year}.jsonl",
                                  lines=True)[["pdf_url", "status"]]
        if (index.empty or index.pdf_url.duplicated().any()
                or extraction.pdf_url.duplicated().any()
                or not index.title.map(strict_plan_title).all()):
            raise ValueError("Malformed original buyback source or title screen")
        joined = index.merge(extraction, on="pdf_url", how="left",
                             validate="one_to_one")
        if joined.status.isna().any():
            raise ValueError("Buyback candidate lacks original PDF audit")
        raw_counts[str(year)] = {
            "title_candidates": len(index),
            "original_pdf_confirmed": int(joined.status.eq("ok").sum()),
        }
        frames.append(joined.loc[joined.status.eq("ok")].copy())
    events = pd.concat(frames, ignore_index=True)
    duplicate = events.duplicated(["code", "notice_date"], keep=False)
    duplicate_rows = int(duplicate.sum())
    events = events.loc[~duplicate].copy()
    events["date"] = events.notice_date.map(lambda day: calendar[
        bisect.bisect_right(calendar, day)])
    events = events.loc[
        events.date.str[:4].eq(events.notice_date.str[:4])
        & events.date.str[5:].le("12-17")
    ].copy()
    if (events.empty or events.duplicated(["date", "code"]).any()
            or not events.date.gt(events.notice_date).all()):
        raise ValueError("Malformed next-session repurchase events")
    return events, {"source": raw_counts,
                    "duplicate_code_notice_rows_removed": duplicate_rows,
                    "mapped_events": len(events)}


def _universe(snapshot_dir: Path, daily_dir: Path, industry_path: Path,
              events: pd.DataFrame, calendar: list[str]) -> pd.DataFrame:
    next_days = pd.DataFrame({"trade_date": calendar[:-1],
                              "date": calendar[1:]})
    event_days = pd.DataFrame({"date": sorted(events.date.unique())})
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.register("next_days", next_days)
    connection.register("event_days", event_days)
    connection.read_parquet(str(snapshot_dir / "*.parquet")).create_view(
        "snapshots")
    connection.read_parquet(str(industry_path)).create_view("industry")
    connection.execute("""
        CREATE TEMP TABLE liquidity AS
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
            FROM read_parquet(?)
            WHERE date BETWEEN '2023-10-01' AND '2025-12-17'
              AND tradestatus = 1 AND amount > 0 AND turn > 0
        )
    """, [str(daily_dir / "*.parquet")])
    frame = connection.execute("""
        WITH eligible AS (
          SELECT s.date, n.trade_date, s.code,
                 CASE WHEN s.code LIKE 'sh.60%' THEN 'sh_main'
                      WHEN s.code LIKE 'sh.68%' THEN 'sh_star'
                      WHEN s.code LIKE 'sz.00%' THEN 'sz_main'
                      WHEN s.code LIKE 'sz.30%' THEN 'sz_gem'
                      ELSE 'other' END AS board,
                 i.industry, d.amount / (d.turn / 100.0) AS float_mv,
                 d.avg20_amount, s.amount_1450, s.price_1450,
                 s.return20_prior_adjusted, s.return_1450,
                 s.open_1450 / s.preclose - 1 AS open_gap,
                 s.isST, s.reference_gap, s.quote_outside_traded_range,
                 s.listing_age_sessions
          FROM snapshots s
          JOIN event_days e ON e.date = s.date
          JOIN next_days n ON n.date = s.date
          JOIN liquidity d ON d.date = n.trade_date AND d.code = s.code
          JOIN industry i ON i.code = s.code
            AND i.effective_date <= s.date
            AND (i.next_effective_date > s.date
                 OR i.next_effective_date IS NULL)
          WHERE i.industry NOT LIKE 'J%'
            AND (s.code LIKE 'sh.60%' OR s.code LIKE 'sh.68%'
              OR s.code LIKE 'sz.00%' OR s.code LIKE 'sz.30%')
            AND d.traded_count = 20 AND d.avg20_amount >= 30000000
            AND s.amount_1450 >= 30000000
            AND s.tradestatus = 1 AND s.isST = 0
            AND s.listing_age_sessions >= 20
            AND NOT s.reference_gap AND NOT s.quote_outside_traded_range
            AND s.price_1450 >= 5 AND s.preclose > 0 AND s.open_1450 > 0
            AND s.open_1450 BETWEEN s.low_1450 - .005 AND s.high_1450 + .005
            AND s.return20_prior_adjusted IS NOT NULL
            AND d.amount > 0 AND d.turn > 0
        ), sized AS (
          SELECT *, NTILE(5) OVER (
            PARTITION BY date, board ORDER BY float_mv, code
          ) AS size_bucket FROM eligible
        )
        SELECT *, NTILE(3) OVER (
          PARTITION BY date, board, size_bucket
          ORDER BY return20_prior_adjusted, code
        ) AS momentum_bucket
        FROM sized ORDER BY date, code
    """).df()
    if (frame.empty or frame.duplicated(["date", "code"]).any()
            or not np.isfinite(frame.float_mv).all()
            or frame.float_mv.le(0).any()
            or not frame.date.gt(frame.trade_date).all()):
        raise ValueError("Malformed point-in-time buyback comparison universe")
    return frame


def _capacity_events(signal: pd.DataFrame,
                     calendar: list[str],
                     cooldown_sessions: int = COOLDOWN) -> pd.DataFrame:
    if cooldown_sessions < 0:
        raise ValueError("Cooldown sessions must be nonnegative")
    index = {day: number for number, day in enumerate(calendar)}
    last_kept: dict[str, int] = {}
    chunks = []
    for day, daily in signal.groupby("date", sort=True):
        session = index[day]
        available = daily.loc[daily.code.map(
            lambda code: session - last_kept.get(code, -1000) > cooldown_sessions
        )].sort_values(["avg20_amount", "code"], ascending=[False, True])
        chosen = available.head(CAPACITY)
        chunks.append(chosen)
        last_kept.update({code: session for code in chosen.code})
    return pd.concat(chunks, ignore_index=True)


def _match(universe: pd.DataFrame, events: pd.DataFrame,
           calendar: list[str], same_industry: bool,
           event_label: str = EVENT,
           max_current_gap: float | None = None,
           cooldown_sessions: int = COOLDOWN) -> tuple[pd.DataFrame, dict]:
    if max_current_gap is not None and max_current_gap <= 0:
        raise ValueError("Current-return match gap must be positive")
    event_keys = events[["date", "code", "notice_date", "pdf_url"]]
    signal = universe.merge(event_keys, on=["date", "code"], how="inner",
                            validate="one_to_one")
    if signal.empty:
        raise ValueError("No buyback plan has an eligible 14:50 quote")
    selected_events = _capacity_events(signal, calendar, cooldown_sessions)
    event_keys = pd.MultiIndex.from_frame(events[["date", "code"]])
    is_event = pd.MultiIndex.from_frame(
        universe[["date", "code"]]).isin(event_keys)
    controls_by_day = {
        day: group.copy()
        for day, group in universe.loc[~is_event].groupby("date", sort=False)
    }
    high_rows, low_rows = [], []
    unmatched = 0
    for day, candidates in selected_events.groupby("date", sort=True):
        controls = controls_by_day[day].copy()
        for _, item in candidates.iterrows():
            choices = controls.loc[
                controls.board.eq(item.board)
                & controls.size_bucket.eq(item.size_bucket)
                & controls.momentum_bucket.eq(item.momentum_bucket)
            ]
            if max_current_gap is not None:
                choices = choices.loc[
                    (choices.return_1450 - item.return_1450).abs().le(
                        max_current_gap)
                ]
            if same_industry:
                choices = choices.loc[choices.industry.eq(item.industry)]
            matched = _match_one(item, choices)
            if matched is None:
                unmatched += 1
                continue
            peer, distance = matched
            left = item.copy()
            left["pair_code"] = item.code
            left["candidate"] = event_label
            left["match_distance"] = distance
            high_rows.append(left)
            right = peer.copy()
            right["pair_code"] = item.code
            right["candidate"] = CONTROL
            right["match_distance"] = distance
            low_rows.append(right)
            controls = controls.loc[controls.code.ne(peer.code)]
    if not high_rows:
        raise ValueError("No buyback event has a matched nonannouncer")
    selected = pd.concat([pd.DataFrame(high_rows), pd.DataFrame(low_rows)],
                         ignore_index=True)
    pairs = selected.groupby(["date", "pair_code"]).candidate.nunique()
    if (not pairs.eq(2).all() or len(selected) != 2 * len(pairs)
            or selected.duplicated(["date", "code", "candidate"]).any()):
        raise ValueError("Malformed buyback event/control pair")
    treated = selected.loc[selected.candidate.eq(event_label)]
    control = selected.loc[selected.candidate.eq(CONTROL)]
    report = {"eligible_event_quotes": len(signal),
              "cooldown_sessions": cooldown_sessions,
              "capacity_selected": len(selected_events),
              "matched_pairs": len(treated), "unmatched": unmatched,
              "matched_days": int(treated.date.nunique()),
              "by_year": {},
              "balance": {}}
    for year in ("2024", "2025"):
        period = treated.loc[treated.date.str.startswith(year)]
        report["by_year"][year] = {
            "pairs": len(period), "days": int(period.date.nunique()),
            "unique_codes": int(period.code.nunique()),
            "peak_month_share": (float(period.date.str[:7].value_counts().max()
                                       / len(period)) if len(period) else None),
        }
    joined = treated.merge(control, on=["date", "pair_code"],
                           suffixes=("_event", "_peer"), validate="one_to_one")
    for field in ("float_mv", "avg20_amount", "price_1450"):
        report["balance"][field + "_median_event_peer_ratio"] = float(
            (joined[field + "_event"] / joined[field + "_peer"]).median())
    for field in ("return20_prior_adjusted", "return_1450"):
        report["balance"][field + "_median_event_minus_peer"] = float(
            (joined[field + "_event"] - joined[field + "_peer"]).median())
    return selected, report


def build(source_dir: Path, snapshot_dir: Path, daily_dir: Path,
          calendar_path: Path, industry_path: Path, output_dir: Path) -> dict:
    calendar = trading_dates(calendar_path, "2024-01-01", "2026-01-15")
    events, source_report = _events(source_dir, calendar)
    universe = _universe(snapshot_dir, daily_dir, industry_path,
                         events, calendar)
    pairs, main_report = _match(universe, events, calendar, False)
    industry_pairs, industry_report = _match(universe, events, calendar, True)
    audit = {"source": source_report, "universe_stock_days": len(universe),
             "main": main_report, "same_industry": industry_report,
             "note": "Inputs only; no future returns opened"}
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(output_dir / "pairs.parquet", index=False,
                     compression="zstd")
    industry_pairs.to_parquet(output_dir / "industry_pairs.parquet", index=False,
                              compression="zstd")
    (output_dir / "input_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(
        "data/research/buyback"))
    parser.add_argument("--snapshots", type=Path, default=Path(
        "data/research/market_snapshots_ci"))
    parser.add_argument("--daily", type=Path, default=Path(
        "data/baostock/market_2020_2026/daily"))
    parser.add_argument("--calendar", type=Path, default=Path(
        "data/baostock/market_2020_2026/metadata/calendar.parquet"))
    parser.add_argument("--industry", type=Path, default=Path(
        "data/research/industry_intervals.parquet"))
    parser.add_argument("--output", type=Path, default=Path(
        "data/research/buyback"))
    args = parser.parse_args()
    audit = build(args.source, args.snapshots, args.daily, args.calendar,
                  args.industry, args.output)
    print({"source": audit["source"], "main": audit["main"],
           "same_industry": audit["same_industry"]})


if __name__ == "__main__":
    main()
