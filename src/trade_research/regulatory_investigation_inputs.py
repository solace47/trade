"""Match issuer investigation notices using information available at 14:50.

This module never reads returns after the signal. It applies the previously
fixed input feasibility gate before any original-minute outcomes may be read.
"""

from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .block_trade_inputs import _distance, _lag_industry
from .buyback_inputs import _capacity_events, _match, _universe
from .exchange_public_events import trading_dates


EVENT = "issuer_investigation"
CONTROL = "same_day_noninvestigated"
COOLDOWN = 120
LAST_SIGNAL = "2025-12-17"
MIN_EVENTS_PER_YEAR = 25
MIN_MONTHS_PER_YEAR = 6


def _events(source_dir: Path, calendar: list[str]) -> tuple[pd.DataFrame, dict]:
    audit = json.loads((source_dir / "source_audit.json").read_text(
        encoding="utf-8"))
    frames = []
    for year in (2024, 2025):
        frame = pd.read_parquet(source_dir / f"confirmed_{year}.parquet")
        if (frame.empty or len(frame) != audit[str(year)][
                    "confirmed_issuer_disclosure_notices"]
                or frame.pdf_url.duplicated().any()
                or frame.duplicated(["code", "notice_date"]).any()
                or not frame.notice_date.str.startswith(str(year)).all()):
            raise ValueError("Confirmed investigation originals changed")
        frames.append(frame)
    events = pd.concat(frames, ignore_index=True)
    if events.pdf_url.duplicated().any():
        raise ValueError("Duplicate investigation original PDF")
    events["date"] = events.notice_date.map(
        lambda day: calendar[bisect.bisect_right(calendar, day)])
    events = events.loc[events.date.le(LAST_SIGNAL)].copy()
    if (events.empty or not events.date.gt(events.notice_date).all()
            or not events.date.str[:4].isin(("2024", "2025")).all()
            or events.duplicated(["date", "code"]).any()):
        raise ValueError("Investigation signal is not a unique next session")
    return events, {
        "confirmed_originals": sum(len(frame) for frame in frames),
        "mapped_2024_2025_stock_days": len(events),
        "by_signal_year": events.date.str[:4].value_counts().sort_index().to_dict(),
        "note": "Source identity and signal times only; no returns opened",
    }


def _five_peer_pairs(universe: pd.DataFrame, events: pd.DataFrame,
                     calendar: list[str]) -> tuple[pd.DataFrame, dict]:
    event_keys = events[["date", "code", "notice_date", "pdf_url"]]
    signal = universe.merge(event_keys, on=["date", "code"], how="inner",
                            validate="one_to_one")
    if signal.empty:
        raise ValueError("No issuer notice has an eligible 14:50 quote")
    selected = _capacity_events(signal, calendar, COOLDOWN)
    event_index = pd.MultiIndex.from_frame(events[["date", "code"]])
    is_event = pd.MultiIndex.from_frame(universe[["date", "code"]]).isin(
        event_index)
    by_day = {day: frame.sort_values("code") for day, frame in
              universe.loc[~is_event].groupby("date", sort=False)}
    rows = []
    missing_industry = insufficient_peers = 0
    for day, daily in selected.groupby("date", sort=True):
        peers = by_day[day]
        for _, event in daily.iterrows():
            choices = peers.loc[peers.board.eq(event.board)
                                & peers.industry.eq(event.industry)]
            if choices.empty:
                missing_industry += 1
                continue
            choices = choices.loc[
                (choices.avg20_amount / event.avg20_amount).between(.5, 2)
                & (choices.float_mv / event.float_mv).between(.5, 2)
                & (choices.return20_prior_adjusted
                   - event.return20_prior_adjusted).abs().le(.15)
                & (choices.return_1450 - event.return_1450).abs().le(.05)
            ].copy()
            if len(choices) < 3:
                insufficient_peers += 1
                continue
            choices["match_distance"] = _distance(choices, event)
            choices = choices.sort_values(["match_distance", "code"]).head(5)
            if (len(choices) < 3 or not np.isfinite(
                    choices.match_distance).all()):
                insufficient_peers += 1
                continue
            treated = event.copy()
            treated["pair_code"] = event.code
            treated["candidate"] = EVENT
            treated["match_distance"] = 0.0
            rows.append(treated)
            for _, peer in choices.iterrows():
                control = peer.copy()
                control["pair_code"] = event.code
                control["candidate"] = CONTROL
                rows.append(control)
    if not rows:
        raise ValueError("No issuer event has three same-industry controls")
    pairs = pd.DataFrame(rows)
    grouped = pairs.groupby(["date", "pair_code"])
    sizes = grouped.size()
    if (not sizes.between(4, 6).all()
            or not grouped.candidate.nunique().eq(2).all()
            or pairs.duplicated(["date", "pair_code", "code"]).any()
            or not pairs.date.gt(pairs.trade_date).all()):
        raise ValueError("Malformed five-peer issuer investigation matches")
    treated = pairs.loc[pairs.candidate.eq(EVENT)]
    controls = pairs.loc[pairs.candidate.eq(CONTROL)]
    report = {
        "eligible_event_quotes": len(signal),
        "capacity_selected": len(selected),
        "matched_events": len(treated),
        "no_same_industry_board": missing_industry,
        "fewer_than_three_quality_peers": insufficient_peers,
        "peer_count_distribution": (sizes - 1).value_counts().sort_index().to_dict(),
        "median_peer_distance": float(controls.match_distance.median()),
        "industry_counts": treated.industry.value_counts().to_dict(),
        "by_year": {},
    }
    for year in ("2024", "2025"):
        frame = treated.loc[treated.date.str.startswith(year)]
        report["by_year"][year] = {
            "events": len(frame), "days": int(frame.date.nunique()),
            "months": int(frame.date.str[:7].nunique()),
            "peak_month_share": (float(frame.date.str[:7].value_counts().max()
                                       / len(frame)) if len(frame) else None),
        }
    report["input_gate_passed"] = all(
        report["by_year"][year]["events"] >= MIN_EVENTS_PER_YEAR
        and report["by_year"][year]["months"] >= MIN_MONTHS_PER_YEAR
        for year in ("2024", "2025"))
    return pairs, report


def _snapshot_funnel(snapshot_dir: Path, events: pd.DataFrame) -> dict:
    """Report observable exclusions before historical daily/industry joins."""
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.register("event_keys", events[["date", "code"]])
    frame = connection.execute("""
        SELECT e.date, e.code, s.price_1450, s.tradestatus, s.isST,
               s.listing_age_sessions, s.amount_1450, s.reference_gap,
               s.quote_outside_traded_range, s.return20_prior_adjusted
        FROM event_keys e LEFT JOIN read_parquet(?) s USING(date, code)
    """, [str(snapshot_dir / "*.parquet")]).df()
    if len(frame) != len(events):
        raise ValueError("Duplicate 14:50 snapshots for investigation event")
    output = {}
    for year in ("2024", "2025"):
        group = frame.loc[frame.date.str.startswith(year)].copy()
        counts = {"source_events": len(group),
                  "no_snapshot": int(group.price_1450.isna().sum()),
                  "snapshot_isST": int(group.isST.eq(1).sum())}
        stages = (
            ("snapshot", group.price_1450.notna()),
            ("trading", group.tradestatus.eq(1)),
            ("non_ST", group.isST.eq(0)),
            ("listing_age_20", group.listing_age_sessions.ge(20)),
            ("amount_30m", group.amount_1450.ge(30_000_000)),
            ("clean_quote", ~group.reference_gap.fillna(True)
             & ~group.quote_outside_traded_range.fillna(True)),
            ("price_5", group.price_1450.ge(5)),
            ("prior_momentum", group.return20_prior_adjusted.notna()),
        )
        selected = pd.Series(True, index=group.index)
        for name, mask in stages:
            selected &= mask
            counts[name] = int(selected.sum())
        output[year] = counts
    return output


def build(source_dir: Path, snapshot_dir: Path, daily_dir: Path,
          calendar_path: Path, industry_path: Path, output_dir: Path) -> dict:
    calendar = trading_dates(calendar_path, "2024-01-01", "2026-01-15")
    events, source = _events(source_dir, calendar)
    output_dir.mkdir(parents=True, exist_ok=True)
    funnel = _snapshot_funnel(snapshot_dir, events)
    lagged_industry = _lag_industry(industry_path,
                                    output_dir / "lagged_industry.parquet")
    universe = _universe(snapshot_dir, daily_dir, lagged_industry,
                         events, calendar)
    main, main_report = _five_peer_pairs(universe, events, calendar)
    secondary, secondary_report = _match(
        universe, events, calendar, False, EVENT,
        cooldown_sessions=COOLDOWN)
    audit = {"source": source, "snapshot_funnel": funnel,
             "universe_stock_days": len(universe),
             "main": main_report, "one_to_one": secondary_report,
             "note": "Input feasibility only; no post-signal returns opened"}
    main.to_parquet(output_dir / "pairs.parquet", index=False,
                    compression="zstd")
    secondary.to_parquet(output_dir / "one_to_one_pairs.parquet", index=False,
                         compression="zstd")
    (output_dir / "input_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(
        "data/research/regulatory_investigation"))
    parser.add_argument("--snapshots", type=Path, default=Path(
        "data/research/market_snapshots_ci"))
    parser.add_argument("--daily", type=Path, default=Path(
        "data/baostock/market_2020_2026/daily"))
    parser.add_argument("--calendar", type=Path, default=Path(
        "data/baostock/market_2020_2026/metadata/calendar.parquet"))
    parser.add_argument("--industry", type=Path, default=Path(
        "data/research/industry_intervals.parquet"))
    parser.add_argument("--output", type=Path, default=Path(
        "data/research/regulatory_investigation"))
    args = parser.parse_args()
    result = build(args.source, args.snapshots, args.daily, args.calendar,
                   args.industry, args.output)
    print({"source": result["source"], "main": result["main"],
           "one_to_one": result["one_to_one"]})


if __name__ == "__main__":
    main()
