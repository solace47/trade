"""Freeze next-session first-insider-buy pairs without future outcomes."""

from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path
import re

import pandas as pd

from .buyback_inputs import _match, _universe
from .exchange_public_events import trading_dates
from .first_insider_buy_source import strict_first_title


EVENT = "first_actual_insider_buy"
RELATED_TITLE = re.compile(r"一致行动人|控制的企业|间接控股股东")


def _events(source_dir: Path, calendar: list[str]) -> tuple[pd.DataFrame, dict]:
    frames = []
    source_counts = {}
    for year in (2024, 2025):
        index = pd.read_parquet(source_dir / f"title_candidates_{year}.parquet")
        checked = pd.read_json(source_dir / f"pdf_audit_{year}.jsonl",
                               lines=True)[["pdf_url", "status", "buy_date",
                                            "shares"]]
        if (index.empty or index.pdf_url.duplicated().any()
                or checked.pdf_url.duplicated().any()
                or not index.title.map(strict_first_title).all()):
            raise ValueError("Malformed first-insider-buy original index")
        joined = index.merge(checked, on="pdf_url", how="left",
                             validate="one_to_one")
        if joined.status.isna().any():
            raise ValueError("First-insider-buy original lacks PDF audit")
        good = joined.loc[joined.status.eq("ok")].copy()
        if (good.buy_date.isna().any() or good.shares.le(0).any()
                or good.buy_date.gt(good.notice_date).any()):
            raise ValueError("Unconfirmed executed purchase or future buy date")
        source_counts[str(year)] = {
            "title_candidates": len(index), "original_pdf_confirmed": len(good),
            "unconfirmed": joined.status.value_counts().to_dict(),
        }
        frames.append(good)
    events = pd.concat(frames, ignore_index=True)
    duplicate = events.duplicated(["code", "notice_date"], keep=False)
    duplicate_rows = int(duplicate.sum())
    events = events.loc[~duplicate].copy()
    events["date"] = events.notice_date.map(
        lambda day: calendar[bisect.bisect_right(calendar, day)])
    events = events.loc[
        events.date.str[:4].eq(events.notice_date.str[:4])
        & events.date.str[5:].le("12-17")
    ].copy()
    same_signal = events.duplicated(["date", "code"], keep=False)
    same_signal_rows = int(same_signal.sum())
    events = events.loc[~same_signal].copy()
    if (events.empty or events.duplicated(["date", "code"]).any()
            or not events.date.gt(events.notice_date).all()):
        raise ValueError("First-insider-buy signal did not lag official notice")
    return events, {"source": source_counts,
                    "duplicate_code_notice_rows_removed": duplicate_rows,
                    "duplicate_code_signal_rows_removed": same_signal_rows,
                    "mapped_events": len(events)}


def _tags(events: pd.DataFrame, calendar: list[str],
          plan_dir: Path) -> pd.DataFrame:
    plans = pd.concat([
        pd.read_json(plan_dir / f"pdf_audit_{year}.jsonl", lines=True)[
            ["code", "notice_date", "status"]]
        for year in (2024, 2025)
    ], ignore_index=True)
    plans = plans.loc[plans.status.eq("ok")]
    dates_by_code = plans.groupby("code").notice_date.apply(
        lambda dates: sorted(set(dates))).to_dict()
    position = {day: i for i, day in enumerate(calendar)}
    rows = []
    for event in events.itertuples(index=False):
        known = dates_by_code.get(event.code, [])
        offset = bisect.bisect_left(known, event.notice_date) - 1
        prior = known[offset] if offset >= 0 else None
        plan_signal = (calendar[bisect.bisect_right(calendar, prior)]
                       if prior is not None else None)
        gap = (position[event.date] - position[plan_signal]
               if plan_signal is not None else None)
        if gap is not None and gap < 0:
            raise ValueError("Future insider plan entered earlier signal")
        rows.append({"date": event.date, "pair_code": event.code,
                     "buy_date": event.buy_date,
                     "related_actor_title": bool(RELATED_TITLE.search(event.title)),
                     "buy_notice_lag_days": int((pd.Timestamp(event.notice_date)
                                                 - pd.Timestamp(event.buy_date)).days),
                     "prior_plan_notice": prior,
                     "plan_gap_sessions": gap,
                     "recent_plan_10": gap is not None and gap <= 10})
    return pd.DataFrame(rows)


def _tag_pairs(pairs: pd.DataFrame, tags: pd.DataFrame) -> pd.DataFrame:
    result = pairs.merge(tags, on=["date", "pair_code"],
                         validate="many_to_one")
    if len(result) != len(pairs):
        raise ValueError("A first-insider-buy pair lacks provenance tags")
    return result


def build(source_dir: Path, snapshot_dir: Path, daily_dir: Path,
          calendar_path: Path, industry_path: Path, output_dir: Path,
          plan_dir: Path) -> dict:
    calendar = trading_dates(calendar_path, "2024-01-01", "2026-01-15")
    events, source = _events(source_dir, calendar)
    universe = _universe(snapshot_dir, daily_dir, industry_path,
                         events, calendar)
    main, main_report = _match(universe, events, calendar, False, EVENT)
    within, within_report = _match(universe, events, calendar, True, EVENT)
    tags = _tags(events, calendar, plan_dir)
    main = _tag_pairs(main, tags)
    within = _tag_pairs(within, tags)
    treated = main.loc[main.candidate.eq(EVENT)]
    audit = {"source": source, "universe_stock_days": len(universe),
             "main": main_report, "same_industry": within_report,
             "related_actor_titles": int(treated.related_actor_title.sum()),
             "known_prior_plan": int(treated.prior_plan_notice.notna().sum()),
             "known_recent_plan_10": int(treated.recent_plan_10.sum()),
             "notice_lag_over_7_days": int(treated.buy_notice_lag_days.gt(7).sum()),
             "note": "First actual insider buy inputs only; no future returns opened"}
    output_dir.mkdir(parents=True, exist_ok=True)
    main.to_parquet(output_dir / "pairs.parquet", index=False,
                    compression="zstd")
    within.to_parquet(output_dir / "industry_pairs.parquet", index=False,
                      compression="zstd")
    (output_dir / "input_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(
        "data/research/first_insider_buy"))
    parser.add_argument("--snapshots", type=Path, default=Path(
        "data/research/market_snapshots_ci"))
    parser.add_argument("--daily", type=Path, default=Path(
        "data/baostock/market_2020_2026/daily"))
    parser.add_argument("--calendar", type=Path, default=Path(
        "data/baostock/market_2020_2026/metadata/calendar.parquet"))
    parser.add_argument("--industry", type=Path, default=Path(
        "data/research/industry_intervals.parquet"))
    parser.add_argument("--plan-source", type=Path, default=Path(
        "data/research/insider_buy"))
    parser.add_argument("--output", type=Path, default=Path(
        "data/research/first_insider_buy"))
    args = parser.parse_args()
    result = build(args.source, args.snapshots, args.daily, args.calendar,
                   args.industry, args.output, args.plan_source)
    print({"source": result["source"], "main": result["main"],
           "same_industry": result["same_industry"],
           "known_prior_plan": result["known_prior_plan"]})


if __name__ == "__main__":
    main()
