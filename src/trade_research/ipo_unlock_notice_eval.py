"""Evaluate frozen post-notice IPO-unlock pairs with original minutes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .buyback_eval import _score, _summary
from .buyback_inputs import CONTROL
from .buyback_reprice import _exact
from .exchange_public_events import trading_dates
from .ipo_unlock_notice_inputs import EVENT, MAX_CURRENT_RETURN_GAP
from .strategy_scan import _stressed_returns


FREEZE_NOTE = "Original IPO unlock notice inputs only; no future returns opened"


def _check(pairs: pd.DataFrame, expected: dict,
           session: dict[str, int]) -> None:
    if (pairs.empty or len(pairs) != 2 * expected["matched_pairs"]
            or pairs.duplicated(["date", "code"]).any()
            or set(pairs.candidate) != {EVENT, CONTROL}):
        raise ValueError("Frozen post-notice pair membership changed")
    group = pairs.groupby(["date", "pair_code"]).candidate.agg(
        ["size", "nunique"])
    if (not group["size"].eq(2).all()
            or not group["nunique"].eq(2).all()):
        raise ValueError("IPO-unlock notice lacks exactly one control")
    event = pairs.loc[pairs.candidate.eq(EVENT)]
    control = pairs.loc[pairs.candidate.eq(CONTROL)]
    joined = event.merge(control, on=["date", "pair_code"],
                         suffixes=("_event", "_peer"),
                         validate="one_to_one")
    if (event.date.nunique() != expected["matched_days"]
            or event.groupby("date").size().gt(5).any()
            or not event.date.str[:4].isin(("2024", "2025")).all()
            or event.date.str[5:].gt("12-17").any()
            or not event.date.gt(event.notice_date).all()
            or not event.date.lt(event.unlock_date).all()
            or not set(event.date).issubset(session)
            or not set(event.unlock_date).issubset(session)
            or any(session[unlock] - session[day] < 2
                   for day, unlock in zip(event.date, event.unlock_date))
            or (joined.return_1450_event
                - joined.return_1450_peer).abs().gt(
                    MAX_CURRENT_RETURN_GAP + 1e-12).any()):
        raise ValueError("IPO-unlock notice timing or pairing changed")
    for year in ("2024", "2025"):
        part = event.loc[event.date.str.startswith(year)]
        if (len(part) != expected["by_year"][year]["pairs"]
                or part.date.nunique() != expected["by_year"][year]["days"]):
            raise ValueError("IPO-unlock notice yearly sample changed")


def _attach(raw: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    unlock = pairs.loc[pairs.candidate.eq(EVENT),
                       ["date", "pair_code", "unlock_date"]]
    rows = raw.merge(pairs[["date", "code", "candidate", "pair_code",
                            "industry"]],
                     on=["date", "code"], validate="many_to_one")
    rows = rows.merge(unlock, on=["date", "pair_code"],
                      validate="many_to_one")
    if len(rows) != 4 * len(pairs):
        raise ValueError("Raw-minute IPO-unlock notice leg is absent")
    rows["cash_return"] = np.where(rows.quality_clean_exit,
                                   rows.net_return, 0.0)
    rows["cash_stress10"] = 0.0
    clean = rows.quality_clean_exit
    rows.loc[clean, "cash_stress10"] = _stressed_returns(rows.loc[clean], 10)
    if (not np.isfinite(rows.cash_return).all()
            or not np.isfinite(rows.cash_stress10).all()):
        raise ValueError("IPO-unlock notice cash return is nonfinite")
    return rows


def _exit_timing(rows: pd.DataFrame, year: str) -> dict:
    frame = rows.loc[rows.date.str.startswith(year)
                     & rows.horizon.eq(1)]
    return {
        "clean_exit_on_or_after_unlock": int((
            frame.quality_clean_exit
            & frame.exit_date.ge(frame.unlock_date)).sum()),
        "clean_exit_total": int(frame.quality_clean_exit.sum()),
    }


def _posthoc_peer_swap(main: pd.DataFrame,
                       industry: pd.DataFrame) -> dict:
    def pair_cash(rows: pd.DataFrame) -> pd.DataFrame:
        event = rows.loc[rows.candidate.eq(EVENT),
                         ["date", "code", "cash_return"]].rename(
            columns={"code": "pair_code", "cash_return": "event_cash"})
        peer = rows.loc[rows.candidate.eq(CONTROL),
                        ["date", "pair_code", "cash_return"]].rename(
            columns={"cash_return": "peer_cash"})
        return event.merge(peer, on=["date", "pair_code"],
                           validate="one_to_one")

    shared = pair_cash(main).merge(pair_cash(industry),
                                   on=["date", "pair_code"],
                                   suffixes=("_broad", "_industry"),
                                   validate="one_to_one")
    if (shared.empty or not np.isclose(shared.event_cash_broad,
                                       shared.event_cash_industry).all()):
        raise ValueError("Post-notice common-event returns differ")
    result = {}
    for year in ("2024", "2025"):
        part = shared.loc[shared.date.str.startswith(year)]
        daily = part.groupby("date")[["event_cash_broad", "peer_cash_broad",
                                      "peer_cash_industry"]].mean()
        result[year] = {
            "shared_events": len(part), "days": len(daily),
            "event_cash_mean": float(daily.event_cash_broad.mean()),
            "broad_peer_cash_mean": float(daily.peer_cash_broad.mean()),
            "industry_peer_cash_mean": float(
                daily.peer_cash_industry.mean()),
            "broad_edge_mean": float((daily.event_cash_broad
                                      - daily.peer_cash_broad).mean()),
            "industry_edge_mean": float((daily.event_cash_broad
                                         - daily.peer_cash_industry).mean()),
        }
    return result


def _load(source_dir: Path, calendar_path: Path
          ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    audit = json.loads((source_dir / "input_audit.json").read_text(
        encoding="utf-8"))
    if audit.get("note") != FREEZE_NOTE:
        raise ValueError("IPO-unlock notice input list was not frozen")
    calendar = trading_dates(calendar_path, "2024-01-01", "2026-01-15")
    session = {day: index for index, day in enumerate(calendar)}
    main = pd.read_parquet(source_dir / "pairs.parquet")
    industry = pd.read_parquet(source_dir / "industry_pairs.parquet")
    events = pd.read_parquet(source_dir / "events.parquet")
    unlock = events[["date", "code", "unlock_date"]].rename(
        columns={"code": "pair_code"})
    if unlock.duplicated(["date", "pair_code"]).any():
        raise ValueError("Frozen IPO-unlock notice events are duplicated")
    main = main.merge(unlock, on=["date", "pair_code"],
                      validate="many_to_one")
    industry = industry.merge(unlock, on=["date", "pair_code"],
                              validate="many_to_one")
    _check(main, audit["main"], session)
    _check(industry, audit["same_industry"], session)
    return main, industry, audit


def prepare(source_dir: Path, calendar_path: Path,
            signals_path: Path) -> int:
    main, industry, _ = _load(source_dir, calendar_path)
    unique = pd.concat([main, industry], ignore_index=True).drop_duplicates(
        ["date", "code"])
    signals_path.parent.mkdir(parents=True, exist_ok=True)
    unique.to_parquet(signals_path, index=False, compression="zstd")
    return len(unique)


def evaluate(source_dir: Path, raw_path: Path, outcome_dir: Path,
             issues_dir: Path, report_path: Path,
             calendar_path: Path) -> dict:
    main, industry, _ = _load(source_dir, calendar_path)
    unique = pd.concat([main, industry], ignore_index=True).drop_duplicates(
        ["date", "code"])
    raw = pd.read_parquet(raw_path)
    if (len(raw) != len(unique) * 4
            or set(raw.target_notional) != {20_000.0, 100_000.0}
            or set(raw.horizon) != {1, 5}
            or raw.duplicated(["date", "code", "target_notional",
                               "horizon"]).any()):
        raise ValueError("Original-minute post-notice list incomplete")
    stored = pd.concat([
        _score(unique, outcome_dir, issues_dir, horizon)
        for horizon in (1, 5)
    ], ignore_index=True)
    exact = _exact(stored, raw.loc[raw.target_notional.eq(100_000)])
    selected = _attach(raw, main)
    within = _attach(raw, industry)
    calendar = trading_dates(calendar_path, "2024-01-01", "2026-01-15")
    session = {day: index for index, day in enumerate(calendar)}
    report = {"note": "Exploratory 2024/2025; 2025 not blind; 2026 not read",
              "exact_100k_rows": exact,
              "main": {}, "same_industry": {}}
    for size in (20_000, 100_000):
        report["main"][str(size)] = {}
        report["same_industry"][str(size)] = {}
        for horizon in (1, 5):
            part = selected.loc[selected.target_notional.eq(size)
                                & selected.horizon.eq(horizon)]
            subset = within.loc[within.target_notional.eq(size)
                                & within.horizon.eq(horizon)]
            report["main"][str(size)][str(horizon)] = {
                year: {segment: _summary(part, year, segment, EVENT)
                       for segment in ("full", "H1", "H2")}
                for year in ("2024", "2025")
            }
            report["same_industry"][str(size)][str(horizon)] = {
                year: _summary(subset, year, "full", EVENT)
                for year in ("2024", "2025")
            }
    report["t1_exit_timing_20k"] = {
        year: _exit_timing(selected.loc[
            selected.target_notional.eq(20_000)], year)
        for year in ("2024", "2025")
    }
    report["posthoc_t5_unlock_overlap"] = {}
    events = main.loc[main.candidate.eq(EVENT)]
    for year in ("2024", "2025"):
        part = events.loc[events.date.str.startswith(year)]
        report["posthoc_t5_unlock_overlap"][year] = {
            "events": len(part),
            "scheduled_t5_on_or_after_unlock": sum(
                session[unlock] - session[day] <= 5
                for day, unlock in zip(part.date, part.unlock_date)),
        }
    report["posthoc_common_event_peer_swap_20k_t5"] = _posthoc_peer_swap(
        selected.loc[selected.target_notional.eq(20_000)
                     & selected.horizon.eq(5)],
        within.loc[within.target_notional.eq(20_000)
                   & within.horizon.eq(5)])
    t1 = selected.loc[selected.target_notional.eq(20_000)
                      & selected.horizon.eq(1)].copy()
    crossed = t1.loc[t1.quality_clean_exit
                     & t1.exit_date.ge(t1.unlock_date),
                     ["date", "pair_code"]].drop_duplicates()
    t1 = t1.merge(crossed.assign(crossed=True),
                  on=["date", "pair_code"], how="left")
    t1 = t1.loc[t1.crossed.isna()]
    report["posthoc_t1_excluding_delayed_exit_past_unlock"] = {
        year: _summary(t1, year, "full", EVENT)
        for year in ("2024", "2025")
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(
        "data/research/unlock_notice"))
    parser.add_argument("--raw", type=Path, default=Path(
        "data/research/unlock_notice/repriced.parquet"))
    parser.add_argument("--outcomes", type=Path, default=Path(
        "data/research/market_outcomes_ci"))
    parser.add_argument("--issues", type=Path, default=Path(
        "data/research/market_issues_ci"))
    parser.add_argument("--report", type=Path, default=Path(
        "data/research/unlock_notice/report.json"))
    parser.add_argument("--calendar", type=Path, default=Path(
        "data/baostock/market_2020_2026/metadata/calendar.parquet"))
    parser.add_argument("--prepare-signals", type=Path)
    args = parser.parse_args()
    if args.prepare_signals:
        print({"unique_signals": prepare(args.source, args.calendar,
                                         args.prepare_signals)})
        return
    report = evaluate(args.source, args.raw, args.outcomes,
                      args.issues, args.report, args.calendar)
    print({"exact_100k_rows": report["exact_100k_rows"],
           "main_20k_t1": {
               year: report["main"]["20000"]["1"][year]["full"]
               for year in ("2024", "2025")}})


if __name__ == "__main__":
    main()
