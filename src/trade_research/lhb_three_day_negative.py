"""Input-only audit for next-session three-day negative abnormal disclosures.

The hypothesis and matching constraints were fixed in
docs/lhb-three-day-negative-plan.md before opening this event's returns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .buyback_inputs import _universe
from .exchange_public_events import load_complete_notices, trading_dates


CAPACITY = 5
COOLDOWN = 10
MAX_DISTANCE = 6.0
EVENT = "three_day_negative_notice"
CONTROL = "same_day_no_recent_notice"


def _events(notices: pd.DataFrame, sessions: list[str]) -> pd.DataFrame:
    sizes = notices.groupby(["trade_date", "code"]).size().rename("reason_count")
    rows = notices.join(sizes, on=["trade_date", "code"])
    is_target = (
        (rows.board.eq("main") & rows.notice_type.eq("2")
         & rows.code.str.startswith("sh.60"))
        | (rows.board.eq("szse") & rows.code.str.startswith("sz.00")
           & rows.notice_type.str.startswith("异常期间价格跌幅偏离值累计达到"))
    )
    events = rows.loc[is_target & rows.reason_count.eq(1),
                      ["trade_date", "code"]].copy()
    following = dict(zip(sessions[:-1], sessions[1:]))
    position = {day: index for index, day in enumerate(sessions)}
    events["date"] = events.trade_date.map(following)
    events = events.loc[
        events.date.notna()
        & events.date.str[:4].eq(events.trade_date.str[:4])
        & events.date.str[5:].le("12-17")
        & events.trade_date.map(lambda day: position[day] >= 2
                                and sessions[position[day] - 2][:4] == day[:4])
    ].copy()
    if events.empty or events.duplicated(["date", "code"]).any():
        raise ValueError("No unique next-session negative abnormal events")
    return events


def _recent_notice_keys(notices: pd.DataFrame, sessions: list[str]) -> pd.DataFrame:
    position = {day: index for index, day in enumerate(sessions)}
    original = notices[["trade_date", "code"]].drop_duplicates()
    rows = []
    for offset in range(3):
        shifted = original.copy()
        shifted["trade_date"] = shifted.trade_date.map(
            lambda day: sessions[position[day] + offset]
            if position[day] + offset < len(sessions) else None)
        rows.append(shifted.dropna())
    return pd.concat(rows, ignore_index=True).drop_duplicates()


def _prior_three(daily_dir: Path, keys: pd.DataFrame,
                 sessions: list[str]) -> pd.DataFrame:
    c = duckdb.connect()
    c.execute("SET threads = 4")
    c.register("session_index", pd.DataFrame(
        {"trade_date": sessions, "session_number": range(len(sessions))}))
    c.register("keys", keys[["trade_date", "code"]].drop_duplicates())
    result = c.execute("""
        WITH raw AS (
          SELECT d.date AS trade_date, d.code, d.pctChg, d.turn AS t_turn,
                 d.amount AS t_amount, d.tradestatus, d.isST,
                 s.session_number
          FROM read_parquet(?) d JOIN session_index s
            ON s.trade_date = d.date
          WHERE d.code LIKE 'sh.60%' OR d.code LIKE 'sz.00%'
        ), lagged AS (
          SELECT *,
                 LAG(pctChg, 1) OVER w AS return_one_prior,
                 LAG(pctChg, 2) OVER w AS return_two_prior,
                 LAG(session_number, 2) OVER w AS two_prior_session,
                 LAG(tradestatus, 1) OVER w AS trade_one_prior,
                 LAG(tradestatus, 2) OVER w AS trade_two_prior,
                 LAG(isST, 1) OVER w AS st_one_prior,
                 LAG(isST, 2) OVER w AS st_two_prior
          FROM raw WINDOW w AS (PARTITION BY code ORDER BY trade_date)
        )
        SELECT l.trade_date, l.code, l.t_turn, l.t_amount,
               l.pctChg / 100.0 AS return_t,
               (1 + l.pctChg / 100.0)
               * (1 + l.return_one_prior / 100.0)
               * (1 + l.return_two_prior / 100.0) - 1 AS return_three
        FROM lagged l JOIN keys k USING (trade_date, code)
        WHERE l.session_number - l.two_prior_session = 2
          AND l.tradestatus = 1 AND l.trade_one_prior = 1
          AND l.trade_two_prior = 1
          AND l.isST = 0 AND l.st_one_prior = 0 AND l.st_two_prior = 0
          AND l.pctChg IS NOT NULL AND l.return_one_prior IS NOT NULL
          AND l.return_two_prior IS NOT NULL
          AND l.t_amount > 0 AND l.t_turn > 0
    """, [str(daily_dir / "*.parquet")]).df()
    if result.duplicated(["trade_date", "code"]).any():
        raise ValueError("Duplicate three-day daily input")
    return result


def _capacity(signal: pd.DataFrame, sessions: list[str]) -> pd.DataFrame:
    position = {day: index for index, day in enumerate(sessions)}
    last_kept: dict[str, int] = {}
    chosen = []
    for day, daily in signal.groupby("date", sort=True):
        index = position[day]
        valid = daily.loc[daily.code.map(
            lambda code: index - last_kept.get(code, -1000) > COOLDOWN)]
        selected = valid.sort_values(["t_amount", "code"],
                                     ascending=[False, True]).head(CAPACITY)
        chosen.append(selected)
        last_kept.update({code: index for code in selected.code})
    return pd.concat(chosen, ignore_index=True)


def _match(signals: pd.DataFrame, pool: pd.DataFrame,
           same_industry: bool) -> tuple[pd.DataFrame, dict]:
    controls_by_day = {day: frame.copy() for day, frame in
                       pool.loc[~pool.recent_notice].groupby("date")}
    paired = []
    unmatched = 0
    for day, daily in signals.groupby("date", sort=True):
        choices = controls_by_day[day]
        for _, event in daily.iterrows():
            peers = choices.loc[
                choices.board.eq(event.board)
                & (choices.float_mv / event.float_mv).between(.5, 2)
                & (choices.avg20_amount / event.avg20_amount).between(.5, 2)
                & (choices.return20_prior_adjusted
                   - event.return20_prior_adjusted).abs().le(.15)
                & (choices.return_three - event.return_three).abs().le(.03)
                & (choices.return_t - event.return_t).abs().le(.02)
                & (choices.return_1450 - event.return_1450).abs().le(.015)
                & (choices.open_gap - event.open_gap).abs().le(.03)
            ]
            if same_industry:
                peers = peers.loc[peers.industry.eq(event.industry)]
            if peers.empty:
                unmatched += 1
                continue
            distance = (
                np.abs(np.log(peers.float_mv / event.float_mv)) / np.log(2)
                + np.abs(np.log(peers.avg20_amount / event.avg20_amount)) / np.log(2)
                + (peers.return20_prior_adjusted
                   - event.return20_prior_adjusted).abs() / .15
                + (peers.return_three - event.return_three).abs() / .03
                + (peers.return_t - event.return_t).abs() / .02
                + (peers.return_1450 - event.return_1450).abs() / .015
                + (peers.open_gap - event.open_gap).abs() / .03
            )
            best = distance.idxmin()
            if not np.isfinite(distance.loc[best]) or distance.loc[best] > MAX_DISTANCE:
                unmatched += 1
                continue
            left = event.copy()
            right = peers.loc[best].copy()
            for row, label in ((left, EVENT), (right, CONTROL)):
                row["candidate"] = label
                row["pair_code"] = event.code
                row["match_distance"] = float(distance.loc[best])
                paired.append(row)
            choices = choices.drop(index=best)
    result = pd.DataFrame(paired)
    if not result.empty:
        counts = result.groupby(["date", "pair_code"]).candidate.nunique()
        if not counts.eq(2).all() or len(result) != 2 * len(counts):
            raise ValueError("Incomplete three-day negative pairs")
    treated = result.loc[result.candidate.eq(EVENT)] if not result.empty else result
    report = {"pairs": len(treated), "unmatched": unmatched,
              "days": int(treated.date.nunique()) if not treated.empty else 0,
              "by_year": {year: int(treated.date.str.startswith(year).sum())
                          if not treated.empty else 0
                          for year in ("2024", "2025")}}
    return result, report


def _gate_coverage(signals: pd.DataFrame, pool: pd.DataFrame) -> dict:
    """Count signals with at least one unused-agnostic peer at each input gate."""
    controls_by_day = {day: frame for day, frame in
                       pool.loc[~pool.recent_notice].groupby("date")}
    names = ("same_board", "prior_three_within_3pp", "prior_day_within_2pp",
             "signal_return_within_1_5pp", "open_gap_within_3pp",
             "size_ratio", "liquidity_ratio", "prior_twenty_within_15pp")
    counts = dict.fromkeys(names, 0)
    for _, event in signals.iterrows():
        choices = controls_by_day.get(event.date)
        if choices is None:
            continue
        masks = (
            choices.board.eq(event.board),
            (choices.return_three - event.return_three).abs().le(.03),
            (choices.return_t - event.return_t).abs().le(.02),
            (choices.return_1450 - event.return_1450).abs().le(.015),
            (choices.open_gap - event.open_gap).abs().le(.03),
            (choices.float_mv / event.float_mv).between(.5, 2),
            (choices.avg20_amount / event.avg20_amount).between(.5, 2),
            (choices.return20_prior_adjusted
             - event.return20_prior_adjusted).abs().le(.15),
        )
        keep = pd.Series(True, index=choices.index)
        for name, mask in zip(names, masks):
            keep &= mask
            counts[name] += bool(keep.any())
    return counts


def build(snapshot_dir: Path, daily_dir: Path, calendar_path: Path,
          industry_path: Path, sse_dir: Path, szse_export: Path,
          output_dir: Path) -> dict:
    sessions = trading_dates(calendar_path, "2024-01-01", "2026-01-15")
    notices = load_complete_notices(calendar_path, sse_dir, szse_export)
    events = _events(notices, sessions)
    universe = _universe(snapshot_dir, daily_dir, industry_path,
                         events, sessions)
    universe = universe.loc[universe.board.isin(("sh_main", "sz_main"))].copy()
    daily = _prior_three(daily_dir, universe[["trade_date", "code"]], sessions)
    pool = universe.merge(daily, on=["trade_date", "code"],
                          validate="one_to_one")
    pool = pool.loc[
        pool.open_gap.between(-.05, .05)
        & pool.return_1450.between(-.08, .08)
    ].copy()
    event_keys = pd.MultiIndex.from_frame(events[["trade_date", "code"]])
    pool["is_event"] = pd.MultiIndex.from_frame(
        pool[["trade_date", "code"]]).isin(event_keys)
    recent = _recent_notice_keys(notices, sessions)
    recent_keys = pd.MultiIndex.from_frame(recent[["trade_date", "code"]])
    pool["recent_notice"] = pd.MultiIndex.from_frame(
        pool[["trade_date", "code"]]).isin(recent_keys)
    signal = pool.loc[pool.is_event]
    if signal.empty:
        raise ValueError("No eligible negative abnormal notice quotes")
    selected = _capacity(signal, sessions)
    main, main_report = _match(selected, pool, False)
    industry, industry_report = _match(selected, pool, True)
    report = {"source_events": len(events), "eligible_event_quotes": len(signal),
              "capacity_selected": len(selected), "universe_stock_days": len(pool),
              "gate_coverage": _gate_coverage(selected, pool),
              "main": main_report, "same_industry": industry_report,
              "note": "Input only; no future stock returns or execution outcomes read"}
    output_dir.mkdir(parents=True, exist_ok=True)
    main.to_parquet(output_dir / "pairs.parquet", index=False, compression="zstd")
    industry.to_parquet(output_dir / "industry_pairs.parquet", index=False,
                        compression="zstd")
    (output_dir / "input_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


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
    parser.add_argument("--sse", type=Path, default=Path(
        "data/research/lhb/sse_daily"))
    parser.add_argument("--szse", type=Path, default=Path(
        "data/research/lhb/szse_2024_2025.xlsx"))
    parser.add_argument("--output", type=Path, default=Path(
        "data/research/lhb_three_day_negative"))
    args = parser.parse_args()
    report = build(args.snapshots, args.daily, args.calendar, args.industry,
                   args.sse, args.szse, args.output)
    print(report)


if __name__ == "__main__":
    main()
