"""Freeze next-session discounted block-trade events and same-day controls.

This stage reads only original exchange trades, unadjusted event-day bars,
historical industries, prior daily features, and signal-time 14:50 snapshots.
It never opens future return or execution files.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .block_trade_source import validate_saved
from .buyback_inputs import _universe
from .exchange_public_events import trading_dates


MIN_BLOCK_AMOUNT_WAN = 1_000.0
MAX_PRICE_TO_LOW = 0.99
CAPACITY = 5
COOLDOWN = 5
EVENT = "discounted_block"
CONTROL = "same_day_same_industry_no_block"


def _is_a_share(code: str) -> bool:
    return code.startswith(("sh.60", "sh.68", "sz.00", "sz.30"))


def _load_trades(source_dir: Path, dates: list[str]) -> tuple[pd.DataFrame, dict]:
    records = []
    raw = {"sse_rows": 0, "szse_rows": 0, "non_a_rows": 0}
    for day in dates:
        saved = validate_saved(source_dir / f"{day}.json", day)
        for exchange in ("sse", "szse"):
            raw[f"{exchange}_rows"] += len(saved[exchange])
            for row in saved[exchange]:
                code = ("sh." + row["stockid"] if exchange == "sse"
                        else "sz." + row["zqdh"])
                if not _is_a_share(code):
                    raw["non_a_rows"] += 1
                    continue
                price = row["tradeprice"] if exchange == "sse" else row["cjjg"]
                amount = (row["tradeamount"] if exchange == "sse"
                          else row["cjjenew"])
                records.append({"trade_date": day, "code": code,
                                "price": float(price.replace(",", "")),
                                "amount_wan": float(amount.replace(",", "")),
                                "exchange": exchange})
    frame = pd.DataFrame(records)
    if (frame.empty or not frame.trade_date.isin(dates).all()
            or not np.isfinite(frame[["price", "amount_wan"]].to_numpy()).all()
            or frame[["price", "amount_wan"]].le(0).any().any()):
        raise ValueError("Malformed A-share block trade source")
    raw["a_share_rows"] = len(frame)
    raw["a_share_stock_days"] = len(frame[["trade_date", "code"]].drop_duplicates())
    return frame, raw


def _attach_event_bar(trades: pd.DataFrame, daily_dir: Path) -> tuple[pd.DataFrame, dict]:
    connection = duckdb.connect()
    connection.register("trades", trades)
    joined = connection.execute("""
        SELECT b.*, d.low, d.high, d.close, d.tradestatus, d.adjustflag
        FROM trades b
        LEFT JOIN read_parquet(?) d
          ON b.trade_date = d.date AND b.code = d.code
    """, [str(daily_dir / "*.parquet")]).df()
    if len(joined) != len(trades):
        raise ValueError("Duplicate event-day daily bars joined to block trades")
    missing = int(joined.low.isna().sum())
    valid = joined.loc[
        joined.low.notna() & joined.adjustflag.eq(3)
        & joined.tradestatus.eq(1) & joined.low.gt(0)
        & joined.high.ge(joined.low) & joined.close.ge(joined.low)
        & joined.close.le(joined.high)
    ].copy()
    if valid.empty or not np.isfinite(valid[["low", "high", "close"]].to_numpy()).all():
        raise ValueError("No valid unadjusted event-day block trade bars")
    return valid, {"missing_daily_rows": missing,
                   "invalid_or_missing_daily_rows": len(joined) - len(valid),
                   "valid_daily_rows": len(valid)}


def _discount_events(valid: pd.DataFrame, calendar: list[str]) -> pd.DataFrame:
    below = valid.loc[valid.price.lt(MAX_PRICE_TO_LOW * valid.low)].copy()
    if below.empty:
        raise ValueError("No block trade below the frozen low-price threshold")
    below["price_times_amount"] = below.price * below.amount_wan
    grouped = below.groupby(["trade_date", "code"], as_index=False).agg(
        qualifying_amount_wan=("amount_wan", "sum"),
        price_times_amount=("price_times_amount", "sum"),
        event_low=("low", "first"),
        block_rows=("price", "size"),
    )
    grouped = grouped.loc[grouped.qualifying_amount_wan.ge(
        MIN_BLOCK_AMOUNT_WAN)].copy()
    grouped["weighted_block_price"] = (
        grouped.price_times_amount / grouped.qualifying_amount_wan)
    grouped["discount_depth"] = (
        1 - grouped.weighted_block_price / grouped.event_low)
    next_day = dict(zip(calendar[:-1], calendar[1:]))
    grouped["date"] = grouped.trade_date.map(next_day)
    grouped = grouped.loc[grouped.date.notna()].copy()
    grouped = grouped.loc[
        grouped.date.str[:4].eq(grouped.trade_date.str[:4])].copy()
    # The tenth subsequent session must remain in the entry's calendar year.
    session = {day: i for i, day in enumerate(calendar)}
    grouped = grouped.loc[grouped.date.map(
        lambda day: session[day] + 10 < len(calendar)
        and calendar[session[day] + 10][:4] == day[:4])].copy()
    if (grouped.empty or grouped.duplicated(["date", "code"]).any()
            or not grouped.date.gt(grouped.trade_date).all()
            or not grouped.discount_depth.gt(.01).all()
            or not grouped.date.str[:4].isin(("2024", "2025")).all()):
        raise ValueError("Malformed next-session discounted block events")
    return grouped.drop(columns="price_times_amount")


def _select_capacity(signals: pd.DataFrame, calendar: list[str]) -> pd.DataFrame:
    session = {day: i for i, day in enumerate(calendar)}
    last_selected: dict[str, int] = {}
    selected = []
    for day, group in signals.groupby("date", sort=True):
        order = group.sort_values(
            ["discount_depth", "qualifying_amount_wan", "code"],
            ascending=[False, False, True])
        available = order.loc[order.code.map(
            lambda code: session[day] - last_selected.get(code, -1000) > COOLDOWN)]
        chosen = available.head(CAPACITY)
        selected.append(chosen)
        last_selected.update({code: session[day] for code in chosen.code})
    if not selected:
        raise ValueError("No 14:50 eligible discounted block events")
    return pd.concat(selected, ignore_index=True)


def _distance(choices: pd.DataFrame, event: pd.Series) -> pd.Series:
    return (
        np.abs(np.log(choices.avg20_amount / event.avg20_amount)) / .5
        + np.abs(np.log(choices.float_mv / event.float_mv)) / .5
        + np.abs(choices.return20_prior_adjusted
                 - event.return20_prior_adjusted) / .10
        + np.abs(choices.return_1450 - event.return_1450) / .02
    )


def _match(selected: pd.DataFrame, universe: pd.DataFrame,
           blocked_keys: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    candidate = universe.merge(blocked_keys.assign(has_block=True),
                               on=["trade_date", "code"], how="left",
                               validate="many_to_one")
    candidate = candidate.loc[candidate.has_block.isna()].drop(
        columns="has_block")
    by_day = {day: frame.sort_values("code").copy()
              for day, frame in candidate.groupby("date", sort=False)}
    matched = []
    unmatched = {"no_same_industry_board": 0,
                 "outside_input_limits": 0}
    for day, events in selected.groupby("date", sort=True):
        peers = by_day.get(day)
        if peers is None:
            unmatched["no_same_industry_board"] += len(events)
            continue
        peers = peers.copy()
        for _, event in events.iterrows():
            choices = peers.loc[
                peers.industry.eq(event.industry)
                & peers.board.eq(event.board)]
            if choices.empty:
                unmatched["no_same_industry_board"] += 1
                continue
            choices = choices.loc[
                (choices.avg20_amount / event.avg20_amount).between(.5, 2)
                & (choices.float_mv / event.float_mv).between(.5, 2)
                & (choices.return20_prior_adjusted
                   - event.return20_prior_adjusted).abs().le(.15)
                & (choices.return_1450 - event.return_1450).abs().le(.05)]
            if choices.empty:
                unmatched["outside_input_limits"] += 1
                continue
            distances = _distance(choices, event)
            best = choices.assign(distance=distances).sort_values(
                ["distance", "code"]).iloc[0]
            if not math.isfinite(best.distance):
                raise ValueError("Nonfinite block control match distance")
            left = event.copy()
            left["candidate"], left["pair_code"] = EVENT, event.code
            left["match_distance"] = best.distance
            right = best.copy()
            right["candidate"], right["pair_code"] = CONTROL, event.code
            right["match_distance"] = best.distance
            matched.extend((left, right))
            peers = peers.loc[peers.code.ne(best.code)]
    if not matched:
        raise ValueError("No discounted block event has a matched control")
    pairs = pd.DataFrame(matched)
    counts = pairs.groupby(["date", "pair_code"]).candidate.agg(
        ["size", "nunique"])
    if (not counts["size"].eq(2).all() or not counts["nunique"].eq(2).all()
            or pairs.duplicated(["date", "code"]).any()
            or not pairs.date.gt(pairs.trade_date).all()):
        raise ValueError("Malformed block trade event/control pairs")
    return pairs, unmatched


def _lag_industry(industry_path: Path, output: Path) -> Path:
    """Delay a historical classification snapshot past its query day."""
    frame = pd.read_parquet(industry_path)
    if (frame.empty or frame.duplicated(["code", "effective_date"]).any()
            or not {"code", "industry", "effective_date",
                    "next_effective_date"}.issubset(frame.columns)):
        raise ValueError("Historical industry intervals are malformed")
    for field in ("effective_date", "next_effective_date"):
        frame[field] = (pd.to_datetime(frame[field]) + pd.Timedelta(days=1)
                        ).dt.strftime("%Y-%m-%d")
    if frame.effective_date.isna().any():
        raise ValueError("Historical industry effective date is missing")
    frame.to_parquet(output, index=False, compression="zstd")
    return output


def build(source_dir: Path, snapshot_dir: Path, daily_dir: Path,
          calendar_path: Path, industry_path: Path,
          output_dir: Path, source_end: str = "2025-12-31") -> dict:
    if source_end not in ("2024-12-31", "2025-12-31"):
        raise ValueError("Source end must be a complete study calendar year")
    source_days = trading_dates(calendar_path, "2024-01-01", source_end)
    calendar = trading_dates(calendar_path, "2024-01-01", "2026-01-15")
    trades, raw = _load_trades(source_dir, source_days)
    valid, bars = _attach_event_bar(trades, daily_dir)
    events = _discount_events(valid, calendar)
    output_dir.mkdir(parents=True, exist_ok=True)
    lagged_industry = _lag_industry(
        industry_path, output_dir / "lagged_industry.parquet")
    universe = _universe(snapshot_dir, daily_dir, lagged_industry,
                         events, calendar)
    eligible = universe.merge(events, on=["date", "code", "trade_date"],
                              validate="one_to_one")
    if eligible.empty:
        raise ValueError("No discounted block event reached the eligible universe")
    selected = _select_capacity(eligible, calendar)
    all_block_keys = trades[["trade_date", "code"]].drop_duplicates()
    pairs, unmatched = _match(selected, universe, all_block_keys)
    all_pairs, all_unmatched = _match(
        eligible.sort_values(["date", "code"]), universe, all_block_keys)
    if not pairs.date.str[:4].isin(("2024", "2025")).all():
        raise ValueError("Block trade inputs escaped the 2024–2025 study window")
    diagnostics = {}
    for year in ("2024", "2025"):
        yearly_events = events.loc[events.date.str.startswith(year)]
        yearly_eligible = eligible.loc[eligible.date.str.startswith(year)]
        yearly_selected = selected.loc[selected.date.str.startswith(year)]
        yearly_pairs = pairs.loc[pairs.date.str.startswith(year)
                                 & pairs.candidate.eq(EVENT)]
        diagnostics[year] = {
            "qualified_stock_days": len(yearly_events),
            "eligible_quotes": len(yearly_eligible),
            "capacity_selected": len(yearly_selected),
            "peak_selected_month": (
                yearly_selected.date.str[:7].value_counts()
                .sort_values(ascending=False).index[0]
                if not yearly_selected.empty else None),
            "matched_pairs": len(yearly_pairs),
            "matched_days": yearly_pairs.date.nunique(),
            "matched_months": yearly_pairs.date.str[:7].nunique(),
            "matched_boards": yearly_pairs.board.value_counts().to_dict(),
        }
    audit = {"pair_scope": "capacity", "source_end": source_end,
             "source_days": len(source_days),
             "source": raw,
             "daily_join": bars, "universe_stock_days": len(universe),
             "unmatched": unmatched, "year": diagnostics,
             "note": "Input-only; no future return or execution files read"}
    events.to_parquet(output_dir / "events.parquet", index=False,
                      compression="zstd")
    eligible.to_parquet(output_dir / "eligible.parquet", index=False,
                        compression="zstd")
    selected.to_parquet(output_dir / "selected.parquet", index=False,
                        compression="zstd")
    pairs.to_parquet(output_dir / "pairs.parquet", index=False,
                     compression="zstd")
    pairs[["date", "code", "isST", "reference_gap",
           "quote_outside_traded_range", "listing_age_sessions"]].to_parquet(
               output_dir / "signals.parquet", index=False,
               compression="zstd")
    (output_dir / "input_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    secondary = output_dir / "all_eligible"
    secondary.mkdir(parents=True, exist_ok=True)
    all_audit = copy.deepcopy(audit)
    all_audit["pair_scope"] = "all_eligible"
    all_audit["unmatched"] = all_unmatched
    for year in ("2024", "2025"):
        annual = all_pairs.loc[all_pairs.date.str.startswith(year)
                               & all_pairs.candidate.eq(EVENT)]
        section = all_audit["year"][year]
        section["matched_pairs"] = len(annual)
        section["matched_days"] = annual.date.nunique()
        section["matched_months"] = annual.date.str[:7].nunique()
        section["matched_boards"] = annual.board.value_counts().to_dict()
        eligible_year = eligible.loc[eligible.date.str.startswith(year)]
        section["peak_selected_month"] = (
            eligible_year.date.str[:7].value_counts()
            .sort_values(ascending=False).index[0]
            if not eligible_year.empty else None)
    all_pairs.to_parquet(secondary / "pairs.parquet", index=False,
                         compression="zstd")
    all_pairs[["date", "code", "isST", "reference_gap",
               "quote_outside_traded_range", "listing_age_sessions"]].to_parquet(
                   secondary / "signals.parquet", index=False,
                   compression="zstd")
    (secondary / "input_audit.json").write_text(
        json.dumps(all_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(
        "data/research/block_trade/daily"))
    parser.add_argument("--snapshots", type=Path, default=Path(
        "data/research/market_snapshots_ci"))
    parser.add_argument("--daily", type=Path, default=Path(
        "data/baostock/market_2020_2026/daily"))
    parser.add_argument("--calendar", type=Path, default=Path(
        "data/baostock/market_2020_2026/metadata/calendar.parquet"))
    parser.add_argument("--industry", type=Path, default=Path(
        "data/research/industry_intervals.parquet"))
    parser.add_argument("--output", type=Path, default=Path(
        "data/research/block_trade"))
    parser.add_argument("--source-end", choices=("2024-12-31", "2025-12-31"),
                        default="2025-12-31")
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.snapshots, args.daily,
                           args.calendar, args.industry, args.output,
                           args.source_end),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
