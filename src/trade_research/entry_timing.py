"""Compare two T+1-compliant entry windows for the same 14:50 stock list.

The selected stock and T+2 target exit are identical across arms. The next
morning's bars are execution inputs only; they never choose the stock. This is
exploratory because 2025 has already been consulted in earlier research.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .hf_outcomes import (
    Assumptions, EXECUTION_LABELS, EXIT_WINDOWS, _fees, _fill,
    _order_shares, _window_quotes,
)
from .intraday_scan import SCREENS
from .market_study import _week_bootstrap
from .size_sensitivity import _attach_quality
from .strategy_scan import _stressed_returns


ROOT = Path("data/research")
MINUTE_ROOT = Path("data/hf/pilot/data/stock_1m")
DAILY_ROOT = Path("data/baostock/market_2020_2026/daily")
CALENDAR_FILE = Path("data/baostock/market_2020_2026/metadata/calendar.parquet")
PERIOD_ENDS = {"2024": "2024-12-17", "2025": "2025-12-17"}
NOTIONALS = (20_000.0, 100_000.0)
MORNING_LABELS = EXIT_WINDOWS["morning"]
CAPACITY = 5


def select_signals() -> pd.DataFrame:
    """Select existing late screens and one deterministic same-pool control."""
    c = duckdb.connect()
    c.read_parquet(str(ROOT / "market_snapshots_ci" / "*.parquet")
                   ).create_view("s")
    c.read_parquet(str(ROOT / "intraday_features" / "*.parquet")
                   ).create_view("i")
    c.execute("""
        CREATE TEMP TABLE eligible AS
        SELECT s.*, i.return_last30, i.volume_share_last30
        FROM s JOIN i USING (date, code)
        WHERE ((s.date BETWEEN '2024-01-01' AND '2024-12-17')
            OR (s.date BETWEEN '2025-01-01' AND '2025-12-17'))
          AND s.isST = 0 AND s.listing_age_sessions >= 20
          AND NOT s.reference_gap AND NOT s.quote_outside_traded_range
          AND s.amount_1450 BETWEEN 100000000 AND 1000000000
          AND abs(s.price_1450 - i.price_1450) <= .005
    """)
    pieces = []
    rules = {name: SCREENS[name] for name in ("late_push", "quiet_trend")}
    rules["same_day_random"] = (
        "TRUE", "md5('entry-timing-v1' || date || code)"
    )
    for name, (condition, ranking) in rules.items():
        selected = c.execute(f"""
            SELECT date, code, daily_rank FROM (
                SELECT date, code, ROW_NUMBER() OVER (
                    PARTITION BY date ORDER BY {ranking}, code
                ) AS daily_rank
                FROM eligible WHERE {condition}
            ) WHERE daily_rank <= {CAPACITY}
        """).df()
        selected["candidate"] = name
        pieces.append(selected)
    signals = pd.concat(pieces, ignore_index=True)
    if (signals.empty or signals.duplicated(["candidate", "date", "code"]).any()
            or not signals.date.str[:4].isin(PERIOD_ENDS).all()):
        raise ValueError("Invalid 2024–2025 signal selection")
    return signals.sort_values(["candidate", "date", "daily_rank"])


def _one_trade(code: str, signal_date: str, entry_date: str,
               target_index: int, calendar: list[str],
               entry_quote: pd.Series | None,
               exit_quotes: dict[str, pd.Series],
               daily_by_date: dict[str, pd.Series],
               reference_gap_dates: set[str],
               assumptions: Assumptions) -> dict:
    """Use the repository's existing minute fill and fee model."""
    result = {
        "entry_date": entry_date, "entry_status": None,
        "entry_price": None, "shares": 0,
        "target_exit_date": calendar[target_index],
        "exit_date": None, "exit_delay_sessions": None,
        "exit_status": None, "exit_price": None,
        "net_return": None,
    }
    if entry_quote is None:
        result["entry_status"] = "no_trading_bar"
    elif float(entry_quote["vwap"]) <= 0:
        result["entry_status"] = "no_liquidity"
    elif entry_date != signal_date and bool(daily_by_date.get(
            entry_date, pd.Series(dtype=object)).get("isST", 0)):
        result["entry_status"] = "became_st"
    else:
        shares = _order_shares(
            code, float(entry_quote["vwap"]), assumptions.target_notional
        )
        if shares == 0:
            result["entry_status"] = "below_minimum_lot"
        else:
            price, status = _fill(
                entry_quote, daily_by_date.get(entry_date), code,
                "buy", shares, assumptions,
            )
            result["entry_status"] = status
            result["entry_price"] = price
            result["shares"] = shares if price is not None else 0
    if result["entry_price"] is None:
        result["exit_status"] = "entry_not_filled"
        return result

    stop = min(target_index + assumptions.maximum_exit_delay_sessions,
               len(calendar) - 1)
    result["exit_status"] = "unfilled_within_delay"
    for index in range(target_index, stop + 1):
        exit_date = calendar[index]
        exit_price, status = _fill(
            exit_quotes.get(exit_date), daily_by_date.get(exit_date),
            code, "sell", result["shares"], assumptions,
        )
        if exit_price is None:
            result["exit_status"] = status
            continue
        result["exit_date"] = exit_date
        result["exit_delay_sessions"] = index - target_index
        result["exit_price"] = exit_price
        start_index = calendar.index(signal_date)
        if any(day in reference_gap_dates
               for day in calendar[start_index + 1:index + 1]):
            result["exit_status"] = "corporate_action_unadjusted"
        else:
            buy_value = result["shares"] * result["entry_price"]
            sell_value = result["shares"] * exit_price
            result["net_return"] = (
                (sell_value - _fees(sell_value, "sell", assumptions, exit_date))
                / (buy_value + _fees(buy_value, "buy", assumptions, entry_date))
                - 1
            )
            result["exit_status"] = "filled"
        break
    return result


def price_symbol(code: str, selections: pd.DataFrame,
                 calendar: list[str], index: dict[str, int]) -> list[dict]:
    exchange, symbol = code.split(".")
    dates = {
        day
        for signal_date in selections.date.unique()
        for day in calendar[index[signal_date]:
                            index[signal_date] + 3
                            + Assumptions().maximum_exit_delay_sessions]
    }
    minute_file = MINUTE_ROOT / exchange.upper() / f"{symbol}.parquet"
    daily_file = DAILY_ROOT / f"{exchange}_{symbol}.parquet"
    minute = pd.read_parquet(minute_file,
                             columns=["timestamp", "volume", "turnover"],
                             filters=[
                                 ("timestamp", ">=", pd.Timestamp(min(dates))),
                                 ("timestamp", "<", pd.Timestamp(max(dates))
                                  + pd.Timedelta(days=1)),
                             ])
    minute["date"] = minute.timestamp.dt.strftime("%Y-%m-%d")
    minute["label"] = minute.timestamp.dt.strftime("%H%M")
    minute = minute.loc[minute.date.isin(dates)].sort_values("timestamp")
    tail_quotes = _window_quotes(minute, EXECUTION_LABELS)
    morning_quotes = _window_quotes(minute, MORNING_LABELS)
    daily = pd.read_parquet(daily_file).sort_values("date")
    active = daily.loc[daily.tradestatus.eq(1)]
    reference_gap_dates = set(active.loc[
        (active.preclose - active.close.shift(1)).abs().gt(.005), "date"
    ])
    daily_by_date = {row["date"]: row for _, row in daily.iterrows()}
    rows = []
    for selected in selections.itertuples(index=False):
        first = index[selected.date]
        if first + 2 >= len(calendar):
            raise ValueError("Selected stock lacks a T+2 market day")
        for notional in NOTIONALS:
            assumptions = Assumptions(target_notional=notional)
            for arm, entry_date, quote in (
                ("tail", selected.date, tail_quotes.get(selected.date)),
                ("next_morning", calendar[first + 1],
                 morning_quotes.get(calendar[first + 1])),
            ):
                priced = _one_trade(
                    code, selected.date, entry_date, first + 2, calendar,
                    quote, tail_quotes, daily_by_date, reference_gap_dates,
                    assumptions,
                )
                rows.append({
                    "candidate": selected.candidate, "date": selected.date,
                    "code": code, "arm": arm, "target_notional": notional,
                    **priced,
                })
    return rows


def _daily_cash(frame: pd.DataFrame, dates: list[str],
                column: str) -> pd.Series:
    count = frame.groupby("date").size().reindex(dates)
    if count.isna().any():
        raise ValueError("An entry arm is missing selected stock-days")
    valid = frame.loc[frame.exit_status.eq("filled")
                      & frame.quality_clean_exit]
    return valid.groupby("date")[column].sum().reindex(
        dates, fill_value=0
    ) / count


def summarize(trades: pd.DataFrame) -> dict:
    expected = len(trades.drop_duplicates(["candidate", "date", "code"])) \
        * len(NOTIONALS) * 2
    if len(trades) != expected or trades.duplicated(
            ["candidate", "date", "code", "arm", "target_notional"]).any():
        raise ValueError("Entry timing arms have incomplete coverage")
    valid = trades.exit_status.eq("filled") & trades.quality_clean_exit
    trades = trades.copy()
    trades["stress10"] = 0.0
    if valid.any():
        stress_input = trades.loc[valid].copy()
        stress_input["date"] = stress_input.entry_date
        if not np.allclose(_stressed_returns(stress_input, 5),
                           stress_input.net_return, atol=1e-12):
            raise ValueError("Timing reprice disagrees with the fee model")
        trades.loc[valid, "stress10"] = _stressed_returns(stress_input, 10)
    trades["period"] = trades.date.str[:4] + "-" + (
        trades.date.str[5:7].astype(int).le(6).map({True: "H1", False: "H2"})
    )
    report = {"order_sizes": NOTIONALS, "groups": []}
    daily_differences = {}
    for (candidate, period, notional), group in trades.groupby(
            ["candidate", "period", "target_notional"]):
        dates = sorted(group.date.unique())
        arms = {}
        for arm in ("tail", "next_morning"):
            subset = group.loc[group.arm.eq(arm)]
            arms[arm] = {
                "entry_rate": float(subset.entry_status.eq("filled").mean()),
                "clean_exit_rate": float((subset.exit_status.eq("filled")
                                          & subset.quality_clean_exit).mean()),
                "cash": _daily_cash(subset, dates, "net_return"),
                "stress10": _daily_cash(subset, dates, "stress10"),
            }
        paired = group.loc[group.arm.eq("tail")].merge(
            group.loc[group.arm.eq("next_morning")],
            on=["candidate", "date", "code", "period", "target_notional"],
            suffixes=("_tail", "_morning"), validate="one_to_one",
        )
        common = paired.loc[
            paired.exit_status_tail.eq("filled")
            & paired.exit_status_morning.eq("filled")
            & paired.quality_clean_exit_tail
            & paired.quality_clean_exit_morning
            & paired.exit_date_tail.eq(paired.exit_date_morning)
        ]
        difference = arms["next_morning"]["cash"] - arms["tail"]["cash"]
        stress_delta = (arms["next_morning"]["stress10"]
                        - arms["tail"]["stress10"])
        daily_differences[(candidate, period, notional)] = difference
        report["groups"].append({
            "candidate": candidate, "period": period, "notional": notional,
            "signals": len(group) // 2, "days": len(dates),
            "tail_entry_rate": arms["tail"]["entry_rate"],
            "morning_entry_rate": arms["next_morning"]["entry_rate"],
            "tail_clean_exit_rate": arms["tail"]["clean_exit_rate"],
            "morning_clean_exit_rate": arms["next_morning"]["clean_exit_rate"],
            "tail_cash": float(arms["tail"]["cash"].mean()),
            "morning_cash": float(arms["next_morning"]["cash"].mean()),
            "common_exit_pairs": len(common),
            "common_pair_entry_discount": float(
                (1 - common.entry_price_morning / common.entry_price_tail).mean()
            ) if len(common) else None,
            "common_pair_net_difference": float(
                (common.net_return_morning - common.net_return_tail).mean()
            ) if len(common) else None,
            "morning_minus_tail": float(difference.mean()),
            "difference_week_ci": _week_bootstrap(
                difference.reset_index(drop=True), pd.Series(dates), 20260925
            ),
            "difference_stress10": float(stress_delta.mean()),
        })
    report["relative_to_random"] = []
    for (candidate, period, notional), difference in daily_differences.items():
        if candidate == "same_day_random":
            continue
        random = daily_differences[("same_day_random", period, notional)]
        common = difference.index.intersection(random.index)
        contrast = difference.reindex(common) - random.reindex(common)
        report["relative_to_random"].append({
            "candidate": candidate, "period": period,
            "notional": notional, "days": len(common),
            "extra_timing_benefit": float(contrast.mean()),
            "week_ci": _week_bootstrap(
                contrast.reset_index(drop=True), pd.Series(common), 20260926
            ),
        })
    return report


def verify_tail_reprice(trades: pd.DataFrame) -> dict:
    """The 100k tail arm must reproduce stored T+2 outcomes exactly."""
    c = duckdb.connect()
    c.register("tail", trades.loc[
        trades.arm.eq("tail") & trades.target_notional.eq(100_000)
    ])
    c.read_parquet(str(ROOT / "market_outcomes_ci" / "*.parquet")
                   ).create_view("original")
    row = c.execute("""
        SELECT COUNT(*) AS checked, COUNT(o.horizon) AS matched,
               COUNT(*) FILTER (WHERE n.entry_status IS DISTINCT FROM o.entry_status
                 OR n.exit_status IS DISTINCT FROM o.exit_status
                 OR n.shares IS DISTINCT FROM o.shares
                 OR n.entry_price IS DISTINCT FROM o.entry_price
                 OR n.exit_price IS DISTINCT FROM o.exit_price
                 OR n.net_return IS DISTINCT FROM o.net_return) AS mismatches
        FROM tail n LEFT JOIN original o
          ON o.date = n.date AND o.code = n.code AND o.horizon = 2
    """).fetchone()
    audit = {"checked": row[0], "matched": row[1], "mismatches": row[2]}
    if audit["checked"] != audit["matched"] or audit["mismatches"]:
        raise ValueError(f"Tail reprice disagrees with stored outcomes: {audit}")
    return audit


def run(workers: int = 4) -> dict:
    if workers < 1:
        raise ValueError("Workers must be positive")
    output = ROOT / "entry_timing"
    output.mkdir(parents=True, exist_ok=True)
    signals = select_signals()
    signals.to_parquet(output / "signals.parquet", index=False,
                       compression="zstd")
    trading = pd.read_parquet(CALENDAR_FILE)
    calendar = sorted(trading.loc[
        trading.is_trading_day.eq("1")
        & trading.calendar_date.between("2024-01-01", "2025-12-31"),
        "calendar_date",
    ].tolist())
    index = {day: offset for offset, day in enumerate(calendar)}
    if not set(signals.date).issubset(index):
        raise ValueError("Signal date absent from exchange calendar")
    grouped = list(signals.groupby("code", sort=True))
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = ((code, frame, calendar, index) for code, frame in grouped)
        for count, priced in enumerate(pool.map(lambda args: price_symbol(*args),
                                                jobs), start=1):
            rows.extend(priced)
            if count % 200 == 0 or count == len(grouped):
                print(f"Priced {count}/{len(grouped)} symbols", flush=True)
    trades = _attach_quality(pd.DataFrame(rows), ROOT / "market_issues_ci")
    audit = verify_tail_reprice(trades)
    trades.to_parquet(output / "trades.parquet", index=False,
                      compression="zstd")
    report = summarize(trades)
    report["stored_tail_agreement"] = audit
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    report = run(args.workers)
    print({"groups": len(report["groups"]),
           "relative_contrasts": len(report["relative_to_random"])})


if __name__ == "__main__":
    main()
