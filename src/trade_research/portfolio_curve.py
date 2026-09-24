"""Mark-to-market risk curve for quality-clean, completed model trades.

The result is conditional on a verified exit and is therefore a risk diagnostic,
not a live portfolio return. Entry fills rejected by the model and unresolved
exits remain in the coverage counts rather than being silently marked as wins.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .hf_outcomes import Assumptions, _fees
from .study_periods import DEVELOPMENT_YEAR


def _calendar(path: Path, first: str, last: str) -> list[str]:
    dates = pd.read_parquet(path)
    return sorted(dates.loc[
        dates["is_trading_day"].eq("1")
        & dates["calendar_date"].between(first, last), "calendar_date"
    ].tolist())


def curve(trades: pd.DataFrame, daily_root: Path, calendar: list[str],
          initial_capital: float = 1_000_000.0,
          assumptions: Assumptions = Assumptions()) -> tuple[pd.DataFrame, dict]:
    if initial_capital <= 0 or not calendar:
        raise ValueError("A positive capital base and nonempty calendar are required")
    if trades.empty:
        raise ValueError("No model signals in this period")
    if trades["date"].min() < f"{DEVELOPMENT_YEAR}-01-01":
        raise ValueError("Portfolio diagnostics require recent research dates")
    if trades.duplicated(["date", "code"]).any():
        raise ValueError("Each stock may have only one signal per date")
    completed = trades.loc[
        trades["entry_status"].eq("filled")
        & trades["exit_status"].eq("filled")
        & trades["quality_clean_exit"]
    ].copy()
    if completed.empty:
        raise ValueError("No quality-clean completed model trades")
    completed = completed.sort_values(["date", "daily_rank", "code"])
    active_dates = set(calendar)
    if not set(completed["date"]).issubset(active_dates):
        raise ValueError("An entry date is absent from the trading calendar")
    if not set(completed["exit_date"]).issubset(active_dates):
        raise ValueError("An exit date is absent from the trading calendar")

    codes = completed["code"].unique()
    close_by_code = {}
    for code in codes:
        path = daily_root / f"{code.replace('.', '_')}.parquet"
        daily = pd.read_parquet(path, columns=["date", "close", "tradestatus"])
        daily = daily.loc[
            daily["tradestatus"].eq(1) & daily["close"].gt(0),
            ["date", "close"],
        ].drop_duplicates("date")
        close_by_code[code] = dict(zip(daily["date"], daily["close"], strict=True))

    entries = {date: group.to_dict("records")
               for date, group in completed.groupby("date", sort=False)}
    cash = float(initial_capital)
    positions = []
    marks = {}
    records = []
    purchased = 0
    capital_blocked = 0
    maximum_positions = 0
    for date in calendar:
        still_open = []
        for position in positions:
            if position["exit_date"] != date:
                still_open.append(position)
                continue
            value = position["shares"] * position["exit_price"]
            cash += value - _fees(value, "sell", assumptions, date)
        positions = still_open
        for trade in entries.get(date, []):
            shares = int(trade["shares"])
            value = shares * float(trade["entry_price"])
            cost = value + _fees(value, "buy", assumptions, date)
            if cost > cash + 1e-9:
                capital_blocked += 1
                continue
            cash -= cost
            positions.append({
                "code": trade["code"], "shares": shares,
                "exit_date": trade["exit_date"],
                "exit_price": float(trade["exit_price"]),
                "entry_price": float(trade["entry_price"]),
            })
            purchased += 1
        maximum_positions = max(maximum_positions, len(positions))
        market_value = 0.0
        for position in positions:
            code = position["code"]
            if date in close_by_code[code]:
                marks[code] = float(close_by_code[code][date])
            price = marks.get(code, position["entry_price"])
            value = position["shares"] * price
            market_value += value - _fees(value, "sell", assumptions, date)
        equity = cash + market_value
        records.append({
            "date": date, "equity": equity, "cash": cash,
            "open_positions": len(positions), "marked_exposure": market_value,
        })
    if positions:
        raise ValueError("The trading calendar ends before all model exits")
    frame = pd.DataFrame(records)
    frame["peak_equity"] = frame["equity"].cummax().clip(lower=initial_capital)
    frame["drawdown"] = frame["equity"] / frame["peak_equity"] - 1
    frame["daily_change"] = frame["equity"].pct_change().fillna(
        frame["equity"].iloc[0] / initial_capital - 1
    )
    summary = {
        "initial_capital": initial_capital,
        "final_equity": float(frame["equity"].iloc[-1]),
        "total_return": float(frame["equity"].iloc[-1] / initial_capital - 1),
        "maximum_marked_drawdown": float(frame["drawdown"].min()),
        "worst_marked_day": float(frame["daily_change"].min()),
        "maximum_simultaneous_positions": maximum_positions,
        "signals": len(trades),
        "model_entry_fills": int(trades["entry_status"].eq("filled").sum()),
        "clean_completed_model_trades": len(completed),
        "purchased_after_cash_limit": purchased,
        "capital_blocked": capital_blocked,
        "omitted_after_entry_fill": int((
            trades["entry_status"].eq("filled")
            & ~(trades["exit_status"].eq("filled") & trades["quality_clean_exit"])
        ).sum()),
        "qualification": "Curve includes only quality-clean completed model exits; "
                         "unresolved exits and corporate actions are counted, not priced",
    }
    return frame, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trades", type=Path,
                        default=Path("data/research/strategy_scan_trades.parquet"))
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--first-date", required=True)
    parser.add_argument("--last-entry-date", required=True)
    parser.add_argument("--daily-root", type=Path,
                        default=Path("data/baostock/market_2020_2026/daily"))
    parser.add_argument("--calendar", type=Path,
                        default=Path("data/baostock/market_2020_2026/metadata/calendar.parquet"))
    parser.add_argument("--capital", type=float, default=1_000_000.0)
    parser.add_argument("--output", type=Path,
                        default=Path("data/research/portfolio_curve.parquet"))
    args = parser.parse_args()
    trades = pd.read_parquet(args.trades)
    trades = trades.loc[
        trades["candidate"].eq(args.candidate)
        & trades["horizon"].eq(args.horizon)
        & trades["date"].between(args.first_date, args.last_entry_date)
    ]
    latest_exit = trades["exit_date"].dropna().max()
    if pd.isna(latest_exit):
        raise ValueError("No candidate exits are available")
    calendar = _calendar(args.calendar, args.first_date, latest_exit)
    result, summary = curve(trades, args.daily_root, calendar, args.capital)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False, compression="zstd")
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
