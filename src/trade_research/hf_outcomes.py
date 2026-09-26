"""Generate auditable future trade outcomes, separate from 14:50 signals.

These are per-stock hypothetical orders, not a portfolio backtest. The model
uses the 14:52--14:55 VWAP after a completed 14:50 signal, observes T+1, and
refuses fills at estimated price limits or above a window-volume cap. It has
no order book, so even accepted fills are estimates.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_HALF_UP
import json
from math import isfinite
from pathlib import Path

import pandas as pd

from .hf_audit import FIRST_DATE, LAST_DATE, read_window
from .hf_download import selected_paths


HORIZONS = (1, 2, 3, 5)
RESEARCH_HORIZONS = HORIZONS + (4, 20)
EXECUTION_LABELS = ("1452", "1453", "1454", "1455")
EXECUTION_LABEL = "1452-1455"
ENTRY_WINDOWS = {
    "baseline": EXECUTION_LABELS,
    "delay_one_minute": ("1453", "1454", "1455", "1456"),
    "auction": ("1500",),
}
EXIT_WINDOWS = {
    "close": EXECUTION_LABELS,
    "morning": ("0935", "0936", "0937", "0938"),
    "late_morning": ("1000", "1001", "1002", "1003"),
    "auction": ("1500",),
}


@dataclass(frozen=True)
class Assumptions:
    target_notional: float = 100_000.0
    maximum_minute_volume_fraction: float = 0.1
    slippage_bps_each_side: float = 5.0
    commission_bps_each_side: float = 3.0
    commission_minimum_each_side: float = 5.0
    transfer_bps_each_side: float = 0.1
    transfer_bps_each_side_before_cutover: float = 0.2
    transfer_cutover_date: str = "2022-04-29"
    stamp_tax_bps_on_sale: float = 5.0
    stamp_tax_bps_on_sale_before_cutover: float = 10.0
    stamp_tax_cutover_date: str = "2023-08-28"
    maximum_exit_delay_sessions: int = 5


def _board_limit_rate(code: str, is_st: int, date: str) -> float:
    if code.startswith("sh.68"):
        return 0.2
    # The ChiNext reform took effect on 2020-08-24 for existing shares too.
    if code.startswith("sz.30") and date >= "2020-08-24":
        return 0.2
    # Both mainland mainboards aligned risk-warning shares with the ordinary
    # 10% band on 2026-07-06. Entry screens already exclude ST shares, but a
    # position can become risk-warning before its modeled exit.
    return 0.05 if is_st and date < "2026-07-06" else 0.1


def _limit_price(preclose: float, rate: float, upper: bool) -> float:
    multiplier = 1 + rate if upper else 1 - rate
    return float((Decimal(str(preclose)) * Decimal(str(multiplier))).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    ))


def _order_shares(code: str, price: float, target_notional: float) -> int:
    if price <= 0:
        return 0
    affordable = int(target_notional // price)
    # STAR Market orders start at 200 shares and may increase one share at a time.
    if code.startswith("sh.68"):
        return affordable if affordable >= 200 else 0
    return affordable // 100 * 100


def _fill(quote: pd.Series | None, daily_row: pd.Series | None, code: str,
          side: str, shares: int, assumptions: Assumptions) -> tuple[float | None, str]:
    if quote is None or daily_row is None or daily_row["tradestatus"] != 1:
        return None, "no_trading_bar"
    price = float(quote["vwap"])
    volume = float(quote["volume"])
    preclose = float(daily_row["preclose"])
    if price <= 0 or volume <= 0 or preclose <= 0:
        return None, "no_liquidity"
    if shares > volume * assumptions.maximum_minute_volume_fraction:
        return None, "volume_cap"
    rate = _board_limit_rate(code, int(daily_row["isST"]), str(daily_row["date"]))
    if side == "buy":
        fill_price = price * (1 + assumptions.slippage_bps_each_side / 10_000)
        upper = _limit_price(preclose, rate, upper=True)
        if fill_price >= upper - 0.005:
            return None, "estimated_upper_limit"
    else:
        fill_price = price * (1 - assumptions.slippage_bps_each_side / 10_000)
        lower = _limit_price(preclose, rate, upper=False)
        if fill_price <= lower + 0.005:
            return None, "estimated_lower_limit"
    return fill_price, "filled"


def _fees(value: float, side: str, assumptions: Assumptions,
          date: str | None = None) -> float:
    commission = max(assumptions.commission_minimum_each_side,
                     value * assumptions.commission_bps_each_side / 10_000)
    transfer_rate = assumptions.transfer_bps_each_side
    if date is not None and date < assumptions.transfer_cutover_date:
        transfer_rate = assumptions.transfer_bps_each_side_before_cutover
    transfer = value * transfer_rate / 10_000
    stamp_rate = assumptions.stamp_tax_bps_on_sale
    if date is not None and date < assumptions.stamp_tax_cutover_date:
        stamp_rate = assumptions.stamp_tax_bps_on_sale_before_cutover
    stamp = value * stamp_rate / 10_000 if side == "sell" else 0
    return commission + transfer + stamp


def _window_quotes(minute: pd.DataFrame,
                   labels: tuple[str, ...]) -> dict[str, pd.Series]:
    execution_bars = minute.loc[minute["label"].isin(labels)]
    quotes = {}
    for date, bars in execution_bars.groupby("date", sort=False):
        if bars["label"].tolist() != list(labels):
            continue
        volume = int(bars["volume"].sum())
        turnover = float(bars["turnover"].sum())
        quotes[date] = pd.Series({
            "volume": volume,
            "vwap": turnover / volume if volume > 0 else 0.0,
        })
    return quotes


def outcomes_for_symbol(signals: pd.DataFrame, minute: pd.DataFrame,
                        daily: pd.DataFrame, calendar: list[str],
                        assumptions: Assumptions = Assumptions(),
                        exit_labels: tuple[str, ...] = EXECUTION_LABELS,
                        horizons: tuple[int, ...] = HORIZONS,
                        entry_labels: tuple[str, ...] = EXECUTION_LABELS,
                        sizing_price_column: str | None = None,
                        ) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()
    if not exit_labels or len(set(exit_labels)) != len(exit_labels):
        raise ValueError("Exit labels must be nonempty and unique")
    if not entry_labels or len(set(entry_labels)) != len(entry_labels):
        raise ValueError("Entry labels must be nonempty and unique")
    if not horizons or any(horizon not in RESEARCH_HORIZONS for horizon in horizons):
        raise ValueError("Unsupported holding period")
    if sizing_price_column is not None and sizing_price_column not in signals.columns:
        raise ValueError("The decision-time sizing price is missing")
    code = str(signals["code"].iloc[0])
    daily = daily.sort_values("date").copy()
    active = daily.loc[daily["tradestatus"] == 1].copy()
    active["reference_gap"] = (
        active["preclose"] - active["close"].shift(1)
    ).abs() > 0.005
    reference_gap_dates = set(active.loc[active["reference_gap"], "date"])
    daily_by_date = {row["date"]: row for _, row in daily.iterrows()}
    entry_quotes = _window_quotes(minute, entry_labels)
    exit_quotes = (entry_quotes if exit_labels == entry_labels
                   else _window_quotes(minute, exit_labels))
    entry_label = f"{entry_labels[0]}-{entry_labels[-1]}"
    exit_label = f"{exit_labels[0]}-{exit_labels[-1]}"
    calendar_index = {date: n for n, date in enumerate(calendar)}
    rows = []
    for signal in signals.itertuples(index=False):
        if signal.date not in calendar_index:
            continue
        entry_date = signal.date
        entry_quote = entry_quotes.get(entry_date)
        estimated_price = float(entry_quote["vwap"]) if entry_quote is not None else 0.0
        sizing_price = (float(getattr(signal, sizing_price_column))
                        if sizing_price_column is not None else estimated_price)
        if sizing_price_column is not None and (not isfinite(sizing_price)
                                                or sizing_price <= 0):
            raise ValueError("Invalid decision-time sizing price")
        shares = _order_shares(code, sizing_price, assumptions.target_notional)
        if shares == 0:
            entry_price, entry_status = None, "below_minimum_lot"
        elif getattr(signal, "listing_age_sessions", 5) < 5:
            # New listings have special price-limit rules that vary by era.
            entry_price, entry_status = None, "new_listing_window"
        elif bool(signal.isST):
            entry_price, entry_status = None, "st_excluded"
        elif (bool(signal.reference_gap)
              and getattr(signal, "entry_reference_verified", False) is not True):
            # A caller may explicitly certify a decision-time corporate-action
            # reference after checking its original terms. Missing markers fail closed.
            entry_price, entry_status = None, "entry_corporate_action"
        elif bool(getattr(signal, "quote_outside_traded_range", False)):
            entry_price, entry_status = None, "source_quote_anomaly"
        else:
            entry_price, entry_status = _fill(
                entry_quote, daily_by_date.get(entry_date), code, "buy", shares, assumptions
            )
        for horizon in horizons:
            result = {
                "date": entry_date, "code": code, "horizon": horizon,
                "entry_label": entry_label, "exit_label": exit_label,
                "entry_status": entry_status, "entry_price": entry_price,
                "shares": shares if entry_price is not None else 0,
                "target_exit_date": None, "exit_date": None,
                "exit_delay_sessions": None, "exit_status": None,
                "exit_price": None, "gross_return": None,
                "net_return": None, "corporate_action_crossed": None,
            }
            start = calendar_index[entry_date] + horizon
            if start >= len(calendar):
                result["exit_status"] = "right_censored"
                rows.append(result)
                continue
            result["target_exit_date"] = calendar[start]
            if entry_price is None:
                result["exit_status"] = "entry_not_filled"
                rows.append(result)
                continue
            stop = min(start + assumptions.maximum_exit_delay_sessions, len(calendar) - 1)
            exit_status = "right_censored" if stop == len(calendar) - 1 else "unfilled_within_delay"
            for index in range(start, stop + 1):
                date = calendar[index]
                exit_price, reason = _fill(
                    exit_quotes.get(date), daily_by_date.get(date), code, "sell", shares, assumptions
                )
                if exit_price is None:
                    exit_status = reason
                    continue
                result["exit_date"] = date
                result["exit_delay_sessions"] = index - start
                result["exit_price"] = exit_price
                crossed = any(start_date in reference_gap_dates for start_date in
                              calendar[calendar_index[entry_date] + 1:index + 1])
                result["corporate_action_crossed"] = crossed
                if crossed:
                    exit_status = "corporate_action_unadjusted"
                else:
                    buy_value = shares * entry_price
                    sell_value = shares * exit_price
                    result["gross_return"] = sell_value / buy_value - 1
                    result["net_return"] = (
                        (sell_value - _fees(sell_value, "sell", assumptions, date)) /
                        (buy_value + _fees(buy_value, "buy", assumptions, entry_date)) - 1
                    )
                    exit_status = "filled"
                break
            result["exit_status"] = exit_status
            rows.append(result)
    return pd.DataFrame(rows)


def build(hf_root: Path, bao_root: Path,
          assumptions: Assumptions = Assumptions()) -> dict:
    snapshot_file = bao_root / "hf_snapshots_1450.parquet"
    signals = pd.read_parquet(snapshot_file)
    pilot_file = bao_root / "metadata" / "pilot_symbols.parquet"
    pilot = pd.read_parquet(pilot_file)
    trading = pd.read_parquet(bao_root / "metadata" / "calendar.parquet")
    calendar = sorted(trading.loc[
        (trading["is_trading_day"] == "1") &
        trading["calendar_date"].between(FIRST_DATE, LAST_DATE), "calendar_date"
    ].tolist())
    frames = []
    for code, relative in zip(pilot["code"], selected_paths(pilot_file), strict=True):
        stock_signals = signals.loc[signals["code"] == code]
        if stock_signals.empty:
            continue
        minute = read_window(hf_root / relative)
        daily = pd.read_parquet(bao_root / "daily" / f"{code.replace('.', '_')}.parquet")
        frame = outcomes_for_symbol(stock_signals, minute, daily, calendar, assumptions)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise RuntimeError("No future outcomes were computed")
    result = pd.concat(frames, ignore_index=True).sort_values(["date", "code", "horizon"])
    output = bao_root / "hf_outcomes.parquet"
    result.to_parquet(output, index=False, compression="zstd")
    summary = {
        "assumptions": asdict(assumptions),
        "signal_rows": len(signals), "outcome_rows": len(result),
        "filled_outcomes": int((result["exit_status"] == "filled").sum()),
        "entry_status_counts": result.drop_duplicates(["date", "code"])["entry_status"].value_counts().to_dict(),
        "exit_status_counts": result["exit_status"].value_counts().to_dict(),
        "output": str(output),
    }
    (bao_root / "metadata" / "hf_outcome_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-root", type=Path, default=Path("data/hf/pilot"))
    parser.add_argument("--bao-root", type=Path, default=Path("data/baostock/pilot_2025_2026"))
    args = parser.parse_args()
    print(json.dumps(build(args.hf_root, args.bao_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
