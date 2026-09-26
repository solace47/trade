"""Audit shared execution capacity and cash needs of eight fixed order books.

Minimum funding is an ex-post cash-flow statistic, not capital optimized for
strategy returns. Unknown terminal assets remain unvalued.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .corporate_cash import MINUTES, save_json, sha
from .fill_accounting import CALENDAR, KEY
from .hf_outcomes import Assumptions, EXECUTION_LABELS, _fees, _limit_price, _order_shares, _window_quotes


ROOT = Path("data/research/orderbook_funding")
SOURCE = Path("data/research/holding_exceptions/accounted.parquet")
SIGNALS = Path("data/research/absolute_ridge_1449_target/repricing_signals.parquet")
BOOK = ["target_notional", "horizon", "candidate"]


def freeze(output: Path = ROOT) -> dict:
    report = json.loads(SOURCE.with_name("report.json").read_text())
    if sha(SOURCE) != report["accounted_sha256"]:
        raise ValueError("Verified holding accounting changed")
    rows = pd.read_parquet(SOURCE)
    if (len(rows) != 12648 or rows.duplicated(KEY).any()
            or not rows.date.between("2024-01-01", "2025-12-31").all()
            or set(rows.target_notional) != {20000, 100000}
            or set(rows.horizon) != {1, 5}
            or set(rows.candidate) != {"absolute_model", "same_day_control"}):
        raise ValueError("Unexpected frozen order-book grid")
    manifest = {"rule_commit": "def097b", "rows": len(rows),
                "sha256": {str(p): sha(p) for p in (SOURCE, SOURCE.with_name("report.json"),
                                                     SIGNALS, CALENDAR)},
                "holdout_read": False, "funding_results_read": False}
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Cannot replace the frozen funding inputs")
    save_json(path, manifest)
    return manifest


def verify(output: Path) -> dict:
    manifest = json.loads((output / "manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        if sha(Path(name)) != expected:
            raise ValueError(f"Frozen order-book input changed: {name}")
    return manifest


def execution_events(rows: pd.DataFrame) -> pd.DataFrame:
    bought = rows.loc[rows.entry_status.eq("filled")].copy()
    bought["signal_date"] = bought.date
    if bought.loc[bought.unknown_after_buy, "accounting_exit_date"].notna().any():
        raise ValueError("Unknown recorded sale proceeds cannot be used as cash")
    buys = bought.assign(side="buy", quantity=bought.shares,
                         price=bought.entry_price)
    sells = bought.loc[~bought.unknown_after_buy].assign(
        side="sell", quantity=lambda f: f.sold_shares,
        price=lambda f: f.accounting_exit_price,
        date=lambda f: f.accounting_exit_date)
    columns = [*BOOK, "signal_date", "date", "code", "side", "quantity", "price"]
    events = pd.concat([buys[columns], sells[columns]], ignore_index=True)
    if (events.date.isna().any() or not events.date.between("2024-01-01", "2025-12-31").all()
            or not np.isfinite(events[["quantity", "price"]]).all().all()
            or events.quantity.le(0).any() or events.price.le(0).any()
            or events.quantity.mod(1).ne(0).any()):
        raise ValueError("Invalid booked execution event")
    return events.sort_values([*BOOK, "date", "code", "side", "signal_date"]).reset_index(drop=True)


def collect_quotes(output: Path = ROOT) -> dict:
    verify(output)
    events = execution_events(pd.read_parquet(SOURCE))
    needed = events[["code", "date"]].drop_duplicates()
    jobs = [(code, sorted(g.date.tolist())) for code, g in needed.groupby("code", sort=True)]

    def one(job):
        code, dates = job
        market, symbol = code.split(".")
        path = MINUTES / market.upper() / f"{symbol}.parquet"
        signature = sha(path)
        minute = pd.read_parquet(path, columns=["timestamp", "volume", "turnover"], filters=[
            ("timestamp", ">=", pd.Timestamp(dates[0])),
            ("timestamp", "<", pd.Timestamp(dates[-1]) + pd.Timedelta(days=1))])
        minute["date"] = minute.timestamp.dt.strftime("%Y-%m-%d")
        minute["label"] = minute.timestamp.dt.strftime("%H%M")
        quotes = _window_quotes(minute.loc[minute.date.isin(dates)].sort_values("timestamp"),
                                 EXECUTION_LABELS)
        if set(quotes) != set(dates):
            raise ValueError(f"Incomplete raw execution window: {code}")
        records = [{"code": code, "date": day, "volume": int(q.volume),
                    "vwap": float(q.vwap)} for day, q in sorted(quotes.items())]
        return records, str(path), signature

    records, hashes = [], {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for n, (part, path, signature) in enumerate(pool.map(one, jobs), 1):
            records.extend(part)
            hashes[path] = signature
            if n % 100 == 0 or n == len(jobs):
                print(f"Raw four-minute quotes: {n}/{len(jobs)} symbols", flush=True)
    quotes = pd.DataFrame(records).sort_values(["code", "date"]).reset_index(drop=True)
    if len(quotes) != len(needed) or quotes.duplicated(["code", "date"]).any():
        raise ValueError("Raw quote keys changed")
    quotes.to_parquet(output / "quotes.parquet", index=False)
    report = {"symbols": len(jobs), "stock_days": len(quotes), "events": len(events),
              "raw_sha256": hashes, "quotes_sha256": sha(output / "quotes.parquet"),
              "funding_results_read": False, "holdout_read": False}
    save_json(output / "quote_audit.json", report)
    return report


def capacity_audit(events: pd.DataFrame, quotes: pd.DataFrame) -> pd.DataFrame:
    linked = events.merge(quotes, on=["code", "date"], how="left", validate="many_to_one")
    computed = linked.vwap * np.where(linked.side.eq("buy"), 1.0005, .9995)
    if (linked.volume.isna().any() or linked.volume.le(0).any()
            or not np.allclose(computed, linked.price, atol=1e-10, rtol=0)
            or linked.quantity.gt(linked.volume * .1 + 1e-9).any()):
        raise ValueError("Original per-order raw prices or participation limits changed")
    groups = []
    for keys, part in linked.groupby([*BOOK, "date", "code"], sort=True):
        buy = float(part.loc[part.side.eq("buy"), "quantity"].sum())
        sell = float(part.loc[part.side.eq("sell"), "quantity"].sum())
        volume = int(part.volume.iloc[0])
        groups.append({**dict(zip([*BOOK, "date", "code"], keys, strict=True)),
                       "buy_shares": buy, "sell_shares": sell,
                       "window_volume": volume, "orders": len(part),
                       "both_sides": bool(buy and sell),
                       "total_participation": (buy + sell) / volume,
                       "capacity_passed": buy + sell <= volume * .1 + 1e-9})
    return pd.DataFrame(groups)


def attach_reservations(rows: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    rows = rows.merge(signals[["date", "code", "price_1449", "preclose", "isST"]],
                       on=["date", "code"], how="left", validate="many_to_one")
    if (rows.price_1449.isna().any() or rows.preclose.le(0).any() or rows.isST.ne(0).any()
            or not rows.code.str.startswith(("sh.60", "sz.00")).all()):
        raise ValueError("Funding reservation requires the frozen ordinary mainboard inputs")
    quantities, ceilings = [], []
    for r in rows.itertuples():
        quantity = _order_shares(r.code, r.price_1449, r.target_notional)
        if r.entry_status == "filled" and quantity != r.shares:
            raise ValueError("14:49 planned shares changed")
        quantities.append(quantity)
        ceilings.append(_limit_price(r.preclose, .1, upper=True))
    rows["planned_shares"] = quantities
    rows["known_price_ceiling"] = ceilings
    for slip in (5, 15):
        reserves = []
        for r in rows.itertuples():
            # 15bps is the old cost stress applied to the same 5bps fill model.
            value = r.planned_shares * r.known_price_ceiling * (1 + slip / 10000) / 1.0005
            reserves.append(value + _fees(value, "buy", Assumptions(), r.date)
                            if r.planned_shares else 0.0)
        rows[f"reservation{slip}"] = reserves
        if (rows[f"reservation{slip}"] + 1e-8 < rows[f"buy_cost{slip}"]).any():
            raise ValueError("An actual cost exceeds its pretrade upper-price reservation")
    return rows.sort_values(KEY).reset_index(drop=True)


def funding_ledger(rows: pd.DataFrame, calendar: list[str], slip: int) -> tuple[pd.DataFrame, dict]:
    if (slip not in (5, 15) or rows.empty or len(rows[BOOK].drop_duplicates()) != 1
            or calendar != sorted(set(calendar)) or not calendar
            or calendar[0] < "2024-01-01" or calendar[-1] > "2025-12-31"
            or not set(rows.date).issubset(calendar)):
        raise ValueError("Invalid independent funding book")
    bought = rows.loc[rows.entry_status.eq("filled")]
    sold = bought.loc[~bought.unknown_after_buy]
    if (sold[f"verified_proceeds{slip}"].isna().any()
            or not set(sold.accounting_exit_date).issubset(calendar)
            or bought.loc[bought.unknown_after_buy, "accounting_exit_date"].notna().any()):
        raise ValueError("Incomplete or unknown sale cannot be included in cash flows")
    dividends = bought.loc[bought.dividend_gross.notna()]
    if (not set(dividends.dividend_pay_date).issubset(calendar)
            or not set(dividends.dividend_tax_recognition_date).issubset(calendar)
            or dividends.dividend_tax_accrued.isna().any()):
        raise ValueError("Dividend cash or tax timing is missing")
    buy_cash = rows.groupby("date")[f"buy_cost{slip}"].sum().to_dict()
    reserved = rows.groupby("date")[f"reservation{slip}"].sum().to_dict()
    sale_cash = sold.groupby("accounting_exit_date")[f"verified_proceeds{slip}"].sum().to_dict()
    freed_cost = sold.groupby("accounting_exit_date")[f"buy_cost{slip}"].sum().to_dict()
    div_cash = dividends.groupby("dividend_pay_date").dividend_gross.sum().to_dict()
    tax_cash = dividends.groupby("dividend_tax_recognition_date").dividend_tax_accrued.sum().to_dict()
    buy_count = bought.groupby("date").size().to_dict()
    sale_count = sold.groupby("accounting_exit_date").size().to_dict()
    open_lots, cost_basis, flow = {}, 0.0, 0.0
    records = []
    for day in calendar:
        before = flow
        orders = bought.loc[bought.date.eq(day)]
        for r in orders.itertuples():
            open_lots[r.code] = open_lots.get(r.code, 0) + 1
        both = sold.loc[sold.accounting_exit_date.eq(day)]
        spent = float(buy_cash.get(day, 0))
        cost_basis += spent
        after_buys = before - spent
        proceeds, dividend, tax = (float(mapping.get(day, 0)) for mapping in
                                    (sale_cash, div_cash, tax_cash))
        flow = after_buys + proceeds + dividend - tax
        records.append({"date": day, "cash_change_before": before,
                        "actual_buy_cash": spent, "planned_reservation": float(reserved.get(day, 0)),
                        "cash_change_after_buys": after_buys, "sell_proceeds": proceeds,
                        "dividend_received": dividend, "dividend_tax_reserved": tax,
                        "cash_change_end": flow, "open_lots_before_sells": sum(open_lots.values()),
                        "overlapping_codes_before_sells": sum(n > 1 for n in open_lots.values()),
                        "open_purchase_cost_before_sells": cost_basis,
                        "buy_orders": int(buy_count.get(day, 0)),
                        "sell_orders": int(sale_count.get(day, 0))})
        cost_basis -= float(freed_cost.get(day, 0))
        for r in both.itertuples():
            open_lots[r.code] -= 1
            if open_lots[r.code] < 0:
                raise ValueError("Selling a position before it was purchased")
    frame = pd.DataFrame(records)
    actual = max(0.0, -float(frame[["cash_change_after_buys", "cash_change_end"]].min().min()))
    planned_need = frame.planned_reservation - frame.cash_change_before
    planned = max(actual, float(planned_need.max()))
    unknown = bought.loc[bought.unknown_after_buy]
    if (sum(open_lots.values()) != len(unknown)
            or not np.isclose(cost_basis, unknown[f"buy_cost{slip}"].sum(), atol=1e-7, rtol=0)):
        raise ValueError("Unresolved bought positions disappeared from open inventory")
    result = {"slippage_bps": slip, "signals": len(rows), "bought": len(bought), "sold": len(sold),
              "minimum_initial_actual_cash": actual,
              "minimum_initial_planned_reservation": planned,
              "planned_capital_peak_date": frame.loc[planned_need.idxmax(), "date"],
              "maximum_open_lots_before_sells": int(frame.open_lots_before_sells.max()),
              "days_with_overlapping_symbol_lots": int(frame.overlapping_codes_before_sells.gt(0).sum()),
              "ending_cash_change": flow, "unresolved_open_lots": len(unknown),
              "unresolved_purchase_cost": float(unknown[f"buy_cost{slip}"].sum()),
              "complete_portfolio_return": None,
              "terminal_asset_value": None if len(unknown) else 0.0}
    return frame, result


def evaluate(output: Path = ROOT) -> dict:
    verify(output)
    audit = json.loads((output / "quote_audit.json").read_text())
    if sha(output / "quotes.parquet") != audit["quotes_sha256"]:
        raise ValueError("Frozen raw quote slice changed")
    original = pd.read_parquet(SOURCE)
    events = execution_events(original)
    quotes = pd.read_parquet(output / "quotes.parquet")
    capacity = capacity_audit(events, quotes)
    capacity.to_parquet(output / "capacity.parquet", index=False)
    signals = pd.read_parquet(SIGNALS, columns=["date", "code", "price_1449", "preclose", "isST"])
    rows = attach_reservations(original, signals)
    dates = pd.read_parquet(CALENDAR)
    calendar = sorted(dates.loc[dates.is_trading_day.eq("1")
                               & dates.calendar_date.between(rows.date.min(), "2025-12-31"),
                               "calendar_date"].tolist())
    results, frames = [], []
    for key, part in rows.groupby(BOOK, sort=True):
        book = dict(zip(BOOK, key, strict=True))
        cap = capacity.loc[capacity.target_notional.eq(key[0]) & capacity.horizon.eq(key[1])
                           & capacity.candidate.eq(key[2])]
        for slip in (5, 15):
            ledger, summary = funding_ledger(part, calendar, slip)
            results.append({**book, **summary,
                            "raw_price_comparisons": int(cap.orders.sum()),
                            "capacity_exceeded_stock_days": int((~cap.capacity_passed).sum()),
                            "same_window_buy_sell_stock_days": int(cap.both_sides.sum()),
                            "max_aggregate_participation": float(cap.total_participation.max()),
                            "all_original_fills_jointly_within_volume_cap": bool(cap.capacity_passed.all())})
            frames.append(ledger.assign(**book, slippage_bps=slip))
    pd.concat(frames, ignore_index=True).to_parquet(output / "funding_ledger.parquet", index=False)
    report = {"execution_events": len(events), "raw_stock_days": len(quotes),
              "books": results, "holdout_read": False,
              "qualification": "Ex-post minimum financing of fixed books; buys precede same-window "
                               "sale credits and payment-day dividends. Aggregate capacity failure "
                               "invalidates joint fills. No terminal valuation, optimized-capital "
                               "return, bid/ask queue claim, order resizing or reselection."}
    save_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "quotes", "evaluate"))
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    result = {"freeze": freeze, "quotes": collect_quotes, "evaluate": evaluate}[args.stage](args.output)
    print({k: v for k, v in result.items() if k not in {"sha256", "raw_sha256", "books"}})


if __name__ == "__main__":
    main()
