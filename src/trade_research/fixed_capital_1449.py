"""Pretrade cash allocation for frozen, conditionally accounted tail orders.

No ranking is retrained. Source-uncertain fills remain conditional. Cash is not
marked portfolio equity, and an unsettled book has no complete terminal return.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .corporate_cash import save_json, sha
from .hf_outcomes import Assumptions, _fees, _limit_price
from .quote_precision import fixed_quote_shares
from .risk_removal_eval import add_cost_scenarios, cost_price
from .turnover_reference import CALENDAR

ROOT = Path("data/research/fixed_capital_1449")
RULE_COMMIT = "37cb9c2"
SOURCES = {
    "long_ridge": Path("data/research/long_history_ridge_1449/continued/long"),
    "long_tree": Path("data/research/long_history_tree_1449/continued/tree"),
    "robust18": Path("data/research/alpha158_1449/continued/robust18"),
    "alpha158": Path("data/research/alpha158_1449/continued/alpha158"),
}
FIRST, LAST = "2024-07-01", "2025-12-31"


def prepare(output: Path = ROOT) -> dict:
    if (output / "report.json").exists():
        raise ValueError("Do not replace fixed cash inputs after evaluating allocation")
    output.mkdir(parents=True, exist_ok=True)
    manifests, frames, capacities = {}, [], []
    calendar = pd.read_parquet(CALENDAR)
    allowed = set(calendar.loc[calendar.is_trading_day.eq("1")
        & calendar.calendar_date.between(FIRST, LAST), "calendar_date"])
    for name, folder in SOURCES.items():
        selection = json.loads((folder / "input_report.json").read_text())
        execution = json.loads((folder / "execution_report.json").read_text())
        accounting = json.loads((folder / "catalog_scenario_report.json").read_text())
        if (sha(folder / "signals.parquet") != selection["signals_sha256"]
                or sha(folder / "catalog_scenario.parquet") != accounting["output_sha256"]
                or sha(folder / "raw_windows.parquet") != execution["output_sha256"]["raw_windows"]):
            raise ValueError("A frozen source list, accounting table or execution window changed")
        rows = add_cost_scenarios(pd.read_parquet(folder / "catalog_scenario.parquet"))
        rows = rows.loc[rows.horizon.eq(5)].copy()
        signals = pd.read_parquet(folder / "signals.parquet",
            columns=["date", "code", "daily_rank", "price_1449", "preclose", "isST"])
        rows = rows.drop(columns=[c for c in signals if c not in ("date", "code") and c in rows])
        rows = rows.merge(signals, on=["date", "code"], validate="one_to_one")
        if (not set(rows.date).issubset(allowed) or not set(rows.exit_date.dropna()).issubset(allowed)
                or rows.duplicated(["date", "code"]).any() or rows.isST.ne(0).any()
                or not rows.code.str.startswith(("sh.60", "sz.00")).all()):
            raise ValueError("Unexpected dates, identities or board in the frozen funding pool")
        rows["model"] = name
        rows["order_id"] = rows.date + ":" + rows.code
        rows["planned_shares"] = [fixed_quote_shares(r.code, r.price_1449, 20000) for r in rows.itertuples()]
        rows["known_upper"] = rows.preclose.map(lambda p: _limit_price(p, .1, True))
        bought = rows.entry_status.eq("filled")
        sold = bought & rows.exit_price.notna()
        if rows.loc[bought, "shares"].ne(rows.loc[bought, "planned_shares"]).any():
            raise ValueError("Previously fixed lot quantities changed")
        events = pd.read_parquet(folder / "holding_catalog_events.parquet")
        events = events.loc[events.horizon.eq(5), ["date", "code", "dividOperateDate", "dividPayDate"]]
        rows = rows.merge(events, on=["date", "code"], how="left", validate="one_to_one")
        if rows.catalog_action_applied.ne(rows.dividOperateDate.notna()).any():
            raise ValueError("An accounted entitlement lost its event timing")
        event_rows = rows.loc[rows.catalog_action_applied]
        if (event_rows.dividOperateDate.le(event_rows.date).any()
                or event_rows.dividOperateDate.gt(event_rows.exit_date).any()
                or event_rows.dividPayDate.lt(event_rows.dividOperateDate).any()
                or event_rows.dividPayDate.gt(LAST).any()):
            raise ValueError("Distribution dates fall outside their accounted holdings or cash horizon")
        for bps in (5, 15):
            ceiling = cost_price(rows.known_upper / 1.0005, bps, "buy")
            reserved_value = rows.planned_shares * ceiling
            rows[f"reservation{bps}"] = [value + _fees(value, "buy", Assumptions(), date)
                for value, date in zip(reserved_value, rows.date)]
            buy_value = rows.loc[bought, "shares"] * cost_price(rows.loc[bought, "entry_price"] / 1.0005, bps, "buy")
            sell_value = rows.loc[sold, "catalog_sold_shares"] * cost_price(rows.loc[sold, "exit_price"] / .9995, bps, "sell")
            rows[f"cash_buy{bps}"], rows[f"cash_sell{bps}"] = 0., np.nan
            rows.loc[bought, f"cash_buy{bps}"] = [value + _fees(value, "buy", Assumptions(), date)
                for value, date in zip(buy_value, rows.loc[bought, "date"])]
            rows.loc[sold, f"cash_sell{bps}"] = [value - _fees(value, "sell", Assumptions(), date)
                for value, date in zip(sell_value, rows.loc[sold, "exit_date"])]
            if (rows[f"cash_buy{bps}"] > rows[f"reservation{bps}"] + 1e-8).any():
                raise ValueError("A source fill exceeds the decision-time cash ceiling")
            reconstructed = (rows.loc[sold, f"cash_sell{bps}"] + rows.loc[sold, "catalog_dividend_gross"]
                - rows.loc[sold, "catalog_dividend_tax"]) / rows.loc[sold, f"cash_buy{bps}"] - 1
            np.testing.assert_allclose(reconstructed, rows.loc[sold, f"tick_return{bps}"], atol=1e-12, rtol=0)
        bars = pd.read_parquet(folder / "raw_windows.parquet")
        quotes = bars.groupby(["date", "code"]).agg(volume=("volume", "sum"), amount=("turnover", "sum"))
        quotes["vwap"] = quotes.amount / quotes.volume
        for side, mask, key, multiplier in (("entry", bought, "date", 1.0005), ("exit", sold, "exit_date", .9995)):
            selected = quotes.reindex(pd.MultiIndex.from_arrays([rows.loc[mask, key], rows.loc[mask, "code"]]))
            np.testing.assert_allclose(selected.vwap.to_numpy() * multiplier,
                rows.loc[mask, f"{side}_price"].to_numpy(), atol=1e-12, rtol=0)
        for arm, group in rows.groupby("arm"):
            active = group.loc[group.entry_status.eq("filled")]
            buys = active[["date", "code", "shares"]].rename(columns={"shares": "quantity"})
            sells = active.loc[active.exit_price.notna(), ["exit_date", "code", "catalog_sold_shares"]]
            sells = sells.rename(columns={"exit_date": "date", "catalog_sold_shares": "quantity"})
            total = pd.concat([buys, sells]).groupby(["date", "code"]).quantity.sum()
            cap = total.to_frame().join(quotes[["volume"]])
            cap["participation"] = cap.quantity / cap.volume
            if cap.volume.isna().any() or cap.volume.le(0).any() or cap.participation.gt(.1 + 1e-12).any():
                raise ValueError("The complete fixed book exceeds shared execution capacity")
            capacities.append({"model": name, "arm": arm, "stock_windows": len(cap),
                "max_aggregate_participation": float(cap.participation.max())})
        source_names = ("signals.parquet", "input_report.json", "execution_report.json", "raw_windows.parquet",
                        "catalog_scenario.parquet", "catalog_scenario_report.json", "holding_catalog_events.parquet")
        manifests[name] = {str(folder / f): sha(folder / f) for f in source_names}
        frames.append(rows)
    all_rows = pd.concat(frames, ignore_index=True).sort_values(["model", "arm", "date", "daily_rank", "code"])
    all_rows.to_parquet(output / "orders.parquet", index=False, compression="zstd")
    report = {"rule_commit": RULE_COMMIT, "initial_capital": 500000, "first_date": FIRST, "last_date": LAST,
        "source_sha256": manifests, "calendar_sha256": sha(CALENDAR), "rows": len(all_rows),
        "capacity": capacities, "orders_sha256": sha(output / "orders.parquet"),
        "cash_policy_results_read": False, "new_market_prices_read": False, "holdout_prices_read": False}
    save_json(output / "manifest.json", report)
    return report


def cash_book(rows: pd.DataFrame, days: list[str], bps: int,
              initial: float = 500000) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if not np.isfinite(initial) or initial <= 0 or bps not in (5, 15) or not days or days != sorted(set(days)):
        raise ValueError("A positive capital base, fixed scenario and ordered dates are required")
    if rows.order_id.duplicated().any() or not set(rows.date).issubset(days):
        raise ValueError("Cash orders must have unique identities within the accounting dates")
    if not np.isfinite(rows[f"reservation{bps}"]).all() or rows[f"reservation{bps}"].le(0).any():
        raise ValueError("Every proposed order needs a finite positive pretrade reservation")
    cash, positions, receivables = float(initial), {}, {}
    entries = {day: g.sort_values(["daily_rank", "code"]).to_dict("records") for day, g in rows.groupby("date")}
    ledger, decisions = [], []
    counts = {"authorized": 0, "capital_rejected": 0, "entry_rejected": 0, "bought": 0, "sold": 0,
              "actual_unknown_bought": 0, "source_invalid_bought": 0}
    for day in days:
        cash_before = cash
        for uid, position in positions.items():
            if position["dividOperateDate"] == day:
                position["tax_due"] = position["catalog_dividend_tax"]
                receivables[uid] = (position["dividPayDate"], position["catalog_dividend_gross"])
        reserved_tax = sum(p["tax_due"] for p in positions.values())
        available = cash - reserved_tax
        # Decide every instruction before inspecting any same-window fill.
        plans = []
        for row in entries.get(day, []):
            reserve = row[f"reservation{bps}"]
            authorized = available + 1e-8 >= reserve
            decisions.append({"order_id": row["order_id"], "date": day, "code": row["code"],
                "daily_rank": row["daily_rank"], "available_before": available,
                "reservation": reserve, "authorized": authorized})
            counts["authorized" if authorized else "capital_rejected"] += 1
            if authorized:
                available -= reserve
                plans.append(row)
        spent = 0.
        for row in plans:
            if row["entry_status"] != "filled":
                counts["entry_rejected"] += 1
                continue
            cost = row[f"cash_buy{bps}"]
            if not np.isfinite(cost) or cost <= 0 or cost > row[f"reservation{bps}"] + 1e-8:
                raise ValueError("An authorized fill violates its pretrade cash reservation")
            cash -= cost
            spent += cost
            positions[row["order_id"]] = {**row, "tax_due": 0.}
            counts["bought"] += 1
            counts["actual_unknown_bought"] += int(pd.isna(row["known_return5"]))
            counts["source_invalid_bought"] += int(not row["execution_source_valid"])
        before_sells = len(positions)
        cost_before_sells = sum(p[f"cash_buy{bps}"] for p in positions.values())
        cash_after_buys = cash
        sell_cash, taxes = 0., 0.
        for uid, position in list(positions.items()):
            if position["exit_date"] != day:
                continue
            if day <= position["date"] or not np.isfinite(position[f"cash_sell{bps}"]):
                raise ValueError("A recorded sale must be after entry and have known conditional proceeds")
            if abs(position["tax_due"] - position["catalog_dividend_tax"]) > 1e-8:
                raise ValueError("A sale tax was not reserved from its entitlement date")
            sell_cash += position[f"cash_sell{bps}"]
            taxes += position["tax_due"]
            del positions[uid]
            counts["sold"] += 1
        dividend = 0.
        for uid, (payday, gross) in list(receivables.items()):
            if payday == day:
                dividend += gross
                del receivables[uid]
        cash += sell_cash + dividend - taxes
        tax_end = sum(p["tax_due"] for p in positions.values())
        if min(cash_after_buys, cash) < -1e-6:
            raise ValueError("The fixed-capital policy borrowed cash")
        ledger.append({"date": day, "cash_before": cash_before, "tax_reserved_before": reserved_tax,
            "cash_buy": spent, "cash_after_buys": cash_after_buys, "sell_proceeds": sell_cash,
            "dividend_received": dividend, "dividend_tax_paid": taxes, "cash_end": cash,
            "spendable_end": cash - tax_end, "tax_reserved_end": tax_end,
            "open_lots_before_sells": before_sells, "purchase_cost_before_sells": cost_before_sells,
            "open_lots_end": len(positions), "unpaid_dividend_gross": sum(r[1] for r in receivables.values())})
    daily = pd.DataFrame(ledger)
    decisions_frame = pd.DataFrame(decisions)
    complete = not positions and not receivables
    expected_cash = initial + (daily.sell_proceeds + daily.dividend_received - daily.dividend_tax_paid - daily.cash_buy).sum()
    if abs(cash - expected_cash) > 1e-6:
        raise ValueError("The cash book does not reconcile with its recorded flows")
    result = {"initial_capital": initial, "signals": len(rows), "bps": bps, **counts,
        "ending_cash": cash, "complete_conditional_settlement": complete,
        "conditional_profit": cash - initial if complete else None,
        "conditional_capital_return": cash / initial - 1 if complete else None,
        "remaining_open_lots": len(positions), "unpaid_dividend_gross": float(daily.unpaid_dividend_gross.iloc[-1]),
        "remaining_purchase_cost": sum(p[f"cash_buy{bps}"] for p in positions.values()),
        "minimum_cash": float(daily[["cash_after_buys", "cash_end"]].min().min()),
        "minimum_spendable_cash": float(daily.spendable_end.min()),
        "maximum_open_lots_before_sells": int(daily.open_lots_before_sells.max()),
        "maximum_open_purchase_cost": float(daily.purchase_cost_before_sells.max()),
        "marked_drawdown": None, "verified_actual_portfolio_return": None}
    return daily, decisions_frame, result


def evaluate(output: Path = ROOT) -> dict:
    manifest = json.loads((output / "manifest.json").read_text())
    if sha(output / "orders.parquet") != manifest["orders_sha256"]:
        raise ValueError("Frozen cash instructions changed")
    rows = pd.read_parquet(output / "orders.parquet")
    days = pd.date_range(FIRST, LAST).strftime("%Y-%m-%d").tolist()
    ledgers, orders, summaries = [], [], []
    for (model, arm), group in rows.groupby(["model", "arm"]):
        for bps in (5, 15):
            daily, choices, summary = cash_book(group, days, bps, manifest["initial_capital"])
            ledgers.append(daily.assign(model=model, arm=arm, bps=bps))
            orders.append(choices.assign(model=model, arm=arm, bps=bps))
            summaries.append({"model": model, "arm": arm, **summary})
    pd.concat(ledgers, ignore_index=True).to_parquet(output / "cash_ledger.parquet", index=False, compression="zstd")
    pd.concat(orders, ignore_index=True).to_parquet(output / "order_decisions.parquet", index=False, compression="zstd")
    report = {"interpretation": "fixed_capital_conditional_catalogue_and_recorded_fill_scenario_not_verified_actual_return",
        "primary": "T5_500000_yuan_max_15bps_or_half_cent_per_leg", "rule_commit": RULE_COMMIT,
        "manifest_sha256": sha(output / "manifest.json"), "books": summaries,
        "ledger_sha256": sha(output / "cash_ledger.parquet"), "decisions_sha256": sha(output / "order_decisions.parquet"),
        "new_market_prices_read": False, "holdout_prices_read": False}
    save_json(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "evaluate"))
    args = parser.parse_args()
    print(json.dumps({"prepare": prepare, "evaluate": evaluate}[args.stage](), ensure_ascii=False, indent=2))
