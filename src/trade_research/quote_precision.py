"""Audit exact-cent decision quotes; never quantize an execution VWAP this way."""

from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .corporate_cash import DAILY, MINUTES, save_json, sha
from .hf_outcomes import Assumptions, _fill, _order_shares


ROOT = Path("data/research/quote_precision")
SIGNALS = Path("data/research/absolute_ridge_1449_target/repricing_signals.parquet")
SELECTIONS = SIGNALS.with_name("selections.parquet")
TRADES = Path("data/research/holding_exceptions/accounted.parquet")


def quote_cents(price: float) -> int:
    value = Decimal(str(price))
    if not value.is_finite() or value <= 0:
        raise ValueError("Decision quote must be finite and positive")
    cents = (value * 100).to_integral_value(rounding=ROUND_HALF_UP)
    if cents < 1 or abs(value - cents / 100) > Decimal(".0001"):
        raise ValueError("Decision quote is not within tolerance of a valid cent")
    return int(cents)


def fixed_quote_shares(code: str, price: float, target_notional: float) -> int:
    """Exact shares for a single A-share decision quote, not a VWAP estimate."""
    budget = Decimal(str(target_notional)) * 100
    if not budget.is_finite() or budget <= 0 or budget != budget.to_integral_value():
        raise ValueError("Target budget must be a positive whole number of cents")
    affordable = int(budget) // quote_cents(price)
    if code.startswith("sh.68"):
        return affordable if affordable >= 200 else 0
    return affordable // 100 * 100


def freeze(output: Path = ROOT) -> dict:
    report = json.loads(TRADES.with_name("report.json").read_text())
    if sha(TRADES) != report["accounted_sha256"]:
        raise ValueError("Frozen holding source changed")
    selections = pd.read_parquet(SELECTIONS, columns=["date", "code", "candidate"])
    if (len(selections) != 3162 or selections.duplicated(["date", "code"]).any()
            or not selections.date.between("2024-01-01", "2025-12-31").all()):
        raise ValueError("Frozen selection keys changed")
    manifest = {"rule_commit": "a170234", "selected_stock_days": len(selections),
                "sha256": {str(p): sha(p) for p in (SIGNALS, SELECTIONS, TRADES)},
                "quote_tolerance_yuan": .0001, "holdout_read": False,
                "sizing_difference_read": False}
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Cannot replace the frozen sizing-precision inputs")
    save_json(path, manifest)
    return manifest


def verify(output: Path) -> dict:
    manifest = json.loads((output / "manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        if sha(Path(name)) != expected:
            raise ValueError(f"Sizing-precision source changed: {name}")
    return manifest


def audit(output: Path = ROOT) -> dict:
    verify(output)
    selected = pd.read_parquet(SELECTIONS, columns=["date", "code", "candidate"])
    inputs = pd.read_parquet(SIGNALS, columns=["date", "code", "price_1449", "isST",
                                              "listing_age_sessions", "reference_gap",
                                              "quote_outside_traded_range"])
    selected = selected.merge(inputs, on=["date", "code"], how="left", validate="one_to_one")
    valid_cents, invalid = [], []
    for row in selected.itertuples():
        try:
            valid_cents.append(quote_cents(row.price_1449))
        except ValueError:
            invalid.append({"date": row.date, "code": row.code, "price_1449": str(row.price_1449)})
    if invalid:
        report = {"outcome_gate_passed": False, "off_tick_quotes": invalid,
                  "holdout_read": False, "returns_read": False}
        save_json(output / "input_report.json", report)
        return report
    if (selected.isST.ne(0).any() or selected.listing_age_sessions.lt(20).any()
            or selected.reference_gap.any() or selected.quote_outside_traded_range.any()):
        raise ValueError("Entry-only replay requires the unchanged pretrade quality gates")
    selected["quote_cents"] = valid_cents
    records = []
    for row in selected.itertuples():
        for amount in (20000, 100000):
            old = _order_shares(row.code, row.price_1449, amount)
            restored_float = _order_shares(row.code, row.quote_cents / 100, amount)
            exact = fixed_quote_shares(row.code, row.price_1449, amount)
            records.append({"date": row.date, "code": row.code, "candidate": row.candidate,
                            "target_notional": amount, "price_1449": row.price_1449,
                            "quote_cents": row.quote_cents, "old_shares": old,
                            "cent_restored_float_shares": restored_float,
                            "exact_shares": exact, "changed": old != exact,
                            "float_division_difference": restored_float != exact})
    frame = pd.DataFrame(records).sort_values(["date", "code", "target_notional"])
    frame["half"] = frame.date.str[:4] + np.where(frame.date.str[5:7].astype(int) <= 6, "H1", "H2")
    cells = [{"half": half, "arm": arm, "notional": int(amount), "orders": len(part),
              "changed": int(part.changed.sum()), "float_division_difference": int(part.float_division_difference.sum())}
             for (half, arm, amount), part in frame.groupby(["half", "candidate", "target_notional"], sort=True)]
    frame.to_parquet(output / "plans.parquet", index=False)
    frame.loc[frame.changed].to_parquet(output / "changed.parquet", index=False)
    report = {"outcome_gate_passed": True, "planned_orders": len(frame),
              "changed_orders": int(frame.changed.sum()),
              "cent_restored_float_division_differences": int(frame.float_division_difference.sum()),
              "max_quote_storage_error_yuan": float((selected.price_1449 - selected.quote_cents / 100).abs().max()),
              "cells": cells, "changed_sha256": sha(output / "changed.parquet"),
              "holdout_read": False, "returns_read": False}
    save_json(output / "input_report.json", report)
    return report


def entries(output: Path = ROOT) -> dict:
    from .holding_exceptions import load_quotes

    verify(output)
    report = json.loads((output / "input_report.json").read_text())
    if not report["outcome_gate_passed"]:
        raise ValueError("Quote precision input gate failed")
    if sha(output / "changed.parquet") != report["changed_sha256"]:
        raise ValueError("Changed sizing keys drifted")
    changes = pd.read_parquet(output / "changed.parquet")
    if changes.empty:
        result = {"required": False, "changed_orders": 0, "returns_read": False,
                  "holdout_read": False}
        save_json(output / "entry_report.json", result)
        return result
    # Deliberately project only entry fields; no new sale or return is inspected.
    archived = pd.read_parquet(TRADES, columns=["date", "code", "target_notional",
                                               "entry_status", "entry_price", "shares"]).drop_duplicates()
    changes = changes.merge(archived, on=["date", "code", "target_notional"],
                             how="left", validate="one_to_one")
    if changes.entry_status.isna().any():
        raise ValueError("An input change has no frozen original entry")
    rows, hashes = [], {}
    for code, group in changes.groupby("code", sort=True):
        market, symbol = code.split(".")
        for path in (MINUTES / market.upper() / f"{symbol}.parquet", DAILY / f"{market}_{symbol}.parquet"):
            hashes[str(path)] = sha(path)
        quotes, daily = load_quotes(code, group.date.min(), group.date.max())
        days = {row["date"]: row for _, row in daily.iterrows()}
        for row in group.itertuples():
            def fill(quantity):
                if not quantity:
                    return None, "below_minimum_lot"
                return _fill(quotes.get(row.date), days.get(row.date), code,
                             "buy", quantity, Assumptions())
            old_price, old_status = fill(row.old_shares)
            new_price, new_status = fill(row.exact_shares)
            if (old_status != row.entry_status
                    or (old_status == "filled" and (row.shares != row.old_shares
                        or not np.isclose(old_price, row.entry_price, atol=1e-12, rtol=0)))):
                raise ValueError("Original changed-key entry no longer reconciles")
            rows.append({"date": row.date, "code": code, "notional": row.target_notional,
                         "old_shares": row.old_shares, "exact_shares": row.exact_shares,
                         "old_status": old_status, "new_status": new_status,
                         "old_price": old_price, "new_price": new_price,
                         "window_volume": int(quotes[row.date].volume) if row.date in quotes else None})
    pd.DataFrame(rows).to_parquet(output / "entries.parquet", index=False)
    result = {"required": True, "changed_orders": len(rows),
              "entry_status_differences": sum(r["old_status"] != r["new_status"] for r in rows),
              "raw_sha256": hashes, "holdout_read": False, "returns_read": False}
    save_json(output / "entry_report.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "audit", "entries"))
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    result = {"freeze": freeze, "audit": audit, "entries": entries}[args.stage](args.output)
    print({k: v for k, v in result.items() if k not in {"sha256", "raw_sha256", "cells"}})


if __name__ == "__main__":
    main()
