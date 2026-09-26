"""Catalogue-based economic scenarios; never overwrite verified/unknown P&L."""

from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .corporate_cash import save_json, sha
from .fixed_return_ranges import economic_return, corner_return_range
from .hf_outcomes import Assumptions, _fill
from .quote_precision import quote_cents
from .reference_gain_eval import DAILY
from .reference_gain_pairs import ROOT


def complete_mean(frame: pd.DataFrame, column: str) -> float | None:
    return None if frame[column].isna().any() else float(frame.groupby("date")[column].mean().mean())


def weekly_interval(daily: pd.Series) -> list[float] | None:
    if daily.isna().any() or not len(daily):
        return None
    x = daily.to_frame("value")
    x["week"] = pd.to_datetime(x.index).to_period("W-SUN").astype(str)
    blocks = x.groupby("week").value.agg(["sum", "count"])
    rng = np.random.default_rng(20260926)
    indices = rng.integers(len(blocks), size=(10000, len(blocks)))
    means = blocks["sum"].to_numpy()[indices].sum(axis=1) / blocks["count"].to_numpy()[indices].sum(axis=1)
    return np.quantile(means, [.025, .975]).tolist()


def evaluate(output: Path = ROOT, *, bootstrap: bool = True,
             catalog_path: Path = Path("data/research/cash_dividend_catalog/events_augmented.parquet"),
             allow_unsettled_payments: bool = False) -> dict:
    report = json.loads((output / "execution_report.json").read_text())
    for name in ("accounted_initial", "raw_windows", "window_quality"):
        if sha(output / (name + ".parquet")) != report["output_sha256"][name]:
            raise ValueError("Frozen raw execution outputs changed")
    original = pd.read_parquet(output / "accounted_initial.parquet")
    rows = original.copy()
    catalog = pd.read_parquet(catalog_path)
    last_date = report.get("last_date", "2025-12-31")
    events = original.loc[original.entry_status.eq("filled"),
        ["date", "code", "horizon", "exit_date"]].merge(catalog, on="code", how="inner")
    events = events.loc[events.dividOperateDate.gt(events.date)
        & events.dividOperateDate.le(events.exit_date)].copy()
    events.to_parquet(output / "holding_catalog_events.parquet", index=False)
    if events.duplicated(["date", "code", "horizon"]).any():
        raise ValueError("Multiple distributions need chronological accounting")
    indexed = events.set_index(["date", "code", "horizon"])
    rows["catalog_sold_shares"] = rows.shares
    rows["catalog_dividend_gross"] = 0.
    rows["catalog_dividend_tax"] = 0.
    rows["catalog_action_applied"] = False
    rows["catalog_share_fill_checked"] = False
    for i, row in rows.iterrows():
        key = (row.date, row.code, row.horizon)
        if key not in indexed.index:
            continue
        e = indexed.loc[key]
        if not (row.entry_status == "filled" and row.date <= e.dividRegistDate < row.exit_date
                and row.date < e.dividOperateDate <= row.exit_date
                and e.dividOperateDate <= e.dividPayDate
                and (allow_unsettled_payments or e.dividPayDate <= last_date)
                and (pd.Timestamp(row.exit_date) - pd.Timestamp(row.date)).days <= 30):
            raise ValueError("Catalogue entitlement needs separate treatment")
        bonus = Decimal(e.dividStocksPs or "0")
        reserve = Decimal(e.dividReserveToStockPs or "0")
        if bonus:
            raise ValueError("Taxable bonus-share case is outside this cash/reserve scenario")
        quantity = Decimal(int(row.shares)) * (1 + reserve)
        if quantity != quantity.to_integral_value():
            raise ValueError("Fractional share entitlement cannot be inferred")
        if reserve:
            if not e.dividOperateDate <= e.dividStockMarketDate <= row.exit_date:
                raise ValueError("New shares are not yet listed at the recorded exit")
            path = DAILY / (row.code.replace(".", "_") + ".parquet")
            day = pd.read_parquet(path, filters=[("date", "==", row.exit_date)]).iloc[0]
            raw = pd.read_parquet(output / "raw_windows.parquet",
                filters=[("code", "==", row.code), ("date", "==", row.exit_date)])
            quote = pd.Series({"volume": raw.volume.sum(), "vwap": raw.turnover.sum() / raw.volume.sum()})
            price, status = _fill(quote, day, row.code, "sell", int(quantity), Assumptions(target_notional=row.target_notional))
            if status != "filled" or abs(price - row.exit_price) > 1e-12:
                raise ValueError("Changed share count changes the planned exit; replay is required")
            rows.loc[i, "catalog_share_fill_checked"] = True
        gross = float(Decimal(int(row.shares)) * Decimal(e.dividCashPsBeforeTax))
        rows.loc[i, ["catalog_sold_shares", "catalog_dividend_gross", "catalog_dividend_tax"]] = (
            int(quantity), gross, gross * .2)
        rows.loc[i, "catalog_action_applied"] = True
    bought = rows.entry_status.eq("filled")
    unresolved = bought & rows.exit_price.isna()
    usable = bought & ~unresolved
    if rows.loc[usable, ["entry_price", "exit_price"]].isna().any().any():
        raise ValueError("Recorded economic scenario is missing an execution")
    for slip in (5, 15):
        buy_price = rows.entry_price / 1.0005 * (1 + slip / 10000)
        sell_price = rows.exit_price / .9995 * (1 - slip / 10000)
        point = pd.Series(np.nan, index=rows.index)
        point.loc[~bought & rows.entry_window_status.eq("valid")] = 0.
        part = rows.loc[usable]
        point.loc[usable] = economic_return(part.shares, part.catalog_sold_shares,
            buy_price.loc[usable], sell_price.loc[usable], part.catalog_dividend_gross, part.catalog_dividend_tax)
        rows[f"catalog_scenario_return{slip}"] = point
        low, high = point.copy(), point.copy()
        buy_bad = ~part.entry_window_status.eq("valid")
        sell_bad = ~part.exit_window_status.eq("valid")
        buy_low = part.entry_window_low.map(lambda p: quote_cents(p) / 100) * (1 + slip / 10000)
        buy_high = part.entry_window_high.map(lambda p: quote_cents(p) / 100) * (1 + slip / 10000)
        sell_low = part.exit_window_low.map(lambda p: quote_cents(p) / 100) * (1 - slip / 10000)
        sell_high = part.exit_window_high.map(lambda p: quote_cents(p) / 100) * (1 - slip / 10000)
        lo, hi = corner_return_range(part.shares, part.catalog_sold_shares,
            np.where(buy_bad, buy_low, buy_price.loc[usable]), np.where(buy_bad, buy_high, buy_price.loc[usable]),
            np.where(sell_bad, sell_low, sell_price.loc[usable]), np.where(sell_bad, sell_high, sell_price.loc[usable]),
            part.catalog_dividend_gross, part.catalog_dividend_tax)
        low.loc[usable], high.loc[usable] = lo, hi
        rows[f"catalog_scenario_lower{slip}"], rows[f"catalog_scenario_upper{slip}"] = low, high
    # Do not promote a catalogue assumption or a price scenario to verified P&L.
    pd.testing.assert_series_equal(rows.known_return5, original.known_return5)
    pd.testing.assert_series_equal(rows.known_return15, original.known_return15)
    rows.to_parquet(output / "catalog_scenario.parquet", index=False, compression="zstd")
    cells, contrasts, annual = [], [], []
    for (half, horizon, arm), part in rows.groupby(["half", "horizon", "arm"]):
        cells.append({"half": half, "horizon": int(horizon), "arm": arm, "rows": len(part),
            "bought_fraction": float(part.entry_status.eq("filled").mean()),
            "catalogue_actions": int(part.catalog_action_applied.sum()),
            "verified_unknown_rows": int(part.known_return5.isna().sum()),
            **{f"scenario_{kind}{slip}": complete_mean(part, f"catalog_scenario_{kind}{slip}")
               for slip in (5, 15) for kind in ("return", "lower", "upper")}})
    for horizon, part in rows.groupby("horizon"):
        paired = part.loc[part.arm.eq("high")].merge(part.loc[part.arm.eq("low")],
            on=["date", "pair_id", "half"], validate="one_to_one", suffixes=("_high", "_low"))
        for slip in (5, 15):
            paired["edge"] = paired[f"catalog_scenario_return{slip}_high"] - paired[f"catalog_scenario_return{slip}_low"]
            paired["lower"] = paired[f"catalog_scenario_lower{slip}_high"] - paired[f"catalog_scenario_upper{slip}_low"]
            paired["upper"] = paired[f"catalog_scenario_upper{slip}_high"] - paired[f"catalog_scenario_lower{slip}_low"]
            for half, p in paired.groupby("half"):
                contrasts.append({"half": half, "horizon": int(horizon), "slip": slip,
                    "pairs": len(p), **{k: complete_mean(p, k) for k in ("edge", "lower", "upper")}})
            for year, p in paired.groupby(paired.date.str[:4]):
                daily = p.groupby("date").edge.agg(lambda x: x.mean() if x.notna().all() else np.nan)
                annual.append({"year": year, "horizon": int(horizon), "slip": slip,
                    "edge": complete_mean(p, "edge"),
                    "weekly_interval": weekly_interval(daily) if bootstrap else None})
    result = {"interpretation": "conditional_on_catalogue_actions_and_recorded_fills_not_verified_complete_PnL",
        "catalog_sha256": sha(catalog_path),
        "source_invalid_rows": int((~rows.execution_source_valid).sum()),
        "unresolved_terminal_rows": int(unresolved.sum()),
        "catalogue_action_rows": int(rows.catalog_action_applied.sum()),
        "share_change_rechecked_rows": int(rows.catalog_share_fill_checked.sum()),
        "known_returns_unchanged": True, "bootstrap_requested": bootstrap,
        "by_half": cells, "contrasts": contrasts, "annual": annual,
        "holdout_read": report.get("holdout_read", False),
        "allow_unsettled_payments": allow_unsettled_payments,
        "output_sha256": sha(output / "catalog_scenario.parquet")}
    save_json(output / "catalog_scenario_report.json", result)
    return result


if __name__ == "__main__":
    result = evaluate()
    print(json.dumps(result, ensure_ascii=False, indent=2))
