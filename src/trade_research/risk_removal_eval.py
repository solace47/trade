"""Half-cent minimum price impact on the fixed removal-event execution ledger."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .corporate_cash import save_json, sha
from .fixed_return_ranges import economic_return, corner_return_range
from .quote_precision import quote_cents
from .reference_gain_accounting import complete_mean, weekly_interval
from .risk_removal_1449 import ROOT


def cost_price(raw, bps: int, side: str):
    if bps not in (5, 15) or side not in ("buy", "sell"):
        raise ValueError("Only the two predeclared cost scenarios are supported")
    raw = np.asarray(raw, dtype=float)
    if not np.isfinite(raw).all() or (raw <= .005).any():
        raise ValueError("A cost scenario requires finite positive executable prices")
    impact = np.maximum(raw * bps / 10000, .005)
    return raw + impact if side == "buy" else raw - impact


def add_cost_scenarios(original: pd.DataFrame) -> pd.DataFrame:
    rows = original.copy()
    bought = rows.entry_status.eq("filled")
    usable = bought & rows.exit_price.notna()
    part = rows.loc[usable]
    for bps in (5, 15):
        buy = cost_price(part.entry_price / 1.0005, bps, "buy")
        sell = cost_price(part.exit_price / .9995, bps, "sell")
        point = pd.Series(np.nan, index=rows.index)
        point.loc[~bought & rows.entry_window_status.eq("valid")] = 0.
        point.loc[usable] = economic_return(part.shares, part.catalog_sold_shares,
            buy, sell, part.catalog_dividend_gross, part.catalog_dividend_tax)
        lower, upper = point.copy(), point.copy()
        bad_buy, bad_sell = ~part.entry_window_status.eq("valid"), ~part.exit_window_status.eq("valid")
        bounds = {}
        for side in ("entry", "exit"):
            for which in ("low", "high"):
                raw = part[f"{side}_window_{which}"].map(lambda p: quote_cents(p) / 100)
                bounds[side, which] = cost_price(raw, bps, "buy" if side == "entry" else "sell")
        lo, hi = corner_return_range(part.shares, part.catalog_sold_shares,
            np.where(bad_buy, bounds["entry", "low"], buy), np.where(bad_buy, bounds["entry", "high"], buy),
            np.where(bad_sell, bounds["exit", "low"], sell), np.where(bad_sell, bounds["exit", "high"], sell),
            part.catalog_dividend_gross, part.catalog_dividend_tax)
        lower.loc[usable], upper.loc[usable] = lo, hi
        rows[f"tick_return{bps}"], rows[f"tick_lower{bps}"], rows[f"tick_upper{bps}"] = point, lower, upper
    pd.testing.assert_frame_equal(rows[original.columns], original)
    return rows


def evaluate(output: Path = ROOT / "continued") -> dict:
    base = json.loads((output / "catalog_scenario_report.json").read_text())
    if sha(output / "catalog_scenario.parquet") != base["output_sha256"]:
        raise ValueError("The fixed catalogue scenario changed")
    rows = add_cost_scenarios(pd.read_parquet(output / "catalog_scenario.parquet"))
    rows.to_parquet(output / "tick_cost_scenario.parquet", index=False, compression="zstd")
    cells, contrasts, annual = [], [], []
    for (half, horizon, arm), part in rows.groupby(["half", "horizon", "arm"]):
        cells.append({"half": half, "horizon": int(horizon), "arm": arm, "rows": len(part),
            "dates": part.date.nunique(), "bought": int(part.entry_status.eq("filled").sum()),
            "actual_unknown": int(part.known_return5.isna().sum()),
            **{f"{kind}{bps}": complete_mean(part, f"tick_{kind}{bps}")
               for bps in (5, 15) for kind in ("return", "lower", "upper")}})
    pairs = []
    for horizon, part in rows.groupby("horizon"):
        paired = part.loc[part.arm.eq("high")].merge(part.loc[part.arm.eq("low")],
            on=["date", "pair_id", "half"], validate="one_to_one", suffixes=("_high", "_low"))
        for bps in (5, 15):
            paired[f"edge{bps}"] = paired[f"tick_return{bps}_high"] - paired[f"tick_return{bps}_low"]
            paired[f"lower{bps}"] = paired[f"tick_lower{bps}_high"] - paired[f"tick_upper{bps}_low"]
            paired[f"upper{bps}"] = paired[f"tick_upper{bps}_high"] - paired[f"tick_lower{bps}_low"]
            for half, p in paired.groupby("half"):
                contrasts.append({"half": half, "horizon": int(horizon), "bps": bps, "pairs": len(p),
                    "dates": p.date.nunique(), **{k: complete_mean(p, f"{k}{bps}") for k in ("edge", "lower", "upper")}})
            own = part.loc[part.arm.eq("high"), ["date", f"tick_return{bps}"]].rename(columns={f"tick_return{bps}": "value"})
            edge = paired[["date", f"edge{bps}"]].rename(columns={f"edge{bps}": "value"})
            for metric, frame in (("own", own), ("edge", edge)):
                for year, g in frame.groupby(frame.date.str[:4]):
                    daily = g.groupby("date").value.agg(lambda x: x.mean() if x.notna().all() else np.nan)
                    annual.append({"year": year, "horizon": int(horizon), "bps": bps, "metric": metric,
                        "rows": len(g), "dates": len(daily), "weeks": pd.to_datetime(daily.index).to_period("W-SUN").nunique(),
                        "mean": complete_mean(g, "value"), "weekly_interval": weekly_interval(daily)})
        paired["horizon"] = horizon
        pairs.append(paired)
    pd.concat(pairs, ignore_index=True).to_parquet(output / "tick_pairs.parquet", index=False, compression="zstd")
    result = {"interpretation": "catalogue_and_recorded_fill_scenario_not_verified_or_shared_capital_return",
        "primary": "T5_max_15bps_or_half_cent_per_share_per_leg",
        "cost_floor_yuan": .005, "by_half": cells, "contrasts": contrasts, "annual": annual,
        "baseline_report_sha256": sha(output / "catalog_scenario_report.json"),
        "tick_scenario_sha256": sha(output / "tick_cost_scenario.parquet"),
        "original_columns_unchanged": True, "holdout_prices_read": False}
    save_json(output / "tick_report.json", result)
    return result


if __name__ == "__main__":
    result = evaluate()
    print(json.dumps({"by_half": [r for r in result["by_half"] if r["horizon"] == 5 and r["arm"] == "high"],
        "contrasts": [r for r in result["contrasts"] if r["horizon"] == 5 and r["bps"] == 15],
        "annual": [r for r in result["annual"] if r["horizon"] == 5 and r["bps"] == 15]},
        ensure_ascii=False, indent=2))
