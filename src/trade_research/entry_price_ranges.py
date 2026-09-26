"""Conditional price-range sensitivity of already observed entry diagnostics.

Ranges assume the raw minute OHLC bounds are usable. They neither repair the
source nor establish fillability, statistical uncertainty, or holding returns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .corporate_cash import save_json, sha
from .quote_precision import fixed_quote_shares, quote_cents
from .round_number_entry import ROOT as ENTRY


ROOT = Path("data/research/entry_price_ranges")


def freeze(output: Path = ROOT) -> dict:
    entry = json.loads((ENTRY / "entry_report.json").read_text())
    for name, digest in entry["sha256"].items():
        if sha(ENTRY / name) != digest:
            raise ValueError("Frozen entry source changed")
    files = [ENTRY / name for name in ("entry_report.json", "manifest.json", "signals.parquet", "entries.parquet", "raw_windows.parquet")]
    manifest = {"rule_commit": "39ca43c", "sha256": {str(p): sha(p) for p in files},
        "scope": "All fixed integer-side entry stock-days; original contrast weights unchanged",
        "source_repaired": False, "holding_returns_read": False, "new_market_prices_read": False,
        "holdout_read": False}
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Cannot replace the frozen range-analysis sources")
    save_json(path, manifest)
    return manifest


def linear_interval(weights, lower, upper) -> tuple[float, float]:
    w, lo, hi = map(lambda a: np.asarray(a, dtype=float), (weights, lower, upper))
    if (w.ndim != 1 or lo.shape != w.shape or hi.shape != w.shape or len(w) == 0
            or not all(np.isfinite(a).all() for a in (w, lo, hi)) or np.any(lo > hi)):
        raise ValueError("Need matching finite weights and ordered price ranges")
    return (float(np.sum(w * np.where(w >= 0, lo, hi))),
            float(np.sum(w * np.where(w >= 0, hi, lo))))


def hypothetical_buy_cash(shares, raw_price):
    value = np.asarray(shares, dtype=float) * np.asarray(raw_price, dtype=float) * 1.0005
    return value + np.maximum(5., value * .0003) + value * .00001


def build_ranges(signals: pd.DataFrame, entries: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    if (signals.duplicated(["date", "code"]).any() or entries.duplicated(["date", "code"]).any()
            or raw.duplicated(["date", "code", "timestamp"]).any()):
        raise ValueError("Range analysis requires unique fixed keys")
    if set(raw.timestamp.dt.strftime("%H%M")) != {"1452", "1453", "1454", "1455"}:
        raise ValueError("Only the four frozen entry minutes may enter a range")
    values = raw[["volume", "turnover", "low", "high"]].to_numpy(dtype=float)
    if (not np.isfinite(values).all() or np.any(values[:, :2] < 0)
            or np.any(values[:, 2:] <= 0) or np.any(values[:, 2] > values[:, 3])):
        raise ValueError("OHLC range assumptions cannot be evaluated on invalid fields")
    raw = raw.copy()
    for field in ("volume", "turnover", "low", "high"):
        raw[field] = raw[field].astype(float)
    valid_quotes = {value: quote_cents(value) for value in pd.unique(pd.concat([raw.low, raw.high], ignore_index=True))}
    raw["low_quote"] = raw.low.map(valid_quotes) / 100
    raw["high_quote"] = raw.high.map(valid_quotes) / 100
    totals = raw.groupby(["date", "code"], sort=True).agg(
        bars=("timestamp", "size"), volume=("volume", "sum"), amount=("turnover", "sum"))
    positive = raw.loc[raw.volume.gt(0)]
    quotes = positive.groupby(["date", "code"], sort=True).agg(
        lower_raw_price=("low_quote", "min"), upper_raw_price=("high_quote", "max"))
    totals = totals.join(quotes).reset_index()
    if not totals.bars.eq(4).all() or not totals.volume.gt(0).all() or totals.isna().any().any():
        raise ValueError("Every original key needs all four bars and a positive-trade range")
    joined = signals[["date", "code", "half", "side", "quote_cents", "price_1449", "contrast_weight"]].merge(
        entries[["date", "code", "bars_valid", "target_shares"]], on=["date", "code"], how="left", validate="one_to_one")
    joined = joined.merge(totals, on=["date", "code"], how="left", validate="one_to_one")
    if len(joined) != len(signals) or joined.isna().any().any():
        raise ValueError("Range calculation lost frozen source rows")
    expected = [fixed_quote_shares(c, p, 20000) for c, p in zip(joined.code, joined.price_1449)]
    if not np.array_equal(joined.target_shares.to_numpy(), np.array(expected)):
        raise ValueError("Decision-time share quantities changed")
    joined["unverified_raw_vwap"] = joined.amount / joined.volume
    joined["unverified_vwap_outside_ohlc"] = (~joined.unverified_raw_vwap.between(joined.lower_raw_price, joined.upper_raw_price))
    denominator = joined.quote_cents / 100
    for name, price in (("baseline", joined.unverified_raw_vwap), ("lower", joined.lower_raw_price), ("upper", joined.upper_raw_price)):
        joined[name + "_drift_bps"] = (price * 1.0005 / denominator - 1) * 10000
        joined[name + "_cash"] = hypothetical_buy_cash(joined.target_shares, price)
    joined["cash_range_yuan"] = joined.upper_cash - joined.lower_cash
    joined["source_flagged"] = ~joined.bars_valid
    return joined


def evaluate(output: Path = ROOT) -> dict:
    manifest = json.loads((output / "manifest.json").read_text())
    for name, digest in manifest["sha256"].items():
        if sha(Path(name)) != digest:
            raise ValueError(f"Frozen range source changed: {name}")
    signals = pd.read_parquet(ENTRY / "signals.parquet")
    entries = pd.read_parquet(ENTRY / "entries.parquet")
    entries = entries.loc[entries.notional.eq(20000)]
    raw = pd.read_parquet(ENTRY / "raw_windows.parquet")
    ranges = build_ranges(signals, entries, raw)
    if (len(ranges) != 114657 or ranges.date.nunique() != 391 or int(ranges.source_flagged.sum()) != 284
            or not ranges.date.between("2024-01-01", "2025-12-31").all()):
        raise ValueError("Fixed price diagnostic population changed")
    daily_rows = []
    for date, part in ranges.groupby("date", sort=True):
        baseline = float(part.contrast_weight @ part.baseline_drift_bps)
        for scenario in ("flagged_windows_only", "all_windows"):
            vary = part.source_flagged if scenario == "flagged_windows_only" else np.ones(len(part), dtype=bool)
            lo = np.where(vary, part.lower_drift_bps, part.baseline_drift_bps)
            hi = np.where(vary, part.upper_drift_bps, part.baseline_drift_bps)
            lower, upper = linear_interval(part.contrast_weight, lo, hi)
            daily_rows.append({"date": date, "half": part.half.iloc[0], "scenario": scenario,
                "stock_days": len(part), "source_flagged": int(part.source_flagged.sum()),
                "unverified_baseline_bps": baseline, "lower_bps": lower, "upper_bps": upper,
                "width_bps": upper - lower})
    daily = pd.DataFrame(daily_rows)
    summaries = []
    for (half, scenario), part in daily.groupby(["half", "scenario"], sort=True):
        summaries.append({"half": half, "scenario": scenario, "days": len(part),
            **{key: float(part[key].mean()) for key in ("unverified_baseline_bps", "lower_bps", "upper_bps", "width_bps")},
            "largest_day_width_bps": float(part.width_bps.max())})
    cash = []
    for (half, side), part in ranges.groupby(["half", "side"], sort=True):
        cash.append({"half": half, "side": side, "stock_days": len(part),
            "source_flagged": int(part.source_flagged.sum()),
            "raw_vwap_outside_ohlc": int(part.unverified_vwap_outside_ohlc.sum()),
            "flagged_raw_vwap_outside_ohlc": int((part.source_flagged & part.unverified_vwap_outside_ohlc).sum()),
            "median_conditional_cash_range_yuan": float(part.cash_range_yuan.median()),
            "p95_conditional_cash_range_yuan": float(part.cash_range_yuan.quantile(.95)),
            "maximum_conditional_cash_range_yuan": float(part.cash_range_yuan.max())})
    ranges.to_parquet(output / "stock_day_ranges.parquet", index=False)
    daily.to_parquet(output / "daily_ranges.parquet", index=False)
    report = {"rule_commit": manifest["rule_commit"], "stock_days": len(ranges),
        "by_half_scenario": summaries, "cash_and_source_descriptions": cash,
        "qualification": "Conditional OHLC envelope; not fillability, confidence intervals, a source repair or holding returns",
        "sha256": {p.name: sha(p) for p in sorted(output.glob("*.parquet"))},
        "source_repaired": False, "holding_returns_read": False, "new_market_prices_read": False, "holdout_read": False}
    save_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "evaluate"))
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    report = freeze(args.output) if args.stage == "freeze" else evaluate(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
