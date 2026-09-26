"""Conditional execution-price ranges for an unchanged rejected old cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .corporate_cash import save_json, sha
from .fill_accounting import KEY
from .hf_outcomes import Assumptions, _fees
from .fixed_execution_quality import ROOT as QUALITY
from .orderbook_funding import SOURCE
from .quote_precision import quote_cents


ROOT = Path("data/research/fixed_return_ranges")


def freeze(output: Path = ROOT) -> dict:
    report = json.loads((QUALITY / "report.json").read_text())
    for name, digest in report["sha256"].items():
        if sha(QUALITY / name) != digest:
            raise ValueError("Fixed execution quality audit changed")
    source_report = json.loads(SOURCE.with_name("report.json").read_text())
    if sha(SOURCE) != source_report["accounted_sha256"]:
        raise ValueError("Old accounting source changed")
    files = [SOURCE, SOURCE.with_name("report.json"), QUALITY / "report.json",
             QUALITY / "raw_windows.parquet", QUALITY / "window_quality.parquet"]
    manifest = {"rule_commit": "bd2eee7", "sha256": {str(p): sha(p) for p in files},
        "scenarios": ["flagged_windows_only", "all_windows"], "slippage_bps_each_side": 5,
        "legacy_cohort": True, "new_market_prices_read": False, "holdout_read": False,
        "unknown_terminal_values_filled": False, "original_trades_changed": False}
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Cannot replace frozen old-return range sources")
    save_json(path, manifest)
    return manifest


def economic_return(buy_quantity, sell_quantity, buy_price, sell_price, dividend_gross, dividend_tax):
    """The existing 2024--2025 commission/transfer/stamp model; prices include slippage."""
    qbuy, qsell, pbuy, psell, dividend, tax = np.broadcast_arrays(
        *[np.asarray(v, dtype=float) for v in (buy_quantity, sell_quantity, buy_price, sell_price, dividend_gross, dividend_tax)])
    if (not all(np.isfinite(a).all() for a in (qbuy, qsell, pbuy, psell, dividend, tax))
            or any(np.any(a <= 0) for a in (qbuy, qsell, pbuy, psell))
            or np.any(dividend < 0) or np.any(tax < 0)):
        raise ValueError("Known economic returns need finite executed quantities, prices and cash terms")
    buy_value, sell_value = qbuy * pbuy, qsell * psell
    cost = buy_value + np.maximum(5., buy_value * .0003) + buy_value * .00001
    proceeds = sell_value - np.maximum(5., sell_value * .0003) - sell_value * (.00001 + .0005)
    return (proceeds + dividend - tax) / cost - 1


def dated_economic_return(buy_quantity, sell_quantity, buy_price, sell_price,
                          dividend_gross, dividend_tax, buy_dates, sell_dates):
    """Same cash model with the existing execution engine's historical fee dates."""
    # Reuse strict validation, keeping every old caller's 2024--2025 formula intact.
    modern = economic_return(buy_quantity, sell_quantity, buy_price, sell_price,
                             dividend_gross, dividend_tax)
    qb, qs, pb, ps, dividend, tax, bought, sold = np.broadcast_arrays(
        buy_quantity, sell_quantity, buy_price, sell_price, dividend_gross,
        dividend_tax, np.asarray(buy_dates, dtype=str), np.asarray(sell_dates, dtype=str))
    if any(len(day) != 10 or not "2022-01-01" <= day <= "2025-12-31"
           for day in np.concatenate([bought.ravel(), sold.ravel()])) or np.any(sold <= bought):
        raise ValueError("Historical economic returns require ordered supported execution dates")
    if np.all(bought >= "2024-01-01"):
        return modern
    terms = Assumptions()
    buy_value, sell_value = qb.astype(float)*pb.astype(float), qs.astype(float)*ps.astype(float)
    buy_fees = np.asarray([_fees(float(value), "buy", terms, str(day))
                          for value, day in zip(buy_value.ravel(), bought.ravel())]).reshape(buy_value.shape)
    sell_fees = np.asarray([_fees(float(value), "sell", terms, str(day))
                           for value, day in zip(sell_value.ravel(), sold.ravel())]).reshape(sell_value.shape)
    return (sell_value-sell_fees+dividend.astype(float)-tax.astype(float))/(buy_value+buy_fees)-1


def corner_return_range(buy_quantity, sell_quantity, buy_low, buy_high, sell_low, sell_high,
                        dividend_gross, dividend_tax):
    if np.any(np.asarray(buy_low) > np.asarray(buy_high)) or np.any(np.asarray(sell_low) > np.asarray(sell_high)):
        raise ValueError("Price ranges must be ordered")
    corners = [economic_return(buy_quantity, sell_quantity, pbuy, psell, dividend_gross, dividend_tax)
               for pbuy in (buy_low, buy_high) for psell in (sell_low, sell_high)]
    return np.min(corners, axis=0), np.max(corners, axis=0)


def complete_summary(part: pd.DataFrame) -> dict:
    unknown = part.unknown_terminal
    if part.loc[~unknown, ["baseline", "lower", "upper"]].isna().any().any():
        raise ValueError("Unexpected missing value in a supposedly known return range")
    return {"rows": len(part), "days": part.date.nunique(), "unknown_terminal_rows": int(unknown.sum()),
            **{name + "_bps": None if unknown.any() else float(part.groupby("date")[name].mean().mean() * 10000)
               for name in ("baseline", "lower", "upper")}}


def pair_ranges(part: pd.DataFrame) -> pd.DataFrame:
    model = part.loc[part.candidate.eq("absolute_model")]
    control = part.loc[part.candidate.eq("same_day_control")]
    pairs = model.merge(control, on=["date", "pair_id"], validate="one_to_one", suffixes=("_model", "_control"))
    if len(pairs) != len(control):
        raise ValueError("A frozen control lost its model counterpart")
    result = pairs[["date", "pair_id", "code_model", "code_control"]].copy()
    result["unknown_terminal"] = pairs.unknown_terminal_model | pairs.unknown_terminal_control
    result["baseline"] = pairs.baseline_model - pairs.baseline_control
    result["lower"] = pairs.lower_model - pairs.upper_control
    result["upper"] = pairs.upper_model - pairs.lower_control
    return result


def stock_price_ranges(raw: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    if raw.duplicated(["date", "code", "timestamp"]).any() or quality.duplicated(["date", "code"]).any():
        raise ValueError("Raw price keys must be unique")
    positive = raw.loc[raw.volume.gt(0)].copy()
    quotes = {p: quote_cents(p) for p in pd.unique(pd.concat([positive.low, positive.high], ignore_index=True))}
    positive["low_cent"] = positive.low.map(quotes)
    positive["high_cent"] = positive.high.map(quotes)
    ranges = positive.groupby(["date", "code"], sort=True).agg(low_cent=("low_cent", "min"), high_cent=("high_cent", "max")).reset_index()
    ranges["low_price"] = ranges.low_cent / 100
    ranges["high_price"] = ranges.high_cent / 100
    return ranges.merge(quality[["date", "code", "source_checks_passed"]], on=["date", "code"], validate="one_to_one")


def evaluate(output: Path = ROOT) -> dict:
    manifest = json.loads((output / "manifest.json").read_text())
    for name, digest in manifest["sha256"].items():
        if sha(Path(name)) != digest:
            raise ValueError(f"Frozen return-range source changed: {name}")
    rows = pd.read_parquet(SOURCE).sort_values(KEY).reset_index(drop=True)
    raw = pd.read_parquet(QUALITY / "raw_windows.parquet")
    quality = pd.read_parquet(QUALITY / "window_quality.parquet")
    prices = stock_price_ranges(raw, quality)
    if len(rows) != 12648 or rows.duplicated(KEY).any() or not rows.date.between("2024-01-01", "2025-12-31").all():
        raise ValueError("Old fixed accounting grid changed")
    buy = prices.rename(columns={"low_price": "buy_low", "high_price": "buy_high", "source_checks_passed": "buy_source_passed"})
    sell = prices.rename(columns={"date": "accounting_exit_date", "low_price": "sell_low", "high_price": "sell_high", "source_checks_passed": "sell_source_passed"})
    attached = rows.merge(buy[["date", "code", "buy_low", "buy_high", "buy_source_passed"]], on=["date", "code"], how="left", validate="many_to_one")
    attached = attached.merge(sell[["accounting_exit_date", "code", "sell_low", "sell_high", "sell_source_passed"]], on=["accounting_exit_date", "code"], how="left", validate="many_to_one")
    bought = attached.entry_status.eq("filled")
    known = bought & ~attached.unknown_after_buy
    if attached.loc[known, ["buy_low", "buy_high", "sell_low", "sell_high"]].isna().any().any():
        raise ValueError("Known recorded trade lost its executed window")
    current = attached.loc[known]
    dividend, tax = current.dividend_gross.fillna(0), current.dividend_tax_accrued.fillna(0)
    computed = economic_return(current.shares, current.sold_shares, current.entry_price,
                               current.accounting_exit_price, dividend, tax)
    baseline = pd.Series(np.nan, index=attached.index)
    baseline.loc[~bought] = 0.
    baseline.loc[known] = computed
    observed = ~attached.unknown_after_buy
    error = float((baseline.loc[observed] - attached.loc[observed, "known_return5"]).abs().max())
    if error > 1e-12 or baseline.loc[attached.unknown_after_buy].notna().any():
        raise ValueError("Original economic accounting did not reproduce")
    frames = []
    for scenario in manifest["scenarios"]:
        result = attached[[*KEY, "candidate", "half", "pair_id"]].copy()
        result["scenario"] = scenario
        result["unknown_terminal"] = attached.unknown_after_buy
        result["baseline"] = baseline
        result["lower"] = baseline
        result["upper"] = baseline
        vary_buy = ~current.buy_source_passed.eq(True) if scenario == "flagged_windows_only" else np.ones(len(current), dtype=bool)
        vary_sell = ~current.sell_source_passed.eq(True) if scenario == "flagged_windows_only" else np.ones(len(current), dtype=bool)
        lo, hi = corner_return_range(current.shares, current.sold_shares,
            np.where(vary_buy, current.buy_low * 1.0005, current.entry_price),
            np.where(vary_buy, current.buy_high * 1.0005, current.entry_price),
            np.where(vary_sell, current.sell_low * .9995, current.accounting_exit_price),
            np.where(vary_sell, current.sell_high * .9995, current.accounting_exit_price), dividend, tax)
        result.loc[known, "lower"], result.loc[known, "upper"] = lo, hi
        result["changed_buy_range"] = bought & (~attached.buy_source_passed.eq(True) if scenario == "flagged_windows_only" else True)
        result["changed_sell_range"] = known & (~attached.sell_source_passed.eq(True) if scenario == "flagged_windows_only" else True)
        frames.append(result)
    ranges = pd.concat(frames, ignore_index=True)
    cells, contrasts, pair_parts = [], [], []
    for (scenario, amount, horizon, half, arm), part in ranges.groupby(["scenario", "target_notional", "horizon", "half", "candidate"], sort=True):
        cells.append({"scenario": scenario, "notional": int(amount), "horizon": int(horizon), "half": half,
            "arm": arm, **complete_summary(part), "changed_buy_windows": int(part.changed_buy_range.sum()),
            "changed_sell_windows": int(part.changed_sell_range.sum())})
    for (scenario, amount, horizon, half), part in ranges.groupby(["scenario", "target_notional", "horizon", "half"], sort=True):
        pairs = pair_ranges(part)
        descriptors = {"scenario": scenario, "notional": int(amount), "horizon": int(horizon), "half": half}
        contrasts.append({**descriptors, **complete_summary(pairs),
            "unpaired_model_rows": int(part.candidate.eq("absolute_model").sum() - len(pairs))})
        pair_parts.append(pairs.assign(**descriptors))
    ranges.to_parquet(output / "return_ranges.parquet", index=False)
    pd.concat(pair_parts, ignore_index=True).to_parquet(output / "paired_ranges.parquet", index=False)
    prices.to_parquet(output / "price_ranges.parquet", index=False)
    report = {"rule_commit": manifest["rule_commit"], "original_rows": len(rows), "unknown_terminal_rows": int(attached.unknown_after_buy.sum()),
        "maximum_baseline_return_error": error, "by_half_book_scenario": cells, "paired_contrasts": contrasts,
        "qualification": "Conditional independent endpoint outer envelopes; original statuses and cash entitlements fixed; not attainable joint fills, confidence intervals or a revived strategy",
        "new_market_prices_read": False, "holdout_read": False, "original_trades_changed": False,
        "sha256": {p.name: sha(p) for p in sorted(output.glob("*.parquet"))}}
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
