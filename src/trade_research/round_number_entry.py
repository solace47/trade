"""Frozen entry feasibility for the full integer-side conditional diagnostic.

Only the four entry minutes of fixed stock-days are read. Unknown source data
are retained separately from modeled no fills. No exit price is loaded here.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import threading

import duckdb
import numpy as np
import pandas as pd

from .corporate_cash import MINUTES, save_json, sha
from .execution_path import LABELS, minute_participation
from .hf_outcomes import Assumptions, _fees, _fill
from .quote_precision import fixed_quote_shares
from .round_number_1449 import ROOT as SIDES
from .round_number_geometry import ROOT as GEOMETRY


ROOT = GEOMETRY / "entry"
EXPECTED_SIDES = "9d39c32ff75e6ad2d9ff479560608790c455194b2bc679dfc1831be006e4939e"
EXPECTED_WEIGHTS = "2ad79a64cafcd5a8b26c412e0e6c98134fc407318e41a3667ae7593587443eb5"
BAR_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]
_LOCAL = threading.local()


def freeze(output: Path = ROOT) -> dict:
    side_path, weight_path = SIDES / "side_inputs.parquet", GEOMETRY / "contrast_inputs.parquet"
    if sha(side_path) != EXPECTED_SIDES or sha(weight_path) != EXPECTED_WEIGHTS:
        raise ValueError("Preregistered integer-side inputs changed")
    report = json.loads((GEOMETRY / "input_report.json").read_text())
    if not report["input_gate_passed"]:
        raise ValueError("Input geometry did not pass")
    sides = pd.read_parquet(side_path)
    weights = pd.read_parquet(weight_path)
    selected = sides.merge(weights[["date", "code", "contrast_weight", "input_day_passed"]],
                           on=["date", "code"], validate="one_to_one")
    selected = selected.loc[selected.input_day_passed].sort_values(["date", "code"])
    if (len(selected) != 114657 or selected.date.nunique() != 391
            or selected.duplicated(["date", "code"]).any()
            or not selected.date.between("2024-01-01", "2025-12-31").all()
            or not selected.code.str.startswith(("sh.60", "sz.00")).all()):
        raise ValueError("Unexpected fixed entry population")
    sources = [side_path, weight_path, GEOMETRY / "input_report.json", GEOMETRY / "manifest.json"]
    manifest = {"rule_commit": "0b5ff3f", "sha256": {str(p): sha(p) for p in sources},
                "stock_days": len(selected), "days": selected.date.nunique(),
                "entry_labels": list(LABELS), "notionals": [20000, 100000],
                "slippage_bps": 5, "exit_prices_read": False, "holdout_read": False}
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if {k: v for k, v in existing.items() if k != "signals_sha256"} != manifest:
            raise ValueError("Cannot replace frozen entry inputs")
        if sha(output / "signals.parquet") != existing["signals_sha256"]:
            raise ValueError("Existing frozen entry signals changed")
    selected.to_parquet(output / "signals.parquet", index=False)
    manifest["signals_sha256"] = sha(output / "signals.parquet")
    save_json(path, manifest)
    return manifest


def validate_window(bars: pd.DataFrame) -> tuple[str, dict | None]:
    if bars.empty:
        return "missing_window", None
    bars = bars.sort_values("timestamp")
    labels = bars.timestamp.dt.strftime("%H%M").tolist()
    if (labels != list(LABELS) or bars.timestamp.dt.normalize().nunique() != 1
            or not bars.timestamp.eq(bars.timestamp.dt.floor("min")).all()):
        return "incomplete_or_duplicate_window", None
    values = bars[BAR_COLUMNS[1:]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        return "invalid_numeric_bar", None
    o, h, l, c, v, a = values.T
    if (np.any(np.minimum.reduce([o, h, l, c]) <= 0)
            or np.any(h + .0001 < np.maximum.reduce([o, c, l]))
            or np.any(l - .0001 > np.minimum(o, c))
            or np.any(v < 0) or np.any(a < 0) or np.any((v == 0) != (a == 0))):
        return "invalid_numeric_bar", None
    positive = v > 0
    minute_vwap = a[positive] / v[positive]
    if np.any(minute_vwap < l[positive] - .0101) or np.any(minute_vwap > h[positive] + .0101):
        return "vwap_outside_bar_range", None
    volume, turnover = float(v.sum()), float(a.sum())
    return "valid", {"volume": volume, "turnover": turnover,
                     "vwap": turnover / volume if volume else 0.0}


def evaluate_day(signal: dict, bars: pd.DataFrame, source_present: bool = True) -> list[dict]:
    status, quote = validate_window(bars) if source_present else ("missing_source", None)
    valid = status == "valid"
    daily = pd.Series({"date": signal["date"], "preclose": signal["preclose"],
                       "tradestatus": 1, "isST": 0})
    assumptions = Assumptions(slippage_bps_each_side=5)
    vwap = quote["vwap"] if quote is not None else None
    priced = vwap is not None and vwap > 0
    same_side = ((vwap > signal["round_yuan"] if signal["side"] == "above"
                  else vwap < signal["round_yuan"]) if priced else None)
    rows = []
    for notional in (20000, 100000):
        shares = fixed_quote_shares(signal["code"], signal["price_1449"], notional)
        price, aggregate_status = None, "source_unknown"
        path_shares, path_price = None, None
        if valid:
            if shares == 0:
                aggregate_status, path_shares = "below_minimum_lot", 0
            else:
                price, aggregate_status = _fill(pd.Series(quote), daily, signal["code"],
                                               "buy", shares, assumptions)
                path_shares, path_price = minute_participation(
                    bars.sort_values("timestamp"), signal["code"], signal["date"],
                    signal["preclose"], shares, 5)
        filled = aggregate_status == "filled"
        cost = (shares * price + _fees(shares * price, "buy", assumptions, signal["date"])) if filled else None
        rows.append({"date": signal["date"], "code": signal["code"], "half": signal["half"],
            "side": signal["side"], "notional": notional, "target_shares": shares,
            "window_status": status, "bars_valid": valid, "raw_vwap": vwap,
            "window_volume": quote["volume"] if quote else None,
            "decision_drift_bps": 10000 * (vwap / (signal["quote_cents"] / 100) - 1) if priced else None,
            "original_side_retained": same_side, "aggregate_status": aggregate_status,
            "aggregate_filled": filled, "aggregate_price": price, "aggregate_buy_cost": cost,
            "excess_cash_yuan": max(0, cost - notional) if cost is not None else None,
            "participation_shares": path_shares, "participation_price": path_price,
            "participation_filled": path_shares == shares if path_shares is not None else False,
            "participation_extra_bps": (10000 * (path_price / price - 1)
                if filled and path_shares == shares and path_price is not None else None)})
    return rows


def _connection():
    if not hasattr(_LOCAL, "connection"):
        _LOCAL.connection = duckdb.connect()
        _LOCAL.connection.execute("SET threads=1")
    return _LOCAL.connection


def _stock(item: tuple[str, pd.DataFrame], minute_root: Path) -> tuple[list[dict], pd.DataFrame, dict]:
    code, signals = item
    exchange, symbol = code.split(".")
    path = minute_root / exchange.upper() / f"{symbol}.parquet"
    present = path.exists()
    source = {"code": code, "path": str(path), "exists": present}
    if present:
        before = path.stat()
        connection = _connection()
        connection.register("wanted_dates", signals[["date"]])
        bars = connection.execute("""
            SELECT m.timestamp, m.open, m.high, m.low, m.close, m.volume, m.turnover
            FROM read_parquet(?) m JOIN wanted_dates w
              ON cast(m.timestamp AS DATE) = cast(w.date AS DATE)
            WHERE m.timestamp >= ? AND m.timestamp < ?
              AND strftime(m.timestamp, '%H%M') IN ('1452','1453','1454','1455')
            ORDER BY m.timestamp
        """, [str(path), pd.Timestamp(signals.date.min()),
              pd.Timestamp(signals.date.max()) + pd.Timedelta(days=1)]).df()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Raw minute source changed during read: {path}")
        source.update(size=before.st_size, mtime_ns=before.st_mtime_ns)
    else:
        bars = pd.DataFrame({name: pd.Series(dtype="datetime64[ns]" if name == "timestamp" else float)
                             for name in BAR_COLUMNS})
    bars["date"] = bars.timestamp.dt.strftime("%Y-%m-%d")
    bars["code"] = code
    source["extracted_rows"] = len(bars)
    source["extracted_content_sha256"] = hashlib.sha256(
        pd.util.hash_pandas_object(bars, index=False).to_numpy().tobytes()).hexdigest()
    groups = {date: part for date, part in bars.groupby("date", sort=False)}
    empty = bars.iloc[:0]
    rows = []
    for signal in signals.to_dict("records"):
        rows.extend(evaluate_day(signal, groups.get(signal["date"], empty), present))
    return rows, bars, source


def evaluate(output: Path = ROOT, minute_root: Path = MINUTES, workers: int = 4) -> dict:
    manifest = json.loads((output / "manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        if sha(Path(name)) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    if sha(output / "signals.parquet") != manifest["signals_sha256"]:
        raise ValueError("Frozen entry signals changed")
    signals = pd.read_parquet(output / "signals.parquet")
    grouped = list(signals.groupby("code", sort=True))
    entries, windows, sources = [], [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for count, (rows, bars, source) in enumerate(pool.map(lambda item: _stock(item, minute_root), grouped), 1):
            entries.extend(rows)
            windows.append(bars)
            sources.append(source)
            if count % 200 == 0 or count == len(grouped):
                print(f"Entry windows: {count}/{len(grouped)} stocks", flush=True)
    result = pd.DataFrame(entries).sort_values(["date", "code", "notional"])
    raw = pd.concat(windows, ignore_index=True).sort_values(["date", "code", "timestamp"])
    if len(result) != 2 * len(signals) or result.duplicated(["date", "code", "notional"]).any():
        raise ValueError("Entry rows differ from frozen attempts")
    result.to_parquet(output / "entries.parquet", index=False)
    raw.to_parquet(output / "raw_windows.parquet", index=False)
    save_json(output / "source_index.json", sources)
    cells = []
    for (half, side, notional), part in result.groupby(["half", "side", "notional"], sort=True):
        priced = part.loc[part.raw_vwap.gt(0)]
        filled = part.loc[part.aggregate_filled]
        partial = part.participation_shares.gt(0) & part.participation_shares.lt(part.target_shares)
        both = part.loc[part.aggregate_filled & part.participation_filled]
        cells.append({"half": half, "side": side, "notional": int(notional), "stock_days": len(part),
            "days": part.date.nunique(), "valid_window_rate": float(part.bars_valid.mean()),
            "aggregate_fill_rate": float(part.aggregate_filled.mean()),
            "participation_fill_rate": float(part.participation_filled.mean()),
            "partial_participation_count": int(partial.sum()),
            "window_status": part.window_status.value_counts().to_dict(),
            "aggregate_status": part.aggregate_status.value_counts().to_dict(),
            "priced_count": len(priced), "same_original_side_rate": float(priced.original_side_retained.mean()) if len(priced) else None,
            "day_mean_decision_drift_bps": float(priced.groupby("date").decision_drift_bps.mean().mean()) if len(priced) else None,
            "day_mean_participation_extra_bps": float(both.groupby("date").participation_extra_bps.mean().mean()) if len(both) else None,
            "extra_cash_count": int(filled.excess_cash_yuan.gt(0).sum()),
            "maximum_extra_cash_yuan": float(filled.excess_cash_yuan.max()) if len(filled) else None})
    passed = len(cells) == 16 and all(
        c["valid_window_rate"] >= .999 and (c["notional"] != 20000 or
        (c["aggregate_fill_rate"] >= .95 and c["participation_fill_rate"] >= .95)) for c in cells)
    report = {"rule_commit": manifest["rule_commit"], "stock_days": len(signals), "days": signals.date.nunique(),
        "by_half_side_notional": cells, "entry_gate_passed": passed,
        "sha256": {p.name: sha(p) for p in [output / "entries.parquet", output / "raw_windows.parquet", output / "source_index.json"]},
        "exit_prices_read": False, "holding_returns_read": False, "holdout_read": False}
    save_json(output / "entry_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "evaluate"))
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    report = freeze(args.output) if args.stage == "freeze" else evaluate(args.output, workers=args.workers)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
