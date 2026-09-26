"""Diagnose fixed minute amount/volume conflicts without estimating returns."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .corporate_cash import DAILY, MINUTES, save_json, sha
from .hf_audit import EXPECTED_LABELS
from .round_number_entry import ROOT as ENTRY, _connection


ROOT = Path("data/research/minute_amount_consistency")
REVISION = "ba589a11534825044fe5a6b84838f50ba8d8d188"
TOLERANCE = .0101


def freeze(output: Path = ROOT) -> dict:
    report = json.loads((ENTRY / "entry_report.json").read_text())
    for name, expected in report["sha256"].items():
        if sha(ENTRY / name) != expected:
            raise ValueError("Frozen entry audit changed")
    signals = pd.read_parquet(ENTRY / "signals.parquet")
    entries = pd.read_parquet(ENTRY / "entries.parquet")
    bad = entries.loc[entries.notional.eq(20000) & ~entries.bars_valid]
    if len(bad) != 284 or not bad.window_status.eq("vwap_outside_bar_range").all():
        raise ValueError("Unexpected frozen source anomaly population")
    signals["key_hash"] = [hashlib.sha256((d + c).encode()).hexdigest()
                           for d, c in zip(signals.date, signals.code)]
    reference = signals.sort_values("key_hash").groupby(["half", "side"], sort=True).head(5)
    marked = signals.merge(bad[["date", "code"]].assign(affected=True), on=["date", "code"], how="left")
    marked = marked.merge(reference[["date", "code"]].assign(reference=True), on=["date", "code"], how="left")
    for column in ("affected", "reference"):
        marked[column] = marked[column].eq(True)
    selected = marked.loc[marked.affected | marked.reference,
                          ["date", "code", "half", "side", "affected", "reference", "key_hash"]]
    selected = selected.sort_values(["date", "code"]).reset_index(drop=True)
    if (int(selected.affected.sum()) != 284 or int(selected.reference.sum()) != 40
            or selected.duplicated(["date", "code"]).any()
            or not selected.date.between("2024-01-01", "2025-12-31").all()):
        raise ValueError("Invalid fixed diagnostic cohort")
    sources = [ENTRY / name for name in ("entry_report.json", "entries.parquet", "signals.parquet", "raw_windows.parquet")]
    sources += [output / "source_docs" / "README.md", output / "source_docs" / "metadata" / "source_provenance.json"]
    manifest = {"rule_commit": "3c96d7e", "sha256": {str(p): sha(p) for p in sources},
        "stock_days": len(selected), "affected_days": 284, "reference_days": 40,
        "revision": REVISION, "holding_returns_read": False, "holdout_read": False}
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if {k: v for k, v in existing.items() if k != "selected_sha256"} != manifest:
            raise ValueError("Cannot replace fixed diagnostic scope")
        if sha(output / "selected.parquet") != existing["selected_sha256"]:
            raise ValueError("Existing diagnostic keys changed")
    selected.to_parquet(output / "selected.parquet", index=False)
    manifest["selected_sha256"] = sha(output / "selected.parquet")
    save_json(path, manifest)
    return manifest


def read_stock(item: tuple[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    code, keys = item
    exchange, symbol = code.split(".")
    relative = Path("data/stock_1m") / exchange.upper() / f"{symbol}.parquet"
    path = MINUTES.parent.parent / relative
    cache = MINUTES.parent.parent / ".cache/huggingface/download" / (str(relative) + ".metadata")
    metadata = cache.read_text().splitlines()
    if len(metadata) < 2 or metadata[0] != REVISION or re.fullmatch(r"[0-9a-f]{64}", metadata[1]) is None:
        raise ValueError(f"Missing pinned download checksum: {code}")
    digest = sha(path)
    if digest != metadata[1]:
        raise ValueError(f"Raw source differs from downloaded checksum: {code}")
    connection = _connection()
    connection.register("audit_keys", keys[["date"]])
    bars = connection.execute("""
        SELECT m.* FROM read_parquet(?) m JOIN audit_keys k
          ON cast(m.timestamp AS DATE) = cast(k.date AS DATE)
        WHERE m.timestamp >= ? AND m.timestamp < ? ORDER BY timestamp
    """, [str(path), pd.Timestamp(keys.date.min()),
          pd.Timestamp(keys.date.max()) + pd.Timedelta(days=1)]).df()
    bars["date"] = bars.timestamp.dt.strftime("%Y-%m-%d")
    bars["label"] = bars.timestamp.dt.strftime("%H%M")
    if not (bars.exchange.str.upper().eq(exchange.upper()) & bars.symbol.str.zfill(6).eq(symbol)).all():
        raise ValueError(f"Source instrument identity mismatch: {code}")
    bars["code"] = code
    daily_path = DAILY / f"{exchange}_{symbol}.parquet"
    daily = pd.read_parquet(daily_path, columns=["date", "open", "high", "low", "close", "volume", "amount", "tradestatus"],
                            filters=[("date", "in", keys.date.tolist())])
    if len(daily) != len(keys) or daily.date.duplicated().any():
        raise ValueError(f"Independent daily source missing fixed dates: {code}")
    daily["code"] = code
    source = {"code": code, "minute_sha256": digest, "download_checksum_matched": True,
              "minute_path": str(path), "daily_sha256": sha(daily_path), "daily_path": str(daily_path),
              "minute_schema": {name: str(dtype) for name, dtype in bars.dtypes.items()},
              "selected_dates": keys.date.tolist()}
    return bars, daily, source


def mark_consistency(bars: pd.DataFrame) -> pd.DataFrame:
    result = bars.copy()
    for column in ("open", "high", "low", "close", "volume", "turnover"):
        result[column] = result[column].astype(float)
    valid = (np.isfinite(result[["open", "high", "low", "close", "volume", "turnover"]]).all(axis=1)
             & result[["open", "high", "low", "close", "volume", "turnover"]].gt(0).all(axis=1))
    result["positive_finite"] = valid
    result["vwap"] = result.turnover / result.volume.where(valid)
    result["outside_yuan"] = np.maximum(result.vwap - result.high, result.low - result.vwap)
    result["inconsistent"] = valid & result.outside_yuan.gt(TOLERANCE)
    # An intentionally generous float32 rounding envelope, even though the
    # archive's declared storage is float64; this is not a correction rule.
    price_spacing = np.maximum(np.spacing(result.high.to_numpy(dtype=np.float32)),
                               np.spacing(result.low.to_numpy(dtype=np.float32))).astype(float) / 2
    amount_spacing = np.spacing(result.turnover.to_numpy(dtype=np.float32)).astype(float) / 2
    result["float32_envelope_yuan"] = price_spacing + amount_spacing / result.volume.where(valid)
    lower_volume = result.turnover / (result.high + .0001)
    upper_volume = result.turnover / (result.low - .0001)
    result["nearest_100_compatible"] = valid & (lower_volume <= result.volume + 50) & (upper_volume >= result.volume - 50)
    result["truncated_100_compatible"] = valid & (lower_volume <= result.volume + 99) & (upper_volume >= result.volume)
    return result


def amount_envelope(bars: pd.DataFrame) -> dict:
    volume = bars.volume.to_numpy(dtype=float)
    amount = float(bars.turnover.to_numpy(dtype=float).sum())
    lower = float(volume @ (bars.low.to_numpy(dtype=float) - TOLERANCE))
    upper = float(volume @ (bars.high.to_numpy(dtype=float) + TOLERANCE))
    return {"amount": amount, "weighted_low_amount": lower, "weighted_high_amount": upper,
            "within_weighted_price_envelope": lower <= amount <= upper,
            "outside_amount_yuan": max(0., lower - amount, amount - upper)}


def shifted_ranges(marked: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for offset in (-2, -1, 0, 1, 2):
        shifted = marked[["date", "code", "timestamp", "low", "high"]].copy()
        shifted["timestamp"] -= pd.Timedelta(minutes=offset)
        shifted = shifted.rename(columns={"low": "shifted_low", "high": "shifted_high"})
        joined = marked.merge(shifted, on=["date", "code", "timestamp"], how="left", validate="one_to_one")
        joined["matched"] = joined.shifted_low.notna() & joined.shifted_high.notna()
        joined["compatible"] = (joined.positive_finite & joined.matched
            & joined.vwap.ge(joined.shifted_low - TOLERANCE) & joined.vwap.le(joined.shifted_high + TOLERANCE))
        joined["offset_minutes"] = offset
        rows.append(joined[["date", "code", "timestamp", "half", "label", "affected", "reference",
                            "positive_finite", "inconsistent", "vwap", "shifted_low", "shifted_high",
                            "matched", "compatible", "offset_minutes"]])
    return pd.concat(rows, ignore_index=True)


def analyze(bars: pd.DataFrame, daily: pd.DataFrame, selected: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if bars.duplicated(["code", "timestamp"]).any():
        raise ValueError("Duplicate source bar cannot be aligned")
    marked = mark_consistency(bars).merge(selected.drop(columns="key_hash"), on=["date", "code"], validate="many_to_one")
    days = []
    daily_by_key = daily.set_index(["date", "code"])
    for (date, code), part in marked.groupby(["date", "code"], sort=True):
        part = part.sort_values("timestamp")
        source = daily_by_key.loc[(date, code)]
        traded = part.loc[part.volume.gt(0)]
        complete = (part.label.tolist() == list(EXPECTED_LABELS)
                    and part.timestamp.eq(part.timestamp.dt.floor("min")).all())
        volume = float(part.volume.sum())
        amount = float(part.turnover.sum())
        high_error = abs(float(traded.high.max()) - float(source.high))
        low_error = abs(float(traded.low.min()) - float(source.low))
        close_error = abs(float(part.close.iloc[-1]) - float(source.close))
        volume_error = abs(volume - float(source.volume))
        amount_relative_error = abs(amount / float(source.amount) - 1) if source.amount else None
        tail = part.loc[part.label.isin(("1452", "1453", "1454", "1455"))]
        row = {"date": date, "code": code, "half": part.half.iloc[0],
            "affected": bool(part.affected.iloc[0]), "reference": bool(part.reference.iloc[0]),
            "minute_rows": len(part), "complete_labels": bool(complete), "daily_trading": source.tradestatus == 1,
            "volume_error_shares": volume_error, "amount_relative_error": amount_relative_error,
            "high_error": high_error, "low_error": low_error, "close_error": close_error,
            "opening_error": abs(float(traded.open.iloc[0]) - float(source.open)) if len(traded) else None,
            "daily_totals_pass": volume_error <= 100 and amount_relative_error is not None and amount_relative_error <= .0001,
            "daily_hlc_pass": max(high_error, low_error, close_error) <= .0001,
            "positive_bars": int(part.positive_finite.sum()), "inconsistent_bars": int(part.inconsistent.sum()),
            "tail_inconsistent_bars": int(tail.inconsistent.sum()), **amount_envelope(tail)}
        days.append(row)
    day_rows = pd.DataFrame(days)
    if len(day_rows) != len(selected):
        raise ValueError("Full-day diagnostic lost frozen keys")
    shifts = shifted_ranges(marked)
    shift_summary = []
    scopes = {"frozen_anomalous_entry_bars": shifts.inconsistent & shifts.label.isin(("1452", "1453", "1454", "1455")),
              "reference_all_positive_bars": shifts.reference & shifts.positive_finite}
    for scope, mask in scopes.items():
        for offset, part in shifts.loc[mask].groupby("offset_minutes", sort=True):
            matched = part.loc[part.matched]
            shift_summary.append({"scope": scope, "offset_minutes": int(offset), "bars": len(part),
                "matched_bars": len(matched), "compatible_bars": int(part.compatible.sum()),
                "compatible_fraction_all": float(part.compatible.mean()),
                "compatible_fraction_matched": float(matched.compatible.mean()) if len(matched) else None})
    bad = marked.loc[marked.inconsistent & marked.label.isin(("1452", "1453", "1454", "1455"))]
    if len(bad) != 290:
        raise ValueError("Frozen 290 entry inconsistencies did not reproduce")
    by_cohort = []
    for cohort in ("affected", "reference"):
        part = day_rows.loc[day_rows[cohort]]
        by_cohort.append({"cohort": cohort, "stock_days": len(part),
            "complete_days": int(part.complete_labels.sum()), "daily_total_passes": int(part.daily_totals_pass.sum()),
            "daily_hlc_passes": int(part.daily_hlc_pass.sum()), "opening_only_difference_days": int((part.opening_error.gt(.0001) & part.daily_hlc_pass).sum()),
            "maximum_daily_volume_error": float(part.volume_error_shares.max()),
            "maximum_daily_amount_relative_error": float(part.amount_relative_error.max()),
            "positive_bars": int(part.positive_bars.sum()), "inconsistent_bars": int(part.inconsistent_bars.sum()),
            "inconsistent_fraction": float(part.inconsistent_bars.sum() / part.positive_bars.sum()),
            "tail_windows_with_inconsistent_bar": int(part.tail_inconsistent_bars.gt(0).sum()),
            "tail_windows_outside_weighted_envelope": int((~part.within_weighted_price_envelope).sum()),
            "maximum_tail_amount_conflict_yuan": float(part.outside_amount_yuan.max())})
    report = {"by_cohort": by_cohort, "fixed_offset_comparisons": shift_summary,
        "frozen_anomalous_bars": len(bad),
        "float32_envelope_maximum_yuan": float(bad.float32_envelope_yuan.max()),
        "minimum_outside_price_yuan": float(bad.outside_yuan.min()),
        "anomalies_exceeding_float32_envelope": int(bad.outside_yuan.gt(bad.float32_envelope_yuan).sum()),
        "nearest_100_compatible": int(bad.nearest_100_compatible.sum()),
        "truncated_100_compatible": int(bad.truncated_100_compatible.sum()),
        "source_repaired": False, "holding_returns_read": False, "holdout_read": False}
    return report, day_rows, marked, shifts


def evaluate(output: Path = ROOT, workers: int = 4) -> dict:
    manifest = json.loads((output / "manifest.json").read_text())
    for name, digest in manifest["sha256"].items():
        if sha(Path(name)) != digest:
            raise ValueError(f"Frozen diagnostic input changed: {name}")
    if sha(output / "selected.parquet") != manifest["selected_sha256"]:
        raise ValueError("Fixed diagnostic keys changed")
    selected = pd.read_parquet(output / "selected.parquet")
    grouped = list(selected.groupby("code", sort=True))
    minute_parts, daily_parts, sources = [], [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for count, (minute, daily, source) in enumerate(pool.map(read_stock, grouped), 1):
            minute_parts.append(minute)
            daily_parts.append(daily)
            sources.append(source)
            if count % 50 == 0 or count == len(grouped):
                print(f"Fixed source diagnosis: {count}/{len(grouped)} stocks", flush=True)
    bars = pd.concat(minute_parts, ignore_index=True).sort_values(["date", "code", "timestamp"])
    daily = pd.concat(daily_parts, ignore_index=True).sort_values(["date", "code"])
    if not bars.date.between("2024-01-01", "2025-12-31").all() or not daily.date.between("2024-01-01", "2025-12-31").all():
        raise ValueError("Diagnostic cannot inspect prices outside research years")
    previous = pd.read_parquet(ENTRY / "raw_windows.parquet").merge(selected[["date", "code"]], on=["date", "code"], validate="many_to_one")
    current = bars.loc[bars.label.isin(("1452", "1453", "1454", "1455")), previous.columns]
    ordering = ["date", "code", "timestamp"]
    pd.testing.assert_frame_equal(previous.sort_values(ordering).reset_index(drop=True),
                                  current.sort_values(ordering).reset_index(drop=True), check_dtype=False, check_exact=True)
    report, days, marked, shifts = analyze(bars, daily, selected)
    for name, frame in (("raw_days", bars), ("daily_reference", daily), ("day_audit", days),
                         ("marked_bars", marked), ("fixed_offset_checks", shifts)):
        frame.to_parquet(output / f"{name}.parquet", index=False)
    save_json(output / "source_index.json", sources)
    report.update(rule_commit=manifest["rule_commit"], stock_days=len(selected), source_files=len(sources),
                  all_source_checksums_match=all(s["download_checksum_matched"] for s in sources),
                  sha256={p.name: sha(p) for p in sorted(output.glob("*.parquet"))})
    save_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "evaluate"))
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    report = freeze(args.output) if args.stage == "freeze" else evaluate(args.output, args.workers)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
