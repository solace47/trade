"""Prepare a fixed broader feature library without new test-return labels."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import pandas as pd

from .alpha158_asof import DEFINITION, definitions, features_for_symbol
from .corporate_cash import save_json, sha
from .long_history_inputs import ROOT as HISTORY

ROOT = Path("data/research/alpha158_1449")
RULE_COMMIT = "f49715c"
DAILY = Path("data/baostock/market_2020_2026/daily")
PREFIX_COLUMNS = ["date", "code", "price_1449", "high_1449", "low_1449", "volume_1449", "amount_1449"]
DAILY_COLUMNS = ["date", "tradestatus", "preclose", "open", "high", "low", "close", "volume", "amount"]


def build_part(item) -> dict:
    code, prefix, output, adapter_hash = item
    folder = Path(output) / "parts" / adapter_hash[:16]
    path = folder / f"{code}.parquet"
    source = DAILY / (code.replace(".", "_") + ".parquet")
    input_hash = hashlib.sha256(pd.util.hash_pandas_object(prefix, index=False).values.tobytes()).hexdigest()
    expected = {"code": code, "prefix_sha256": input_hash, "daily_sha256": sha(source), "adapter_sha256": adapter_hash}
    report_path = path.with_suffix(".json")
    if path.exists() and report_path.exists():
        cached = json.loads(report_path.read_text())
        if all(cached.get(k) == v for k, v in expected.items()) and sha(path) == cached["output_sha256"]:
            return cached
        raise ValueError(f"A cached feature part changed: {code}")
    daily = pd.read_parquet(source, columns=DAILY_COLUMNS,
        filters=[("date", ">=", "2019-01-01"), ("date", "<=", "2025-12-31")])
    result = features_for_symbol(daily, prefix)
    result.insert(1, "code", code)
    if len(result) != len(prefix) or not result.date.equals(prefix.date.reset_index(drop=True)):
        raise ValueError("Feature dates changed")
    folder.mkdir(parents=True, exist_ok=True)
    result.to_parquet(path, index=False, compression="zstd")
    report = {**expected, "path": str(path), "rows": len(result), "output_sha256": sha(path),
        "missing_cells": int(result.drop(columns=["date", "code"]).isna().sum().sum())}
    save_json(report_path, report)
    return report


def build(output: Path = ROOT, workers: int = 4) -> dict:
    if any(output.glob("*/repriced.parquet")):
        raise ValueError("Cannot replace model features after execution outcomes exist")
    if not 1 <= workers <= 4:
        raise ValueError("Feature preparation uses at most four workers")
    report = json.loads((HISTORY / "feature_report.json").read_text())
    source = HISTORY / "features.parquet"
    if sha(source) != report["features_sha256"]:
        raise ValueError("The fixed common risk pool changed")
    output.mkdir(parents=True, exist_ok=True)
    prefix = pd.read_parquet(source, columns=PREFIX_COLUMNS).sort_values(["code", "date"]).reset_index(drop=True)
    if (prefix.duplicated(["date", "code"]).any() or not prefix.date.between("2022-01-01", "2025-12-31").all()
            or len(prefix) != 1002329):
        raise ValueError("The common historical feature universe changed")
    adapter = Path(__file__).with_name("alpha158_asof.py")
    adapter_hash = sha(adapter)
    manifest = {"rule_commit": RULE_COMMIT, "source_features_sha256": sha(source),
        "definition_sha256": sha(DEFINITION), "adapter_sha256": adapter_hash,
        "warmup_first": "2019-01-01", "last_allowed": "2025-12-31", "rows": len(prefix),
        "new_test_labels_read": False, "holdout_prices_read": False}
    save_json(output / "feature_manifest.json", manifest)
    jobs = [(code, g.reset_index(drop=True), str(output), adapter_hash) for code, g in prefix.groupby("code", sort=True)]
    parts = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, result in enumerate(pool.map(build_part, jobs), 1):
            parts.append(result)
            if i % 100 == 0 or i == len(jobs):
                progress = {"processed": i, "total": len(jobs), "rows": sum(x["rows"] for x in parts)}
                save_json(output / "feature_progress.json", progress)
                print(progress, flush=True)
    frame = pd.concat([pd.read_parquet(part["path"]) for part in parts], ignore_index=True)
    frame = frame.sort_values(["date", "code"]).reset_index(drop=True)
    expected = prefix[["date", "code"]].sort_values(["date", "code"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(frame[["date", "code"]], expected)
    columns = [name for name, _ in definitions()]
    if frame.columns.tolist() != ["date", "code", *columns]:
        raise ValueError("A factor column was omitted or reordered")
    frame.to_parquet(output / "features.parquet", index=False, compression="zstd")
    save_json(output / "feature_parts.json", parts)
    result = {**manifest, "symbols": len(parts), "features": len(columns),
        "by_year": frame.groupby(frame.date.str[:4]).size().to_dict(),
        "missing_by_feature": frame[columns].isna().sum().to_dict(),
        "features_sha256": sha(output / "features.parquet"), "parts_sha256": sha(output / "feature_parts.json")}
    save_json(output / "feature_report.json", result)
    return result


if __name__ == "__main__":
    r = build()
    print(json.dumps({k: r[k] for k in ("rows", "symbols", "features", "by_year", "features_sha256")}, indent=2))
