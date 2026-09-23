"""Compare BaoStock five-minute and independent one-minute 14:50 states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .hf_audit import FIRST_DATE, LAST_DATE, read_window
from .hf_download import selected_paths


def compare(hf_root: Path, bao_root: Path) -> dict:
    pilot_file = bao_root / "metadata" / "pilot_symbols.parquet"
    pilot = pd.read_parquet(pilot_file)
    rows = []
    for code, relative in zip(pilot["code"], selected_paths(pilot_file), strict=True):
        hf_path = hf_root / relative
        if not hf_path.exists():
            continue
        stem = code.replace(".", "_")
        minute5 = pd.read_parquet(bao_root / "minute_5m" / f"{stem}.parquet")
        daily = pd.read_parquet(bao_root / "daily" / f"{stem}.parquet")
        active = daily.loc[daily["tradestatus"] == 1, ["date", "preclose"]]
        minute5 = minute5.loc[minute5["date"].between(FIRST_DATE, LAST_DATE)].copy()
        minute5["label"] = minute5["time"].str[8:12]
        price5 = minute5.loc[minute5["label"] == "1450", ["date", "close"]].rename(
            columns={"close": "price_5m"}
        )
        minute1 = read_window(hf_path)
        price1 = minute1.loc[minute1["label"] == "1450", ["date", "close"]].rename(
            columns={"close": "price_1m"}
        )
        comparison = active.merge(price5, on="date").merge(price1, on="date")
        comparison["code"] = code
        comparison["price_difference"] = comparison["price_5m"] - comparison["price_1m"]
        comparison["screen5"] = (comparison["price_5m"] / comparison["preclose"] - 1).between(0.015, 0.05)
        comparison["screen1"] = (comparison["price_1m"] / comparison["preclose"] - 1).between(0.015, 0.05)
        rows.append(comparison)
    if not rows:
        raise RuntimeError("No overlapping one-minute files")
    joined = pd.concat(rows, ignore_index=True)
    output = bao_root / "source_comparison_1450.parquet"
    joined.to_parquet(output, index=False, compression="zstd")
    summary = {
        "symbols": int(joined["code"].nunique()),
        "overlapping_active_days": len(joined),
        "price_disagreement_days": int(joined["price_difference"].abs().gt(0.005).sum()),
        "screen_1_5_to_5pct_classification_disagreements": int((joined["screen5"] != joined["screen1"]).sum()),
        "maximum_absolute_price_difference": float(joined["price_difference"].abs().max()),
        "output": str(output),
    }
    (bao_root / "metadata" / "source_comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-root", type=Path, default=Path("data/hf/pilot"))
    parser.add_argument("--bao-root", type=Path, default=Path("data/baostock/pilot_2025_2026"))
    args = parser.parse_args()
    print(json.dumps(compare(args.hf_root, args.bao_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
