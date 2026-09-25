"""Build features through the archive's 14:50 label, assuming bar-end stamps."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from .hf_audit import EXPECTED_LABELS, FIRST_DATE, LAST_DATE, read_window
from .hf_download import selected_paths
from .snapshot import prior_daily_features


CUTOFF_LABELS = EXPECTED_LABELS[:231]
assert CUTOFF_LABELS[-1] == "1450"


def snapshot_for_symbol(minute: pd.DataFrame, daily: pd.DataFrame, code: str) -> pd.DataFrame:
    """Build signals using only records stamped no later than 14:50."""
    candidates = minute.loc[minute["label"] <= "1450"].sort_values(["date", "timestamp"])
    if candidates.empty:
        return pd.DataFrame()
    valid_bar = (
        candidates["label"].isin(CUTOFF_LABELS)
        & candidates[["open", "high", "low", "close", "volume", "turnover"]].notna().all(axis=1)
        & candidates["low"].gt(0)
        & candidates["volume"].ge(0)
    )
    candidates = candidates.assign(valid_bar=valid_bar)
    grouped = candidates.groupby("date", sort=True)
    pattern = grouped.agg(
        bar_count=("label", "size"), unique_labels=("label", "nunique"),
        first_label=("label", "first"), last_label=("label", "last"),
        all_valid=("valid_bar", "all"),
    )
    complete_dates = pattern.index[
        pattern["bar_count"].eq(231)
        & pattern["unique_labels"].eq(231)
        & pattern["first_label"].eq("0930")
        & pattern["last_label"].eq("1450")
        & pattern["all_valid"]
    ]
    candidates = candidates.loc[candidates["date"].isin(complete_dates)]
    traded = candidates.loc[candidates["volume"] > 0]
    if traded.empty:
        return pd.DataFrame()
    all_aggregate = candidates.groupby("date", sort=True).agg(
        signal_time=("timestamp", "last"), bar_count=("label", "size"),
        price_1450=("close", "last"), volume_1450=("volume", "sum"),
        amount_1450=("turnover", "sum"),
    )
    traded_aggregate = traded.groupby("date", sort=True).agg(
        high_1450=("high", "max"), low_1450=("low", "min"),
    )
    signal = all_aggregate.join(traded_aggregate, how="inner").reset_index()
    signal["code"] = code
    history = prior_daily_features(daily)
    signal = signal.merge(history, on=["date", "code"], how="inner", validate="one_to_one")
    # A zero-volume 09:30 placeholder can carry a quote different from the
    # actual opening trade. Today's daily open is known before the cutoff.
    opening = daily[["date", "code", "open"]].rename(columns={"open": "daily_open"})
    signal = signal.merge(opening, on=["date", "code"], how="inner", validate="one_to_one")
    signal["open_1450"] = signal["daily_open"]
    signal = signal.drop(columns="daily_open")
    signal = signal.loc[signal["tradestatus"] == 1].copy()
    signal["return_1450"] = signal["price_1450"] / signal["preclose"] - 1
    signal["quote_outside_traded_range"] = (
        signal["price_1450"].gt(signal["high_1450"] + 0.005)
        | signal["price_1450"].lt(signal["low_1450"] - 0.005)
    )
    amplitude = signal["high_1450"] - signal["low_1450"]
    signal["position_1450"] = (
        (signal["price_1450"] - signal["low_1450"]) / amplitude.where(amplitude > 0)
    )
    signal["volume_ratio_est"] = (
        signal["volume_1450"] / signal["volume5_prior"] * (241 / 231)
    )
    signal["distance_ma20_raw"] = signal["price_1450"] / signal["ma20_prior"] - 1
    signal["distance_ma20_adjusted"] = (
        signal["price_1450"] / signal["ma20_prior_adjusted"] - 1
    )
    signal["reference_gap"] = (
        signal["preclose"] - signal["prev_traded_close"]
    ).abs() > 0.005
    return signal.reset_index(drop=True)


def build(hf_root: Path, bao_root: Path, first_date: str = FIRST_DATE,
          last_date: str = LAST_DATE) -> dict:
    pilot_file = bao_root / "metadata" / "pilot_symbols.parquet"
    pilot = pd.read_parquet(pilot_file)
    frames, missing = [], []
    for code, relative in zip(pilot["code"], selected_paths(pilot_file), strict=True):
        minute_file = hf_root / relative
        daily_file = bao_root / "daily" / f"{code.replace('.', '_')}.parquet"
        if not minute_file.exists() or not daily_file.exists():
            missing.append(code)
            continue
        minute = read_window(minute_file, first_date, last_date)
        daily = pd.read_parquet(daily_file)
        frame = snapshot_for_symbol(minute, daily, code)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise RuntimeError("No complete 14:50 one-minute snapshots")
    result = pd.concat(frames, ignore_index=True).sort_values(["date", "code"])
    if result.duplicated(["date", "code"]).any():
        raise ValueError("Duplicate signal keys")
    output = bao_root / "hf_snapshots_1450.parquet"
    result.to_parquet(output, index=False, compression="zstd")
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cutoff": "14:50 Asia/Shanghai; 231 source bars including 09:30 auction",
        "first_date": first_date, "last_date": last_date,
        "snapshots": len(result), "symbols": int(result["code"].nunique()),
        "dates": int(result["date"].nunique()),
        "missing_download_symbols": missing,
        "reference_gap_rows": int(result["reference_gap"].sum()),
        "quote_outside_traded_range_rows": int(result["quote_outside_traded_range"].sum()),
        "output": str(output),
    }
    (bao_root / "metadata" / "hf_snapshot_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-root", type=Path, default=Path("data/hf/pilot"))
    parser.add_argument("--bao-root", type=Path, default=Path("data/baostock/pilot_2025_2026"))
    parser.add_argument("--first-date", default=FIRST_DATE)
    parser.add_argument("--last-date", default=LAST_DATE)
    args = parser.parse_args()
    print(json.dumps(build(args.hf_root, args.bao_root, args.first_date, args.last_date),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
