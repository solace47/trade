"""Build point-in-time 14:50 features from completed five-minute bars."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from .audit import EXPECTED_LABELS


CUTOFF_LABELS = EXPECTED_LABELS[:46]
assert CUTOFF_LABELS[-1] == "1450"


def prior_daily_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Use only completed trading days before each row's date."""
    traded = daily.loc[daily["tradestatus"] == 1].sort_values("date").copy()
    previous = traded["close"].shift(1)
    traded["prev_traded_close"] = previous
    traded["ma5_prior"] = previous.rolling(5, min_periods=5).mean()
    traded["ma20_prior"] = previous.rolling(20, min_periods=20).mean()
    traded["ma60_prior"] = previous.rolling(60, min_periods=60).mean()
    traded["volume5_prior"] = traded["volume"].shift(1).rolling(5, min_periods=5).mean()
    traded["return5_prior"] = previous / traded["close"].shift(6) - 1
    traded["return20_prior"] = previous / traded["close"].shift(21) - 1
    # preclose is today's exchange reference price, available before open.
    # Its difference from the prior raw close carries split/dividend effects.
    # Convert past closes to today's raw-price basis without using any future
    # adjustment factor. Ignore sub-half-tick rounding noise.
    ratio = traded["preclose"] / previous
    ratio = ratio.where((traded["preclose"] - previous).abs() > 0.005, 1.0).fillna(1.0)
    factor = ratio.cumprod()
    adjusted_close = traded["close"] / factor
    traded["ma5_prior_adjusted"] = adjusted_close.shift(1).rolling(5, min_periods=5).mean() * factor
    traded["ma20_prior_adjusted"] = adjusted_close.shift(1).rolling(20, min_periods=20).mean() * factor
    traded["ma60_prior_adjusted"] = adjusted_close.shift(1).rolling(60, min_periods=60).mean() * factor
    traded["return5_prior_adjusted"] = adjusted_close.shift(1) / adjusted_close.shift(6) - 1
    traded["return20_prior_adjusted"] = adjusted_close.shift(1) / adjusted_close.shift(21) - 1
    return traded[[
        "date", "code", "preclose", "isST", "tradestatus", "prev_traded_close",
        "ma5_prior", "ma20_prior", "ma60_prior", "volume5_prior",
        "return5_prior", "return20_prior",
        "ma5_prior_adjusted", "ma20_prior_adjusted", "ma60_prior_adjusted",
        "return5_prior_adjusted", "return20_prior_adjusted",
    ]]


def snapshot_for_symbol(minute: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    minute = minute.copy()
    minute["label"] = minute["time"].str[8:12]
    candidates = minute.loc[minute["label"] <= "1450"].sort_values(["date", "time"])
    rows = []
    for date, bars in candidates.groupby("date", sort=True):
        if bars["label"].tolist() != list(CUTOFF_LABELS):
            continue
        rows.append({
            "date": date,
            "code": bars["code"].iloc[0],
            "signal_time": bars["time"].iloc[-1],
            "bar_count": len(bars),
            "open_1450": float(bars["open"].iloc[0]),
            "high_1450": float(bars["high"].max()),
            "low_1450": float(bars["low"].min()),
            "price_1450": float(bars["close"].iloc[-1]),
            "volume_1450": int(bars["volume"].sum()),
            "amount_1450": float(bars["amount"].sum()),
        })
    if not rows:
        return pd.DataFrame()
    snapshot = pd.DataFrame(rows)
    snapshot = snapshot.merge(prior_daily_features(daily), on=["code", "date"], how="inner")
    snapshot = snapshot.loc[snapshot["tradestatus"] == 1].copy()
    snapshot["return_1450"] = snapshot["price_1450"] / snapshot["preclose"] - 1
    amplitude = snapshot["high_1450"] - snapshot["low_1450"]
    snapshot["position_1450"] = (snapshot["price_1450"] - snapshot["low_1450"]) / amplitude.where(amplitude > 0)
    snapshot["volume_ratio_est"] = (
        snapshot["volume_1450"] / snapshot["volume5_prior"] * (48 / 46)
    )
    snapshot["distance_ma20_raw"] = snapshot["price_1450"] / snapshot["ma20_prior"] - 1
    snapshot["distance_ma20_adjusted"] = (
        snapshot["price_1450"] / snapshot["ma20_prior_adjusted"] - 1
    )
    snapshot["reference_gap"] = (
        snapshot["preclose"] - snapshot["prev_traded_close"]
    ).abs() > 0.005
    # Raw prior closes can jump on corporate-action dates. Keep the diagnostic
    # and avoid treating these rows as clean trend-factor observations.
    return snapshot.reset_index(drop=True)


def build(root: Path) -> dict:
    metadata = root / "metadata"
    pilot = pd.read_parquet(metadata / "pilot_symbols.parquet")
    selection = json.loads((metadata / "selection.json").read_text(encoding="utf-8"))
    snapshots = []
    missing = []
    for code in pilot["code"]:
        stem = code.replace(".", "_")
        minute_file = root / "minute_5m" / f"{stem}.parquet"
        daily_file = root / "daily" / f"{stem}.parquet"
        if not minute_file.exists() or not daily_file.exists():
            missing.append(code)
            continue
        minute = pd.read_parquet(minute_file)
        daily = pd.read_parquet(daily_file)
        frame = snapshot_for_symbol(minute, daily)
        if not frame.empty:
            snapshots.append(frame)
    if not snapshots:
        raise RuntimeError("No complete 14:50 snapshots")
    result = pd.concat(snapshots, ignore_index=True)
    result = result.loc[result["date"].between(
        selection["minute_start"], selection["minute_end"]
    )].sort_values(["date", "code"]).reset_index(drop=True)
    if result.duplicated(["date", "code"]).any():
        raise ValueError("Duplicate 14:50 snapshot keys")
    output = root / "snapshots_1450.parquet"
    result.to_parquet(output, index=False, compression="zstd")
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cutoff": "14:50 Asia/Shanghai; completed five-minute bars through 14:50",
        "snapshots": len(result),
        "symbols": int(result["code"].nunique()),
        "dates": int(result["date"].nunique()),
        "first_date": result["date"].min(),
        "last_date": result["date"].max(),
        "missing_download_symbols": missing,
        "reference_gap_rows": int(result["reference_gap"].sum()),
        "output": str(output),
    }
    (metadata / "snapshot_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/baostock/pilot_2025_2026")
    args = parser.parse_args()
    print(json.dumps(build(Path(args.root)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
