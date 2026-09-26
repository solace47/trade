"""Split 2024–2025 daily OHLC issue keys before quality-policy sensitivity."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import json
from pathlib import Path

import pandas as pd

from trade_research.hf_audit import EXPECTED_LABELS


ROOT = Path("data/research")
MINUTES = Path("data/hf/pilot/data/stock_1m")
DAILY = Path("data/baostock/market_2020_2026/daily")


def _classify(bars: pd.DataFrame, daily: pd.Series | None) -> dict:
    if daily is None or int(daily.tradestatus) != 1:
        return {"classification": "missing_active_daily"}
    if (len(bars) != len(EXPECTED_LABELS)
            or tuple(bars.label) != EXPECTED_LABELS
            or bars[["open", "high", "low", "close", "volume",
                     "turnover"]].isna().any().any()
            or bars.volume.lt(0).any() or bars.turnover.lt(0).any()
            or bars.low.le(0).any()
            or bars.high.lt(bars[["open", "low", "close"]].max(axis=1)).any()
            or bars.low.gt(bars[["open", "high", "close"]].min(axis=1)).any()):
        return {"classification": "invalid_minute_grid"}
    traded = bars.loc[bars.volume.gt(0)]
    if traded.empty:
        return {"classification": "no_executed_minute"}
    minute = {
        "open": float(traded.open.iloc[0]),
        "high": float(traded.high.max()),
        "low": float(traded.low.min()),
        "close": float(bars.close.iloc[-1]),
    }
    mismatched = {field: abs(float(getattr(daily, field)) - price) > .0001
                  for field, price in minute.items()}
    volume_delta = abs(float(daily.volume) - float(traded.volume.sum()))
    amount_delta = abs(float(daily.amount) - float(traded.turnover.sum()))
    volume_ok = volume_delta <= 100
    amount_ok = amount_delta <= abs(float(daily.amount)) * .0001
    open_only = mismatched["open"] and not any(
        mismatched[field] for field in ("high", "low", "close"))
    benign = open_only and volume_ok and amount_ok
    return {
        "classification": ("opening_only_volume_matched" if benign
                           else "other_ohlc_or_turnover_issue"),
        "mismatch_open": mismatched["open"],
        "mismatch_high": mismatched["high"],
        "mismatch_low": mismatched["low"],
        "mismatch_close": mismatched["close"],
        "volume_delta": volume_delta,
        "amount_delta": amount_delta,
        "opening_bar_volume": int(bars.volume.iloc[0]),
        "opening_bar_range": float(bars.high.iloc[0] - bars.low.iloc[0]),
    }


def _symbol(item: tuple[str, list[str]]) -> list[dict]:
    code, dates = item
    exchange, symbol = code.split(".")
    minute_path = MINUTES / exchange.upper() / f"{symbol}.parquet"
    daily_path = DAILY / f"{exchange}_{symbol}.parquet"
    if not minute_path.exists() or not daily_path.exists():
        return [{"date": date, "code": code,
                 "classification": "missing_source_file"} for date in dates]
    first = pd.Timestamp(dates[0])
    last_exclusive = pd.Timestamp(dates[-1]) + timedelta(days=1)
    minute = pd.read_parquet(
        minute_path,
        columns=["timestamp", "open", "high", "low", "close",
                 "volume", "turnover"],
        filters=[("timestamp", ">=", first),
                 ("timestamp", "<", last_exclusive)],
    )
    minute["date"] = minute.timestamp.dt.strftime("%Y-%m-%d")
    minute["label"] = minute.timestamp.dt.strftime("%H%M")
    grouped = {date: group.sort_values("timestamp") for date, group
               in minute.loc[minute.date.isin(dates)].groupby("date")}
    daily = pd.read_parquet(
        daily_path,
        columns=["date", "open", "high", "low", "close", "volume",
                 "amount", "tradestatus"],
        filters=[("date", ">=", dates[0]), ("date", "<=", dates[-1])],
    )
    if daily.date.duplicated().any():
        raise ValueError(f"Duplicate daily row for {code}")
    reference = {row.date: row for row in daily.itertuples(index=False)}
    rows = []
    for date in dates:
        bar = grouped.get(date, pd.DataFrame())
        row = _classify(bar, reference.get(date))
        rows.append({"date": date, "code": code, **row})
    return rows


def run(output: Path = ROOT / "opening_quality") -> dict:
    issue_files = sorted((ROOT / "market_issues_ci").glob("shard_*.csv"))
    if len(issue_files) != 20:
        raise FileNotFoundError("Twenty frozen issue shards are required")
    issues = pd.concat((pd.read_csv(path, dtype=str) for path in issue_files),
                       ignore_index=True)
    flagged = issues.loc[
        issues.kind.eq("ohlc_disagreement")
        & issues.date.between("2024-01-01", "2025-12-31"),
        ["date", "code"],
    ]
    if flagged.empty or flagged.duplicated(["date", "code"]).any():
        raise ValueError("OHLC issue keys are empty or duplicated")
    work = [(code, sorted(group.date.tolist())) for code, group
            in flagged.groupby("code", sort=True)]
    rows = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i, chunk in enumerate(pool.map(_symbol, work), 1):
            rows.extend(chunk)
            if i % 200 == 0 or i == len(work):
                print(f"Rechecked {i}/{len(work)} symbols", flush=True)
    frame = pd.DataFrame(rows).sort_values(["date", "code"])
    if (len(frame) != len(flagged)
            or frame.duplicated(["date", "code"]).any()
            or not set(map(tuple, frame[["date", "code"]].to_numpy()))
            == set(map(tuple, flagged[["date", "code"]].to_numpy()))):
        raise ValueError("Recheck did not preserve the frozen issue keys")
    by_year = frame.groupby([frame.date.str[:4], "classification"]).size(
    ).unstack(fill_value=0).to_dict("index")
    by_exchange = frame.groupby([frame.code.str[:2], "classification"]).size(
    ).unstack(fill_value=0).to_dict("index")
    count = int(frame.classification.eq("opening_only_volume_matched").sum())
    report = {
        "period": "2024-01-01 through 2025-12-31",
        "legacy_ohlc_issue_days": len(frame),
        "opening_only_volume_matched_days": count,
        "opening_only_fraction": count / len(frame),
        "by_year": by_year,
        "by_exchange": by_exchange,
        "sensitivity_input_gate_passed": count / len(frame) >= .90,
        "note": ("Source-quality audit uses full-day bars and daily OHLC; "
                 "no strategy returns are read and this is not a signal input"),
    }
    output.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output / "day_classifications.parquet", index=False,
                     compression="zstd")
    (output / "input_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "opening_quality")
    args = parser.parse_args()
    print(json.dumps(run(args.output), ensure_ascii=False, indent=2))
