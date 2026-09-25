"""Audit actual 15:00 auction prints before repricing a frozen signal list."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .hf_outcomes import _order_shares
from .next_morning_exit import _half


SOURCE = Path("data/research/late_to_open_reversal")
OUTPUT = Path("data/research/closing_auction_entry")
MINUTES = Path("data/hf/pilot/data/stock_1m")
CANDIDATES = ("late_decline", "late_rally_control")
HALVES = ("2024H1", "2024H2", "2025H1", "2025H2")


def build_inputs(output_dir: Path = OUTPUT, minute_root: Path = MINUTES,
                 source_dir: Path = SOURCE, workers: int = 4) -> dict:
    if workers < 1:
        raise ValueError("Workers must be positive")
    signals = pd.read_parquet(source_dir / "selections.parquet", columns=[
        "date", "code", "candidate", "pair_id",
    ])
    pair_sizes = signals.groupby(["date", "pair_id"]).candidate.agg(
        ["size", "nunique"])
    if (signals.empty or signals.duplicated(["date", "code"]).any()
            or not signals.date.str[:4].isin(("2024", "2025")).all()
            or set(signals.candidate) != set(CANDIDATES)
            or pair_sizes["size"].ne(2).any()
            or pair_sizes["nunique"].ne(2).any()):
        raise ValueError("Invalid frozen all-market pair list")

    def one(item: tuple[str, pd.DataFrame]) -> pd.DataFrame:
        code, stock_signals = item
        exchange, symbol = code.split(".")
        dates = set(stock_signals.date)
        path = minute_root / exchange.upper() / f"{symbol}.parquet"
        minute = pd.read_parquet(path,
                                 columns=["timestamp", "close", "volume", "turnover"],
                                 filters=[
                                     ("timestamp", ">=", pd.Timestamp(min(dates))),
                                     ("timestamp", "<", pd.Timestamp(max(dates))
                                      + pd.Timedelta(days=1)),
                                 ])
        minute = minute.loc[minute.timestamp.dt.strftime("%H%M").eq("1500")
                            ].copy()
        minute["date"] = minute.timestamp.dt.strftime("%Y-%m-%d")
        minute = minute.loc[minute.date.isin(dates)]
        grouped = minute.groupby("date", sort=False).agg(
            bar_count=("close", "size"), auction_close=("close", "first"),
            auction_volume=("volume", "first"),
            auction_turnover=("turnover", "first"),
        ).reset_index()
        grouped["code"] = code
        return grouped

    with ThreadPoolExecutor(max_workers=workers) as pool:
        frames = list(pool.map(one, signals.groupby("code", sort=True)))
    bars = pd.concat(frames, ignore_index=True)
    inputs = signals.merge(bars, on=["date", "code"], how="left",
                           validate="one_to_one")
    inputs["auction_vwap"] = inputs.auction_turnover / inputs.auction_volume
    inputs["valid_auction_bar"] = (
        inputs.bar_count.eq(1) & inputs.auction_close.gt(0)
        & inputs.auction_volume.gt(0) & inputs.auction_turnover.gt(0)
        & (inputs.auction_vwap - inputs.auction_close).abs().le(.010000001)
    )
    inputs["order_shares_20k"] = [
        _order_shares(code, price, 20_000) if np.isfinite(price) else 0
        for code, price in zip(inputs.code, inputs.auction_vwap)
    ]
    inputs["capacity_20k"] = (
        inputs.valid_auction_bar & inputs.order_shares_20k.gt(0)
        & inputs.order_shares_20k.le(inputs.auction_volume * .1)
    )
    inputs["half"] = _half(inputs.date)
    sections = inputs.groupby(["half", "candidate"], sort=True).agg(
        signals=("code", "size"),
        valid_bar_rate=("valid_auction_bar", "mean"),
        capacity_20k_rate=("capacity_20k", "mean"),
        days=("date", "nunique"),
    ).reset_index().to_dict("records")
    cells = {(row["half"], row["candidate"]): row for row in sections}
    gate = all(
        (half, candidate) in cells
        and cells[(half, candidate)]["valid_bar_rate"] >= .99
        and cells[(half, candidate)]["capacity_20k_rate"] >= .90
        for half in HALVES for candidate in CANDIDATES
    )
    audit = {
        "pairs": len(signals) // 2,
        "invalid_auction_bars": int((~inputs.valid_auction_bar).sum()),
        "capacity_20k_failures": int((~inputs.capacity_20k).sum()),
        "by_half_candidate": sections,
        "outcome_gate_passed": gate,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").unlink(missing_ok=True)
    inputs.to_parquet(output_dir / "auction_inputs.parquet", index=False,
                      compression="zstd")
    (output_dir / "input_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--minute-root", type=Path, default=MINUTES)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(build_inputs(args.output, args.minute_root, args.source, args.workers))


if __name__ == "__main__":
    main()
