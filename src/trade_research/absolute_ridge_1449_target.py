"""Train the fixed 14:49 ridge on its executable 20k, T+5 target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .absolute_ridge import ROOT, freeze as freeze_ridge
from .absolute_ridge_1449 import feature_frame
from .absolute_ridge_1449_eval import apply_period_quality
from .strategy_scan import _stressed_returns


OUTPUT = ROOT / "absolute_ridge_1449_target"
TRAINING_SIGNALS = OUTPUT / "training_signals.parquet"
TRAINING_TRADES = OUTPUT / "training_repriced.parquet"


def prepare(output_dir: Path = OUTPUT) -> dict:
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    features = feature_frame(connection)
    keys = features.loc[
        features.date.between("2024-01-01", "2024-12-17"),
        ["date", "code", "price_1449"],
    ].copy()
    if keys.empty or keys.duplicated(["date", "code"]).any():
        raise ValueError("Invalid 2024 training universe")
    connection.register("training_keys", keys[["date", "code"]])
    snapshots = connection.execute("""
        SELECT s.* FROM training_keys t JOIN snapshots s USING (date, code)
    """).df()
    signals = snapshots.merge(keys, on=["date", "code"], validate="one_to_one")
    if len(signals) != len(keys):
        raise ValueError("Training feature missing a snapshot")
    # The 14:49 builder has already validated its own traded range. A 14:50
    # quote cannot retroactively remove an otherwise eligible 14:49 signal.
    signals["quote_outside_traded_range"] = False
    output_dir.mkdir(parents=True, exist_ok=True)
    signals.to_parquet(output_dir / TRAINING_SIGNALS.name, index=False,
                       compression="zstd")
    audit = {"training_stock_days": len(signals),
             "first": signals.date.min(), "last": signals.date.max(),
             "sizing_price_column": "price_1449",
             "target_notional": 20000, "horizon": 5,
             "exit_window": "close"}
    (output_dir / "training_input_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def exact_training_labels(trades_path: Path = TRAINING_TRADES) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_parquet(trades_path)
    if (raw.empty or raw.duplicated(["date", "code"]).any()
            or not raw.date.str.startswith("2024").all()
            or not raw.target_notional.eq(20000).all()
            or not raw.horizon.eq(5).all()
            or not raw.entry_window.eq("baseline").all()
            or not raw.exit_window.eq("close").all()
            or raw.target_exit_date.isna().any()
            or raw.target_exit_date.ge("2025-01-01").any()
            or raw.exit_date.dropna().ge("2025-01-01").any()):
        raise ValueError("Training raw-minute trade grid is invalid")
    period, released = apply_period_quality(raw)
    valid = (period.exit_status.eq("filled") & period.quality_clean_exit
             & period.exit_delay_sessions.eq(0))
    net = np.zeros(len(period), dtype=float)
    net[valid.to_numpy()] = _stressed_returns(period.loc[valid], 15)
    labels = period[["date", "code", "target_exit_date", "exit_date"]].copy()
    labels["net_return"] = net
    labels["clean_ontime_exit"] = valid.to_numpy()
    audit = {"training_trades": len(labels),
             "clean_ontime_fraction": float(valid.mean()),
             "quality_released_trades": released,
             "mean_cash_stress15": float(net.mean()),
             "first": labels.date.min(), "last": labels.date.max()}
    return labels, audit


def freeze(output_dir: Path = OUTPUT) -> dict:
    input_audit = json.loads((output_dir / "training_input_audit.json").read_text(
        encoding="utf-8"))
    labels, label_audit = exact_training_labels(
        output_dir / TRAINING_TRADES.name
    )
    if (len(labels) != input_audit["training_stock_days"]
            or input_audit["last"] != "2024-12-17"):
        raise ValueError("Training repricing does not cover its frozen universe")
    audit = freeze_ridge(
        output_dir, main_only=True, feature_builder=feature_frame,
        decision_price_column="price_1449", cutoff_label="1449",
        training_label_frame=labels,
    )
    audit["training_target"] = label_audit
    audit["training_target_name"] = "20k_fixed_shares_T5_close_stress15"
    (output_dir / "input_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "freeze"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = (prepare(args.output_dir) if args.phase == "prepare"
              else freeze(args.output_dir))
    print(result if args.phase == "prepare" else {
        key: result[key] for key in (
            "signals", "controls", "control_fraction", "by_half",
            "outcome_gate_passed", "training_target",
        )
    })


if __name__ == "__main__":
    main()
