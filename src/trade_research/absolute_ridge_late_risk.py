"""Freeze a ridge ranking that cannot reward late absolute price moves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .absolute_ridge import (
    COOLDOWN, CAPACITY, FEATURES, MAIN_OUTPUT, PERIODS, ROOT,
    choose, feature_frame, fit, match_controls, score, training_labels,
)
from .market_study import _quality_keys


OUTPUT = ROOT / "absolute_ridge_late_risk"
RISK_FEATURE = "abs_return_last30"


def constrain_late_risk(model: dict, train: pd.DataFrame,
                        training_mean: float) -> dict:
    """Cap one coefficient at zero and preserve the training mean forecast."""
    index = FEATURES.index(RISK_FEATURE)
    coefficients = model["coefficients"].copy()
    coefficients[index] = min(float(coefficients[index]), 0.0)
    x = ((train[list(FEATURES)] - model["median"])
         / model["scale"]).clip(-5, 5).to_numpy(dtype=float)
    intercept = float(training_mean - x.mean(axis=0) @ coefficients)
    return {**model, "coefficients": coefficients,
            "intercept": intercept}


def freeze(output_dir: Path = OUTPUT) -> dict:
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    features = feature_frame(connection, main_only=True)
    connection.read_parquet(str(ROOT / "market_outcomes_ci" / "*.parquet")
                            ).create_view("outcomes")
    connection.register("bad_days", _quality_keys(ROOT / "market_issues_ci"))
    calendar = connection.execute("""
        SELECT DISTINCT date FROM snapshots
        WHERE date BETWEEN '2024-01-01' AND '2025-12-31'
        ORDER BY date
    """).df().date.tolist()
    picks, training_audits = [], []
    for train_end, test_first, test_last, name in PERIODS:
        train = features.loc[features.date.le(train_end)].copy()
        labels = training_labels(connection, train, test_first)
        original, audit = fit(train, labels)
        model = constrain_late_risk(
            original, train, audit["mean_training_cash_clipped"])
        candidates = features.loc[features.date.between(test_first, test_last)]
        scored = score(candidates, model)
        chosen = choose(scored, calendar)
        if not chosen.empty:
            chosen["test_period"] = name
            picks.append(chosen)
        audit["test_period"] = name
        audit["original_risk_coefficient"] = float(
            original["coefficients"][FEATURES.index(RISK_FEATURE)])
        audit["constrained_risk_coefficient"] = float(
            model["coefficients"][FEATURES.index(RISK_FEATURE)])
        audit["constrained_intercept"] = model["intercept"]
        audit["selected"] = len(chosen)
        training_audits.append(audit)
    if not picks:
        raise ValueError("The constrained model has no positive-score signals")
    chosen = pd.concat(picks, ignore_index=True)
    controls = match_controls(chosen, features)
    chosen["candidate"] = "absolute_model"
    chosen["pair_id"] = chosen.code
    controls["candidate"] = "same_day_control"
    membership = pd.concat([chosen, controls], ignore_index=True)
    if membership.duplicated(["date", "code"]).any():
        raise ValueError("Duplicated treatment or control stock-day")
    selected_keys = set(zip(chosen.date, chosen.code))
    control_keys = set(zip(controls.date, controls.pair_id))
    chosen["matched_control"] = [key in control_keys for key in
                                 zip(chosen.date, chosen.code)]
    chosen["half"] = chosen.date.str[:4] + "H" + np.where(
        chosen.date.str[5:7].astype(int) <= 6, "1", "2")
    by_half = chosen.groupby("half").agg(
        signals=("code", "size"), signal_days=("date", "nunique"),
        matched_controls=("matched_control", "sum"),
    ).reset_index().to_dict("records")
    prior = pd.read_parquet(MAIN_OUTPUT / "selections.parquet")
    prior = prior.loc[prior.candidate.eq("absolute_model")]
    old_keys = set(zip(prior.date, prior.code))
    overlap_fraction = len(selected_keys & old_keys) / len(selected_keys)
    volatility = features[["date", "code", "return_last30"]]
    new_late_abs = float(chosen.merge(volatility, on=["date", "code"],
                                      validate="one_to_one").return_last30.abs().median())
    old_late_abs = float(prior.merge(volatility, on=["date", "code"],
                                     validate="one_to_one").return_last30.abs().median())
    control_fraction = len(controls) / len(chosen)
    gate = (len(by_half) == 3 and all(
        item["signals"] >= 100 and item["signal_days"] >= 40
        for item in by_half)
        and control_fraction >= .70 and overlap_fraction < .90
        and new_late_abs < old_late_abs)
    audit = {
        "training": training_audits, "signals": len(chosen),
        "controls": len(controls), "control_fraction": control_fraction,
        "by_half": by_half, "overlap_with_original": overlap_fraction,
        "new_median_abs_return_last30": new_late_abs,
        "old_median_abs_return_last30": old_late_abs,
        "capacity": CAPACITY, "cooldown": COOLDOWN,
        "treatment_label": "absolute_model",
        "control_label": "same_day_control",
        "outcome_gate_passed": bool(gate),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("repricing_signals.parquet", "repriced.parquet", "report.json"):
        (output_dir / stale).unlink(missing_ok=True)
    membership.to_parquet(output_dir / "selections.parquet", index=False,
                          compression="zstd")
    if gate:
        connection.register("members", membership[["date", "code"]])
        snapshots = connection.execute("""
            SELECT s.* FROM members m JOIN snapshots s USING (date, code)
        """).df()
        if len(snapshots) != len(membership):
            raise ValueError("A constrained-model signal lacks a snapshot")
        snapshots.to_parquet(output_dir / "repricing_signals.parquet",
                             index=False, compression="zstd")
    (output_dir / "input_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    audit = freeze(args.output)
    print({key: audit[key] for key in (
        "signals", "controls", "control_fraction", "by_half",
        "overlap_with_original", "new_median_abs_return_last30",
        "old_median_abs_return_last30", "outcome_gate_passed")})


if __name__ == "__main__":
    main()
