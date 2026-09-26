"""Describe resampling resolution of an already rejected, accounted cohort.

No new strategy outcomes are read. Artificial additive shifts of centered old
daily differences illustrate conditional noise; they do not forecast power.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .corporate_cash import save_json, sha


ROOT = Path("data/research/statistical_resolution")
SOURCE = Path("data/research/holding_exceptions/accounted.parquet")
HALVES = ("2024H2", "2025H1", "2025H2")
MODES = ("independent_days", "calendar_weeks", "calendar_fortnights")
SEED = 20260926
DRAWS = 20000


def freeze(output: Path = ROOT) -> dict:
    report_path = SOURCE.with_name("report.json")
    report = json.loads(report_path.read_text())
    if sha(SOURCE) != report["accounted_sha256"]:
        raise ValueError("The verified accounting source changed")
    result = {"rule_commit": "6cb5e22", "sha256": {str(SOURCE): report["accounted_sha256"],
        str(report_path): sha(report_path)}, "notional": 20000, "horizon": 1, "slippage_bps": 5,
        "halves": list(HALVES), "draws": DRAWS, "seed": SEED, "modes": list(MODES),
        "artificial_shifts_bps": [0, 10, 25, 50], "new_strategy_outcomes_read": False,
        "raw_windows_read": False, "holdout_read": False}
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != result:
        raise ValueError("Cannot replace the fixed resolution-audit source or methods")
    save_json(path, result)
    return result


def paired_days(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if (rows.duplicated(["date", "code", "candidate"]).any()
            or rows.unknown_after_buy.any() or not np.isfinite(rows.known_return5).all()):
        raise ValueError("Unique complete economic outcomes are required; unknown is not zero")
    if not rows.date.between("2024-01-01", "2025-12-31").all():
        raise ValueError("Old-cohort audit is restricted to 2024–2025 outcomes")
    if not set(rows.candidate) <= {"absolute_model", "same_day_control"}:
        raise ValueError("Unexpected arm in the fixed source")
    not_bought = rows.entry_status.ne("filled")
    if rows.loc[not_bought, "known_return5"].ne(0).any():
        raise ValueError("An unbought record acquired an economic trading return")
    model = rows.loc[rows.candidate.eq("absolute_model")]
    control = rows.loc[rows.candidate.eq("same_day_control")]
    if rows.pair_id.isna().any():
        raise ValueError("Original pair identities cannot be missing")
    pairs = model.merge(control, on=["date", "pair_id"], how="outer", validate="one_to_one",
                         suffixes=("_model", "_control"), indicator=True)
    if pairs._merge.eq("right_only").any():
        raise ValueError("A fixed control has no original counterpart")
    unpaired = pairs.loc[pairs._merge.eq("left_only")].copy()
    pairs = pairs.loc[pairs._merge.eq("both")].copy()
    if pairs.half_model.ne(pairs.half_control).any():
        raise ValueError("A pair crosses analysis periods")
    pairs["half"] = pairs.half_model
    pairs["gap"] = pairs.known_return5_model - pairs.known_return5_control
    daily = pairs.groupby(["half", "date"], sort=True).agg(
        gap=("gap", "mean"), pairs=("pair_id", "size")).reset_index()
    return pairs, daily, {"model_rows": len(model), "control_rows": len(control), "pairs": len(pairs),
        "unpaired_model_rows": len(unpaired),
        "unpaired_model_by_half": unpaired.groupby("half_model").size().to_dict()}


def block_table(daily: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode not in MODES or daily.date.duplicated().any() or not np.isfinite(daily.gap).all():
        raise ValueError("Invalid daily series or resampling mode")
    rows = daily.sort_values("date").copy()
    rows["residual"] = rows.gap - rows.gap.mean()
    dates = pd.to_datetime(rows.date)
    monday = dates - pd.to_timedelta(dates.dt.dayofweek, unit="D")
    if mode == "independent_days":
        rows["block"] = rows.date
    elif mode == "calendar_weeks":
        rows["block"] = monday.dt.strftime("%Y-%m-%d")
    else:
        rows["block"] = ((monday - monday.min()).dt.days // 14).astype(str)
    blocks = rows.groupby("block", sort=True).agg(
        residual_sum=("residual", "sum"), days=("residual", "size")).reset_index()
    if len(blocks) < 2 or abs(blocks.residual_sum.sum()) > 1e-10:
        raise ValueError("Need at least two blocks with the original daily mean removed")
    return blocks


def sample_block_means(blocks: pd.DataFrame, rng: np.random.Generator, draws: int) -> np.ndarray:
    if draws < 1 or blocks.days.le(0).any():
        raise ValueError("Positive block counts and draws are required")
    indices = rng.integers(0, len(blocks), size=(draws, len(blocks)))
    sums, counts = blocks.residual_sum.to_numpy(), blocks.days.to_numpy()
    # The mean of block means is wrong when holidays/missing signal days make
    # week lengths unequal. Preserve the signal-day denominator in each draw.
    return sums[indices].sum(axis=1) / counts[indices].sum(axis=1)


def noise_summary(errors: np.ndarray) -> dict:
    q025, q20, q50, q80, q975 = np.quantile(errors, [.025, .20, .50, .80, .975])
    return {"mean_resampling_error": float(errors.mean()),
            "standard_deviation": float(errors.std(ddof=1)),
            "error_q025": float(q025), "error_q20": float(q20), "error_median": float(q50),
            "error_q80": float(q80), "error_q975": float(q975),
            "noise_interval_width_bps": float((q975 - q025) * 10000),
            "shift_for_80pct_exceedance_bps": float((q975 - q20) * 10000),
            "shift_exceedance": [{"shift_bps": bps,
                "fraction_above_zero_shift_q975": float((errors + bps / 10000 > q975).mean())}
                for bps in (0, 10, 25, 50)]}


def evaluate(output: Path = ROOT) -> dict:
    manifest = json.loads((output / "manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        if sha(Path(name)) != expected:
            raise ValueError("Frozen accounting evidence changed")
    rows = pd.read_parquet(SOURCE, columns=["date", "code", "candidate", "pair_id", "half",
        "entry_status", "unknown_after_buy", "known_return5"],
        filters=[("target_notional", "==", 20000.), ("horizon", "==", 1),
                 ("date", ">=", "2024-01-01"), ("date", "<=", "2025-12-31")])
    if set(rows.half) != set(HALVES):
        raise ValueError("Fixed audit half-years changed")
    pairs, daily, counts = paired_days(rows)
    old = json.loads(SOURCE.with_name("report.json").read_text())
    expected = {row["half"]: row["known_edge5"] for row in old["pairs"]
                if row["notional"] == 20000 and row["horizon"] == 1}
    reproduced = daily.groupby("half").gap.mean().to_dict()
    if set(expected) != set(reproduced) or any(abs(reproduced[k] - expected[k]) > 1e-12 for k in expected):
        raise ValueError("Daily paired differences do not reproduce the completed accounting")
    pairs.to_parquet(output / "fixed_pairs.parquet", index=False)
    daily.to_parquet(output / "daily_gaps.parquet", index=False)
    rng = np.random.default_rng(SEED)
    cells, draws = [], []
    for half in HALVES:
        part = daily.loc[daily.half.eq(half)]
        for mode in MODES:
            blocks = block_table(part, mode)
            errors = sample_block_means(blocks, rng, DRAWS)
            cells.append({"half": half, "mode": mode, "days": len(part), "blocks": len(blocks),
                "paired_stock_days": int(part.pairs.sum()), "old_observed_gap": reproduced[half],
                **noise_summary(errors)})
            draws.append(pd.DataFrame({"half": half, "mode": mode, "error": errors}))
    pd.concat(draws, ignore_index=True).to_parquet(output / "resampling_errors.parquet", index=False)
    result = {"cohort": counts, "old_accounting_means_reproduced": True, "cells": cells,
        "draws": DRAWS, "seed": SEED, "outputs_sha256": {name: sha(output / name) for name in
            ("fixed_pairs.parquet", "daily_gaps.parquet", "resampling_errors.parquet")},
        "new_strategy_outcomes_read": False, "raw_windows_read": False, "holdout_read": False,
        "qualification": "Conditional additive-shift sensitivity of a previously rejected, selected cohort. "
            "Not predicted future power, a new alpha test, a portfolio return or multiplicity correction."}
    save_json(output / "report.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "evaluate"))
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    result = {"freeze": freeze, "evaluate": evaluate}[args.stage](args.output)
    if args.stage == "freeze":
        print(result)
    else:
        print(result["cohort"])
        for cell in result["cells"]:
            print({key: cell[key] for key in ("half", "mode", "days", "blocks", "paired_stock_days",
                  "noise_interval_width_bps", "shift_for_80pct_exceedance_bps", "shift_exceedance")})


if __name__ == "__main__":
    main()
