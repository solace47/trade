"""Pre-outcome geometry of the full integer-price-side population.

Residual contrast weights identify a conditional linear association, not an
investable portfolio. This module never loads a response or a later price.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .corporate_cash import save_json, sha
from .round_number_1449 import ROOT as PREVIOUS


ROOT = Path("data/research/round_number_geometry")
HALVES = ("2024H1", "2024H2", "2025H1", "2025H2")


def controls(frame: pd.DataFrame) -> tuple[np.ndarray, list[str], list[str]]:
    variables = {"shanghai": frame.code.str.startswith("sh.").astype(float).to_numpy()}
    for cents in range(2, 20):
        variables[f"absolute_distance_{cents}"] = frame.distance_cents.eq(cents).astype(float).to_numpy()
    for feature, scale in (("day_return", .03), ("return_last29", .01), ("prior20_return", .20)):
        value = frame[feature].to_numpy() / scale
        variables[feature] = value
        variables[feature + "_squared"] = value ** 2
    for feature, scale in (("price_1449", 10), ("amount_1449", 300000000)):
        value = np.log(frame[feature].to_numpy() / scale)
        variables["log_" + feature] = value
        variables["log_" + feature + "_squared"] = value ** 2
    columns, names, dropped = [np.ones(len(frame))], ["intercept"], []
    for name, values in variables.items():
        if not np.isfinite(values).all():
            raise ValueError("Input controls must be finite")
        if np.ptp(values) == 0:
            dropped.append(name)
        else:
            columns.append((values - values.mean()) / values.std(ddof=0))
            names.append(name)
    return np.column_stack(columns), names, dropped


def fit_day(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    if (frame.date.nunique() != 1 or frame.half.nunique() != 1 or frame.code.duplicated().any()
            or not set(frame.side) <= {"above", "below"} or not frame.distance_cents.between(1, 19).all()):
        raise ValueError("Need one unique, valid decision-date cross-section")
    matrix, names, dropped = controls(frame)
    treatment = frame.side.eq("above").astype(float).to_numpy()
    coefficients, _, rank, singular = np.linalg.lstsq(matrix, treatment, rcond=1e-10)
    residual = treatment - matrix @ coefficients
    ss = float(residual @ residual)
    residual_variance = ss / len(frame)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else None
    orthogonality = float(np.max(np.abs(matrix.T @ residual)))
    residual_sum_error = float(abs(residual.sum()))
    weights = residual / ss if residual_variance >= .10 else np.full(len(frame), np.nan)
    normalized = bool(np.isfinite(weights).all())
    contrast_error = float(abs(weights @ treatment - 1)) if normalized else None
    dominance = float(np.max(np.abs(weights)) / np.sum(np.abs(weights))) if normalized else None
    checks = {"both_arms": int(treatment.sum()) >= 30 and int((1 - treatment).sum()) >= 30,
        "stock_days": len(frame) >= 120, "full_rank": int(rank) == len(names),
        "condition": condition is not None and condition <= 1e6, "residual_variance": residual_variance >= .10,
        "residual_sum": residual_sum_error <= 1e-10, "orthogonality": orthogonality <= 1e-10,
        "contrast_normalization": contrast_error is not None and contrast_error <= 1e-10,
        "concentration": dominance is not None and dominance <= .05}
    passed = bool(all(checks.values()))
    audit = {"date": frame.date.iloc[0], "half": frame.half.iloc[0], "rows": len(frame),
        "above": int(treatment.sum()), "below": int((1 - treatment).sum()), "columns": len(names),
        "rank": int(rank), "condition_number": condition, "residual_variance": residual_variance,
        "residual_sum_error": residual_sum_error, "orthogonality_error": orthogonality,
        "contrast_normalization_error": contrast_error, "largest_absolute_weight_fraction": dominance,
        "coefficients": dict(zip(names, map(float, coefficients))), "dropped_constant_controls": dropped,
        "checks": {name: bool(value) for name, value in checks.items()}, "passed": passed}
    result = frame[["date", "code", "half", "side", "distance_cents"]].copy()
    result["treatment"] = treatment
    result["residual"] = residual
    result["contrast_weight"] = weights
    result["input_day_passed"] = passed
    return audit, result


def freeze(output: Path = ROOT) -> dict:
    source = PREVIOUS / "side_inputs.parquet"
    previous_report = json.loads((PREVIOUS / "input_report.json").read_text())
    expected = previous_report["output_sha256"][source.name]
    if sha(source) != expected:
        raise ValueError("The complete frozen price-side population changed")
    manifest = {"rule_commit": "ab73b89", "input_sha256": {str(source): expected,
        str(PREVIOUS / "manifest.json"): sha(PREVIOUS / "manifest.json")}, "history_start": "2024-01-01",
        "scope": list(HALVES), "response_read": False, "entry_prices_read": False, "holdout_read": False}
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Cannot replace the preregistered geometry inputs")
    save_json(path, manifest)
    rows = pd.read_parquet(source)
    if (rows.duplicated(["date", "code"]).any() or set(rows.half) != set(HALVES)
            or not rows.date.between("2024-01-01", "2025-12-31").all()):
        raise ValueError("Input population has duplicate keys or unexpected dates")
    audits, weights = [], []
    for _, group in rows.groupby("date", sort=True):
        audit, result = fit_day(group)
        audits.append(audit)
        weights.append(result)
    weight_rows = pd.concat(weights, ignore_index=True)
    weight_rows.to_parquet(output / "contrast_inputs.parquet", index=False)
    save_json(output / "daily_geometry.json", audits)
    summaries = []
    for half in HALVES:
        part = [row for row in audits if row["half"] == half]
        passed = [row for row in part if row["passed"]]
        all_rows, covered = sum(row["rows"] for row in part), sum(row["rows"] for row in passed)
        failed = [{"date": row["date"], "rows": row["rows"],
                   "failed_checks": [name for name, value in row["checks"].items() if not value]}
                  for row in part if not row["passed"]]
        summaries.append({"half": half, "all_days": len(part), "all_stock_days": all_rows,
            "eligible_days": len(passed), "covered_stock_days": covered, "coverage": covered / all_rows,
            "minimum_eligible_residual_variance": min((row["residual_variance"] for row in passed), default=None),
            "maximum_eligible_weight_fraction": max((row["largest_absolute_weight_fraction"] for row in passed), default=None),
            "failed_days": failed, "passed": len(passed) >= 40 and covered >= 5000 and covered / all_rows >= .80})
    report = {"rule_commit": manifest["rule_commit"], "total_stock_days": len(rows), "by_half": summaries,
        "input_gate_passed": all(row["passed"] for row in summaries),
        "contrast_inputs_sha256": sha(output / "contrast_inputs.parquet"),
        "daily_geometry_sha256": sha(output / "daily_geometry.json"),
        "response_read": False, "entry_prices_read": False, "holdout_read": False}
    save_json(output / "input_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    report = freeze(args.output)
    print({key: value for key, value in report.items() if key != "by_half"})
    for row in report["by_half"]:
        print({key: value for key, value in row.items() if key != "failed_days"})
        print("Failed input days:", row["failed_days"])


if __name__ == "__main__":
    main()
