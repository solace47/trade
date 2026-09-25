"""Freeze a morning-led versus afternoon-led path test without outcomes."""

from __future__ import annotations

import argparse
from hashlib import md5
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path("data/research")
SOURCE = ROOT / "morning_burst" / "inputs.parquet"
OUTPUT = ROOT / "morning_path"
HALVES = ("2024H1", "2024H2", "2025H1", "2025H2")
INPUT_COLUMNS = (
    "date", "code", "board", "max5", "morning_pp", "day_pp", "tail_pp",
    "prior20_pp", "price_1449", "amount_1449",
)


def audit(source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Use only the pre-14:49 fields from the earlier morning input extract."""
    if not set(INPUT_COLUMNS).issubset(source.columns):
        raise ValueError("Missing pre-decision input columns")
    inputs = source.loc[:, INPUT_COLUMNS].copy()
    if (inputs.empty or inputs.duplicated(["date", "code"]).any()
            or not inputs.date.between("2024-01-01", "2025-12-17").all()
            or not np.isfinite(inputs.select_dtypes(include="number").to_numpy()).all()):
        raise ValueError("Invalid, duplicate, or out-of-period inputs")
    inputs["half"] = inputs.date.str[:4] + "H" + np.where(
        inputs.date.str[5:7].astype(int) <= 6, "1", "2")
    if set(inputs.half) != set(HALVES):
        raise ValueError("Incomplete development periods")
    # These are path-timing groups, independent of the old max5 burst labels.
    inputs["arm"] = np.select(
        [inputs.morning_pp.ge(1.5), inputs.morning_pp.le(.5)],
        ["morning", "afternoon"], default="middle")
    inputs["day_bin"] = np.floor(inputs.day_pp / .5).astype(int)
    arms = inputs.loc[inputs.arm.ne("middle")].copy()
    counts = arms.groupby(
        ["half", "date", "board", "day_bin", "arm"]
    ).size().unstack("arm", fill_value=0)
    for arm in ("morning", "afternoon"):
        if arm not in counts:
            counts[arm] = 0
    eligible = counts.loc[counts.morning.ge(5) & counts.afternoon.ge(5)]
    keys = eligible.reset_index()[["half", "date", "board", "day_bin"]]
    matched = arms.merge(keys, on=["half", "date", "board", "day_bin"],
                         how="inner", validate="many_to_one")
    report = {}
    for half in HALVES:
        all_half = arms.loc[arms.half.eq(half)]
        paired_half = matched.loc[matched.half.eq(half)]
        strata = (eligible.loc[half]
                  if half in eligible.index.get_level_values("half") else None)
        arm_report = {}
        for arm in ("morning", "afternoon"):
            total = int(all_half.arm.eq(arm).sum())
            covered = int(paired_half.arm.eq(arm).sum())
            arm_report[arm] = {
                "candidates": total,
                "matched_stock_days": covered,
                "coverage": covered / total if total else 0.0,
            }
        report[half] = {
            "comparable_strata": len(strata) if strata is not None else 0,
            "comparable_days": (strata.index.get_level_values("date").nunique()
                                if strata is not None else 0),
            "arms": arm_report,
        }
    gate = all(
        part["comparable_strata"] >= 50
        and part["comparable_days"] >= 30
        and all(item["matched_stock_days"] >= 500
                and item["coverage"] >= .20
                for item in part["arms"].values())
        for part in report.values()
    )
    return arms, matched, {
        "cutoff": "14:49", "years": [2024, 2025],
        "base_stock_days": len(inputs), "by_half": report,
        "outcome_gate_passed": bool(gate),
    }


def freeze(source_path: Path = SOURCE, output_dir: Path = OUTPUT) -> dict:
    arms, matched, report = audit(pd.read_parquet(source_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    arms.to_parquet(output_dir / "all_arms.parquet", index=False,
                    compression="zstd")
    matched.to_parquet(output_dir / "matched.parquet", index=False,
                       compression="zstd")
    if report["outcome_gate_passed"]:
        selected = matched.copy()
        selected["audit_order"] = [md5(
            ("morning-path-v1" + row.date + row.code).encode()
        ).hexdigest() for row in selected.itertuples(index=False)]
        sample = selected.sort_values("audit_order").groupby(
            ["half", "arm"], sort=True).head(40)
        if len(sample) != 320:
            raise ValueError("Incomplete raw-minute verification sample")
        sample.drop(columns="audit_order").to_parquet(
            output_dir / "raw_signals.parquet", index=False,
            compression="zstd")
    else:
        (output_dir / "raw_signals.parquet").unlink(missing_ok=True)
    (output_dir / "input_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return report


def opening_gap_audit(
        input_dir: Path = OUTPUT,
        morning_dir: Path = ROOT / "morning_burst" / "parts",
        snapshot_dir: Path = ROOT / "market_snapshots_ci") -> dict:
    """Post-freeze, input-only support check; never touches trade outcomes."""
    all_arms = pd.read_parquet(input_dir / "all_arms.parquet")
    matched = pd.read_parquet(input_dir / "matched.parquet")
    connection = duckdb.connect()
    try:
        connection.execute("SET threads = 4")
        connection.register("arms", all_arms)
        connection.from_parquet(str(morning_dir / "*.parquet")
                                ).create_view("morning")
        connection.from_parquet(str(snapshot_dir / "*.parquet")
                                ).create_view("snapshots")
        inputs = connection.execute("""
            SELECT a.*, 100 * (m.price_0930 / s.preclose - 1)
                           AS opening_gap_pp,
                   100 * (a.price_1449 / m.price_1130 - 1)
                           AS afternoon_pp
            FROM arms a JOIN morning m USING (date, code)
            JOIN snapshots s USING (date, code)
            WHERE s.preclose > 0 AND m.price_0930 > 0
              AND m.price_1130 > 0
        """).df()
    finally:
        connection.close()
    if (len(inputs) != len(all_arms)
            or inputs.duplicated(["date", "code"]).any()
            or not np.isfinite(inputs[["opening_gap_pp", "afternoon_pp"]]
                               .to_numpy()).all()):
        raise ValueError("Incomplete opening-gap input join")
    original = inputs.merge(matched[["date", "code"]],
                            on=["date", "code"], validate="one_to_one")
    medians = original.groupby(["half", "arm"])[
        ["opening_gap_pp", "afternoon_pp"]].median()
    means = original.groupby(["half", "arm"])[
        ["opening_gap_pp", "afternoon_pp"]].mean()
    report = {"original_matched": {
        half: {arm: {
            "opening_gap_mean_pp": float(means.loc[(half, arm),
                                                 "opening_gap_pp"]),
            "opening_gap_median_pp": float(medians.loc[(half, arm),
                                                     "opening_gap_pp"]),
            "afternoon_mean_pp": float(means.loc[(half, arm),
                                               "afternoon_pp"]),
        } for arm in ("morning", "afternoon")}
        for half in HALVES}, "gap_bin_checks": {}}
    for width in (.5, 1.0):
        inputs["opening_gap_bin"] = np.floor(
            inputs.opening_gap_pp / width).astype(int)
        counts = inputs.groupby(
            ["half", "date", "board", "day_bin", "opening_gap_bin", "arm"]
        ).size().unstack("arm", fill_value=0)
        by_half = {}
        for half in HALVES:
            part = counts.loc[half]
            both = part.morning.ge(5) & part.afternoon.ge(5)
            candidates = all_arms.loc[all_arms.half.eq(half)]
            by_half[half] = {
                "comparable_strata": int(both.sum()),
                "comparable_days": int(part.index.get_level_values(
                    "date")[both].nunique()),
                "morning_coverage": float(part.morning[both].sum()
                                          / candidates.arm.eq("morning").sum()),
                "afternoon_coverage": float(part.afternoon[both].sum()
                                            / candidates.arm.eq("afternoon").sum()),
            }
        report["gap_bin_checks"][str(width)] = by_half
    broad = report["gap_bin_checks"]["1.0"]
    report["stop_before_outcomes"] = any(
        part["comparable_strata"] < 50
        or part["comparable_days"] < 30
        or part["morning_coverage"] < .20
        or part["afternoon_coverage"] < .20
        for part in broad.values()
    )
    inputs.to_parquet(input_dir / "opening_gap_inputs.parquet", index=False,
                      compression="zstd")
    (input_dir / "opening_gap_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "opening-gap-audit"),
                        nargs="?", default="freeze")
    args = parser.parse_args()
    action = freeze if args.stage == "freeze" else opening_gap_audit
    print(json.dumps(action(), ensure_ascii=False, indent=2))
