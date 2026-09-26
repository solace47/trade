"""Re-evaluate fixed morning-path selections after the frozen opening audit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from trade_research.market_study import _quality_keys
from trade_research.morning_path_continuous_eval import _one_grid, _period_quality


ROOT = Path("data/research")
AUDIT = ROOT / "opening_quality"
SOURCE = ROOT / "morning_path_open_anchor"
ISSUES = ROOT / "market_issues_ci"
PERIOD_REPORT = ROOT / "quality_period_2024_2025.json"
METRICS = ("morning_cash", "edge_cash")


def _released_days(classified: pd.DataFrame, issues: pd.DataFrame) -> pd.DataFrame:
    keys = ["date", "code"]
    if classified.duplicated(keys).any():
        raise ValueError("Duplicate classified day")
    safe = classified.loc[
        classified.classification.eq("opening_only_volume_matched"), keys]
    other_bad = issues.loc[issues.kind.isin((
        "partial_minute_day", "unexpected_minute_on_suspended_day",
        "active_no_trade", "missing_active_minute",
    )), keys].drop_duplicates()
    merged = safe.merge(other_bad, on=keys, how="left", indicator=True,
                        validate="one_to_one")
    return merged.loc[merged._merge.eq("left_only"), keys].reset_index(drop=True)


def _compare(saved: dict, recomputed: dict) -> None:
    for half, reference in saved["by_half"].items():
        current = recomputed["by_half"][half]
        for key in (*METRICS, "morning_completion", "afternoon_completion"):
            if not np.isclose(reference[key], current[key], atol=1e-12,
                              rtol=0):
                raise ValueError(f"Strict baseline drifted: {half} {key}")


def _summary(strict: dict, relaxed: dict) -> tuple[dict, bool]:
    comparison = {}
    material = False
    for half, old in strict["by_half"].items():
        new = relaxed["by_half"][half]
        comparison[half] = {}
        for key in METRICS:
            delta = new[key] - old[key]
            flipped = np.sign(old[key]) != np.sign(new[key])
            comparison[half][key] = {
                "strict": old[key], "relaxed": new[key],
                "change": delta, "sign_changed": bool(flipped),
            }
            material |= bool(abs(delta) >= .0002 or flipped)
    return comparison, material


def run() -> dict:
    audit = json.loads((AUDIT / "input_audit.json").read_text("utf-8"))
    if (audit["legacy_ohlc_issue_days"] != 14856
            or not audit["sensitivity_input_gate_passed"]):
        raise ValueError("Frozen input gate failed; outcomes stay closed")
    classified = pd.read_parquet(AUDIT / "day_classifications.parquet")
    if (len(classified) != 14856
            or int(classified.classification.eq(
                "opening_only_volume_matched").sum())
            != audit["opening_only_volume_matched_days"]):
        raise ValueError("Opening audit changed since the input gate")
    issue_files = sorted(ISSUES.glob("shard_*.csv"))
    if len(issue_files) != 20:
        raise FileNotFoundError("Frozen issue shards are incomplete")
    issues = pd.concat((pd.read_csv(path, dtype=str) for path in issue_files),
                       ignore_index=True)
    release = _released_days(classified, issues)
    bad = _quality_keys(ISSUES)
    remaining = bad.merge(release, on=["date", "code"], how="left",
                          indicator=True, validate="one_to_one")
    remaining = remaining.loc[remaining._merge.eq("left_only"),
                              ["date", "code"]]
    selections = pd.read_parquet(SOURCE / "selections.parquet")
    trades = pd.read_parquet(SOURCE / "repriced.parquet")
    if (len(selections) != 29558 or len(trades) != len(selections) * 4
            or not selections.date.between("2024-01-01", "2025-12-17").all()):
        raise ValueError("Frozen selection or raw-minute grid changed")
    strict = _period_quality(trades, ISSUES, PERIOD_REPORT, bad)
    relaxed = _period_quality(trades, ISSUES, PERIOD_REPORT, remaining)
    row_keys = ["date", "code", "target_notional", "entry_window",
                "exit_window", "horizon"]
    statuses = strict[row_keys + ["quality_clean_exit"]].merge(
        relaxed[row_keys + ["quality_clean_exit"]], on=row_keys,
        suffixes=("_strict", "_relaxed"), validate="one_to_one")
    if len(statuses) != len(trades):
        raise ValueError("Strict and relaxed outcomes have different keys")
    improved = (~statuses.quality_clean_exit_strict
                & statuses.quality_clean_exit_relaxed)
    if (statuses.quality_clean_exit_strict
            & ~statuses.quality_clean_exit_relaxed).any():
        raise ValueError("Relaxing bad-day keys made a trade invalid")
    source_report = json.loads((SOURCE / "report.json").read_text("utf-8"))
    result = {
        "released_bad_day_keys": len(release),
        "selected_entry_days_released": len(selections.merge(
            release, on=["date", "code"])),
        "quality_status_improved_trade_grid_rows": int(improved.sum()),
        "baseline": "fixed selections and fixed raw-minute outcomes",
    }

    def grid(frame: pd.DataFrame, notional: float, window: str) -> dict:
        subset = frame.loc[frame.target_notional.eq(notional)
                           & frame.exit_window.eq(window)]
        return _one_grid(selections, subset)

    old = grid(strict, 100000., "morning")
    _compare(source_report["results"]["100000"]["morning"], old)
    new = grid(relaxed, 100000., "morning")
    result["primary_by_half"], material = _summary(old, new)
    result["primary_material_change"] = material
    result["primary_by_year"] = {
        year: {"strict_edge": old["by_year"][year]["edge_cash"],
               "relaxed_edge": new["by_year"][year]["edge_cash"]}
        for year in ("2024", "2025")
    }
    if material:
        result["additional_grids"] = {}
        for notional, window in ((20000., "morning"), (20000., "close"),
                                 (100000., "close")):
            name = f"{int(notional)}_{window}"
            before = grid(strict, notional, window)
            _compare(source_report["results"][str(int(notional))][window],
                     before)
            after = grid(relaxed, notional, window)
            result["additional_grids"][name], _ = _summary(before, after)
    AUDIT.mkdir(parents=True, exist_ok=True)
    (AUDIT / "sensitivity.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", "utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
