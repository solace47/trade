"""Apply the frozen same-day matching gate before ex-date outcome reads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .cash_dividend_catalog import ROOT as CATALOG
from .cash_ex_inputs import ROOT
from .corporate_cash import save_json, sha


RETURN_CALIPERS = {"day_return": .005, "return_last29": .002, "prior20_return": .05}
RATIO_FIELDS = ("price_1449", "amount_1449")


def match_candidates(attempts: pd.DataFrame, controls: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if attempts.duplicated(["date", "code"]).any() or controls.duplicated(["date", "code"]).any():
        raise ValueError("Matching requires unique stock days")
    common = attempts[["date", "code"]].merge(controls[["date", "code"]], on=["date", "code"])
    if len(common):
        raise ValueError("Treatment cannot also appear in its control universe")
    features = list(RETURN_CALIPERS) + list(RATIO_FIELDS)
    if any(not np.isfinite(frame[features].to_numpy()).all() for frame in (attempts, controls)):
        raise ValueError("Nonfinite matching features")
    if any((frame[list(RATIO_FIELDS)] <= 0).any().any() for frame in (attempts, controls)):
        raise ValueError("Price and amount must be positive before logarithmic matching")
    groups = {key: group for key, group in controls.groupby(["date", controls.code.str[:2]], sort=False)}
    matched, missed = [], []
    used: dict[str, set[str]] = {}
    for row in attempts.sort_values(["date", "daily_rank", "code"]).to_dict("records"):
        pool = groups.get((row["date"], row["code"][:2]))
        if pool is not None:
            pool = pool.loc[~pool.code.isin(used.setdefault(row["date"], set()))].copy()
            for feature, caliper in RETURN_CALIPERS.items():
                pool = pool.loc[(pool[feature] - row[feature]).abs() <= caliper + 1e-12]
            for feature in RATIO_FIELDS:
                pool = pool.loc[(row[feature] / pool[feature]).between(.5, 2)]
        if pool is None or pool.empty:
            missed.append({"date": row["date"], "code": row["code"], "half": row["half"],
                           "daily_rank": row["daily_rank"], "reason": "no_unused_control_within_calipers"})
            continue
        distance = sum((pool[feature] - row[feature]).abs() / caliper
                       for feature, caliper in RETURN_CALIPERS.items())
        distance += sum(np.abs(np.log(row[feature] / pool[feature])) / np.log(2)
                        for feature in RATIO_FIELDS)
        choice = pool.assign(distance=distance).sort_values(["distance", "code"]).iloc[0]
        used[row["date"]].add(choice.code)
        matched.append({"date": row["date"], "half": row["half"], "code": row["code"],
            "control_code": choice.code, "daily_rank": row["daily_rank"], "distance": choice.distance,
            **{feature: row[feature] for feature in features},
            **{"control_" + feature: choice[feature] for feature in features},
            **{feature + "_difference": row[feature] - choice[feature] for feature in RETURN_CALIPERS},
            **{feature + "_ratio": row[feature] / choice[feature] for feature in RATIO_FIELDS}})
    pair_columns = ["date", "half", "code", "control_code", "daily_rank", "distance"] + features
    pair_columns += ["control_" + feature for feature in features]
    pair_columns += [feature + "_difference" for feature in RETURN_CALIPERS]
    pair_columns += [feature + "_ratio" for feature in RATIO_FIELDS]
    return pd.DataFrame(matched, columns=pair_columns), pd.DataFrame(missed,
        columns=["date", "code", "half", "daily_rank", "reason"])


def assess(attempts: pd.DataFrame, pairs: pd.DataFrame) -> list[dict]:
    summaries = []
    for half in ("2024H1", "2024H2", "2025H1", "2025H2"):
        attempted = attempts.loc[attempts.half.eq(half)]
        part = pairs.loc[pairs.half.eq(half)]
        coverage = len(part) / len(attempted) if len(attempted) else 0.0
        metrics = {feature + "_mean_difference": float(part[feature + "_difference"].mean())
                   if len(part) else None for feature in RETURN_CALIPERS}
        metrics.update({feature + "_mean_ratio": float(part[feature + "_ratio"].mean())
                        if len(part) else None for feature in RATIO_FIELDS})
        checks = {"pair_days": part.date.nunique() >= 30, "pairs": len(part) >= 50, "coverage": coverage >= .60}
        for feature, limit in (("day_return", .001), ("return_last29", .0005), ("prior20_return", .015)):
            value = metrics[feature + "_mean_difference"]
            checks[feature + "_balance"] = value is not None and abs(value) <= limit
        for feature in RATIO_FIELDS:
            value = metrics[feature + "_mean_ratio"]
            checks[feature + "_balance"] = value is not None and 2 / 3 <= value <= 1.5
        summaries.append({"half": half, "attempted_rows": len(attempted),
            "attempted_days": int(attempted.date.nunique()), "pairs": len(part),
            "paired_days": int(part.date.nunique()), "coverage": coverage, **metrics,
            "checks": {key: bool(value) for key, value in checks.items()}, "passed": all(checks.values())})
    return summaries


def run(output: Path = ROOT) -> dict:
    primary = json.loads((output / "notice_template_report.json").read_text())
    base_report = json.loads((output / "base_report.json").read_text())
    supplement = json.loads((CATALOG / "primary_supplement_report.json").read_text())
    if not primary["all_decision_terms_verified"] or not base_report["association_passed"]:
        raise ValueError("Source verification and input association must pass before matching")
    hashes = {str(output / "verified_candidates.parquet"): primary["verified_candidates_sha256"],
              str(output / "base.parquet"): base_report["base_sha256"],
              str(CATALOG / "events_augmented.parquet"): supplement["augmented_events_sha256"]}
    for name, expected in hashes.items():
        if sha(Path(name)) != expected:
            raise ValueError("A verified matching input changed")
    candidates = pd.read_parquet(output / "verified_candidates.parquet")
    attempts = candidates.loc[candidates.daily_rank.le(5)].copy()
    base = pd.read_parquet(output / "base.parquet")
    events = pd.read_parquet(CATALOG / "events_augmented.parquet")[["code", "dividOperateDate"]].rename(
        columns={"dividOperateDate": "date"})
    controls = base.merge(events.assign(has_event=True), on=["code", "date"], how="left", validate="one_to_one")
    controls = controls.loc[controls.has_event.isna() & ~controls.reference_gap].copy()
    pairs, missed = match_candidates(attempts, controls)
    if (len(pairs) + len(missed) != len(attempts) or pairs.duplicated(["date", "code"]).any()
            or pairs.duplicated(["date", "control_code"]).any()):
        raise ValueError("Matching lost an attempt or reused a stock day")
    summaries = assess(attempts, pairs)
    pairs.to_parquet(output / "input_pairs.parquet", index=False)
    missed.to_parquet(output / "unmatched_attempts.parquet", index=False)
    report = {"rule_commit": "1599017", "inputs_sha256": hashes, "by_half": summaries,
              "input_gate_passed": all(row["passed"] for row in summaries),
              "pairs_sha256": sha(output / "input_pairs.parquet"),
              "unmatched_sha256": sha(output / "unmatched_attempts.parquet"),
              "strategy_returns_read": False, "holdout_read": False}
    save_json(output / "matching_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    print(run(args.output))


if __name__ == "__main__":
    main()
