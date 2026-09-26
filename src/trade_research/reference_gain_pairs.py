"""Match comparable stocks first, then rank their pre-decision reference gap."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .cash_ex_matching import RETURN_CALIPERS
from .corporate_cash import save_json, sha
from .turnover_reference_long import ROOT as INPUT_ROOT

ROOT = Path("data/research/reference_gain_pairs")
RULE_COMMIT = "8c75ac3"
RATIOS = ("price_1449", "amount_1449", "prior20_turnover_pct")


def comparable_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    """Nearest unused pairs depend on controls, never on reference scores."""
    if frame.duplicated(["date", "code"]).any():
        raise ValueError("Duplicate input keys")
    features = [*RETURN_CALIPERS, *RATIOS]
    if (not np.isfinite(frame[features].to_numpy()).all()
            or not (frame[list(RATIOS)] > 0).all().all()):
        raise ValueError("Invalid matching features")
    rows = []
    for (date, _), group in frame.groupby(["date", frame.code.str[:2]], sort=True):
        group = group.sort_values("code").reset_index(drop=True)
        left, right = np.triu_indices(len(group), 1)
        valid = np.ones(len(left), dtype=bool)
        distance = np.zeros(len(left))
        for name, caliper in RETURN_CALIPERS.items():
            values = group[name].to_numpy()
            delta = np.abs(values[left] - values[right])
            valid &= delta <= caliper + 1e-12
            distance += delta / caliper
        for name in RATIOS:
            values = group[name].to_numpy()
            ratio = values[left] / values[right]
            valid &= (ratio >= .5) & (ratio <= 2)
            distance += np.abs(np.log(ratio)) / np.log(2)
        possible = np.flatnonzero(valid)
        order = possible[np.lexsort((right[possible], left[possible], distance[possible]))]
        used = set()
        for edge in order:
            i, j = int(left[edge]), int(right[edge])
            if i in used or j in used:
                continue
            used.update((i, j))
            a, b = group.iloc[i], group.iloc[j]
            if a.gain_seed_one < b.gain_seed_one:
                a, b = b, a
            rows.append({"date": date, "half": a.half,
                "code": a.code, "control_code": b.code,
                "distance": float(distance[edge]),
                "scenario_gap": float(a.screen_gain_low - b.screen_gain_high),
                "reference_gap_one": float(a.gain_seed_one - b.gain_seed_one),
                **{f: a[f] for f in features},
                **{"control_" + f: b[f] for f in features}})
    return pd.DataFrame(rows, columns=["date", "half", "code", "control_code", "distance",
        "scenario_gap", "reference_gap_one", *features, *("control_" + f for f in features)])


def freeze(output: Path = ROOT) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    report = json.loads((INPUT_ROOT / "report.json").read_text())
    path = INPUT_ROOT / "screened_inputs.parquet"
    expected = report["output_sha256"]["screened_inputs"]
    if sha(path) != expected:
        raise ValueError("Long-history inputs changed")
    frame = pd.read_parquet(path)
    eligible = frame.loc[frame.screen_eligible].copy()
    # Recompute the orientation from the restored-cent decision quote too.
    eligible["gain_seed_one"] = 1 - eligible.reference_seed_one / eligible.price_1449
    pairs = comparable_pairs(eligible)
    separated = pairs.loc[pairs.scenario_gap.ge(.02)].sort_values(
        ["date", "scenario_gap", "code", "control_code"], ascending=[True, False, True, True]).copy()
    separated["daily_rank"] = separated.groupby("date").cumcount() + 1
    selected = separated.loc[separated.daily_rank.le(5)].copy()
    selected["pair_id"] = selected.date + ":" + selected.code + ":" + selected.control_code
    arms = []
    for arm, key in (("high", "code"), ("low", "control_code")):
        part = selected[["date", key, "pair_id", "daily_rank"]].rename(columns={key: "code"})
        part["arm"] = arm
        arms.append(part)
    assignments = pd.concat(arms, ignore_index=True)
    signals = assignments.merge(eligible, on=["date", "code"], validate="one_to_one")
    signals["isST"] = 0
    signals["listing_age_sessions"] = signals.prior_sessions
    if len(signals) != 2 * len(selected) or signals.duplicated(["date", "code"]).any():
        raise ValueError("Pair endpoints missing or reused")
    summary = []
    for half, all_rows in frame.groupby("half"):
        part = selected.loc[selected.half.eq(half)]
        summary.append({"half": half, "all_rows": len(all_rows),
            "eligible_rows": int(eligible.half.eq(half).sum()),
            "comparable_pairs": int(pairs.half.eq(half).sum()),
            "separated_pairs": int(separated.half.eq(half).sum()),
            "selected_pairs": len(part), "selected_dates": int(part.date.nunique()),
            "mean_scenario_gap": float(part.scenario_gap.mean()) if len(part) else None,
            "mean_return_differences": {f: float((part[f] - part["control_" + f]).mean())
                if len(part) else None for f in RETURN_CALIPERS},
            "mean_feature_ratios": {f: float((part[f] / part["control_" + f]).mean())
                if len(part) else None for f in RATIOS}})
    tables = {"comparable_pairs": pairs, "separated_pairs": separated,
        "selected_pairs": selected, "signals": signals}
    manifest = {"rule_commit": RULE_COMMIT, "input_sha256": expected,
        "entry_prices_read": False, "holding_returns_read": False, "holdout_read": False}
    if (output / "manifest.json").exists() and json.loads((output / "manifest.json").read_text()) != manifest:
        raise ValueError("Cannot overwrite a different experiment")
    save_json(output / "manifest.json", manifest)
    for name, table in tables.items():
        table.to_parquet(output / f"{name}.parquet", index=False, compression="zstd")
    result = {**manifest, "by_half": summary,
        "output_sha256": {k: sha(output / f"{k}.parquet") for k in tables}}
    save_json(output / "input_report.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(freeze()["by_half"], ensure_ascii=False, indent=2))
