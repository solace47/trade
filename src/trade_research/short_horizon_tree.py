"""One predeclared shallow nonlinear model on the fixed next-day target."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from .absolute_ridge import FEATURES, choose, match_controls
from .corporate_cash import save_json, sha
from .long_history_inputs import ROOT as HISTORY
from .short_horizon_labels import ROOT as TARGET, features
from .short_horizon_target import training_fold
from .short_horizon_target_eval import execute as execute_models, compare as compare_models
from .turnover_reference import CALENDAR

ROOT = Path("data/research/short_horizon_tree")
PROTOCOL = Path("config/short_horizon_tree_protocol.json")
RULE_COMMIT = "d3227c4"
MODELS = ("t1_tree", "t1_target")


def freeze(output: Path = ROOT) -> dict:
    if (output / "input_report.json").exists():
        raise ValueError("Do not replace a frozen tree comparison")
    protocol = json.loads(PROTOCOL.read_text())
    base = json.loads(Path(protocol["base_protocol"]).read_text())
    baseline = Path(protocol["linear_baseline"])
    for path, expected in ((Path(protocol["label_file"]), protocol["label_sha256"]),
            (baseline / "signals.parquet", protocol["linear_signals_sha256"]),
            (baseline / "comparison_report.json", protocol["linear_comparison_sha256"])):
        if sha(path) != expected:
            raise ValueError("Frozen next-day labels or the actual linear baseline changed")
    label_checks = json.loads((TARGET / "independent_label_checks.json").read_text())
    if label_checks["label_report_sha256"] != sha(TARGET / "label_report.json"):
        raise ValueError("The independently verified labels changed")
    output.mkdir(parents=True, exist_ok=True)
    frame = features(); labels = pd.read_parquet(protocol["label_file"])
    raw_calendar = pd.read_parquet(CALENDAR)
    calendar = sorted(raw_calendar.loc[raw_calendar.is_trading_day.eq("1") &
        raw_calendar.calendar_date.between("2022-01-01", "2025-12-31"), "calendar_date"])
    catalog = pd.read_parquet(HISTORY / "training_catalog.parquet")
    baseline_audits = json.loads((TARGET / "input_report.json").read_text())["training"]
    picks, scores, audits = [], [], []
    for fold in base["folds"]:
        train, boundary = training_fold(frame, labels, fold, calendar, catalog)
        baseline_fit = next(x for x in baseline_audits if x["model"] == "t1_target" and x["fold"] == fold["name"])
        # Reuse the exact audited training normalization, not test-year quantiles.
        median = pd.Series(baseline_fit["median"])[list(FEATURES)]
        scale = pd.Series(baseline_fit["scale"])[list(FEATURES)]
        train_x = ((train[list(FEATURES)]-median)/scale).clip(-5, 5)
        target = train.downside_score.clip(-.15, .15)
        test = frame.loc[frame.date.between(fold["test_first"], fold["test_last"])].copy()
        test_x = ((test[list(FEATURES)]-median)/scale).clip(-5, 5)
        with threadpool_limits(limits=4):
            model = HistGradientBoostingRegressor(**protocol["tree_parameters"]).fit(train_x, target)
            test["score"] = model.predict(test_x)
        selected = choose(test, calendar)
        if selected.empty:
            selected = pd.DataFrame(columns=["date", "code", "daily_rank", "score"])
        model_file = output / ("tree_"+fold["name"]+".joblib")
        joblib.dump({"model": model, "median": median, "scale": scale, "features": FEATURES}, model_file)
        picks.append(selected); scores.append(test[["date", "code", "score"]])
        audits.append({"fold": fold["name"], **boundary, "training_rows": len(train), "test_rows": len(test),
            "selected": len(selected), "positive_test_predictions": int(test.score.gt(0).sum()),
            "iterations": int(model.n_iter_), "model_sha256": sha(model_file)})
        print("Trained fixed T+1 tree", audits[-1], flush=True)
    high = pd.concat(picks, ignore_index=True)
    eligible = frame.loc[frame.date.ge("2024-01-01")]
    low = match_controls(high, eligible)
    high["arm"], high["pair_id"] = "high", high.code
    low["arm"] = "low"
    members = pd.concat([high, low], ignore_index=True); members["pair_id"] = members.date+":"+members.pair_id
    signals = members.merge(eligible, on=["date", "code"], validate="one_to_one")
    signals["half"] = signals.date.str[:4]+np.where(signals.date.str[5:7].le("06"), "H1", "H2")
    signals = signals.sort_values(["date", "arm", "daily_rank", "code"]).reset_index(drop=True)
    if len(signals) != len(members) or signals.duplicated(["date", "code"]).any():
        raise ValueError("A tree or control identity changed")
    folder = output / "t1_tree"; folder.mkdir(exist_ok=True)
    signals.to_parquet(folder / "signals.parquet", index=False, compression="zstd")
    pd.concat(scores, ignore_index=True).to_parquet(folder / "all_scores.parquet", index=False, compression="zstd")
    result = {"model": "t1_tree", "candidates": len(high), "controls": len(low),
        "signals_sha256": sha(folder / "signals.parquet"), "all_scores_sha256": sha(folder / "all_scores.parquet"),
        "by_half": signals.groupby(["half", "arm"]).agg(rows=("code", "size"), days=("date", "nunique")).reset_index().to_dict("records")}
    save_json(folder / "input_report.json", result)
    destination = output / "t1_target"
    if not destination.exists():
        shutil.copytree(baseline, destination)
    reused = {}
    for name in ("signals.parquet", "comparison_report.json", "continued/repriced.parquet",
            "continued/tick_cost_scenario.parquet", "continued/execution_queue_audit.parquet"):
        if sha(baseline/name) != sha(destination/name):
            raise ValueError("The copied linear execution differs from its existing ledger")
        reused[name] = sha(baseline/name)
    for name in ("label_report.json", "independent_label_checks.json"):
        shutil.copyfile(TARGET/name, output/name)
    baseline_input = json.loads((baseline / "input_report.json").read_text())
    report = {"rule_commit": RULE_COMMIT, "models": {"t1_tree": result, "t1_target": baseline_input},
        "training": audits, "protocol_sha256": sha(PROTOCOL), "base_protocol_sha256": sha(Path(protocol["base_protocol"])),
        "labels_sha256": protocol["label_sha256"], "baseline_reused_sha256": reused,
        "new_tree_test_returns_read": False, "new_2026_prices_read": False}
    save_json(output / "input_report.json", report)
    return report


def execute(output: Path = ROOT) -> dict:
    return execute_models(output, model_names=MODELS, rule_commit=RULE_COMMIT)


def compare(output: Path = ROOT) -> dict:
    return compare_models(output, model_names=MODELS, rule_commit=RULE_COMMIT,
        contrast_key="common_date_tree_minus_linear", preserve_comparison_models=("t1_target",))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze", "execute", "compare"])
    args = parser.parse_args()
    print(json.dumps(globals()[args.stage](), ensure_ascii=False, indent=2))
