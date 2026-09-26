"""Fixed shallow-tree comparison using the same long-history economic risk labels."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from .absolute_ridge import FEATURES, PERIODS, choose, match_controls, score
from .corporate_cash import save_json, sha
from .downside_ridge_1449 import fit_score
from .downside_ridge_inputs import ROOT as RECENT
from .long_history_inputs import ROOT as HISTORY
from .quote_precision import quote_cents
from .shallow_tree_1449 import TREE_PARAMETERS
from .turnover_reference import CALENDAR

ROOT = Path("data/research/long_history_tree_1449")
RULE_COMMIT = "7ef4059"


def training_block(features: pd.DataFrame, labels: pd.DataFrame, last: str,
                   test_first: str, calendar: list[str]) -> tuple[pd.DataFrame, str]:
    """Keep conservative labels intact and purge the complete observation window."""
    expected = features.loc[features.date.le(last)]
    train = expected.merge(labels[["date", "code", "downside_score", "score_origin",
        "target_exit_date", "exit_date"]], on=["date", "code"], validate="one_to_one")
    if train.empty or len(train) != len(expected) or not np.isfinite(train.downside_score).all():
        raise ValueError("Incomplete long-history training labels")
    index = {date: i for i, date in enumerate(calendar)}
    endpoint = index[train.date.max()] + 10
    if endpoint >= len(calendar):
        raise ValueError("No complete ten-session training observation window")
    last_possible = calendar[endpoint]
    if (train.target_exit_date.isna().any() or train.target_exit_date.ge(test_first).any()
            or train.exit_date.dropna().ge(test_first).any() or last_possible >= test_first):
        raise ValueError("Training labels or full observation windows overlap the test")
    return train.sort_values(["date", "code"]), last_possible


def freeze(output: Path = ROOT) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    if (output / "tree/repriced.parquet").exists():
        raise ValueError("Cannot replace a tree list after its returns exist")
    original = json.loads((HISTORY / "input_report.json").read_text())
    history_labels = json.loads((HISTORY / "historical_label_report.json").read_text())
    feature_report = json.loads((HISTORY / "feature_report.json").read_text())
    recent_labels = json.loads((RECENT / "label_report.json").read_text())
    sources = [(HISTORY / "features.parquet", feature_report["features_sha256"]),
        (HISTORY / "historical_training_labels.parquet", history_labels["labels_sha256"]),
        (RECENT / "training_labels.parquet", recent_labels["labels_sha256"]),
        (HISTORY / "long/signals.parquet", original["models"]["long"]["signals_sha256"])]
    for path, expected_sha in sources:
        if sha(path) != expected_sha:
            raise ValueError("The existing long-history input or comparison list changed")
    features = pd.read_parquet(sources[0][0])
    labels = pd.concat([pd.read_parquet(sources[1][0]), pd.read_parquet(sources[2][0])], ignore_index=True)
    old_signals = pd.read_parquet(sources[3][0])
    if features.duplicated(["date", "code"]).any() or labels.duplicated(["date", "code"]).any():
        raise ValueError("Duplicate long-history inputs")
    eligible = features.loc[features.date.ge("2024-01-01")].copy()
    eligible["price_1449"] = eligible.price_1449.map(lambda p: quote_cents(p) / 100)
    eligible["price_signal"] = eligible.price_1449
    table = pd.read_parquet(CALENDAR)
    calendar = sorted(table.loc[table.is_trading_day.eq("1")
        & table.calendar_date.between("2022-01-01", "2025-12-31"), "calendar_date"])
    picks, scores, audits = [], [], []
    with threadpool_limits(limits=4):
        for last, first, end, period in PERIODS:
            train, last_possible = training_block(features, labels, last, first, calendar)
            test = eligible.loc[eligible.date.between(first, end)].copy()
            ridge, audit = fit_score(train, train.downside_score)
            expected = next(a for a in original["training"] if a["period"] == period)
            error = max([abs(audit["coefficients"][f] - expected["coefficients"][f])
                for f in FEATURES] + [abs(audit["intercept"] - expected["intercept"])])
            if error > 1e-12:
                raise ValueError("The fixed long-history linear fit no longer reproduces")
            reproduced = choose(score(test, ridge), calendar)
            prior = old_signals.loc[old_signals.arm.eq("high") & old_signals.date.between(first, end),
                ["date", "code", "daily_rank", "score"]]
            pd.testing.assert_frame_equal(reproduced.sort_values(["date", "code"]).reset_index(drop=True),
                prior.sort_values(["date", "code"]).reset_index(drop=True), check_dtype=False, atol=1e-12, rtol=0)
            x = ((train[list(FEATURES)] - ridge["median"]) / ridge["scale"]).clip(-5, 5)
            y = train.downside_score.clip(-.15, .15)
            tree = HistGradientBoostingRegressor(**TREE_PARAMETERS).fit(x, y)
            tx = ((test[list(FEATURES)] - ridge["median"]) / ridge["scale"]).clip(-5, 5)
            test["score"] = tree.predict(tx)
            selected = choose(test, calendar)
            if selected.empty:
                selected = pd.DataFrame(columns=["date", "code", "daily_rank", "score"])
            picks.append(selected)
            scores.append(test[["date", "code", "score"]])
            model_path = output / f"tree_{period}.joblib"
            joblib.dump({"model": tree, "median": ridge["median"], "scale": ridge["scale"],
                "features": list(FEATURES)}, model_path)
            audits.append({"period": period, "train_first": train.date.min(), "train_last": last,
                "test_first": first, "test_last": end, "last_possible_label_day": last_possible,
                "linear_fit_max_difference": error, "linear_candidates_reproduced": len(reproduced),
                "tree_candidates": len(selected), "tree_iterations": int(tree.n_iter_),
                "positive_test_predictions": int(test.score.gt(0).sum()),
                "training_rows_by_year": train.groupby(train.date.str[:4]).size().to_dict(),
                "score_origins": train.score_origin.value_counts().to_dict(),
                "model_sha256": sha(model_path), **audit})
            print(f"Fitted {period}: reproduced linear {len(reproduced)}, tree {len(selected)}", flush=True)
    chosen = pd.concat(picks, ignore_index=True)
    controls = match_controls(chosen, eligible)
    chosen["arm"], chosen["pair_id"] = "high", chosen.code
    controls["arm"] = "low"
    members = pd.concat([chosen, controls], ignore_index=True)
    members["pair_id"] = members.date + ":" + members.pair_id
    signals = members.merge(eligible, on=["date", "code"], validate="one_to_one")
    signals["isST"], signals["reference_gap"], signals["listing_age_sessions"] = 0, False, 20
    signals["half"] = signals.date.str[:4] + np.where(signals.date.str[5:7].astype(int).le(6), "H1", "H2")
    if len(signals) != len(members) or signals.duplicated(["date", "code"]).any():
        raise ValueError("The selected tree/control identities changed")
    folder = output / "tree"
    folder.mkdir(exist_ok=True)
    signals.to_parquet(folder / "signals.parquet", index=False, compression="zstd")
    pd.concat(scores, ignore_index=True).to_parquet(folder / "all_scores.parquet", index=False, compression="zstd")
    model_report = {"candidates": len(chosen), "controls": len(controls),
        "signals_sha256": sha(folder / "signals.parquet"),
        "by_half": signals.groupby(["half", "arm"]).agg(rows=("code", "size"),
            days=("date", "nunique")).reset_index().to_dict("records")}
    save_json(folder / "input_report.json", model_report)
    result = {"rule_commit": RULE_COMMIT, "tree_parameters": TREE_PARAMETERS,
        "features": list(FEATURES), "training": audits,
        "models": {"linear": original["models"]["long"], "tree": model_report},
        "source_sha256": {str(path): value for path, value in sources},
        "linear_input_report_sha256": sha(HISTORY / "input_report.json"),
        "new_test_returns_read": False, "holdout_read": False}
    save_json(output / "input_report.json", result)
    return result


if __name__ == "__main__":
    report = freeze()
    print(json.dumps(report["models"], ensure_ascii=False, indent=2))
