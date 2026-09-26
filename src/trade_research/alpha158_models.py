"""One fixed feature-library comparison with identical robust normalization."""

from __future__ import annotations

import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .absolute_ridge import FEATURES, PERIODS, choose, match_controls, score
from .alpha158_asof import definitions
from .alpha158_inputs import ROOT, RULE_COMMIT
from .corporate_cash import save_json, sha
from .downside_ridge_1449 import fit_score
from .downside_ridge_inputs import ROOT as RECENT
from .long_history_inputs import ROOT as HISTORY
from .long_history_tree_1449 import training_block
from .quote_precision import quote_cents
from .turnover_reference import CALENDAR


def transform(frame: pd.DataFrame, model: dict) -> np.ndarray:
    values = frame[model["columns"]].to_numpy(float)
    if np.isinf(values).any():
        raise ValueError("Infinite feature values must be explicitly marked missing")
    values = (values - np.asarray(model["median"])) / np.asarray(model["scale"])
    np.clip(values, -3, 3, out=values)
    for column in model["all_missing_columns"]:
        values[:, model["columns"].index(column)] = 0.
    return np.nan_to_num(values, copy=False, nan=0.)


def fit_robust(frame: pd.DataFrame, target: pd.Series) -> dict:
    if not frame.index.equals(target.index) or not np.isfinite(target).all():
        raise ValueError("Training targets must align with every feature row")
    raw = frame.to_numpy(float)
    if np.isinf(raw).any():
        raise ValueError("Infinite training inputs are not silently normalized")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(raw, axis=0)
        mad = np.nanmedian(np.abs(raw - median), axis=0)
    all_missing = np.isnan(median)
    median[all_missing], mad[all_missing] = 0., 0.
    model = {"columns": frame.columns.tolist(), "median": median.tolist(),
        "scale": ((mad + 1e-12) * 1.4826).tolist(),
        "all_missing_columns": frame.columns[all_missing].tolist(),
        "zero_mad_columns": frame.columns[mad == 0].tolist(), "penalty": .05, "clip": 3}
    del raw
    x = transform(frame, model)
    y = target.clip(-.15, .15).to_numpy(float)
    x_mean, y_mean = x.mean(axis=0), y.mean()
    x -= x_mean
    coefficients = np.linalg.solve(x.T @ x + len(frame) * .05 * np.eye(x.shape[1]), x.T @ (y - y_mean))
    model.update(coefficients=coefficients.tolist(), intercept=float(y_mean - x_mean @ coefficients),
                 training_rows=len(frame), clipped_target_mean=float(y_mean))
    return model


def freeze(output: Path = ROOT) -> dict:
    names = ("robust18", "alpha158")
    if any((output / name / "repriced.parquet").exists() for name in names):
        raise ValueError("Do not replace model lists after outcomes exist")
    original = json.loads((HISTORY / "input_report.json").read_text())
    historical = json.loads((HISTORY / "historical_label_report.json").read_text())
    old_features = json.loads((HISTORY / "feature_report.json").read_text())
    recent = json.loads((RECENT / "label_report.json").read_text())
    new_features = json.loads((output / "feature_report.json").read_text())
    checks = json.loads((output / "independent_feature_checks.json").read_text())
    if checks["samples"] != 64 or checks["factor_values_checked"] != 10112 or checks["maximum_difference"] > 5e-8:
        raise ValueError("The real feature sample has not passed its independent check")
    sources = [(HISTORY / "features.parquet", old_features["features_sha256"]),
        (HISTORY / "historical_training_labels.parquet", historical["labels_sha256"]),
        (RECENT / "training_labels.parquet", recent["labels_sha256"]),
        (HISTORY / "long/signals.parquet", original["models"]["long"]["signals_sha256"]),
        (output / "features.parquet", new_features["features_sha256"])]
    for path, expected in sources:
        if sha(path) != expected:
            raise ValueError("A frozen common input changed")
    base, alpha = pd.read_parquet(sources[0][0]), pd.read_parquet(sources[4][0])
    pd.testing.assert_frame_equal(base[["date", "code"]].sort_values(["date", "code"]).reset_index(drop=True),
                                  alpha[["date", "code"]].reset_index(drop=True))
    alpha_index = pd.MultiIndex.from_frame(alpha[["date", "code"]])
    labels = pd.concat([pd.read_parquet(sources[1][0]), pd.read_parquet(sources[2][0])], ignore_index=True)
    old_signals = pd.read_parquet(sources[3][0])
    eligible = base.loc[base.date.ge("2024-01-01")].copy()
    eligible["price_1449"] = eligible.price_1449.map(lambda p: quote_cents(p) / 100)
    eligible["price_signal"] = eligible.price_1449
    table = pd.read_parquet(CALENDAR)
    calendar = sorted(table.loc[table.is_trading_day.eq("1")
        & table.calendar_date.between("2022-01-01", "2025-12-31"), "calendar_date"])
    columns = {"robust18": list(FEATURES), "alpha158": [name for name, _ in definitions()]}
    picks, scores, audits = {name: [] for name in names}, {name: [] for name in names}, []
    with threadpool_limits(limits=4):
        for last, first, end, period in PERIODS:
            train, endpoint = training_block(base, labels, last, first, calendar)
            test = eligible.loc[eligible.date.between(first, end)].copy()
            legacy, legacy_audit = fit_score(train, train.downside_score)
            expected = next(a for a in original["training"] if a["period"] == period)
            legacy_error = max([abs(legacy_audit["coefficients"][f] - expected["coefficients"][f]) for f in FEATURES]
                               + [abs(legacy_audit["intercept"] - expected["intercept"])])
            if legacy_error > 1e-12:
                raise ValueError("Original long-history fit no longer reproduces")
            reproduced = choose(score(test, legacy), calendar)
            prior = old_signals.loc[old_signals.arm.eq("high") & old_signals.date.between(first, end),
                ["date", "code", "daily_rank", "score"]]
            pd.testing.assert_frame_equal(reproduced.sort_values(["date", "code"]).reset_index(drop=True),
                prior.sort_values(["date", "code"]).reset_index(drop=True), check_dtype=False, atol=1e-12, rtol=0)
            train_indices = alpha_index.get_indexer(pd.MultiIndex.from_frame(train[["date", "code"]]))
            test_indices = alpha_index.get_indexer(pd.MultiIndex.from_frame(test[["date", "code"]]))
            if (train_indices < 0).any() or (test_indices < 0).any():
                raise ValueError("New factors lost a common training or evaluation identity")
            for name in names:
                x = (train[columns[name]] if name == "robust18" else alpha.iloc[train_indices][columns[name]]).reset_index(drop=True)
                model = fit_robust(x, train.downside_score.reset_index(drop=True))
                del x
                tx = test[columns[name]] if name == "robust18" else alpha.iloc[test_indices][columns[name]]
                ranked = test[["date", "code"]].copy()
                ranked["score"] = transform(tx, model) @ np.asarray(model["coefficients"]) + model["intercept"]
                selected = choose(ranked, calendar)
                if selected.empty:
                    selected = pd.DataFrame(columns=["date", "code", "daily_rank", "score"])
                model_path = output / f"{name}_{period}.json"
                save_json(model_path, model)
                picks[name].append(selected)
                scores[name].append(ranked)
                audits.append({"model": name, "period": period, "train_last": last, "test_first": first,
                    "test_last": end, "last_possible_label_day": endpoint, "training_rows": len(train),
                    "training_rows_by_year": train.groupby(train.date.str[:4]).size().to_dict(),
                    "score_origins": train.score_origin.value_counts().to_dict(), "legacy_fit_max_error": legacy_error,
                    "legacy_candidates_reproduced": len(reproduced), "candidates": len(selected),
                    "positive_predictions": int(ranked.score.gt(0).sum()), "model_sha256": sha(model_path),
                    "all_missing_columns": model["all_missing_columns"], "zero_mad_columns": model["zero_mad_columns"]})
                print({"period": period, "model": name, "training_rows": len(train), "candidates": len(selected)}, flush=True)
    models = {}
    for name in names:
        chosen = pd.concat(picks[name], ignore_index=True)
        controls = match_controls(chosen, eligible)
        chosen["arm"], chosen["pair_id"] = "high", chosen.code
        controls["arm"] = "low"
        members = pd.concat([chosen, controls], ignore_index=True)
        members["pair_id"] = members.date + ":" + members.pair_id
        signals = members.merge(eligible, on=["date", "code"], validate="one_to_one")
        signals["isST"], signals["reference_gap"], signals["listing_age_sessions"] = 0, False, 20
        signals["half"] = signals.date.str[:4] + np.where(signals.date.str[5:7].le("06"), "H1", "H2")
        if len(signals) != len(members) or signals.duplicated(["date", "code"]).any():
            raise ValueError("Selected model/control identities changed")
        folder = output / name
        folder.mkdir(exist_ok=True)
        signals.to_parquet(folder / "signals.parquet", index=False, compression="zstd")
        pd.concat(scores[name], ignore_index=True).to_parquet(folder / "all_scores.parquet", index=False, compression="zstd")
        model_report = {"candidates": len(chosen), "controls": len(controls), "signals_sha256": sha(folder / "signals.parquet"),
            "by_half": signals.groupby(["half", "arm"]).agg(rows=("code", "size"), days=("date", "nunique"))
                .reset_index().to_dict("records")}
        save_json(folder / "input_report.json", model_report)
        models[name] = model_report
    report = {"rule_commit": RULE_COMMIT, "models": models, "training": audits,
        "sources": {str(path): expected for path, expected in sources},
        "independent_feature_check_sha256": sha(output / "independent_feature_checks.json"),
        "new_test_returns_read": False, "holdout_prices_read": False}
    save_json(output / "input_report.json", report)
    return report


if __name__ == "__main__":
    report = freeze()
    print(json.dumps(report["models"], ensure_ascii=False, indent=2))
