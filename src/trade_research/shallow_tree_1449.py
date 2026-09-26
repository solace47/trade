"""A fixed shallow nonlinear ablation with a shared decision-time buy filter."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from .absolute_ridge import FEATURES, PERIODS, fit, score, choose, match_controls
from .absolute_ridge_1449 import feature_frame
from .absolute_ridge_1449_target import exact_training_labels, TRAINING_TRADES
from .corporate_cash import save_json, sha
from .hf_outcomes import _limit_price
from .quote_precision import quote_cents, fixed_quote_shares
from .turnover_reference import CALENDAR

ROOT = Path("data/research/shallow_tree_1449")
RULE_COMMIT = "491a012"
TREE_PARAMETERS = dict(loss="squared_error", learning_rate=.05, max_iter=100,
    max_depth=3, max_leaf_nodes=7, min_samples_leaf=500, l2_regularization=10,
    max_bins=63, early_stopping=False, random_state=20260926)


def decision_buyable(code: str, price: float, preclose: float) -> bool:
    if not code.startswith(("sh.60", "sz.00")) or not np.isfinite(preclose) or preclose <= 0:
        return False
    try:
        quote = quote_cents(price) / 100
        shares = fixed_quote_shares(code, quote, 20000)
    except ValueError:
        return False
    return shares >= 100 and quote * 1.0015 < _limit_price(preclose, .1, True) - .005


def freeze(output: Path = ROOT) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name / "repriced.parquet").exists() for name in ("ridge", "tree")):
        raise ValueError("Model outcomes already exist; do not overwrite the selected lists")
    c = duckdb.connect()
    c.execute("SET threads=4")
    features = feature_frame(c).sort_values(["date", "code"]).reset_index(drop=True)
    c.close()
    labels, target_audit = exact_training_labels()
    labels = labels.sort_values(["date", "code"]).reset_index(drop=True)
    features["decision_buyable"] = [decision_buyable(code, p, prior) for code, p, prior
        in zip(features.code, features.price_1449, features.preclose)]
    eligible = features.loc[features.decision_buyable].copy()
    eligible["price_1449"] = [quote_cents(p) / 100 for p in eligible.price_1449]
    eligible["price_signal"] = eligible.price_1449
    table = pd.read_parquet(CALENDAR)
    calendar = sorted(table.loc[table.is_trading_day.eq("1")
        & table.calendar_date.between("2024-01-01", "2025-12-31"), "calendar_date"].tolist())
    labels.to_parquet(output / "training_labels.parquet", index=False)
    features.to_parquet(output / "features.parquet", index=False)
    picks = {"ridge": [], "tree": []}
    scores = {"ridge": [], "tree": []}
    audits = []
    with threadpool_limits(limits=4):
        for train_end, test_first, test_last, period in PERIODS:
            train = features.loc[features.date.le(train_end)].copy()
            targets = train[["date", "code"]].merge(labels, on=["date", "code"], validate="one_to_one")
            if (len(targets) != len(train) or targets.target_exit_date.ge(test_first).any()
                    or targets.exit_date.ge(test_first).any() or targets.net_return.isna().any()):
                raise ValueError("Incomplete training labels or overlapping test time")
            model, audit = fit(train, targets)
            test = eligible.loc[eligible.date.between(test_first, test_last)].copy()
            ridge = score(test, model)
            joined = train.merge(targets, on=["date", "code"], validate="one_to_one")
            x = ((joined[list(FEATURES)] - model["median"]) / model["scale"]).clip(-5, 5)
            y = joined.net_return.where(joined.clean_ontime_exit, 0.).clip(-.15, .15)
            tree_model = HistGradientBoostingRegressor(**TREE_PARAMETERS).fit(x, y)
            tree = test.copy()
            tx = ((test[list(FEATURES)] - model["median"]) / model["scale"]).clip(-5, 5)
            tree["score"] = tree_model.predict(tx)
            joblib.dump({"model": tree_model, "median": model["median"], "scale": model["scale"]},
                output / f"tree_{period}.joblib")
            audit.update({"period": period, "test_first": test_first, "test_last": test_last,
                "test_eligible_rows": len(test), "tree_iterations": int(tree_model.n_iter_)})
            for name, scored in (("ridge", ridge), ("tree", tree)):
                selected = choose(scored, calendar)
                if selected.empty:
                    selected = pd.DataFrame(columns=["date", "code", "daily_rank", "score"])
                picks[name].append(selected)
                scores[name].append(scored[["date", "code", "score"]])
                audit[name + "_selected"] = len(selected)
            audits.append(audit)
            print(f"Trained {period}: ridge={audit['ridge_selected']}, tree={audit['tree_selected']}", flush=True)
    result = {"rule_commit": RULE_COMMIT, "features": list(FEATURES), "tree_parameters": TREE_PARAMETERS,
        "feature_rows": len(features), "buyable_rows": len(eligible),
        "excluded_at_decision": len(features) - len(eligible), "training": audits,
        "training_target_is_old_score_not_economic_PnL": True, "target_audit": target_audit,
        "training_trades_sha256": sha(TRAINING_TRADES),
        "features_sha256": sha(output / "features.parquet"),
        "labels_sha256": sha(output / "training_labels.parquet"),
        "new_test_outcomes_read": False, "holdout_read": False, "models": {}}
    for name in picks:
        directory = output / name
        directory.mkdir(exist_ok=True)
        chosen = pd.concat(picks[name], ignore_index=True)
        controls = match_controls(chosen, eligible)
        chosen["arm"], chosen["pair_id"] = "high", chosen.code
        controls["arm"] = "low"
        members = pd.concat([chosen, controls], ignore_index=True)
        members["pair_id"] = members.date + ":" + members.pair_id
        signals = members.merge(eligible, on=["date", "code"], validate="one_to_one")
        signals["isST"], signals["reference_gap"], signals["listing_age_sessions"] = 0, False, 20
        signals["half"] = signals.date.str[:4] + np.where(signals.date.str[5:7].astype(int).le(6), "H1", "H2")
        if len(signals) != len(members) or signals.duplicated(["date", "code"]).any():
            raise ValueError("The model/control list changed during input joins")
        signals.to_parquet(directory / "signals.parquet", index=False, compression="zstd")
        pd.concat(scores[name], ignore_index=True).to_parquet(directory / "all_scores.parquet", index=False)
        summary = signals.groupby(["half", "arm"]).agg(rows=("code", "size"), days=("date", "nunique")).reset_index().to_dict("records")
        model_report = {"rule_commit": RULE_COMMIT, "signals_sha256": sha(directory / "signals.parquet"),
            "candidates": len(chosen), "controls": len(controls), "by_half": summary,
            "new_test_outcomes_read": False, "holdout_read": False}
        save_json(directory / "input_report.json", model_report)
        result["models"][name] = model_report
    save_json(output / "input_report.json", result)
    return result


if __name__ == "__main__":
    result = freeze()
    print(json.dumps(result["models"], ensure_ascii=False, indent=2))
