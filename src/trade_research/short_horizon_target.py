"""Fit T+1 and T+5 targets with one feature set and shared conservative labels."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .absolute_ridge import FEATURES, score, choose, match_controls
from .corporate_cash import save_json, sha
from .downside_ridge_1449 import fit_score
from .long_history_inputs import ROOT as HISTORY
from .short_horizon_labels import ROOT, PROTOCOL, RULE_COMMIT, features
from .turnover_reference import CALENDAR


def training_fold(frame: pd.DataFrame, labels: pd.DataFrame, fold: dict,
                  calendar: list[str], catalog: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    fields = ["date", "code", "downside_score", "score_origin", "target_exit_date", "exit_date", "action_status"]
    base = frame.loc[frame.date.between(fold["train_first"], fold["train_last"])]
    train = base.merge(labels[fields], on=["date", "code"], validate="one_to_one").sort_values(["date", "code"])
    position = {d: i for i, d in enumerate(calendar)}
    last_possible = calendar[position[train.date.max()]+10]
    if (len(train) != len(base) or train.downside_score.isna().any() or last_possible >= fold["test_first"]
            or train.target_exit_date.ge(fold["test_first"]).any() or train.exit_date.dropna().ge(fold["test_first"]).any()):
        raise ValueError("A complete training label overlaps the test period")
    # A cash payment used in a training target must also precede the fit boundary.
    known = train.loc[train.score_origin.eq("recorded_economic_scenario") & train.action_status.eq("catalogue_scenario")]
    actions = known[["date", "code", "exit_date"]].merge(catalog, on="code")
    actions = actions.loc[actions.dividOperateDate.gt(actions.date) & actions.dividOperateDate.le(actions.exit_date)]
    if actions.dividPayDate.ge(fold["test_first"]).any():
        raise ValueError("A training dividend payment occurs in the test period")
    return train, {"train_first": train.date.min(), "train_last": train.date.max(),
        "last_possible_label_day": last_possible,
        "last_recorded_exit": None if train.exit_date.dropna().empty else train.exit_date.dropna().max(),
        "last_used_dividend_pay_date": None if actions.empty else actions.dividPayDate.max(),
        "rows_by_year": train.groupby(train.date.str[:4]).size().to_dict(),
        "score_origins": train.score_origin.value_counts().to_dict()}


def freeze(output: Path = ROOT) -> dict:
    if (output / "input_report.json").exists():
        raise ValueError("Frozen model selections cannot be overwritten")
    protocol = json.loads(PROTOCOL.read_text())
    report = json.loads((output / "label_report.json").read_text())
    frame = features()
    calendar_raw = pd.read_parquet(CALENDAR)
    calendar = sorted(calendar_raw.loc[calendar_raw.is_trading_day.eq("1") &
        calendar_raw.calendar_date.between("2022-01-01", "2025-12-31"), "calendar_date"])
    catalog = pd.read_parquet(HISTORY / "training_catalog.parquet")
    output_models, audits = {}, []
    for model in ("t1_target", "t5_target"):
        stem = model[:2]
        path = output / (stem+"_labels.parquet")
        if sha(path) != report["models"][stem]["labels_sha256"]:
            raise ValueError("A frozen target label file changed")
        labels = pd.read_parquet(path)
        picks, scores = [], []
        with threadpool_limits(limits=4):
            for fold in protocol["folds"]:
                train, boundary = training_fold(frame, labels, fold, calendar, catalog)
                fitted, audit = fit_score(train, train.downside_score)
                test = frame.loc[frame.date.between(fold["test_first"], fold["test_last"])]
                ranked = score(test, fitted)
                selected = choose(ranked, calendar)
                if selected.empty:
                    selected = pd.DataFrame(columns=["date", "code", "daily_rank", "score"])
                picks.append(selected); scores.append(ranked[["date", "code", "score"]])
                audits.append({"model": model, "fold": fold["name"], **boundary, **audit,
                    "test_first": fold["test_first"], "test_last": fold["test_last"],
                    "test_rows": len(test), "positive_test_predictions": int(ranked.score.gt(0).sum()),
                    "selected": len(selected)})
        high = pd.concat(picks, ignore_index=True)
        eligible = frame.loc[frame.date.ge("2024-01-01")]
        low = match_controls(high, eligible)
        high["arm"], high["pair_id"] = "high", high.code
        low["arm"] = "low"
        members = pd.concat([high, low], ignore_index=True)
        members["pair_id"] = members.date+":"+members.pair_id
        signals = members.merge(eligible, on=["date", "code"], validate="one_to_one")
        signals["half"] = signals.date.str[:4]+np.where(signals.date.str[5:7].le("06"), "H1", "H2")
        signals = signals.sort_values(["date", "arm", "daily_rank", "code"]).reset_index(drop=True)
        if len(signals) != len(members) or signals.duplicated(["date", "code"]).any():
            raise ValueError("A model or control identity changed")
        folder = output / model; folder.mkdir(exist_ok=True)
        signals.to_parquet(folder / "signals.parquet", index=False, compression="zstd")
        pd.concat(scores, ignore_index=True).to_parquet(folder / "all_scores.parquet", index=False, compression="zstd")
        result = {"model": model, "candidates": len(high), "controls": len(low),
            "by_half": signals.groupby(["half", "arm"]).agg(rows=("code", "size"), days=("date", "nunique")).reset_index().to_dict("records"),
            "signals_sha256": sha(folder / "signals.parquet"), "all_scores_sha256": sha(folder / "all_scores.parquet")}
        save_json(folder / "input_report.json", result); output_models[model] = result
    result = {"rule_commit": RULE_COMMIT, "models": output_models, "training": audits,
        "protocol_sha256": sha(PROTOCOL), "label_report_sha256": sha(output / "label_report.json"),
        "test_labels_used_to_fit_same_period": False, "new_2026_prices_read": False,
        "new_selected_test_returns_read": False}
    save_json(output / "input_report.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(freeze(), ensure_ascii=False, indent=2))
