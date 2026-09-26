"""Freeze equal-risk-pool ridge models with original or conservative labels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .absolute_ridge import FEATURES, PERIODS, score, choose, match_controls
from .corporate_cash import save_json, sha
from .downside_ridge_inputs import ROOT, RULE_COMMIT
from .quote_precision import quote_cents
from .turnover_reference import CALENDAR


def fit_score(frame: pd.DataFrame, values: pd.Series) -> tuple[dict, dict]:
    if values.isna().any() or not frame.index.equals(values.index):
        raise ValueError("Every training feature needs its aligned risk score")
    median = frame[list(FEATURES)].median()
    scale = (frame[list(FEATURES)].quantile(.75)-frame[list(FEATURES)].quantile(.25)).clip(lower=.01)
    x = ((frame[list(FEATURES)]-median)/scale).clip(-5,5).to_numpy(float)
    y = values.clip(-.15,.15).to_numpy(float)
    x_mean, y_mean = x.mean(axis=0), y.mean()
    centered = x-x_mean
    coefficients = np.linalg.solve(centered.T@centered + len(frame)*.05*np.eye(len(FEATURES)), centered.T@(y-y_mean))
    model = {"median": median, "scale": scale, "coefficients": coefficients, "intercept": float(y_mean-x_mean@coefficients)}
    audit = {"rows": len(frame), "target_clipped_mean": float(y_mean), "intercept": model["intercept"],
        "median": median.to_dict(), "scale": scale.to_dict(), "coefficients": dict(zip(FEATURES,coefficients.tolist()))}
    return model, audit


def freeze(output: Path = ROOT) -> dict:
    names = ("old_score", "downside")
    if any((output/name/"repriced.parquet").exists() for name in names):
        raise ValueError("Do not change lists after test outcomes exist")
    inputs = json.loads((output/"label_report.json").read_text())
    for name,key in (("features","features_sha256"),("training_labels","labels_sha256")):
        if sha(output/(name+".parquet")) != inputs[key]:
            raise ValueError("Prepared training inputs changed")
    features, labels = pd.read_parquet(output/"features.parquet"), pd.read_parquet(output/"training_labels.parquet")
    table = pd.read_parquet(CALENDAR)
    calendar = sorted(table.loc[table.is_trading_day.eq("1") & table.calendar_date.between("2024-01-01","2025-12-31"),"calendar_date"])
    eligible = features.copy()
    eligible["price_1449"] = eligible.price_1449.map(lambda p: quote_cents(p)/100)
    eligible["price_signal"] = eligible.price_1449
    picks, scores, audits = {n:[] for n in names}, {n:[] for n in names}, []
    with threadpool_limits(limits=4):
        for last,first,end,period in PERIODS:
            train = features.loc[features.date.le(last)].merge(labels[["date","code","old_score","downside_score","target_exit_date","exit_date"]],on=["date","code"],validate="one_to_one").sort_values(["date","code"])
            if len(train) != features.date.le(last).sum() or train.target_exit_date.ge(first).any() or train.exit_date.dropna().ge(first).any():
                raise ValueError("Training labels are incomplete or overlap test")
            test = eligible.loc[eligible.date.between(first,end)]
            for name in names:
                column = "old_score" if name == "old_score" else "downside_score"
                model,audit = fit_score(train,train[column])
                ranked = score(test,model)
                selected = choose(ranked,calendar)
                if selected.empty:
                    selected = pd.DataFrame(columns=["date","code","daily_rank","score"])
                picks[name].append(selected)
                scores[name].append(ranked[["date","code","score"]])
                audits.append({"model":name,"period":period,"train_last":last,"test_first":first,
                    "candidates":len(selected),"positive_predictions":int(ranked.score.gt(0).sum()),**audit})
    result = {"rule_commit":RULE_COMMIT,"training":audits,"label_report_sha256":sha(output/"label_report.json"),
        "new_test_outcomes_read":False,"holdout_read":False,"models":{}}
    for name in names:
        folder = output/name
        folder.mkdir(exist_ok=True)
        chosen = pd.concat(picks[name],ignore_index=True)
        controls = match_controls(chosen,eligible)
        chosen["arm"], chosen["pair_id"] = "high", chosen.code
        controls["arm"] = "low"
        members = pd.concat([chosen,controls],ignore_index=True)
        members["pair_id"] = members.date+":"+members.pair_id
        signals = members.merge(eligible,on=["date","code"],validate="one_to_one")
        signals["isST"],signals["reference_gap"],signals["listing_age_sessions"] = 0,False,20
        signals["half"] = signals.date.str[:4]+np.where(signals.date.str[5:7].astype(int).le(6),"H1","H2")
        if len(signals) != len(members) or signals.duplicated(["date","code"]).any():
            raise ValueError("A fixed signal key changed")
        signals.to_parquet(folder/"signals.parquet",index=False,compression="zstd")
        pd.concat(scores[name],ignore_index=True).to_parquet(folder/"all_scores.parquet",index=False)
        result["models"][name] = {"candidates":len(chosen),"controls":len(controls),"signals_sha256":sha(folder/"signals.parquet"),
            "by_half":signals.groupby(["half","arm"]).agg(rows=("code","size"),days=("date","nunique")).reset_index().to_dict("records")}
        save_json(folder/"input_report.json",result["models"][name])
    save_json(output/"input_report.json",result)
    return result


if __name__ == "__main__":
    print(json.dumps(freeze()["models"],ensure_ascii=False,indent=2))
