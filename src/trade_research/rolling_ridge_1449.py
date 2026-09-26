"""Preregistered quarterly expanding fits of the fixed conservative ridge."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from pathlib import Path
import shutil

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .absolute_ridge import FEATURES, score, choose, match_controls
from .corporate_cash import save_json, sha
from .downside_ridge_1449 import fit_score
from .downside_ridge_inputs import ROOT as STATIC, training_scores, window_summary
from .hf_outcomes import Assumptions, outcomes_for_symbol
from .quote_precision import quote_cents
from .reference_gain_eval import DAILY, MINUTES
from .turnover_reference import CALENDAR

ROOT = Path("data/research/rolling_ridge_1449")
RULE_COMMIT = "681ad90"


def calendar_and_folds() -> tuple[list[str], list[dict]]:
    data = pd.read_parquet(CALENDAR)
    days = sorted(data.loc[data.is_trading_day.eq("1")
        & data.calendar_date.between("2024-01-01", "2025-12-31"), "calendar_date"])
    folds = []
    for period in pd.period_range("2024Q3", "2025Q4", freq="Q"):
        start = str(period.start_time.date())
        last = str(period.end_time.date())
        first = next(date for date in days if date >= start)
        cut = days[days.index(first)-11]
        folds.append({"period": str(period), "train_last": cut, "test_first": first,
            "test_last": min(last, first[:4]+"-12-17"),
            "last_possible_label_day": days[days.index(cut)+10]})
    return days, folds


def prepare(output: Path = ROOT) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    if (output/"rolling"/"repriced.parquet").exists():
        raise ValueError("Do not rebuild training after rolling-model outcomes exist")
    source_report = json.loads((STATIC/"label_report.json").read_text())
    for name, key in (("features", "features_sha256"), ("training_labels", "labels_sha256")):
        if sha(STATIC/(name+".parquet")) != source_report[key]:
            raise ValueError("The fixed static-model training source changed")
    features = pd.read_parquet(STATIC/"features.parquet")
    calendar, folds = calendar_and_folds()
    index = {date:i for i,date in enumerate(calendar)}
    signals = features.loc[features.date.between("2025-01-01", folds[-1]["train_last"])].copy()
    signals["price_1449"] = signals.price_1449.map(lambda price:quote_cents(price)/100)
    signals["isST"], signals["reference_gap"], signals["listing_age_sessions"] = 0, False, 20
    signals.to_parquet(output/"training_signals_2025.parquet", index=False, compression="zstd")

    def one(item):
        code, group = item
        exchange, symbol = code.split(".")
        minute_path = MINUTES/exchange.upper()/(symbol+".parquet")
        daily_path = DAILY/(code.replace(".","_")+".parquet")
        dates = sorted({date for signal in group.date for date in calendar[index[signal]:index[signal]+11]})
        if dates[-1] > folds[-1]["last_possible_label_day"]:
            raise ValueError("A training source window crosses the fourth-quarter boundary")
        c = duckdb.connect()
        c.execute("SET threads=1")
        c.read_parquet(str(minute_path)).create_view("minutes")
        c.register("needed", pd.DataFrame({"date":dates}))
        c.execute("""CREATE TEMP TABLE selected_bars AS SELECT timestamp,open,high,low,close,volume,turnover
            FROM minutes WHERE timestamp>=?::TIMESTAMP AND timestamp<?::DATE+INTERVAL 1 DAY
              AND strftime(timestamp,'%H%M') IN ('1452','1453','1454','1455')
              AND strftime(timestamp,'%Y-%m-%d') IN (SELECT date FROM needed)""", [dates[0],dates[-1]])
        minute = c.execute("SELECT * FROM selected_bars ORDER BY timestamp").df()
        windows = pd.DataFrame({"date":dates}).merge(window_summary(c),on="date",how="left",validate="one_to_one")
        windows["window_status"],windows["code"] = windows.window_status.fillna("missing_window"),code
        c.read_parquet(str(daily_path)).create_view("daily")
        prior = c.execute("SELECT max(date) FROM daily WHERE date<? AND tradestatus=1",[dates[0]]).fetchone()[0]
        if prior is None:
            raise ValueError("A training stock lacks its previous trading day")
        daily = c.execute("SELECT * FROM daily WHERE date BETWEEN ? AND ? ORDER BY date",[prior,dates[-1]]).df()
        c.close()
        minute["date"],minute["label"] = minute.timestamp.dt.strftime("%Y-%m-%d"),minute.timestamp.dt.strftime("%H%M")
        trades = outcomes_for_symbol(group,minute,daily,calendar,Assumptions(target_notional=20000),horizons=(5,),sizing_price_column="price_1449")
        trades["target_notional"],trades["entry_window"],trades["exit_window"] = 20000,"baseline","close"
        return trades,windows,{"code":code,"first":dates[0],"last":dates[-1],
            "minute_sha256":sha(minute_path),"daily_sha256":sha(daily_path)}

    records, windows, sources = [],[],[]
    groups = list(signals.groupby("code",sort=True))
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,(trades,quality,source) in enumerate(pool.map(one,groups),1):
            records.append(trades);windows.append(quality);sources.append(source)
            if i % 300 == 0 or i == len(groups):
                print(f"Quarterly training sources {i}/{len(groups)}",flush=True)
    raw = pd.concat(records,ignore_index=True)
    all_windows = pd.concat(windows,ignore_index=True)
    raw.to_parquet(output/"training_repriced_2025.parquet",index=False,compression="zstd")
    all_windows.to_parquet(output/"training_windows_2025.parquet",index=False,compression="zstd")
    save_json(output/"training_sources_2025.json",sources)
    labels,events = training_scores(raw,all_windows,last_catalog_pay_date="2025-12-31")
    labels = labels.sort_values(["date","code"])
    labels.to_parquet(output/"training_labels_2025.parquet",index=False,compression="zstd")
    events.to_parquet(output/"training_actions_2025.parquet",index=False)
    joined = pd.concat([pd.read_parquet(STATIC/"training_labels.parquet"),labels],ignore_index=True)
    for fold in folds:
        part = joined.loc[joined.date.le(fold["train_last"])]
        if part.target_exit_date.ge(fold["test_first"]).any() or part.exit_date.dropna().ge(fold["test_first"]).any():
            raise ValueError("An actual label exit overlaps its test fold")
        fold["training_rows"] = len(part)
        fold["score_origins"] = part.score_origin.value_counts().to_dict()
    report = {"rule_commit":RULE_COMMIT,"folds":folds,"new_training_signals":len(signals),
        "new_training_labels":len(labels),"new_training_first":labels.date.min(),"new_training_last":labels.date.max(),
        "latest_source_date":max(source["last"] for source in sources),
        "features_sha256":sha(STATIC/"features.parquet"),"labels_2024_sha256":sha(STATIC/"training_labels.parquet"),
        "labels_2025_sha256":sha(output/"training_labels_2025.parquet"),
        "source_windows_sha256":sha(output/"training_windows_2025.parquet"),
        "window_statuses":all_windows.window_status.value_counts().to_dict(),"new_test_outcomes_read":False,"holdout_read":False}
    save_json(output/"training_report.json",report)
    return report


def freeze(output: Path = ROOT) -> dict:
    if (output/"rolling"/"repriced.parquet").exists():
        raise ValueError("Do not change a rolling list after its outcomes exist")
    report = json.loads((output/"training_report.json").read_text())
    for path,key in ((STATIC/"features.parquet","features_sha256"),
        (STATIC/"training_labels.parquet","labels_2024_sha256"),
        (output/"training_labels_2025.parquet","labels_2025_sha256")):
        if sha(path) != report[key]:
            raise ValueError("Frozen rolling-training input changed")
    features = pd.read_parquet(STATIC/"features.parquet")
    labels = pd.concat([pd.read_parquet(STATIC/"training_labels.parquet"),
        pd.read_parquet(output/"training_labels_2025.parquet")],ignore_index=True)
    if labels.duplicated(["date","code"]).any():
        raise ValueError("Duplicate historical labels")
    eligible = features.copy()
    eligible["price_1449"] = eligible.price_1449.map(lambda price:quote_cents(price)/100)
    eligible["price_signal"] = eligible.price_1449
    calendar,folds = calendar_and_folds()
    scored,audits = [],[]
    with threadpool_limits(limits=4):
        for fold in folds:
            train = features.loc[features.date.le(fold["train_last"])].merge(
                labels[["date","code","downside_score","exit_date","target_exit_date"]],
                on=["date","code"],validate="one_to_one").sort_values(["date","code"])
            if (len(train) != features.date.le(fold["train_last"]).sum()
                or train.exit_date.dropna().ge(fold["test_first"]).any()
                or train.target_exit_date.ge(fold["test_first"]).any()):
                raise ValueError("Training is incomplete or overlaps the test fold")
            model,audit = fit_score(train,train.downside_score)
            test = eligible.loc[eligible.date.between(fold["test_first"],fold["test_last"])]
            ranked = score(test,model)
            scored.append(ranked)
            audits.append({**fold,**audit,"test_rows":len(test),"positive_predictions":int(ranked.score.gt(0).sum())})
    ranked = pd.concat(scored,ignore_index=True)
    if ranked.duplicated(["date","code"]).any():
        raise ValueError("Quarterly prediction intervals overlap")
    # Select once across all folds, so refitting never resets the cooldown.
    chosen = choose(ranked,calendar)
    controls = match_controls(chosen,eligible)
    chosen["arm"],chosen["pair_id"] = "high",chosen.code
    controls["arm"] = "low"
    members = pd.concat([chosen,controls],ignore_index=True)
    members["pair_id"] = members.date+":"+members.pair_id
    signals = members.merge(eligible,on=["date","code"],validate="one_to_one")
    signals["isST"],signals["reference_gap"],signals["listing_age_sessions"] = 0,False,20
    signals["half"] = signals.date.str[:4]+np.where(signals.date.str[5:7].astype(int).le(6),"H1","H2")
    if len(signals) != len(members) or signals.duplicated(["date","code"]).any():
        raise ValueError("A selected signal key changed")
    static_report = json.loads((STATIC/"input_report.json").read_text())
    static_model = static_report["models"]["downside"]
    if sha(STATIC/"downside"/"signals.parquet") != static_model["signals_sha256"]:
        raise ValueError("The selected static comparator changed")
    static_audits = {a["period"]:a for a in static_report["training"] if a["model"]=="downside"}
    reproduction = {}
    for quarter,period in (("2024Q3","2024H2"),("2025Q1","2025")):
        current = next(a for a in audits if a["period"]==quarter)
        original = static_audits[period]
        difference = max(abs(current["coefficients"][k]-original["coefficients"][k]) for k in FEATURES)
        difference = max(difference,abs(current["intercept"]-original["intercept"]))
        if difference>1e-12:
            raise ValueError("An unchanged quarterly fit failed to reproduce the static model")
        reproduction[quarter] = difference
    shutil.copytree(STATIC/"downside",output/"static",dirs_exist_ok=True)
    folder = output/"rolling"
    folder.mkdir(exist_ok=True)
    signals.to_parquet(folder/"signals.parquet",index=False,compression="zstd")
    ranked[["date","code","score"]].to_parquet(folder/"all_scores.parquet",index=False)
    model_report = {"candidates":len(chosen),"controls":len(controls),"signals_sha256":sha(folder/"signals.parquet"),
        "by_half":signals.groupby(["half","arm"]).agg(rows=("code","size"),days=("date","nunique")).reset_index().to_dict("records")}
    save_json(folder/"input_report.json",model_report)
    result = {"rule_commit":RULE_COMMIT,"training_report_sha256":sha(output/"training_report.json"),
        "training":audits,"unchanged_fit_max_differences":reproduction,
        "models":{"static":static_model,"rolling":model_report},
        "static_results_previously_read":True,"new_rolling_test_outcomes_read":False,"holdout_read":False}
    save_json(output/"input_report.json",result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("prepare","freeze"),nargs="?",default="prepare")
    args = parser.parse_args()
    result = prepare() if args.phase=="prepare" else freeze()
    print(json.dumps(result if args.phase=="prepare" else result["models"],ensure_ascii=False,indent=2))
