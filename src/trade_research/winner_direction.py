"""One frozen classifier, upside-only and two-sided rankings, actual T+1 orders."""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from .absolute_ridge import choose,match_controls
from .cash_dividend_catalog import fetch as fetch_catalog,assemble as assemble_catalog
from .corporate_cash import save_json,sha
from .next_day_winner import ROOT as PORTRAIT,calendar
from .next_day_winner_analysis import FEATURES
from .price_limit_queue_audit import audit
from .reference_gain_accounting import evaluate as account
from .reference_gain_eval import reprice
from .risk_removal_eval import evaluate as tick
from .shallow_tree_continuation import continue_model
from .short_horizon_target_eval import compare as compare_models

ROOT=Path("data/research/winner_direction")
PROTOCOL=Path("config/winner_direction_protocol.json")
RULE_COMMIT="66e011b"
MODELS=("balanced","upside_only")


def features() -> tuple[pd.DataFrame,pd.DataFrame]:
    allowed=list(dict.fromkeys(["date","code","half","board","industry","necessary_tradeable",
        "preclose","price_1449","high_1449","low_1449","volume_1449","amount_1449","decision_shares",
        "isST","reference_gap","listing_age_sessions",
        "market_return","market_rising",*FEATURES]))
    frame=pd.read_parquet(PORTRAIT/"cohort.parquet",columns=allowed)
    ranked=frame.groupby(["date","board"])[list(FEATURES)].rank(pct=True,method="average").add_prefix("rank_")
    ranked["log_price"]=np.log(frame.price_1449)
    ranked["log_amount"]=np.log(frame.amount_1449)
    ranked["day_range"]=(frame.high_1449-frame.low_1449)/frame.preclose
    for col in ("return_1449","return5_prior_adjusted","return20_prior_adjusted","market_return","market_rising"):
        ranked[col]=frame[col]
    for board in ("chinext","star"):
        ranked["board_"+board]=frame.board.eq(board).astype(float)
    mask=frame.necessary_tradeable & (frame.date.le("2024-12-30")|frame.date.between("2025-01-02","2025-12-17"))
    return frame.loc[mask].reset_index(drop=True),ranked.loc[mask].replace([np.inf,-np.inf],np.nan).reset_index(drop=True)


def freeze(output:Path=ROOT) -> dict:
    if (output/"input_report.json").exists():
        raise ValueError("Do not replace a fixed probability-ranked list")
    output.mkdir(parents=True,exist_ok=True)
    audit_report=json.loads((PORTRAIT/"analysis_report.json").read_text())
    checks=json.loads((PORTRAIT/"independent_paths_checks.json").read_text())
    if (sha(PORTRAIT/"cohort.parquet")!=audit_report["cohort_sha256"]
        or checks["daily_labels_rebuilt"]!=audit_report["rows"]):
        raise ValueError("The independently checked portrait changed")
    pool,x=features();train=pool.date.lt("2025-01-01");test=~train
    labels=pd.read_parquet(PORTRAIT/"labels.parquet",filters=[("date","<=","2024-12-30")],
        columns=["date","code","next_date","known_label","next_close","next_preclose"])
    labels=pool.loc[train,["date","code"]].merge(labels,on=["date","code"],how="left",validate="one_to_one")
    if labels.next_date.isna().any() or labels.next_date.ge("2025-01-01").any():
        raise ValueError("A training label is missing or crosses the fit boundary")
    close=np.rint(labels.next_close.fillna(0)*100).astype("int64")
    reference=np.rint(labels.next_preclose.fillna(0)*100).astype("int64")
    labels["target"]=np.select([~labels.known_label,close*100>=reference*105,close*100<=reference*95],
        ["unknown","up","down"],default="flat")
    medians=x.loc[train].median()
    if medians.isna().any():raise ValueError("An entire training feature is missing")
    filled=x.fillna(medians)
    mean=filled.loc[train].mean();scale=filled.loc[train].std(ddof=0).clip(lower=1e-12)
    z=((filled-mean)/scale).to_numpy(dtype="float64")
    with warnings.catch_warnings(),threadpool_limits(limits=4):
        warnings.simplefilter("error",ConvergenceWarning)
        fitted=LogisticRegression(C=1.,solver="newton-cholesky",tol=1e-8,max_iter=100,random_state=20260927)
        fitted.fit(z[train],labels.target)
        probabilities=fitted.predict_proba(z[test])
    model={"feature_names":list(x),"median":medians.tolist(),"mean":mean.tolist(),"scale":scale.tolist(),
        "classes":fitted.classes_.tolist(),"coefficients":fitted.coef_.tolist(),"intercepts":fitted.intercept_.tolist(),
        "iterations":fitted.n_iter_.tolist(),"C":1.,"solver":"newton-cholesky","tol":1e-8,"max_iter":100}
    save_json(output/"model.json",model)
    labels.to_parquet(output/"training_labels.parquet",index=False,compression="zstd")
    encoded=pd.concat([pool[["date","code"]],x],axis=1)
    encoded.to_parquet(output/"model_features.parquet",index=False,compression="zstd")
    pool.to_parquet(output/"visible_pool.parquet",index=False,compression="zstd")
    label_report={"kind":"next_market_day_reference_price_class_not_trading_pnl","rows":len(labels),
        "target_counts":labels.target.value_counts().to_dict(),"last_signal":labels.date.max(),
        "last_label":labels.next_date.max(),"test_labels_used":False,
        "source_labels_sha256":sha(PORTRAIT/"labels.parquet"),"labels_sha256":sha(output/"training_labels.parquet")}
    save_json(output/"label_report.json",label_report)
    scores=pool.loc[test,["date","code"]].reset_index(drop=True)
    for i,name in enumerate(fitted.classes_):scores["p_"+name]=probabilities[:,i]
    scores["balanced"]=scores.p_up-scores.p_down;scores["upside_only"]=scores.p_up
    scores.to_parquet(output/"all_scores.parquet",index=False,compression="zstd")
    eligible=pool.loc[test].copy()
    eligible["price_signal"]=eligible.price_1449;eligible["amount_signal"]=eligible.amount_1449
    eligible["return_1450"]=eligible.return_1449  # Compatibility alias, strictly 14:49.
    results={}
    for name in MODELS:
        chosen=choose(scores[["date","code",name]].rename(columns={name:"score"}),calendar())
        if chosen.empty:
            chosen=pd.DataFrame(columns=["date","code","daily_rank","score"])
        low=match_controls(chosen,eligible)
        chosen["arm"],chosen["pair_id"]="high",chosen.code;low["arm"]="low"
        members=pd.concat([chosen,low],ignore_index=True)
        members["pair_id"]=members.date+":"+members.pair_id
        signals=members.merge(eligible,on=["date","code"],validate="one_to_one").sort_values(["date","arm","daily_rank","code"]).reset_index(drop=True)
        if signals.duplicated(["date","code"]).any() or len(signals)!=len(members):
            raise ValueError("A selected identity was duplicated or lost")
        folder=output/name;folder.mkdir(exist_ok=True)
        signals.to_parquet(folder/"signals.parquet",index=False,compression="zstd")
        info={"model":name,"candidates":len(chosen),"controls":len(low),"signals_sha256":sha(folder/"signals.parquet"),
            "by_half_board":signals.groupby(["half","board","arm"]).agg(rows=("code","size"),days=("date","nunique")).reset_index().to_dict("records")}
        save_json(folder/"input_report.json",info);results[name]=info
    report={"rule_commit":RULE_COMMIT,"models":results,"training":label_report,
        "test_rows":int(test.sum()),"feature_count":len(x.columns),"iterations":model["iterations"],
        "protocol_sha256":sha(PROTOCOL),"source_cohort_sha256":sha(PORTRAIT/"cohort.parquet"),
        "outputs_sha256":{name:sha(output/name) for name in ("model.json","training_labels.parquet","model_features.parquet","visible_pool.parquet","all_scores.parquet")},
        "new_selected_trading_results_computed":False,"new_2026_prices_read":False}
    save_json(output/"input_report.json",report)
    return report


def catalog(output:Path=ROOT) -> dict:
    root=output/"catalog";root.mkdir(exist_ok=True)
    models=json.loads((output/"input_report.json").read_text())["models"]
    frames=[]
    for name,report in models.items():
        path=output/name/"signals.parquet"
        if sha(path)!=report["signals_sha256"]:raise ValueError("A fixed selection changed")
        frames.append(pd.read_parquet(path,columns=["date","code"]))
    chosen=pd.concat(frames,ignore_index=True).drop_duplicates()
    positions={d:i for i,d in enumerate(calendar())};days=calendar()
    needed=pd.DataFrame(sorted({(r.code,str(year)) for r in chosen.itertuples()
        for year in range(int(r.date[:4]),int(days[positions[r.date]+10][:4])+1)}),columns=["code","year"])
    old=Path("data/research/cash_dividend_catalog")
    coverage=pd.read_parquet(old/"query_coverage.parquet")
    missing=needed.merge(coverage[["code","year"]],on=["code","year"],how="left",indicator=True)
    missing=missing.loc[missing._merge.eq("left_only"),["code","year"]].reset_index(drop=True)
    jobs=root/"jobs.parquet"
    if jobs.exists():pd.testing.assert_frame_equal(pd.read_parquet(jobs),missing)
    else:missing.to_parquet(jobs,index=False)
    save_json(root/"manifest.json",{"jobs_sha256":sha(jobs),"rule_commit":RULE_COMMIT,"code_years":len(missing),"strategy_returns_read":False})
    if len(missing):
        fetched=fetch_catalog(root)
        if not fetched["complete"]:return fetched
        assemble_catalog(root)
        new_events=pd.read_parquet(root/"events.parquet")
        new_coverage=pd.read_parquet(root/"query_coverage.parquet")
    else:
        new_events=pd.DataFrame();new_coverage=coverage.iloc[:0]
    old_events=pd.read_parquet(old/"events_augmented.parquet")
    combined=pd.concat([old_events,new_events],ignore_index=True).sort_values(["code","dividOperateDate"])
    if combined.duplicated(["code","dividOperateDate"]).any():
        raise ValueError("Old and new catalogues have overlapping event identities")
    combined.to_parquet(root/"combined_events.parquet",index=False)
    coverage=pd.concat([coverage,new_coverage],ignore_index=True)
    coverage.to_parquet(root/"combined_coverage.parquet",index=False)
    if not needed.merge(coverage,on=["code","year"],how="left",indicator=True)._merge.eq("both").all():
        raise ValueError("A selected security still lacks catalogue coverage")
    result={"needed_code_years":len(needed),"new_code_years":len(missing),
        "events_sha256":sha(root/"combined_events.parquet"),"coverage_sha256":sha(root/"combined_coverage.parquet"),
        "original_events_sha256":sha(old/"events_augmented.parquet"),"complete":True,"new_2026_prices_read":False}
    save_json(root/"coverage_report.json",result)
    return result


def execute(output:Path=ROOT) -> dict:
    checks=json.loads((output/"independent_input_checks.json").read_text())
    inputs=json.loads((output/"input_report.json").read_text())
    cat=json.loads((output/"catalog/coverage_report.json").read_text())
    if sha(output/"catalog/combined_events.parquet")!=cat["events_sha256"]:
        raise ValueError("The checked catalogue changed")
    reports={}
    for name in MODELS:
        if inputs["models"][name]["candidates"]==0:
            reports[name]={"no_positive_selections":True}
            continue
        folder=output/name;expected=inputs["models"][name]["signals_sha256"]
        if sha(folder/"signals.parquet")!=expected or checks["models"][name]["signals_sha256"]!=expected:
            raise ValueError("A checked selection changed")
        if not (folder/"execution_report.json").exists():
            reprice(folder,expected_signal_sha=expected,rule_commit=RULE_COMMIT,horizons=(1,),exact_decision_sizing=True)
        continued=folder/"continued"
        if not (continued/"execution_report.json").exists():continue_model(folder,continued)
        if not (continued/"execution_queue_report.json").exists():audit(continued)
        if not (continued/"catalog_scenario_report.json").exists():account(continued,catalog_path=output/"catalog/combined_events.parquet")
        if not (continued/"tick_report.json").exists():tick(continued,primary_horizon=1)
        reports[name]=json.loads((continued/"execution_report.json").read_text())
    return reports


def compare(output:Path=ROOT) -> dict:
    return compare_models(output,model_names=MODELS,rule_commit=RULE_COMMIT,contrast_key="common_date_balanced_minus_upside_only")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("stage",choices=["freeze","catalog","execute","compare"])
    args=p.parse_args();print(json.dumps(globals()[args.stage](),ensure_ascii=False,indent=2))
