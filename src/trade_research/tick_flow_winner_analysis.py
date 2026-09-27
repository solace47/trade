"""All frozen direction bands and reverse portraits against strict T1 labels."""
from __future__ import annotations

import json
import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .corporate_cash import save_json,sha
from .economic_winner_analysis import number,summarize,METRICS
from .reference_gain_accounting import weekly_interval
from .tick_flow_winner import ROOT,FEATURES

LABELS=Path("data/research/economic_winner/period_quality")
PERIODS=["2024H1","2024H2","2025H1","2025H2","2024","2025"]
BANDS=["low","middle","high","unknown"]


def in_period(frame,period):
    return frame.loc[frame.half.eq(period) if "H" in period else frame.date.str.startswith(period)]


def aggregate(p,keys):
    r=p.groupby(keys,dropna=False).agg(n=("code","size"),known=("known","sum"),unknown=("unknown","sum"),
        no_trade=("no_trade","sum"),winner_count=("winner","sum"),loser_count=("loser","sum"),
        positive_count=("positive","sum"),net_mean=("net_return","mean")).reset_index()
    for name in ["winner","loser","positive"]:
        r[name+"_lower"]=r[name+"_count"]/r.n
        r[name+"_upper"]=(r[name+"_count"]+r.unknown)/r.n
    return r


def summary(daily,rows,identifiers):
    result=summarize(daily,identifiers)
    values=rows.net_return.dropna()
    wins=values[values>0];losses=values[values<0]
    result.update(conditional_win_rate=number((values>0).mean()),median=number(values.median()),
        mean_win=number(wins.mean()),mean_loss=number(losses.mean()),
        payoff_ratio=number(wins.mean()/-losses.mean()) if len(wins) and len(losses) else None,
        worst_five_percent_mean=number(values.nsmallest(max(1,math.ceil(len(values)*.05))).mean()))
    return result


def evaluate():
    if (ROOT/"analysis_report.json").exists():raise ValueError("Do not overwrite inspected direction results")
    gate=json.loads((ROOT/"input_verification.json").read_text())
    assert gate["passed"] and gate["input_report_sha256"]==sha(ROOT/"input_report.json")
    inputs=json.loads((ROOT/"input_report.json").read_text())
    for name,digest in inputs["output_sha256"].items():assert sha(ROOT/name)==digest
    proof=json.loads((LABELS/"analysis_verification.json").read_text())
    assert proof["passed"] and proof["analysis_report_sha256"]==sha(LABELS/"analysis_report.json")
    assert sha(LABELS/"labels.parquet")==json.loads((LABELS/"label_report.json").read_text())["labels_sha256"]
    features=pd.read_parquet(ROOT/"features.parquet")
    c=duckdb.connect();c.register("selected",features[["date","code"]])
    labels=c.execute("""select l.date,l.code,l.label5,l.label15,l.net_return5,l.net_return15
        from read_parquet(?) l join selected s on l.date=s.date and l.code=s.code""",[str(LABELS/"labels.parquet")]).fetchdf()
    frame=features.merge(labels,on=["date","code"],validate="one_to_one")
    assert len(frame)==len(features)
    frame.to_parquet(ROOT/"joined_original.parquet",index=False,compression="zstd")
    scenarios=[];day_tables=[];baselines=[];groups=[];inverse=[]
    for quality in ["original_labels","tick_quality_sensitivity"]:
        for cost in [5,15]:
            p=frame.copy()
            p["label"]=p[f"label{cost}"]
            p["net_return"]=p[f"net_return{cost}"]
            if quality=="tick_quality_sensitivity":
                uncertain=~p.source_valid & ~p.label.eq("no_trade")
                p.loc[uncertain,"label"]="unknown"
                p.loc[uncertain,"net_return"]=np.nan
            p["known"]=p.net_return.notna();p["unknown"]=p.label.eq("unknown")
            p["no_trade"]=p.label.eq("no_trade");p["winner"]=p.label.eq("economic_winner")
            p["loser"]=p.label.eq("economic_loser");p["positive"]=p.net_return.gt(0)
            assert (p.known.astype(int)+p.unknown.astype(int)+p.no_trade.astype(int)).eq(1).all()
            p["quality"]=quality;p["cost_bps"]=cost
            scenarios.append(p)
            base=aggregate(p,["date","half"])
            base["quality"]=quality;base["cost_bps"]=cost;base["feature"]="baseline";base["band"]="all"
            day_tables.append(base)
            for period in PERIODS:
                baselines.append(summary(in_period(base,period),in_period(p,period),dict(quality=quality,cost_bps=cost,period=period)))
            for feature in FEATURES:
                p["band"]=p[feature+"_group"]
                daily=aggregate(p,["date","half","band"])
                daily=daily.merge(base[["date",*METRICS]],on="date",validate="many_to_one",suffixes=("","_baseline"))
                for name in ["winner","loser","positive"]:
                    daily[name+"_lower_delta"]=daily[name+"_lower"]-daily[name+"_upper_baseline"]
                    daily[name+"_upper_delta"]=daily[name+"_upper"]-daily[name+"_lower_baseline"]
                daily["net_mean_delta"]=daily.net_mean-daily.net_mean_baseline
                daily["quality"]=quality;daily["cost_bps"]=cost;daily["feature"]=feature
                day_tables.append(daily)
                for period in PERIODS:
                    for band in BANDS:
                        data=in_period(daily,period);raw=in_period(p,period)
                        groups.append(summary(data.loc[data.band.eq(band)],raw.loc[raw.band.eq(band)],
                            dict(quality=quality,cost_bps=cost,feature=feature,period=period,band=band)))
                if cost==15:
                    for period in PERIODS:
                        part=in_period(p,period)
                        for label in ["economic_winner","economic_loser","middle","unknown","no_trade"]:
                            # The source's middle label is checked below and is not synthesized from NaN.
                            raw=part.loc[part.label.eq(label)]
                            values=raw[feature]
                            day_mean=raw.groupby("date")[feature].mean().sort_index()
                            inverse.append(dict(quality=quality,feature=feature,period=period,label=label,
                                rows=len(raw),feature_missing=int(values.isna().sum()),mean=number(values.mean()),
                                median=number(values.median()),date_mean=number(day_mean.mean()),
                                date_mean_week_interval=weekly_interval(day_mean)))
                print(json.dumps(dict(quality=quality,cost_bps=cost,feature=feature,completed_groups=len(groups))),flush=True)
    assert set(frame.label15.unique())<={"economic_winner","economic_loser","middle","unknown","no_trade"}
    pd.concat(scenarios,ignore_index=True).drop(columns="band",errors="ignore").to_parquet(ROOT/"evaluation_rows.parquet",index=False,compression="zstd")
    pd.concat(day_tables,ignore_index=True).to_parquet(ROOT/"groups_daily.parquet",index=False,compression="zstd")
    report=dict(interpretation="unbiased-by-outcome fixed daily sample; vendor-inferred aggregate direction, not investor identity or a deployed strategy",
        input_report_sha256=sha(ROOT/"input_report.json"),input_verification_sha256=sha(ROOT/"input_verification.json"),
        label_report_sha256=sha(LABELS/"label_report.json"),baselines=baselines,groups=groups,inverse=inverse,
        outputs_sha256={name:sha(ROOT/(name+".parquet")) for name in ["joined_original","evaluation_rows","groups_daily"]},
        prices_2026_read=False,new_strategy_selected=False)
    save_json(ROOT/"analysis_report.json",report)
    return dict(baselines=baselines,group_summaries=len(groups),inverse_summaries=len(inverse))


if __name__=="__main__":print(json.dumps(evaluate(),ensure_ascii=False,indent=2))
