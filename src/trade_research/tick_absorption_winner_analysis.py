"""All nine response/pressure cells plus two fixed within-flat comparisons."""
import json

import pandas as pd

from .corporate_cash import save_json,sha
from .economic_winner_analysis import number
from .reference_gain_accounting import weekly_interval
from .tick_absorption_winner import ROOT
from .tick_flow_winner_analysis import evaluate,in_period,PERIODS

BANDS=[f"{a}:{b}" for a in ["sell","balanced","buy"] for b in ["down","flat","up"]]+["unknown"]


def run():
    evaluate(ROOT=ROOT,FEATURES=["pressure_response"],BANDS=BANDS,include_inverse=False,group_columns={"pressure_response":"group"})
    report=json.loads((ROOT/"analysis_report.json").read_text())
    daily=pd.read_parquet(ROOT/"groups_daily.parquet")
    primary=daily.loc[daily.feature.eq("pressure_response")&daily.band.eq("sell:flat")]
    tables=[];summaries=[]
    for control in ["balanced:flat","buy:flat"]:
        other=daily.loc[daily.feature.eq("pressure_response")&daily.band.eq(control)]
        pairs=primary.merge(other,on=["quality","cost_bps","date","half"],validate="one_to_one",suffixes=("_primary","_control"))
        pairs["comparison"]="sell:flat-minus-"+control
        pairs["net_difference"]=pairs.net_mean_primary-pairs.net_mean_control
        for name in ["winner","loser","positive"]:
            pairs[name+"_lower_difference"]=pairs[name+"_lower_primary"]-pairs[name+"_upper_control"]
            pairs[name+"_upper_difference"]=pairs[name+"_upper_primary"]-pairs[name+"_lower_control"]
        tables.append(pairs)
        for quality in ["original_labels","tick_quality_sensitivity"]:
            for cost in [5,15]:
                for period in PERIODS:
                    p=in_period(pairs.loc[pairs.quality.eq(quality)&pairs.cost_bps.eq(cost)],period)
                    result=dict(quality=quality,cost_bps=cost,period=period,comparison="sell:flat-minus-"+control,
                        dates=len(p),known_dates=int(p.net_difference.notna().sum()),
                        primary_rows=int(p.n_primary.sum()),control_rows=int(p.n_control.sum()))
                    for name in ["net_difference",*[n+"_"+b+"_difference" for n in ["winner","loser","positive"] for b in ["lower","upper"]]]:
                        values=p.set_index("date")[name].sort_index()
                        result[name]=number(values.mean());result[name+"_week_interval"]=weekly_interval(values)
                    summaries.append(result)
    pd.concat(tables,ignore_index=True).to_parquet(ROOT/"comparisons_daily.parquet",index=False,compression="zstd")
    report["comparisons"]=summaries
    report["outputs_sha256"]["comparisons_daily"]=sha(ROOT/"comparisons_daily.parquet")
    report["primary"]="sell:flat"
    save_json(ROOT/"analysis_report.json",report)
    return dict(groups=len(report["groups"]),comparisons=len(summaries),primary=report["primary"])


if __name__=="__main__":print(json.dumps(run(),ensure_ascii=False,indent=2))
