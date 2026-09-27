"""Independent SQL joins/daily groups and numeric reconstruction of all reports."""
from __future__ import annotations

from datetime import date
import json
import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from trade_research.corporate_cash import save_json,sha

ROOT=Path("data/research/tick_flow_winner")
FEATURES=["morning_net","afternoon_net","tail_net","tail_acceleration","tail_positive_minutes","tail_flat_net"]
METRICS=["winner_lower","winner_upper","loser_lower","loser_upper","positive_lower","positive_upper","net_mean"]
checked=0


def clean(x):
    return float(x) if pd.notna(x) and np.isfinite(x) else None


def eq(actual,expected,where):
    global checked
    checked+=1
    if isinstance(expected,list):
        assert actual is not None,where
        np.testing.assert_allclose(actual,expected,atol=2e-12,rtol=0,err_msg=str(where))
    elif expected is None:
        assert actual is None,(where,actual,expected)
    elif isinstance(expected,(float,np.floating)):
        assert actual is not None and abs(actual-expected)<2e-12,(where,actual,expected)
    else:
        assert actual==expected,(where,actual,expected)


def interval(frame,column):
    if len(frame)==0 or frame[column].isna().any():return None
    weeks={}
    for day,value in frame[["date",column]].itertuples(index=False,name=None):
        iso=date.fromisoformat(day).isocalendar()
        key=(iso.year,iso.week)
        total,count=weeks.get(key,(0.,0))
        weeks[key]=(total+value,count+1)
    if len(weeks)<2:return None
    blocks=np.array([weeks[k] for k in sorted(weeks)])
    choices=np.random.default_rng(20260926).choice(len(blocks),size=(10000,len(blocks)),replace=True)
    numerator=np.take(blocks[:,0],choices).sum(axis=1)
    denominator=np.take(blocks[:,1],choices).sum(axis=1)
    return np.percentile(numerator/denominator,[2.5,97.5]).tolist()


def period(p,value):
    return p.loc[p.half.eq(value) if "H" in value else p.date.str.startswith(value)]


def main():
    report=json.loads((ROOT/"analysis_report.json").read_text())
    for name,digest in report["outputs_sha256"].items():assert sha(ROOT/(name+".parquet"))==digest
    c=duckdb.connect()
    c.execute(f"create view features as select * from read_parquet('{ROOT}/features.parquet')")
    labels="data/research/economic_winner/period_quality/labels.parquet"
    c.execute(f"""create view joined as select f.*,l.label5,l.label15,l.net_return5,l.net_return15 from features f
        join read_parquet('{labels}') l on f.date=l.date and f.code=l.code""")
    original=pd.read_parquet(ROOT/"joined_original.parquet").sort_values(["date","code"]).reset_index(drop=True)
    expected=c.sql("select * from joined order by date,code").df()
    pd.testing.assert_frame_equal(original,expected,check_dtype=False)
    c.execute("""create view scenario0 as select j.*,
        q.quality,k.cost_bps,
        case when k.cost_bps=5 then label5 else label15 end as initial_label,
        case when k.cost_bps=5 then net_return5 else net_return15 end as initial_net
        from joined j cross join (values ('original_labels'),('tick_quality_sensitivity')) q(quality)
        cross join (values (5),(15)) k(cost_bps)""")
    c.execute("""create view scenario as select *,
        case when quality='tick_quality_sensitivity' and not source_valid and initial_label!='no_trade' then 'unknown' else initial_label end as label,
        case when quality='tick_quality_sensitivity' and not source_valid and initial_label!='no_trade' then NULL else initial_net end as net_return
        from scenario0""")
    scenario=c.sql("select * from scenario order by quality,cost_bps,date,code").df()
    saved=pd.read_parquet(ROOT/"evaluation_rows.parquet").sort_values(["quality","cost_bps","date","code"]).reset_index(drop=True)
    assert len(scenario)==len(saved)==1936*4
    for name in ["date","code","quality","cost_bps","label","net_return"]:
        pd.testing.assert_series_equal(saved[name],scenario[name],check_dtype=False)
    arms=["select quality,cost_bps,date,half,code,label,net_return,'baseline' as feature,'all' as band from scenario"]
    arms.extend(f"select quality,cost_bps,date,half,code,label,net_return,'{f}' as feature,{f}_group as band from scenario" for f in FEATURES)
    c.execute("create view arms as "+" union all ".join(arms))
    c.execute("""create view grouped as select quality,cost_bps,feature,band,date,half,
        count(*) as n,count(net_return) as known,count(*) filter(where label='unknown') as unknown,
        count(*) filter(where label='no_trade') as no_trade,
        count(*) filter(where label='economic_winner') as winner_count,
        count(*) filter(where label='economic_loser') as loser_count,
        count(*) filter(where net_return>0) as positive_count,avg(net_return) as net_mean
        from arms group by quality,cost_bps,feature,band,date,half""")
    c.execute("""create view daily0 as select *,winner_count::double/n as winner_lower,(winner_count+unknown)::double/n as winner_upper,
        loser_count::double/n as loser_lower,(loser_count+unknown)::double/n as loser_upper,
        positive_count::double/n as positive_lower,(positive_count+unknown)::double/n as positive_upper from grouped""")
    daily=c.sql("select * from daily0 order by quality,cost_bps,feature,band,date").df()
    base=daily.loc[daily.feature.eq("baseline"),["quality","cost_bps","date",*METRICS]]
    daily=daily.merge(base,on=["quality","cost_bps","date"],validate="many_to_one",suffixes=("","_baseline"))
    for name in ["winner","loser","positive"]:
        daily[name+"_lower_delta"]=daily[name+"_lower"]-daily[name+"_upper_baseline"]
        daily[name+"_upper_delta"]=daily[name+"_upper"]-daily[name+"_lower_baseline"]
    daily["net_mean_delta"]=daily.net_mean-daily.net_mean_baseline
    extras=[n for n in daily if n.endswith("_baseline") or n.endswith("_delta")]
    daily.loc[daily.feature.eq("baseline"),extras]=np.nan
    order=["quality","cost_bps","feature","band","date"]
    actual=pd.read_parquet(ROOT/"groups_daily.parquet").sort_values(order).reset_index(drop=True)
    daily=daily.sort_values(order).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual[daily.columns],daily,check_dtype=False,atol=2e-12,rtol=0)
    for category in ["baselines","groups"]:
        for item in report[category]:
            identity={k:item[k] for k in ["quality","cost_bps","period"]}
            identity["feature"]=item.get("feature","baseline");identity["band"]=item.get("band","all")
            p=daily.loc[daily.quality.eq(identity["quality"])&daily.cost_bps.eq(identity["cost_bps"])
                &daily.feature.eq(identity["feature"])&daily.band.eq(identity["band"])]
            p=period(p,identity["period"])
            raw=scenario.loc[scenario.quality.eq(identity["quality"])&scenario.cost_bps.eq(identity["cost_bps"])]
            if identity["feature"]!="baseline":raw=raw.loc[raw[identity["feature"]+"_group"].eq(identity["band"])]
            raw=period(raw,identity["period"])
            counters={"stock_days":int(p.n.sum()),"dates":len(p),"known":int(p.known.sum()),"unknown":int(p.unknown.sum()),
                "no_trade":int(p.no_trade.sum()),"winner_cases":int(p.winner_count.sum()),"loser_cases":int(p.loser_count.sum()),
                "positive_cases":int(p.positive_count.sum()),"valid_net_dates":int(p.net_mean.notna().sum())}
            for name,value in counters.items():eq(item[name],value,(identity,name))
            metrics=METRICS+[x+"_delta" for x in METRICS if x+"_delta" in item]
            for name in metrics:
                eq(item[name],clean(p[name].mean()),(identity,name))
                eq(item[name+"_week_interval"],interval(p,name),(identity,name,"interval"))
            values=raw.net_return.dropna().to_numpy();wins=values[values>0];losses=values[values<0]
            dist=dict(conditional_win_rate=clean(np.mean(values>0)) if len(values) else None,
                median=clean(np.median(values)) if len(values) else None,
                mean_win=clean(np.mean(wins)) if len(wins) else None,mean_loss=clean(np.mean(losses)) if len(losses) else None,
                payoff_ratio=clean(np.mean(wins)/-np.mean(losses)) if len(wins) and len(losses) else None,
                worst_five_percent_mean=clean(np.mean(np.sort(values)[:max(1,math.ceil(len(values)*.05))])) if len(values) else None)
            for name,value in dist.items():eq(item[name],value,(identity,name))
        print(json.dumps(dict(verified=category,statistics=checked)),flush=True)
    for item in report["inverse"]:
        raw=scenario.loc[scenario.quality.eq(item["quality"])&scenario.cost_bps.eq(15)&scenario.label.eq(item["label"])]
        raw=period(raw,item["period"])
        values=raw[item["feature"]]
        perday=raw.groupby("date")[item["feature"]].mean().rename("value").reset_index()
        for name,value in dict(rows=len(raw),feature_missing=int(values.isna().sum()),mean=clean(values.mean()),
            median=clean(values.median()),date_mean=clean(perday.value.mean()),date_mean_week_interval=interval(perday,"value")).items():
            eq(item[name],value,("inverse",item["feature"],item["period"],item["label"],name))
    result=dict(passed=True,analysis_report_sha256=sha(ROOT/"analysis_report.json"),
        label_joins=len(original),scenario_rows=len(scenario),daily_rows=len(daily),
        group_summaries=len(report["groups"]),inverse_summaries=len(report["inverse"]),statistics_checked=checked,
        prices_2026_read=False)
    save_json(ROOT/"analysis_verification.json",result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
