"""Reuse independent cell checks, then independently audit within-flat pairs."""
import json
from pathlib import Path

import duckdb
import pandas as pd

import verify_tick_flow_analysis as audit
from trade_research.corporate_cash import save_json,sha

ROOT=Path("data/research/tick_absorption_winner")


def main():
    result=audit.main(ROOT=ROOT,FEATURES=["pressure_response"],group_columns={"pressure_response":"group"},write_report=False)
    report=json.loads((ROOT/"analysis_report.json").read_text())
    c=duckdb.connect();c.read_parquet(str(ROOT/"groups_daily.parquet")).create_view("daily")
    columns=["net_mean","winner_lower","winner_upper","loser_lower","loser_upper","positive_lower","positive_upper"]
    copied=','.join(f'p.{n} as {n}_primary,c.{n} as {n}_control' for n in columns)
    pairs=c.sql(f'''select p.quality,p.cost_bps,p.date,p.half,
        'sell:flat-minus-'||c.band as comparison,p.n as n_primary,c.n as n_control,{copied}
        from daily p join daily c on p.quality=c.quality and p.cost_bps=c.cost_bps and p.date=c.date and p.half=c.half
        where p.feature='pressure_response' and c.feature='pressure_response' and p.band='sell:flat'
        and c.band in ('balanced:flat','buy:flat')''').df()
    pairs["net_difference"]=pairs.net_mean_primary-pairs.net_mean_control
    for name in ["winner","loser","positive"]:
        pairs[name+"_lower_difference"]=pairs[name+"_lower_primary"]-pairs[name+"_upper_control"]
        pairs[name+"_upper_difference"]=pairs[name+"_upper_primary"]-pairs[name+"_lower_control"]
    order=["quality","cost_bps","comparison","date"]
    pairs=pairs.sort_values(order).reset_index(drop=True)
    actual=pd.read_parquet(ROOT/"comparisons_daily.parquet").sort_values(order).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual[pairs.columns],pairs,check_dtype=False,atol=2e-12,rtol=0)
    for item in report["comparisons"]:
        p=pairs.loc[pairs.quality.eq(item["quality"])&pairs.cost_bps.eq(item["cost_bps"])&pairs.comparison.eq(item["comparison"])]
        p=audit.period(p,item["period"])
        for name,value in dict(dates=len(p),known_dates=int(p.net_difference.notna().sum()),primary_rows=int(p.n_primary.sum()),control_rows=int(p.n_control.sum())).items():
            audit.eq(item[name],value,("comparison",item["comparison"],item["period"],name))
        for name in ["net_difference",*[n+"_"+b+"_difference" for n in ["winner","loser","positive"] for b in ["lower","upper"]]]:
            audit.eq(item[name],audit.clean(p[name].mean()),("comparison",name))
            audit.eq(item[name+"_week_interval"],audit.interval(p,name),("comparison",name,"interval"))
    result.update(passed=True,analysis_report_sha256=sha(ROOT/"analysis_report.json"),
        comparison_days=len(pairs),comparison_summaries=len(report["comparisons"]),statistics_checked=audit.checked)
    save_json(ROOT/"analysis_verification.json",result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
