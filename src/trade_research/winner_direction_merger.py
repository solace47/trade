"""Follow one unchanged rejected order through a documented security conversion."""
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING
import html
import json
from pathlib import Path
import re

import duckdb
import numpy as np
import pandas as pd
import pdfplumber

from .corporate_cash import curl, save_json, sha
from .first_limitup_overnight_eval import daily_statistics, trade_distribution
from .fixed_return_ranges import economic_return
from .hf_outcomes import Assumptions, _fill, _limit_price, _board_limit_rate
from .next_day_winner import calendar
from .reference_gain_eval import MINUTES, DAILY, quality_windows
from .risk_removal_eval import cost_price
from .winner_direction import ROOT


def evaluate() -> dict:
    config=Path("config/winner_direction_merger_review.json")
    review=json.loads(config.read_text());output=ROOT/"balanced/merger_followup";output.mkdir(exist_ok=True)
    for source in review["sources"]:
        path=output/source["file"]
        if not path.exists():path.write_bytes(curl(source["url"]))
        if sha(path)!=source["sha256"]:raise ValueError("A merger primary source changed")
    with pdfplumber.open(output/"last_trading_day.pdf") as doc:
        text=re.sub(r"\s+","","".join(p.extract_text() or "" for p in doc.pages))
    if "2025年8月12日为公司股票最后一个交易日" not in text:
        raise ValueError("The last trading date is not confirmed")
    text=re.sub(r"\s+","",html.unescape(re.sub(r"<[^>]+>","",(output/"conversion_listing.html").read_text())))
    for term in ("公告编号：2025-069","上市流通日期为2025年9月16日","按照1:0.1339的比例转换",
                 "换股实施股权登记日为2025年9月4日","按照其小数点后尾数大小排序"):
        if term not in text:raise ValueError("The listing or fractional-share terms are unconfirmed")
    folder=ROOT/"balanced/continued";path=folder/"tick_cost_scenario.parquet"
    tick=json.loads((folder/"tick_report.json").read_text())
    if sha(path)!=tick["tick_scenario_sha256"]:raise ValueError("The original ledger changed")
    rows=pd.read_parquet(path);pending=rows.loc[rows.entry_status.eq("filled")&rows.exit_price.isna()]
    if len(pending)!=1:raise ValueError("This review is for exactly one unresolved conversion")
    r=pending.iloc[0]
    if (r.date,r.code,int(r.shares),r.arm)!=(review["signal_date"],review["old_code"],review["original_shares"],"high"):
        raise ValueError("The unresolved holding differs from the reviewed event")
    exact=Decimal(int(r.shares))*Decimal(review["conversion_ratio"])
    quantities=sorted({int(exact.to_integral_value(rounding=x)) for x in (ROUND_FLOOR,ROUND_CEILING)})
    if quantities!=review["quantity_scenarios"]:raise ValueError("The possible share allocations changed")
    catpath=ROOT/"catalog/events_reconciled.parquet";cat=pd.read_parquet(catpath)
    coverage=pd.read_parquet(ROOT/"catalog/combined_coverage.parquet")
    for code,start in ((review["old_code"],r.date),(review["new_code"],review["record_date"])):
        if not (coverage.code.eq(code)&coverage.year.eq("2025")).any():raise ValueError("Missing merger security catalogue")
        if (cat.code.eq(code)&cat.dividOperateDate.gt(start)&cat.dividOperateDate.le(review["listing_date"])).any():
            raise ValueError("A distribution during the conversion needs separate accounting")
    code=review["new_code"];exchange,symbol=code.split(".")
    minute_path=MINUTES/exchange.upper()/(symbol+".parquet")
    daily_path=DAILY/(code.replace(".","_")+".parquet")
    c=duckdb.connect();c.read_parquet(str(minute_path)).create_view("m")
    bars=c.sql("""SELECT *,strftime(timestamp,'%Y-%m-%d') AS date,strftime(timestamp,'%H%M') AS label
        FROM m WHERE timestamp>='2025-09-16'::TIMESTAMP AND timestamp<'2026-01-01'::TIMESTAMP
        AND strftime(timestamp,'%H%M') IN ('1452','1453','1454','1455') ORDER BY timestamp""").df();c.close()
    bars["code"]=code
    daily=pd.read_parquet(daily_path,filters=[("date",">=",review["listing_date"]),("date","<=","2025-12-31")]).set_index("date",drop=False)
    days=calendar();results=[];attempts=[];used=[]
    for quantity in quantities:
        for date in (d for d in days if review["listing_date"]<=d<="2025-12-31"):
            raw=bars.loc[bars.date.eq(date)];status="missing_bars";price=None
            if len(raw)==4 and raw.timestamp.nunique()==4 and date in daily.index:
                day=daily.loc[date];volume=float(raw.volume.sum())
                quote=pd.Series({"volume":volume,"vwap":float(raw.turnover.sum()/volume) if volume else np.nan})
                price,status=_fill(quote,day,code,"sell",quantity,Assumptions(target_notional=r.target_notional))
            attempts.append({"date":date,"shares":quantity,"status":status})
            if status!="filled":continue
            quality=quality_windows(raw,[date]).iloc[0]
            lower=_limit_price(float(day.preclose),_board_limit_rate(code,int(day.isST),date),False)
            touch=bool((raw.loc[raw.volume.gt(0),"low"].round(2)<=lower).any())
            if r.entry_window_status!="valid" or quality.window_status!="valid":
                raise ValueError("A conversion source window requires additional price bounds")
            for bps in (5,15):
                buy=float(cost_price(r.entry_price/1.0005,bps,"buy"));sell=float(cost_price(price/.9995,bps,"sell"))
                result=float(economic_return(r.shares,quantity,buy,sell,0.,0.))
                results.append({"date":r.date,"code":r.code,"horizon":int(r.horizon),"new_code":code,
                    "exit_date":date,"sold_shares":quantity,"fractional_allocation_unknown":True,
                    "bps":bps,"net_return":result,"buy_price":buy,"sell_price":sell,"exit_raw_vwap":price/.9995,
                    "exit_window_volume":volume,"exit_limit_touched":touch,
                    "exit_delay_sessions":days.index(date)-days.index(r.target_exit_date)})
            used.append(raw);break
        else:raise ValueError("Converted shares still lack an executable 2025 window")
    scenarios=pd.DataFrame(results);scenarios.to_parquet(output/"share_scenarios.parquet",index=False)
    pd.concat(used).drop_duplicates().to_parquet(output/"raw_exit_windows.parquet",index=False)
    high=rows.loc[rows.arm.eq("high")].copy()
    other=pd.read_parquet(ROOT/"upside_only/continued/tick_cost_scenario.parquet");other=other.loc[other.arm.eq("high")]
    metrics=[];distributions=[]
    for scenario in scenarios.itertuples():
        values=high[["date","code",f"tick_return{scenario.bps}"]].rename(columns={f"tick_return{scenario.bps}":"value"})
        values.loc[values.date.eq(r.date)&values.code.eq(r.code),"value"]=scenario.net_return
        complete=high.copy();mask=complete.date.eq(r.date)&complete.code.eq(r.code)
        complete.loc[mask,f"tick_return{scenario.bps}"]=scenario.net_return
        complete.loc[mask,"exit_date"]=scenario.exit_date
        complete.loc[mask,"exit_delay_sessions"]=scenario.exit_delay_sessions
        for period,first,last in (("2025","2025-01-01","2025-12-31"),("2025H1","2025-01-01","2025-06-30"),("2025H2","2025-07-01","2025-12-31")):
            distributions.append({"sold_shares":scenario.sold_shares,"bps":scenario.bps,"period":period,
                **trade_distribution(complete.loc[complete.date.between(first,last)],f"tick_return{scenario.bps}")})
        own=values.groupby("date").value.agg(lambda x:x.mean() if x.notna().all() else np.nan)
        comparator=other.groupby("date")[f"tick_return{scenario.bps}"].agg(lambda x:x.mean() if x.notna().all() else np.nan)
        for name,series in (("own",own),("balanced_minus_upside_only",own-comparator)):
            for period,first,last in (("2025","2025-01-01","2025-12-31"),("2025H1","2025-01-01","2025-06-30"),("2025H2","2025-07-01","2025-12-31")):
                metrics.append({"sold_shares":scenario.sold_shares,"bps":scenario.bps,"metric":name,"period":period,
                    **daily_statistics(series.loc[(series.index>=first)&(series.index<=last)])})
    result={"interpretation":"unchanged_order_conversion_quantity_bounds_conditional_on_recorded_fills_not_actual_portfolio",
        "original_ledger_unchanged":True,"original_unknown_preserved":True,"new_2026_prices_read":False,
        "review_sha256":sha(config),"original_ledger_sha256":sha(path),"catalogue_sha256":sha(catpath),
        "minute_source_sha256":sha(minute_path),"daily_source_sha256":sha(daily_path),
        "exact_conversion_quantity":str(exact),"attempts":attempts,"scenarios":results,"metrics":metrics,"trade_distributions":distributions,
        "scenarios_sha256":sha(output/"share_scenarios.parquet"),"raw_exit_windows_sha256":sha(output/"raw_exit_windows.parquet")}
    save_json(output/"report.json",result);return result


if __name__=="__main__":print(json.dumps(evaluate(),ensure_ascii=False,indent=2))
