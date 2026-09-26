"""Independent SQL cash flows, separate execution windows and distribution checks."""
from pathlib import Path
import json

import duckdb
import numpy as np
import pandas as pd

from trade_research.corporate_cash import save_json, sha


root=Path("data/research/first_limitup_overnight")
comparison=json.loads((root/"comparison_report.json").read_text())
frames={};checks={};c=duckdb.connect()
for window in ("morning","close"):
    p=root/"continued"/window
    report=json.loads((p/"tick_report.json").read_text())
    c.read_parquet(str(p/"tick_cost_scenario.parquet")).create_view("trades",replace=True)
    c.read_parquet(str(p/"raw_windows.parquet")).create_view("bars",replace=True)
    rows=c.sql("""WITH prices AS(SELECT *,bps,entry_price/1.0005 AS rb,exit_price/.9995 AS rs
      FROM trades CROSS JOIN(VALUES(5),(15))v(bps)),value AS(SELECT *,
      shares*(rb+greatest(rb*bps/10000.,.005)) AS bv,
      catalog_sold_shares*(rs-greatest(rs*bps/10000.,.005)) AS sv FROM prices)
      SELECT *,CASE WHEN entry_status!='filled' AND entry_window_status='valid' THEN 0.
      WHEN entry_status='filled' AND exit_price IS NOT NULL THEN
      (sv-greatest(5.,sv*.0003)-sv*.00051+catalog_dividend_gross-catalog_dividend_tax)
      /(bv+greatest(5.,bv*.0003)+bv*.00001)-1 ELSE NULL END AS computed,
      CASE WHEN bps=5 THEN tick_return5 ELSE tick_return15 END AS stored FROM value""").df()
    assert (rows.computed.isna()==rows.stored.isna()).all()
    assert (rows.computed-rows.stored).abs().max()<1e-12
    labels=["0935","0936","0937","0938"] if window=="morning" else ["1452","1453","1454","1455"]
    e=c.execute("""SELECT code,date,count(*) AS n,count(distinct timestamp) AS nu,
      sum(volume) AS volume,sum(turnover)/nullif(sum(volume),0) AS vwap
      FROM bars WHERE label IN (SELECT unnest(?)) GROUP BY code,date""",[labels]).df()
    b=c.sql("""SELECT code,date,count(*) AS n,count(distinct timestamp) AS nu,
      sum(volume) AS volume,sum(turnover)/nullif(sum(volume),0) AS vwap
      FROM bars WHERE label IN ('1452','1453','1454','1455') GROUP BY code,date""").df()
    c.register("buy_windows",b);c.register("sell_windows",e)
    fills=c.sql("""SELECT count(*) AS n,max(abs(t.entry_price-b.vwap*1.0005)) AS buy_error,
      max(abs(t.exit_price-e.vwap*.9995)) AS sell_error,
      count(*) FILTER(WHERE b.n<>4 OR e.n<>4 OR b.nu<>4 OR e.nu<>4
      OR t.shares>b.volume*.1 OR t.catalog_sold_shares>e.volume*.1) AS invalid
      FROM trades t JOIN buy_windows b ON t.date=b.date AND t.code=b.code
      JOIN sell_windows e ON t.exit_date=e.date AND t.code=e.code
      WHERE t.entry_status='filled'""").df().iloc[0]
    assert fills.invalid==0 and max(fills.buy_error,fills.sell_error)<1e-12
    q=pd.read_parquet(p/"execution_queue_audit.parquet")
    daily=[]
    for code,g in q.groupby("code"):
        dates=sorted(set(g.date)|set(g.exit_date.dropna()))
        d=pd.read_parquet(Path("data/baostock/market_2020_2026/daily")/(code.replace(".","_")+".parquet"),
            filters=[("date","in",dates)],columns=["date","code","preclose","isST"])
        daily.append(d)
    c.register("daily",pd.concat(daily,ignore_index=True));c.register("queues",q)
    # All study stocks are ordinary 10% main-board stocks at entry; sale ST is historical.
    reproduced=c.execute("""WITH limits AS(SELECT code,date,
      round(preclose::DECIMAL(18,6)*CASE WHEN isST=1 THEN 1.05 ELSE 1.10 END,2) AS upper,
      round(preclose::DECIMAL(18,6)*CASE WHEN isST=1 THEN .95 ELSE .90 END,2) AS lower FROM daily),
      buys AS(SELECT b.code,b.date,bool_or(round(b.high,2)>=d.upper) AS touch
      FROM bars b JOIN limits d USING(code,date) WHERE volume>0
      AND label IN ('1452','1453','1454','1455') GROUP BY b.code,b.date),
      sells AS(SELECT b.code,b.date,bool_or(round(b.low,2)<=d.lower) AS touch
      FROM bars b JOIN limits d USING(code,date) WHERE volume>0
      AND label IN (SELECT unnest(?)) GROUP BY b.code,b.date)
      SELECT q.date,q.code,q.horizon,b.touch AS buy_touch,e.touch AS sell_touch,
      q.entry_status='filled' AND (coalesce(b.touch,true) OR coalesce(e.touch,true)) AS unknown
      FROM queues q LEFT JOIN buys b ON q.date=b.date AND q.code=b.code
      LEFT JOIN sells e ON q.exit_date=e.date AND q.code=e.code""",[labels]).df()
    checked=q.merge(reproduced,on=["date","code","horizon"],validate="one_to_one")
    for actual,expected in (("entry_limit_touched","buy_touch"),("exit_limit_touched","sell_touch"),
                            ("queue_allocation_unverified","unknown")):
        assert checked[actual].fillna(True).eq(checked[expected].fillna(True)).all()
    rows=rows.merge(q[["date","code","horizon","queue_allocation_unverified"]],
                    on=["date","code","horizon"],validate="many_to_one")
    frames[window]=rows
    checks[window]={"cost_rows":len(rows),"raw_fills":int(fills.n),"queue_records":len(q),
        "economic_max_error":float((rows.computed-rows.stored).abs().max()),
        "raw_fill_max_error":float(max(fills.buy_error,fills.sell_error)),
        "tick_report_sha256":sha(p/"tick_report.json")}


def period(frame,label):
    return frame if label=="full" else frame.loc[frame.date.str[:4].eq(label)] if len(label)==4 else frame.loc[frame.half.eq(label)]


def verify_number(actual,expected):
    if expected is None:
        assert pd.isna(actual)
    else:
        assert abs(actual-expected)<1e-12,(actual,expected)


def bootstrap(values):
    weeks=pd.to_datetime(values.index).to_period("W-SUN")
    blocks=[g.to_numpy() for _,g in values.groupby(weeks)]
    choices=np.random.default_rng(20260926).integers(len(blocks),size=(10000,len(blocks)))
    weights=np.stack([np.bincount(x,minlength=len(blocks)) for x in choices])
    sums=np.array([x.sum() for x in blocks]);counts=np.array([len(x) for x in blocks])
    return np.quantile((weights@sums)/(weights@counts),[.025,.975])


for item in comparison["metrics"]:
    if item["window"]=="morning_minus_close":
        a=frames["morning"]
        a=a.loc[a.arm.eq("high")&a.bps.eq(item["bps"])]
        b=frames["close"].loc[lambda x:x.arm.eq("high")&x.bps.eq(item["bps"])]
        part=a.merge(b,on=["date","code","horizon","half"],validate="one_to_one",suffixes=("_a","_b"))
        part["value"]=part.computed_a-part.computed_b
    else:
        rows=frames[item["window"]].loc[lambda x:x.bps.eq(item["bps"])]
        if item["metric"]=="same_day_edge":
            part=rows.loc[rows.arm.eq("high")].merge(rows.loc[rows.arm.eq("low")],
                on=["date","pair_id","horizon","half"],validate="one_to_one",suffixes=("_a","_b"))
            part["value"]=part.computed_a-part.computed_b
        else:
            part=rows.loc[rows.arm.eq("high")].copy();part["value"]=part.computed
            if item["metric"]=="own_with_queue_unknown_retained":
                part.loc[part.queue_allocation_unverified,"value"]=np.nan
    part=period(part,item["period"])
    daily=part.groupby("date").value.agg(lambda x:np.nan if x.isna().any() else sum(x)/len(x))
    assert len(daily)==item["signal_dates"] and int(daily.isna().sum())==item["unknown_dates"]
    verify_number(np.nan if daily.isna().any() or daily.empty else daily.mean(),item["daily_mean"])
    if daily.empty or daily.isna().any():
        assert item["weekly_interval"] is None
    else:
        assert np.max(abs(bootstrap(daily)-item["weekly_interval"]))<1e-12
for item in comparison["trade_distributions"]:
    rows=frames[item["window"]].loc[lambda x:x.arm.eq("high")&x.bps.eq(item["bps"])]
    rows=period(rows,item["period"]);bought=rows.loc[rows.entry_status.eq("filled")]
    assert len(rows)==item["orders"] and len(bought)==item["bought"]
    values=np.sort(bought.computed.to_numpy())
    assert np.isfinite(values).all()
    positive,negative=values[values>0],values[values<0]
    expected={"win_rate":len(positive)/len(values),"mean_win":positive.mean(),"mean_loss":negative.mean(),
        "payoff_ratio":positive.mean()/-negative.mean(),"trade_mean":values.mean(),
        "worst_trade_return":values[0],"worst_five_percent_mean":values[:(len(values)+19)//20].mean()}
    for key,value in expected.items():verify_number(value,item[key])
save_json(root/"independent_economic_checks.json",{"windows":checks,"daily_metrics_and_intervals":len(comparison["metrics"]),
    "trade_distributions":len(comparison["trade_distributions"]),"comparison_report_sha256":sha(root/"comparison_report.json")})
print(json.dumps({"windows":checks,"daily_metrics":len(comparison["metrics"]),"trade_distributions":len(comparison["trade_distributions"])},indent=2))
