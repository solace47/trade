"""Independent SQL paths and cash-flow checks for the fixed classifier follow-ups."""
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING, ROUND_HALF_UP
import json
from math import ceil
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from trade_research.corporate_cash import save_json, sha

root=Path("data/research/winner_direction");c=duckdb.connect();c.execute("SET threads=4")
diagnostic=json.loads((root/"diagnostic_report.json").read_text())
c.read_parquet("data/research/next_day_winner/labels.parquet").create_view("labels")
c.read_parquet(str(root/"all_scores.parquet")).create_view("scores")
c.read_parquet(str(root/"visible_pool.parquet")).create_view("pool")
frames=[c.sql("SELECT 'eligible_pool' AS model,* FROM pool WHERE date>='2025-01-02'").df()]
for model in ("balanced","upside_only"):
    selected=pd.read_parquet(root/model/"signals.parquet");selected=selected.loc[selected.arm.eq("high"),list(frames[0].columns)[1:]]
    selected.insert(0,"model",model);frames.append(selected)
c.register("members",pd.concat(frames,ignore_index=True))
c.execute("""CREATE TABLE cases AS SELECT m.*,l.known_label,
    l.known_label AND round(l.next_close*100)::BIGINT*100>=round(l.next_preclose*100)::BIGINT*105 AS up,
    l.known_label AND round(l.next_close*100)::BIGINT*100<=round(l.next_preclose*100)::BIGINT*95 AS down,
    p.p_up,p.p_down FROM members m JOIN labels l USING(date,code) JOIN scores p USING(date,code)""")


def bounds(period):
    return ("2025-01-01","2025-06-30") if period=="2025H1" else (("2025-07-01","2025-12-31") if period=="2025H2" else ("2025-01-01","2025-12-31"))


def number(actual,expected):
    assert pd.isna(actual) if expected is None else abs(actual-expected)<1e-12,(actual,expected)


for row in diagnostic["case_rates"]:
    a,b=bounds(row["period"]);board=row["board"]
    part=c.execute("SELECT * FROM cases WHERE model=? AND date BETWEEN ? AND ? AND (?='all' OR board=?)",
        [row["model"],a,b,board,board]).df();c.register("part",part)
    checked=c.sql("""SELECT count(*) AS "rows",count(distinct date) dates,sum(up::INT) up_cases,sum(down::INT) down_cases,
        sum((NOT known_label)::INT) unknown_cases,median(return_1449) visible_gain_median,
        median(return5_prior_adjusted) prior5_gain_median FROM part""").df().iloc[0]
    for col,value in checked.items():number(value,row[col])
    daily=c.sql("""SELECT date,avg(up::INT) u,avg(down::INT) d,avg((NOT known_label)::INT) missing,
        avg(p_up) pu,avg(p_down) pd FROM part GROUP BY date""").df()
    for col,value in {"up_rate_daily_lower":daily.u.mean(),"up_rate_daily_upper":(daily.u+daily.missing).mean(),
        "down_rate_daily_lower":daily.d.mean(),"down_rate_daily_upper":(daily.d+daily.missing).mean(),
        "predicted_up_daily_mean":daily.pu.mean(),"predicted_down_daily_mean":daily.pd.mean()}.items():number(value,row[col])

paths=[]
for model in ("balanced","upside_only"):
    folder=root/model/"continued";c.read_parquet(str(folder/"tick_cost_scenario.parquet")).create_view("t")
    c.read_parquet(str(folder/"execution_queue_audit.parquet")).create_view("q")
    frame=c.execute("""WITH p AS(SELECT t.*,m.board,l.next_open,l.day_close,l.next_close,
        entry_price/1.0005 rb,exit_price/.9995 rs FROM t JOIN labels l USING(date,code)
        JOIN members m ON m.date=t.date AND m.code=t.code AND m.model=?
        JOIN q ON q.date=t.date AND q.code=t.code AND q.horizon=t.horizon
        WHERE t.arm='high' AND t.entry_status='filled' AND t.exit_date=l.next_date AND l.known_label
        AND t.execution_source_valid AND NOT t.catalog_action_applied AND NOT coalesce(l.next_reference_gap,true)
        AND NOT q.queue_allocation_unverified AND NOT coalesce(t.corporate_action_crossed,true))
        SELECT date,code,board,rb/price_1449-1 entry_drift_from_1449,(day_close-rb)/rb after_entry_contribution,
        (next_open-day_close)/rb overnight_contribution,(rs-next_open)/rb next_session_contribution,
        rs/rb-1 gross_tail_return,(next_close-rs)/rb after_planned_exit_contribution FROM p ORDER BY date,code""",[model]).df()
    frame.insert(0,"model",model);paths.append(frame)
checked=pd.concat(paths,ignore_index=True).sort_values(["model","date","code"]).reset_index(drop=True)
stored=pd.read_parquet(root/"path_diagnostics.parquet").sort_values(["model","date","code"]).reset_index(drop=True)
pd.testing.assert_frame_equal(checked,stored,check_dtype=False,atol=1e-12,rtol=0)
for row in diagnostic["path_metrics"]:
    a,b=bounds(row["period"])
    p=checked.loc[checked.model.eq(row["model"])&checked.date.between(a,b)]
    if row["board"]!="all":p=p.loc[p.board.eq(row["board"])]
    assert len(p)==row["included_rows"] and p.date.nunique()==row["dates"]
    for col in checked.columns[4:]:number(p.groupby("date")[col].mean().mean(),row[col])

folder=root/"balanced/merger_followup";merger=json.loads((folder/"report.json").read_text())
review=json.loads(Path("config/winner_direction_merger_review.json").read_text())
for source in review["sources"]:assert sha(folder/source["file"])==source["sha256"]
old=pd.read_parquet(root/"balanced/continued/tick_cost_scenario.parquet")
pending=old.loc[old.entry_status.eq("filled")&old.exit_price.isna()]
assert len(pending)==1 and sha(root/"balanced/continued/tick_cost_scenario.parquet")==merger["original_ledger_sha256"]
r=pending.iloc[0];exact=Decimal(int(r.shares))*Decimal(review["conversion_ratio"])
assert exact==Decimal("522.2100")
assert review["quantity_scenarios"]==[int(exact.to_integral_value(rounding=x)) for x in (ROUND_FLOOR,ROUND_CEILING)]
raw=pd.read_parquet("data/hf/pilot/data/stock_1m/SH/600150.parquet",filters=[("timestamp",">=",pd.Timestamp("2025-09-16 14:52")),("timestamp","<=",pd.Timestamp("2025-09-16 14:55"))])
assert len(raw)==raw.timestamp.nunique()==4 and raw.volume.min()>0
vwap=raw.turnover.sum()/raw.volume.sum()
day=pd.read_parquet("data/baostock/market_2020_2026/daily/sh_600150.parquet",filters=[("date","==","2025-09-16")]).iloc[0]
assert day.tradestatus==1 and day.isST==0
lower_limit=(Decimal(str(day.preclose))*Decimal(".9")).quantize(Decimal(".01"),rounding=ROUND_HALF_UP)
assert raw.low.round(2).min()>float(lower_limit)
for row in merger["scenarios"]:
    assert row["exit_date"]==review["listing_date"] and row["sold_shares"]<=raw.volume.sum()*.1
    bps=row["bps"];buy=r.entry_price/1.0005;sell=vwap
    buy+=max(buy*bps/10000,.005);sell-=max(sell*bps/10000,.005)
    bv=r.shares*buy;sv=row["sold_shares"]*sell
    value=(sv-max(5.,sv*.0003)-sv*.00051)/(bv+max(5.,bv*.0003)+bv*.00001)-1
    number(value,row["net_return"]);number(buy,row["buy_price"]);number(sell,row["sell_price"])
    assert row["exit_delay_sessions"]==24 and not row["exit_limit_touched"]
other=pd.read_parquet(root/"upside_only/continued/tick_cost_scenario.parquet")
for row in merger["metrics"]:
    col=f"tick_return{row['bps']}";part=old.loc[old.arm.eq("high")].copy()
    scenario=next(x for x in merger["scenarios"] if (x["bps"],x["sold_shares"])==(row["bps"],row["sold_shares"]))
    part.loc[part.date.eq(r.date)&part.code.eq(r.code),col]=scenario["net_return"]
    daily=part.groupby("date")[col].mean()
    if row["metric"]!="own":daily-=other.loc[other.arm.eq("high")].groupby("date")[col].mean()
    a,b=bounds(row["period"]);daily=daily.loc[(daily.index>=a)&(daily.index<=b)]
    assert daily.notna().all() and len(daily)==row["signal_dates"]
    number(daily.mean(),row["daily_mean"])
    blocks=[g.to_numpy() for _,g in daily.groupby(pd.to_datetime(daily.index).to_period("W-SUN"))]
    choices=np.random.default_rng(20260926).integers(len(blocks),size=(10000,len(blocks)))
    sums=np.array([x.sum() for x in blocks]);counts=np.array([len(x) for x in blocks])
    ci=np.quantile(sums[choices].sum(axis=1)/counts[choices].sum(axis=1),[.025,.975])
    assert np.max(abs(ci-row["weekly_interval"]))<1e-12
for row in merger["trade_distributions"]:
    part=old.loc[old.arm.eq("high")&old.entry_status.eq("filled")].copy();col=f"tick_return{row['bps']}"
    scenario=next(x for x in merger["scenarios"] if (x["bps"],x["sold_shares"])==(row["bps"],row["sold_shares"]))
    mask=part.date.eq(r.date)&part.code.eq(r.code)
    part.loc[mask,col]=scenario["net_return"];part.loc[mask,"exit_delay_sessions"]=24
    a,b=bounds(row["period"]);part=part.loc[part.date.between(a,b)];values=part[col]
    assert values.notna().all() and len(part)==row["bought"]
    win=values[values>0];loss=values[values<0]
    for field,value in {"win_rate":len(win)/len(values),"mean_win":win.mean(),"mean_loss":loss.mean(),
        "payoff_ratio":win.mean()/-loss.mean(),"worst_five_percent_mean":np.sort(values)[:ceil(len(values)*.05)].mean(),
        "worst_trade_return":values.min(),"trade_mean":values.mean(),"max_exit_delay_sessions":part.exit_delay_sessions.max()}.items():number(value,row[field])
result={"case_rate_cells":len(diagnostic["case_rates"]),"raw_path_rows":len(checked),"path_mean_cells":len(diagnostic["path_metrics"]),
    "merger_cash_scenarios":len(merger["scenarios"]),"merger_daily_metrics":len(merger["metrics"]),"merger_trade_distributions":len(merger["trade_distributions"]),
    "original_unknown_row_preserved":True,"diagnostic_report_sha256":sha(root/"diagnostic_report.json"),
    "merger_report_sha256":sha(folder/"report.json"),"new_2026_prices_read":False}
save_json(root/"independent_followup_checks.json",result);print(json.dumps(result,indent=2))
