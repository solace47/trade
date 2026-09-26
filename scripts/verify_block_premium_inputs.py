"""Rebuild official disclosures in exact units, all-day weighting and matching."""
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from trade_research.corporate_cash import save_json, sha

root = Path("data/research/block_premium_short")
report = json.loads((root/"input_report.json").read_text())
manifest = json.loads((root/"source_manifest.json").read_text())
assert sha(root/"source_manifest.json")==report["source_manifest_sha256"]
assert sha(root/"signals.parquet")==report["signals_sha256"]
for name,digest in report["output_sha256"].items():
    assert sha(root/name)==digest
for path,digest in manifest["source_sha256"].items():
    assert sha(Path(path))==digest
raw,events,pool,signals=(pd.read_parquet(root/(name+".parquet")) for name in
    ("raw_disclosures","source_events","visible_pool","signals"))
table=pd.read_parquet("data/baostock/market_2020_2026/metadata/calendar.parquet")
calendar=sorted(table.loc[table.is_trading_day.eq("1") & table.calendar_date.between("2024-01-01","2025-12-31"),"calendar_date"])
positions={d:i for i,d in enumerate(calendar)}
rows=[]
for current in calendar[1:-10]:
    previous=calendar[positions[current]-1]
    data=json.loads((root/"sse_archive"/(previous+".json")).read_text())
    assert data["trade_date"]==previous and data["schema_version"]==1 and data["exchange"]=="sse"
    if "original_joint_archive_sha256" in data:
        original=Path("data/research/block_trade/daily")/(previous+".json")
        assert sha(original)==data["original_joint_archive_sha256"]
        assert json.loads(original.read_text())["sse"]==data["rows"]
    else:
        assert data["transport"]=="https" and data["official_pagination_checked"] is True
    for market in ("sse",):
        exchange="sh"
        fields=("stockid","tradeprice","tradeqty","tradeamount","tradedate")
        for index,item in enumerate(data["rows"]):
            code=exchange+"."+item[fields[0]]
            assert item[fields[4]]==previous
            if not code.startswith("sh.60"):
                continue
            values=[Decimal(item[name].replace(",",""))*factor for name,factor in zip(fields[1:4],(100,10000,1000000))]
            assert all(v.is_finite() and v>0 and v==v.to_integral_value() for v in values)
            price,quantity,amount=map(int,values)
            rows.append({"date":current,"trade_date":previous,"code":code,"exchange":exchange,
                "source_row":index,"price_cents":price,"reported_shares":quantity,
                "reported_amount_cents":amount,"quoted_price_times_shares":price*quantity})
pd.testing.assert_frame_equal(pd.DataFrame(rows),raw,check_exact=True)
c=duckdb.connect();c.execute("SET threads=4");c.register("disclosures",raw)
aggregated=c.sql("""SELECT date,trade_date,code,exchange,sum(reported_shares)::BIGINT AS reported_shares,
 sum(reported_amount_cents)::BIGINT AS reported_amount_cents,
 sum(price_cents::HUGEINT*reported_shares)::BIGINT AS quoted_price_times_shares,count(*) AS source_rows
 FROM disclosures GROUP BY ALL ORDER BY date,trade_date,code,exchange""").df()
pd.testing.assert_frame_equal(aggregated,events[aggregated.columns],check_dtype=False,check_exact=True)
c.register("schedule",pd.DataFrame({"date":calendar[1:-10],"trade_date":calendar[:-11]}))
paths=list(Path("data/baostock/market_2020_2026/daily").glob("sh_60*.parquet"))
c.read_parquet([str(p) for p in paths]).create_view("daily")
c.read_parquet("data/research/minute_prefix_1449/202[45]/*.parquet").create_view("prefix")
c.read_parquet("data/research/market_snapshots_ci/*.parquet").create_view("snapshots")
c.read_parquet("data/research/risk_removal_1449/notice_index.parquet").create_view("notices")
past=c.sql("""SELECT code,d.date AS trade_date,high AS prior_high,close AS prior_close,preclose AS prior_reference
 FROM daily d JOIN schedule s ON d.date=s.trade_date WHERE tradestatus=1 AND isST=0 AND adjustflag=3
 AND low>0 AND high>=close AND close>=low""").df()
facts=aggregated.merge(past,on=["code","trade_date"],how="left",validate="one_to_one")
flag=[]
for row in facts.itertuples():
    valid=pd.notna(row.prior_high)
    high=int((Decimal(str(row.prior_high))*100).to_integral_value(rounding=ROUND_HALF_UP)) if valid else 0
    flag.append(valid and row.reported_amount_cents>=1_000_000_000
        and Fraction(row.quoted_price_times_shares,row.reported_shares)>=Fraction(high*101,100))
assert np.array_equal(flag,events.premium_event)
pd.testing.assert_frame_equal(facts[past.columns].reset_index(drop=True),events[past.columns].reset_index(drop=True),check_exact=True)
assert np.allclose(events.block_quote_vwap,events.quoted_price_times_shares/events.reported_shares/100,atol=1e-12,rtol=0)
base=c.sql("""SELECT p.date,p.code,substr(p.code,1,2) AS exchange,d.trade_date,
 p.price_1449,p.high_1449,p.low_1449,p.amount_1449,p.volume_1449,
 s.preclose,s.isST,s.reference_gap,s.listing_age_sessions,s.return20_prior_adjusted
 FROM prefix p JOIN schedule d USING(date) JOIN snapshots s USING(date,code)
 WHERE substr(p.code,1,5)='sh.60' AND s.isST=0 AND s.tradestatus=1
 AND s.listing_age_sessions>=20 AND NOT s.reference_gap AND NOT p.quote_outside_traded_range
 AND p.price_1449>=5 AND p.amount_1449>=30000000 AND p.volume_1449>0 AND s.preclose>0
 AND NOT EXISTS(SELECT 1 FROM notices n WHERE n.code=p.code AND n.notice_date<p.date
 AND regexp_matches(n.title,'进入退市整理|退市整理期交易'))""").df().merge(past,on=["code","trade_date"],validate="many_to_one")
cents=np.floor(base.price_1449*100+.5).astype("int64")
upper=np.array([int((Decimal(str(v))*110).to_integral_value(rounding=ROUND_HALF_UP)) for v in base.preclose])
eligible=((base.price_1449-cents/100).abs().le(.0001)&(2_000_000//cents//100*100).ge(100)
    &(cents*10000+np.maximum(cents*15,5000)<upper*10000-5000))
base=base.loc[eligible].copy();base["price_1449"]=cents[eligible]/100
base=base.sort_values(["date","code"]).reset_index(drop=True)
pd.testing.assert_frame_equal(base,pool[base.columns],check_dtype=False,check_exact=True)
assert np.allclose(pool.return_1450,pool.price_1449/pool.preclose-1,atol=1e-12,rtol=0)
assert np.allclose(pool.prior_day_return,pool.prior_close/pool.prior_reference-1,atol=1e-12,rtol=0)
event_keys=set(zip(events.date,events.code))
assert np.array_equal(pool.has_block,[(d,code) in event_keys for d,code in zip(pool.date,pool.code)])
candidate=[]
for row in pool.itertuples():
    candidate.append(row.has_block and bool(row.premium_event)
        and int(round(row.price_1449*100))*int(row.reported_shares)<=int(row.quoted_price_times_shares))
assert np.array_equal(candidate,pool.candidate)
last,counts,picked={},{},[]
for date,part in pool.loc[pool.candidate].groupby("date",sort=True):
    order=sorted(part.itertuples(),key=lambda r:(-Fraction(int(r.quoted_price_times_shares),
        int(r.reported_shares)*int(round(r.prior_high*100))),-int(r.reported_amount_cents),r.code))
    for row in order:
        if counts.get(date,0)==5 or positions[date]-last.get(row.code,-1000)<=5:
            continue
        counts[date]=counts.get(date,0)+1;last[row.code]=positions[date]
        picked.append((date,row.code,counts[date]))
high,low=signals.loc[signals.arm.eq("high")],signals.loc[signals.arm.eq("low")]
assert set(picked)==set(zip(high.date,high.code,high.daily_rank))
c.register("pool",pool);c.register("high",high)
edges=c.sql("""WITH x AS(SELECT h.date,h.code AS event,h.daily_rank,p.code AS peer,
 abs(h.return20_prior_adjusted-p.return20_prior_adjusted) AS prior_gap,
 abs(h.return_1450-p.return_1450) AS day_gap,abs(h.prior_day_return-p.prior_day_return) AS yesterday_gap,
 p.amount_1449/h.amount_1449 AS ar,p.price_1449/h.price_1449 AS pr
 FROM high h JOIN pool p ON h.date=p.date AND h.exchange=p.exchange WHERE NOT p.has_block)
 SELECT *,prior_gap/.05+day_gap/.02+yesterday_gap/.02+abs(ln(ar))/ln(2)+abs(ln(pr))/ln(2) AS distance
 FROM x WHERE prior_gap<=.05 AND day_gap<=.02 AND yesterday_gap<=.02 AND ar BETWEEN .5 AND 2 AND pr BETWEEN .5 AND 2
 ORDER BY date,daily_rank,distance,peer""").df()
matched,assigned,used=[],set(),set()
for row in edges.itertuples():
    if (row.date,row.event) in assigned or (row.date,row.peer) in used:
        continue
    assigned.add((row.date,row.event));used.add((row.date,row.peer));matched.append((row.date,row.peer,row.date+":"+row.event))
assert set(matched)==set(zip(low.date,low.code,low.pair_id))
assert np.array_equal(signals.decision_shares,2_000_000//np.floor(signals.price_1449*100+.5).astype("int64")//100*100)
coverage=pd.read_parquet("data/research/cash_dividend_catalog/query_coverage.parquet")
needed=pd.DataFrame(sorted({(r.code,str(year)) for r in signals.itertuples()
    for year in range(int(r.date[:4]),int(calendar[positions[r.date]+10][:4])+1)}),columns=["code","year"])
assert needed.merge(coverage,on=["code","year"],how="left",indicator=True)._merge.eq("both").all()
result={"official_days":len(calendar)-11,"raw_rows":len(raw),"whole_day_stock_events":len(events),
    "premium_events":int(events.premium_event.sum()),"candidates":len(high),"controls":len(low),
    "exhaustive_matching_edges":len(edges),"catalogue_code_years":len(needed),
    "exact_rational_event_and_ranking_checks":True,"signals_sha256":sha(root/"signals.parquet"),
    "new_holding_results_read":False,"new_2026_prices_read":False}
save_json(root/"independent_input_checks.json",result)
print(json.dumps(result,ensure_ascii=False,indent=2))
