"""Independently rebuild input identities with calendar joins and integer prices."""
from pathlib import Path
import json
from decimal import Decimal, ROUND_HALF_UP

import duckdb
import numpy as np
import pandas as pd

from trade_research.corporate_cash import save_json, sha


root = Path("data/research/first_limitup_overnight")
report = json.loads((root / "input_report.json").read_text())
for name, key in (("signals.parquet", "signals_sha256"),
                  ("visible_pool.parquet", "pool_sha256"),
                  ("source_manifest.json", "source_manifest_sha256")):
    assert sha(root / name) == report[key]
pool = pd.read_parquet(root / "visible_pool.parquet")
signals = pd.read_parquet(root / "signals.parquet")
cal = pd.read_parquet("data/baostock/market_2020_2026/metadata/calendar.parquet")
days = sorted(cal.loc[cal.is_trading_day.eq("1") &
    cal.calendar_date.between("2023-01-01", "2025-12-31"), "calendar_date"])
position = {d: i for i, d in enumerate(days)}
schedule = pd.DataFrame([(d, days[position[d]-1], days[position[d]-2])
    for d in days[:-10] if d >= "2024-01-01"], columns=["date", "prior1", "prior2"])
c = duckdb.connect(); c.execute("SET threads=4"); c.register("schedule", schedule)
c.read_parquet("data/research/minute_prefix_1449/202[45]/*.parquet").create_view("prefix")
c.read_parquet("data/research/market_snapshots_ci/*.parquet").create_view("snapshots")
c.read_parquet("data/research/risk_removal_1449/notice_index.parquet").create_view("notices")
daily = Path("data/baostock/market_2020_2026/daily")
c.read_parquet([str(p) for p in sorted([*daily.glob("sh_60*.parquet"),
                                      *daily.glob("sz_00*.parquet")])]).create_view("daily")
# Materialize daily fields once; avoid an expensive four-way file-scan join.
past = c.sql("""SELECT code,date,close,preclose,isST FROM daily
 WHERE date BETWEEN '2023-01-01' AND '2025-12-31' AND tradestatus=1""").df()
print("Daily input fields loaded", len(past), flush=True)
rebuilt = c.sql("""SELECT p.date,p.code,p.price_1449,p.amount_1449,p.volume_1449,
 p.price_1420,p.return_last29,s.preclose,s.return20_prior_adjusted
 FROM schedule d JOIN prefix p USING(date) JOIN snapshots s USING(date,code)
 WHERE (p.code LIKE 'sh.60%' OR p.code LIKE 'sz.00%')
 AND s.isST=0 AND s.tradestatus=1 AND s.listing_age_sessions>=20
 AND NOT s.reference_gap
 AND NOT p.quote_outside_traded_range AND p.price_1449>=5 AND p.amount_1449>=30000000
 AND p.volume_1449>0 AND s.preclose>0 AND p.return_last29>=0
 AND p.price_1449>=p.amount_1449/p.volume_1449
 AND NOT EXISTS(SELECT 1 FROM notices n WHERE n.code=p.code AND n.notice_date<p.date
   AND regexp_matches(n.title,'进入退市整理|退市整理期交易'))
 ORDER BY p.date,p.code""").df()
# Join exact previous market dates, independently of the generator's stock lag.
rebuilt = rebuilt.merge(past[["code","date"]],on=["code","date"],validate="one_to_one")
rebuilt = rebuilt.merge(schedule,on="date",validate="many_to_one")
for lag in (1,2):
    fields = past.rename(columns={"date":f"prior{lag}","close":f"close{lag}",
        "preclose":f"reference{lag}","isST":f"st{lag}"})
    rebuilt = rebuilt.merge(fields,on=["code",f"prior{lag}"],validate="many_to_one")
rebuilt = rebuilt.loc[rebuilt.st1.eq(0)&rebuilt.st2.eq(0)&
    (rebuilt.reference1-rebuilt.close2).abs().le(.005)]
rebuilt = rebuilt.drop(columns=["prior1","prior2","st1","st2"])
rebuilt = rebuilt.sort_values(["date","code"])
print("Calendar joins completed", len(rebuilt), flush=True)
assert len(rebuilt) == report["before_buyable_rows"]
cents = np.floor(rebuilt.price_1449 * 100 + .5).astype("int64")
upper = np.array([int((Decimal(str(value))*110).to_integral_value(rounding=ROUND_HALF_UP))
                  for value in rebuilt.preclose], dtype="int64")
shares = 2_000_000 // cents // 100 * 100
allowed = ((rebuilt.price_1449-cents/100).abs().le(.0001) & shares.ge(100)
    & ((cents*10000 + np.maximum(cents*15, 5000)) < upper*10000-5000))
rebuilt = rebuilt.loc[allowed].copy(); rebuilt["price_1449"] = cents[allowed]/100
pool = pool.sort_values(["date", "code"]).reset_index(drop=True)
rebuilt = rebuilt.reset_index(drop=True)
pd.testing.assert_frame_equal(rebuilt, pool[rebuilt.columns], check_dtype=False,
                              atol=1e-12, rtol=0)
integers = {}
for field in ("close1", "reference1", "close2", "reference2"):
    integers[field] = np.floor(rebuilt[field]*100+.5).astype("int64")
    assert (rebuilt[field]-integers[field]/100).abs().le(.0001).all()
c1, r1, c2, r2 = [integers[x] for x in ("close1", "reference1", "close2", "reference2")]
up1, up2 = (11*r1+5)//10, (11*r2+5)//10
first = c1.eq(up1) & c2.ne(up2)
control = (100*c1).ge(106*r1) & (1000*c1).le(1095*r1) & c1.ne(up1) & c2.ne(up2)
assert np.array_equal(first, pool.first_board)
assert np.array_equal(control, pool.strong_unsealed)
c.register("pool", pool)
ordered = c.sql("SELECT date,code FROM pool WHERE first_board ORDER BY date,amount_1449 DESC,code").df()
picked = []; last = {}; used_today = {}
for date, code in ordered.itertuples(index=False, name=None):
    if used_today.get(date, 0) == 5 or position[date]-last.get(code, -1000) <= 5:
        continue
    used_today[date] = used_today.get(date, 0)+1; last[code] = position[date]
    picked.append((date, code, used_today[date]))
high, low = signals.loc[signals.arm.eq("high")], signals.loc[signals.arm.eq("low")]
assert set(picked) == set(zip(high.date, high.code, high.daily_rank))
c.register("high", high)
edges = c.sql("""WITH x AS(SELECT h.date,h.code AS event,h.daily_rank,p.code AS peer,
 abs(h.return20_prior_adjusted-p.return20_prior_adjusted) AS prior_gap,
 abs(h.return_1450-p.return_1450) AS day_gap,
 p.amount_1449/h.amount_1449 AS ar,p.price_1449/h.price_1449 AS pr
 FROM high h JOIN pool p ON h.date=p.date AND h.exchange=p.exchange WHERE p.strong_unsealed)
 SELECT *,prior_gap/.05+day_gap/.02+abs(ln(ar))/ln(2)+abs(ln(pr))/ln(2) AS distance
 FROM x WHERE prior_gap<=.05 AND day_gap<=.02 AND ar BETWEEN .5 AND 2 AND pr BETWEEN .5 AND 2
 ORDER BY date,daily_rank,distance,peer""").df()
matching = []; used = set(); assigned = set()
for row in edges.itertuples():
    if (row.date,row.event) in assigned or (row.date,row.peer) in used:
        continue
    assigned.add((row.date,row.event)); used.add((row.date,row.peer))
    matching.append((row.date,row.peer,row.date+":"+row.event))
assert set(matching) == set(zip(low.date, low.code, low.pair_id))
assert np.array_equal(signals.decision_shares,
    2_000_000//np.floor(signals.price_1449*100+.5).astype("int64")//100*100)
coverage = pd.read_parquet("data/research/cash_dividend_catalog/query_coverage.parquet")
needed = pd.DataFrame(sorted({(row.code,str(year)) for row in signals.itertuples()
    for year in range(int(row.date[:4]),int(days[position[row.date]+10][:4])+1)}), columns=["code","year"])
assert needed.merge(coverage,on=["code","year"],how="left",indicator=True)._merge.eq("both").all()
paired = high.merge(low,on=["date","pair_id"],suffixes=("_high","_low"))
result = {"visible_pool_rows":len(pool),"candidates":len(high),"controls":len(low),
    "exhaustive_matching_edges":len(edges),"catalogue_code_years":len(needed),
    "matched_prior_day_return_gap":float((paired.prior_day_return_high-paired.prior_day_return_low).mean()),
    "signals_sha256":sha(root/"signals.parquet"),"new_holding_results_read":False,"new_2026_prices_read":False}
save_json(root/"independent_input_checks.json",result)
print(json.dumps(result,ensure_ascii=False,indent=2))
