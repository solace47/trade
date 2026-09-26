"""Rebuild industry membership, leave-one-out conditions and every match edge."""
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from trade_research.corporate_cash import save_json, sha

root = Path("data/research/sector_strength_short")
report = json.loads((root / "input_report.json").read_text())
manifest = json.loads((root / "source_manifest.json").read_text())
for name, key in (("signals.parquet", "signals_sha256"), ("visible_pool.parquet", "pool_sha256"),
                  ("peers.parquet", "peers_sha256"), ("source_manifest.json", "source_manifest_sha256")):
    assert sha(root/name) == report[key]
for path, expected in manifest["source_sha256"].items():
    assert sha(Path(path)) == expected
peers, pool, signals = (pd.read_parquet(root / (name+".parquet")) for name in ("peers", "visible_pool", "signals"))
table = pd.read_parquet("data/baostock/market_2020_2026/metadata/calendar.parquet")
calendar = sorted(table.loc[table.is_trading_day.eq("1") & table.calendar_date.between("2024-01-01", "2025-12-31"), "calendar_date"])
positions = {d: i for i, d in enumerate(calendar)}
c = duckdb.connect(); c.execute("SET threads=4")
c.register("dates", pd.DataFrame({"date": calendar[:-10]}))
c.read_parquet("data/research/minute_prefix_1449/202[45]/*.parquet").create_view("prefix")
c.read_parquet("data/research/market_snapshots_ci/*.parquet").create_view("snapshots")
c.read_parquet("data/research/industry_intervals.parquet").create_view("history")
c.read_parquet("data/research/risk_removal_1449/notice_index.parquet").create_view("notices")
base = c.sql("""SELECT p.date,p.code,h.industry,h.asof,h.updateDate,h.effective_date,h.next_effective_date,
 round(p.price_1449,2) AS price_1449,round(p.price_1420,2) AS price_1420,p.amount_1449,p.volume_1449,
 p.high_1449,p.low_1449,s.preclose,s.isST,s.tradestatus,s.listing_age_sessions,s.reference_gap,s.return20_prior_adjusted
 FROM prefix p INNER JOIN dates d USING(date) INNER JOIN snapshots s USING(date,code)
 INNER JOIN history h ON h.code=p.code AND h.effective_date<=p.date
 AND(p.date<h.next_effective_date OR h.next_effective_date IS NULL)
 WHERE substr(p.code,1,5) IN('sh.60','sz.00') AND s.isST=0 AND s.tradestatus=1
 AND s.listing_age_sessions>=20 AND s.reference_gap=false AND p.quote_outside_traded_range=false
 AND length(h.industry)>0 AND h.asof<=p.date AND h.updateDate<p.date
 AND p.date::DATE-h.updateDate::DATE<=370 AND p.amount_1449>=30000000 AND p.volume_1449>0
 AND s.preclose>0 AND p.price_1449>0 AND p.price_1420>0
 AND abs(p.price_1449-round(p.price_1449,2))<=.0001 AND abs(p.price_1420-round(p.price_1420,2))<=.0001
 AND NOT EXISTS(SELECT 1 FROM notices n WHERE n.code=p.code AND n.notice_date<p.date
 AND regexp_matches(n.title,'进入退市整理|退市整理期交易')) ORDER BY p.date,p.code""").df()
pd.testing.assert_frame_equal(base, peers[base.columns], check_dtype=False, atol=1e-12, rtol=0)
c.register("base", base)
statistics = c.sql("""WITH x AS(SELECT *,price_1449/preclose-1 AS ret,
 greatest(-.1,least(.1,price_1449/preclose-1)) AS clipped FROM base)
 SELECT date,code,count(*) OVER sector AS peer_count,count(*) OVER market AS market_peer_count,
 avg(clipped) OVER sector AS sector_return,avg((ret>0)::DOUBLE) OVER sector AS sector_rising_fraction,
 avg(clipped) OVER market AS market_return FROM x
 WINDOW sector AS(PARTITION BY date,industry ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING EXCLUDE CURRENT ROW),
 market AS(PARTITION BY date ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING EXCLUDE CURRENT ROW)
 ORDER BY date,code""").df()
for column in statistics.columns[2:]:
    assert np.allclose(statistics[column], peers[column], atol=1e-12, rtol=0, equal_nan=True), column
assert np.allclose(peers.return_1450, peers.price_1449/peers.preclose-1, rtol=0, atol=1e-12)
assert np.allclose(peers.return_last29, peers.price_1449/peers.price_1420-1, rtol=0, atol=1e-12)
assert np.allclose(peers.vwap_1449, peers.amount_1449/peers.volume_1449, rtol=0, atol=1e-12)
strong = (statistics.peer_count.ge(10) & statistics.sector_return.ge(.01)
    & statistics.sector_rising_fraction.ge(.60) & (statistics.sector_return-statistics.market_return).ge(.005))
assert strong.equals(peers.strong_sector)
eligible = peers.loc[peers.peer_count.ge(10) & peers.price_1449.ge(5) & peers.amount_1449.ge(1e8)
    & peers.return_1450.between(.02,.06) & peers.return20_prior_adjusted.ge(0)
    & peers.return_last29.between(0,.01) & peers.price_1449.ge(peers.vwap_1449)].copy()
cents = np.floor(eligible.price_1449*100+.5).astype("int64")
upper = np.array([int((Decimal(str(p))*110).to_integral_value(rounding=ROUND_HALF_UP)) for p in eligible.preclose])
eligible = eligible.loc[(2_000_000//cents//100*100).ge(100)
    & (cents*10000+np.maximum(cents*15,5000)<upper*10000-5000)]
pd.testing.assert_frame_equal(eligible.reset_index(drop=True), pool.reset_index(drop=True), check_exact=True)
c.register("pool", pool)
order = c.sql("SELECT date,code,industry FROM pool WHERE strong_sector ORDER BY date,amount_1449 DESC,code").df()
last, counts, industries, picked = {}, {}, set(), []
for date, code, industry in order.itertuples(index=False, name=None):
    if counts.get(date,0)==5 or (date,industry) in industries or positions[date]-last.get(code,-1000)<=5:
        continue
    counts[date]=counts.get(date,0)+1; last[code]=positions[date]; industries.add((date,industry))
    picked.append((date,code,counts[date]))
high, low = signals.loc[signals.arm.eq("high")], signals.loc[signals.arm.eq("low")]
assert set(picked)==set(zip(high.date,high.code,high.daily_rank))
c.register("high",high)
edges = c.sql("""WITH x AS(SELECT h.date,h.code AS event,h.daily_rank,p.code AS peer,
 abs(h.return20_prior_adjusted-p.return20_prior_adjusted) AS prior_gap,
 abs(h.return_1450-p.return_1450) AS day_gap,abs(h.return_last29-p.return_last29) AS tail_gap,
 p.amount_1449/h.amount_1449 AS ar,p.price_1449/h.price_1449 AS pr
 FROM high h JOIN pool p ON h.date=p.date AND h.exchange=p.exchange WHERE NOT p.strong_sector)
 SELECT *,prior_gap/.05+day_gap/.02+tail_gap/.005+abs(ln(ar))/ln(2)+abs(ln(pr))/ln(2) AS distance
 FROM x WHERE prior_gap<=.05 AND day_gap<=.02 AND tail_gap<=.005 AND ar BETWEEN .5 AND 2 AND pr BETWEEN .5 AND 2
 ORDER BY date,daily_rank,distance,peer""").df()
matched, assigned, used, last = [], set(), set(), {}
for row in edges.itertuples():
    if (row.date,row.event) in assigned or (row.date,row.peer) in used or positions[row.date]-last.get(row.peer,-1000)<=5:
        continue
    assigned.add((row.date,row.event)); used.add((row.date,row.peer)); last[row.peer]=positions[row.date]
    matched.append((row.date,row.peer,row.date+":"+row.event))
assert set(matched)==set(zip(low.date,low.code,low.pair_id))
assert np.array_equal(signals.decision_shares,2_000_000//np.floor(signals.price_1449*100+.5).astype("int64")//100*100)
coverage = pd.read_parquet("data/research/cash_dividend_catalog/query_coverage.parquet")
needed = pd.DataFrame(sorted({(r.code,str(year)) for r in signals.itertuples()
    for year in range(int(r.date[:4]),int(calendar[positions[r.date]+10][:4])+1)}),columns=["code","year"])
assert needed.merge(coverage,on=["code","year"],how="left",indicator=True)._merge.eq("both").all()

# The raw prefix sample is chosen by identity hash, independently of any outcome.
chosen = signals.assign(check_key=[hashlib.sha256((d+code).encode()).hexdigest() for d,code in zip(signals.date,signals.code)])
chosen = chosen.sort_values("check_key").groupby(["half","arm"],sort=True).head(5)
for row in chosen.itertuples():
    exchange,symbol=row.code.split(".")
    path=Path("data/hf/pilot/data/stock_1m")/exchange.upper()/(symbol+".parquet")
    c.read_parquet(str(path)).create_view("source",replace=True)
    bars=c.execute("""SELECT timestamp,close,volume,turnover FROM source
      WHERE timestamp>=?::TIMESTAMP AND timestamp<=?::TIMESTAMP ORDER BY timestamp""",
      [row.date+" 09:30:00",row.date+" 14:49:00"]).df()
    labels=bars.timestamp.dt.strftime("%H%M")
    bars=bars.loc[labels.between("0930","1130")|labels.between("1301","1449")]
    assert len(bars)==230 and bars.timestamp.nunique()==230
    assert abs(float(bars.iloc[-1].close)-row.price_1449)<.0001
    assert abs(float(bars.volume.sum())-row.volume_1449)<1e-6
    assert abs(float(bars.turnover.sum())-row.amount_1449)<.01
result={"peer_rows":len(peers),"pool_rows":len(pool),"candidates":len(high),"controls":len(low),
    "exhaustive_matching_edges":len(edges),"catalogue_code_years":len(needed),"raw_prefix_checks":len(chosen),
    "leave_one_out_all_rows_verified":True,"signals_sha256":sha(root/"signals.parquet"),
    "new_holding_results_read":False,"new_2026_prices_read":False}
save_json(root/"independent_input_checks.json",result)
print(json.dumps(result,ensure_ascii=False,indent=2))
