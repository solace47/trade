"""Rebuild disclosed cents, lagged inputs, rankings and all matching edges."""
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from trade_research.corporate_cash import save_json, sha

root = Path("data/research/lhb_institutional_short")
report = json.loads((root / "input_report.json").read_text())
manifest = json.loads((root / "source_manifest.json").read_text())
for name, key in (("signals.parquet", "signals_sha256"), ("visible_pool.parquet", "pool_sha256"),
                  ("source_events.parquet", "source_events_sha256"), ("source_manifest.json", "source_manifest_sha256")):
    assert sha(root / name) == report[key]
for name, digest in manifest["source_sha256"].items():
    assert sha(Path(name)) == digest
events = pd.read_parquet(root / "source_events.parquet")
ambiguities = pd.read_parquet(root / "ambiguous_events.parquet")
pool = pd.read_parquet(root / "visible_pool.parquet")
signals = pd.read_parquet(root / "signals.parquet")
cal = pd.read_parquet("data/baostock/market_2020_2026/metadata/calendar.parquet")
days = sorted(cal.loc[cal.is_trading_day.eq("1") & cal.calendar_date.between("2024-01-01", "2025-12-31"), "calendar_date"])
pos = {day: i for i, day in enumerate(days)}
groups = defaultdict(list)
for current in days[1:-10]:
    prior = days[pos[current]-1]
    raw = json.loads((Path("data/research/lhb/sse_daily") / ("sse_"+prior.replace("-", "")+".json")).read_text())
    assert raw["trade_date"] == prior
    for item in raw["main"]:
        if item["secType"] != "A" or not item["secCode"].startswith("60") or item["refType"] not in ("11", "12", "13", "14"):
            continue
        def amount(side):
            values = [Decimal(v) for v in item["branchTxAmt"+side].split(",")]
            names = item["branchName"+side].split(",")
            assert 1 <= len(values) == len(names) <= 5
            return int(sum((v for n, v in zip(names, values) if n.strip() == "机构专用"), Decimal(0))*100)
        facts = int(Decimal(item["secTxAmount"])*100), amount("B"), amount("S")
        groups[(current, prior, "sh."+item["secCode"])].append((item["refType"], facts))
rebuilt_events, ambiguous = [], []
for (current, prior, code), entries in sorted(groups.items()):
    facts = {entry[1] for entry in entries}
    if len(facts) != 1:
        ambiguous.append((current, code)); continue
    total, buy, sell = facts.pop()
    rebuilt_events.append({"date": current, "trade_date": prior, "code": code,
        "reasons": ",".join(sorted(e[0] for e in entries)), "source_reason_count": len(entries),
        "disclosed_turnover_cents": total, "institution_buy_cents": buy, "institution_sell_cents": sell,
        "institution_net_cents": buy-sell, "net_fraction": (buy-sell)/total})
pd.testing.assert_frame_equal(events, pd.DataFrame(rebuilt_events), check_exact=True)
assert set(ambiguous) == set(zip(ambiguities.date, ambiguities.code))
print("Source amounts rebuilt", len(events), flush=True)
c = duckdb.connect(); c.execute("SET threads=4")
c.register("schedule", pd.DataFrame({"date": days[1:-10], "trade_date": days[:-11]}))
c.read_parquet("data/research/minute_prefix_1449/202[45]/*.parquet").create_view("prefix")
c.read_parquet("data/research/market_snapshots_ci/*.parquet").create_view("snapshots")
c.read_parquet("data/baostock/market_2020_2026/daily/sh_60*.parquet").create_view("daily")
c.read_parquet("data/research/risk_removal_1449/notice_index.parquet").create_view("notices")
past = c.sql("""SELECT d.date AS trade_date,d.code,d.close AS prior_close,d.preclose AS prior_reference
 FROM daily d JOIN schedule s ON d.date=s.trade_date WHERE d.isST=0 AND d.tradestatus=1""").df()
rebuilt = c.sql("""SELECT p.date,p.code,p.price_1449,p.high_1449,p.low_1449,p.amount_1449,p.volume_1449,
 s.preclose,s.isST,s.reference_gap,s.listing_age_sessions,s.return20_prior_adjusted,d.trade_date
 FROM schedule d JOIN prefix p USING(date) JOIN snapshots s USING(date,code)
 WHERE p.code LIKE 'sh.60%' AND s.isST=0 AND s.tradestatus=1 AND s.listing_age_sessions>=20
 AND NOT s.reference_gap AND NOT p.quote_outside_traded_range AND p.price_1449>=5
 AND p.amount_1449>=30000000 AND p.volume_1449>0 AND s.preclose>0
 AND NOT EXISTS(SELECT 1 FROM notices n WHERE n.code=p.code AND n.notice_date<p.date
   AND regexp_matches(n.title,'进入退市整理|退市整理期交易'))""").df()
rebuilt = rebuilt.merge(past, on=["code", "trade_date"], validate="many_to_one")
rebuilt = rebuilt.loc[[(d, code) not in set(ambiguous) for d, code in zip(rebuilt.date, rebuilt.code)]]
assert len(rebuilt) == report["before_buyable_rows"]
cents = np.floor(rebuilt.price_1449*100+.5).astype("int64")
upper = np.array([int((Decimal(str(v))*110).to_integral_value(rounding=ROUND_HALF_UP)) for v in rebuilt.preclose])
allowed = ((rebuilt.price_1449-cents/100).abs().le(.0001) & (2_000_000//cents//100*100).ge(100)
    & (cents*10000+np.maximum(cents*15, 5000) < upper*10000-5000))
rebuilt = rebuilt.loc[allowed].copy(); rebuilt["price_1449"] = cents[allowed]/100
rebuilt["return_1450"] = rebuilt.price_1449/rebuilt.preclose-1
rebuilt["prior_day_return"] = rebuilt.prior_close/rebuilt.prior_reference-1
rebuilt = rebuilt.merge(events, on=["date", "trade_date", "code"], how="left", validate="one_to_one")
rebuilt["positive_net"] = rebuilt.institution_net_cents.gt(0)
rebuilt["half"] = rebuilt.date.str[:4]+np.where(rebuilt.date.str[5:7].le("06"), "H1", "H2")
rebuilt = rebuilt.sort_values(["date", "code"]).reset_index(drop=True)
pd.testing.assert_frame_equal(rebuilt, pool[rebuilt.columns], check_dtype=False, atol=1e-12, rtol=0)
c.register("pool", pool)
order = c.sql("SELECT date,code FROM pool WHERE positive_net ORDER BY date,net_fraction DESC,code").df()
last, counts, picked = {}, {}, []
for date, code in order.itertuples(index=False, name=None):
    if counts.get(date, 0) >= 5 or pos[date]-last.get(code, -1000) <= 5:
        continue
    counts[date] = counts.get(date, 0)+1; last[code] = pos[date]
    picked.append((date, code, counts[date]))
high, low = signals.loc[signals.arm.eq("high")], signals.loc[signals.arm.eq("low")]
assert set(picked) == set(zip(high.date, high.code, high.daily_rank))
c.register("high", high)
edges = c.sql("""WITH x AS(SELECT h.date,h.code AS event,h.daily_rank,p.code AS peer,
 abs(h.return20_prior_adjusted-p.return20_prior_adjusted) AS prior_gap,
 abs(h.return_1450-p.return_1450) AS day_gap,abs(h.prior_day_return-p.prior_day_return) AS yesterday_gap,
 p.amount_1449/h.amount_1449 AS ar,p.price_1449/h.price_1449 AS pr
 FROM high h JOIN pool p ON h.date=p.date WHERE NOT p.positive_net)
 SELECT *,prior_gap/.05+day_gap/.02+yesterday_gap/.02+abs(ln(ar))/ln(2)+abs(ln(pr))/ln(2) AS distance
 FROM x WHERE prior_gap<=.05 AND day_gap<=.02 AND yesterday_gap<=.02
 AND ar BETWEEN .5 AND 2 AND pr BETWEEN .5 AND 2 ORDER BY date,daily_rank,distance,peer""").df()
matches, taken, assigned = [], set(), set()
for row in edges.itertuples():
    if (row.date, row.event) in assigned or (row.date, row.peer) in taken:
        continue
    assigned.add((row.date, row.event)); taken.add((row.date, row.peer))
    matches.append((row.date, row.peer, row.date+":"+row.event))
assert set(matches) == set(zip(low.date, low.code, low.pair_id))
assert np.array_equal(signals.decision_shares, 2_000_000//np.floor(signals.price_1449*100+.5).astype("int64")//100*100)
coverage = pd.read_parquet("data/research/cash_dividend_catalog/query_coverage.parquet")
needed = pd.DataFrame(sorted({(r.code, str(year)) for r in signals.itertuples()
    for year in range(int(r.date[:4]), int(days[pos[r.date]+10][:4])+1)}), columns=["code", "year"])
assert needed.merge(coverage, on=["code", "year"], how="left", indicator=True)._merge.eq("both").all()
paired = high.merge(low, on=["date", "pair_id"], suffixes=("_high", "_low"))
result = {"source_events": len(events), "ambiguous_events": len(ambiguous), "visible_pool_rows": len(pool),
    "candidates": len(high), "controls": len(low), "exhaustive_matching_edges": len(edges),
    "catalogue_code_years": len(needed), "matched_listed_controls": int(paired.reasons_low.notna().sum()),
    "matched_prior_day_return_gap": float((paired.prior_day_return_high-paired.prior_day_return_low).mean()),
    "signals_sha256": sha(root / "signals.parquet"), "new_holding_results_read": False,
    "new_2026_holding_prices_read": False}
save_json(root / "independent_input_checks.json", result)
print(json.dumps(result, ensure_ascii=False, indent=2))
