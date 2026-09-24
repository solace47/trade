"""Check two fixed exploratory screens by market breadth in 2024-2025.

Requires the market shards, the recent strategy scan, and repriced low-amount
outcomes. Results are exploratory because 2025 was consulted repeatedly.
"""

import json
from pathlib import Path

import duckdb
import pandas as pd

from trade_research.market_study import _quality_keys, _quality_symbols, _week_bootstrap
from trade_research.size_sensitivity import _attach_quality
from trade_research.strategy_scan import _stressed_returns

root = Path("data/research")
scan = json.loads((root / "strategy_scan.json").read_text())
end24 = scan["last_entry_dates"]["2024"]
end25 = scan["last_entry_dates"]["2025"]
c = duckdb.connect()
c.read_parquet(str(root / "market_snapshots_ci" / "*.parquet")).create_view("snapshots")
c.read_parquet(str(root / "market_outcomes_ci" / "*.parquet")).create_view("outcomes")
c.register("bad_days", _quality_keys(root / "market_issues_ci"))
c.register("bad_symbols", _quality_symbols(root / "market_issues_ci"))
c.execute(f"""
CREATE TEMP TABLE eligible AS
SELECT s.* FROM snapshots s
LEFT JOIN bad_days d ON d.date=s.date AND d.code=s.code
LEFT JOIN bad_symbols x ON x.code=s.code
WHERE ((s.date BETWEEN '2024-01-01' AND '{end24}')
    OR (s.date BETWEEN '2025-01-01' AND '{end25}'))
  AND d.code IS NULL AND x.code IS NULL
  AND s.isST=0 AND s.listing_age_sessions>=20
  AND NOT s.reference_gap AND NOT s.quote_outside_traded_range
""")
c.execute("""
CREATE TEMP TABLE regimes AS
SELECT date,
  CASE WHEN AVG((return_1450>0)::INT)>=.6 THEN 'advance'
       WHEN AVG((return_1450>0)::INT)<=.4 THEN 'decline'
       ELSE 'mixed' END regime,
  AVG((return_1450>0)::INT) breadth
FROM eligible GROUP BY date
""")
neutral = pd.read_parquet(root / "size_sensitivity_neutral.parquet")
if "quality_clean_exit" not in neutral:
    neutral = _attach_quality(neutral, root / "market_issues_ci")
neutral = neutral.loc[neutral.target_notional == 20000].copy()
c.register("neutral", neutral)
mid = c.execute("""
WITH ranked AS (
 SELECT *, ROW_NUMBER() OVER (PARTITION BY date
           ORDER BY return20_prior_adjusted ASC NULLS LAST, code) rank
 FROM eligible WHERE amount_1450>=70000000 AND amount_1450<100000000
)
SELECT s.date,s.code,o.horizon,o.entry_status,o.exit_status,o.exit_date,
       o.entry_price,o.exit_price,o.shares,o.net_return,
       NOT EXISTS (SELECT 1 FROM bad_days q WHERE q.code=s.code
                   AND q.date>=s.date AND q.date<=o.exit_date)
         AS quality_clean_exit
FROM ranked s JOIN outcomes o USING(date,code)
WHERE s.rank<=5
""").df()
c.register("mid", mid)
baseline = c.execute("""
SELECT o.date, o.horizon, AVG(o.net_return) AS baseline_net_return
FROM outcomes o JOIN eligible s USING(date, code)
WHERE s.amount_1450>=30000000 AND o.exit_status='filled'
  AND NOT EXISTS (SELECT 1 FROM bad_days q WHERE q.code=o.code
                  AND q.date>=o.date AND q.date<=o.exit_date)
GROUP BY o.date, o.horizon
""").df()
for name in ["neutral", "mid"]:
    frame = c.execute(f"""
       SELECT a.*, r.regime, r.breadth FROM {name} a
       JOIN regimes r USING(date)
    """).df()
    for (year, regime, horizon), part in frame.groupby(
        [frame.date.str[:4], "regime", "horizon"]
    ):
        valid = part.loc[part.exit_status.eq("filled") & part.quality_clean_exit].copy()
        valid["stress"] = _stressed_returns(valid, 10)
        daily = valid.groupby("date", as_index=False)[["net_return", "stress"]].mean()
        daily = daily.merge(baseline.loc[baseline.horizon.eq(horizon),
                                     ["date", "baseline_net_return"]], on="date")
        interval = _week_bootstrap(daily.net_return, daily.date, 20260924)
        edge = daily.net_return - daily.baseline_net_return
        edge_interval = _week_bootstrap(edge, daily.date, 20260925)
        print(json.dumps({"strategy":name,"year":year,"regime":regime,
                          "horizon":int(horizon),
                          "days":len(daily),"signals":len(part),
                          "fills":int(part.entry_status.eq("filled").sum()),
                          "clean_exits":len(valid),
                          "mean":round(daily.net_return.mean()*100,3),
                          "stress10":round(daily.stress.mean()*100,3),
                          "median":round(valid.net_return.median()*100,3),
                          "ci_low":round(interval[0]*100,3),
                          "ci_high":round(interval[1]*100,3),
                          "same_day_baseline":round(daily.baseline_net_return.mean()*100,3),
                          "edge":round(edge.mean()*100,3),
                          "edge_ci_low":round(edge_interval[0]*100,3)},
                         ensure_ascii=False))
