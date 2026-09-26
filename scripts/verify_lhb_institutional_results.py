"""Independently verify raw fills, queue flags, fees and short-horizon statistics."""
from pathlib import Path
import argparse
import json

import duckdb
import numpy as np
import pandas as pd

from trade_research.corporate_cash import save_json, sha

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, default=Path("data/research/lhb_institutional_short"))
root = parser.parse_args().root
folder = root / "continued"
comparison = json.loads((root / "comparison_report.json").read_text())
c = duckdb.connect()
c.read_parquet(str(folder / "tick_cost_scenario.parquet")).create_view("trades")
c.read_parquet(str(folder / "raw_windows.parquet")).create_view("bars")
rows = c.sql("""WITH prices AS(SELECT *,bps,entry_price/1.0005 AS rb,exit_price/.9995 AS rs
 FROM trades CROSS JOIN(VALUES(5),(15))v(bps)),value AS(SELECT *,
 shares*(rb+greatest(rb*bps/10000.,.005)) AS bv,
 catalog_sold_shares*(rs-greatest(rs*bps/10000.,.005)) AS sv FROM prices)
 SELECT *,CASE WHEN entry_status!='filled' AND entry_window_status='valid' THEN 0.
 WHEN entry_status='filled' AND exit_price IS NOT NULL THEN
 (sv-greatest(5.,sv*.0003)-sv*.00051+catalog_dividend_gross-catalog_dividend_tax)
 /(bv+greatest(5.,bv*.0003)+bv*.00001)-1 ELSE NULL END AS computed,
 CASE WHEN bps=5 THEN tick_return5 ELSE tick_return15 END AS stored FROM value""").df()
assert (rows.computed.isna() == rows.stored.isna()).all()
assert rows.date.between("2024-01-01", "2025-12-31").all()
assert rows.exit_date.dropna().between("2024-01-01", "2025-12-31").all()
assert (rows.computed-rows.stored).abs().max() < 1e-12
windows = c.sql("""SELECT code,date,count(*) AS n,count(distinct timestamp) AS nu,
 sum(volume) AS volume,sum(turnover)/nullif(sum(volume),0) AS vwap
 FROM bars WHERE label IN ('1452','1453','1454','1455') GROUP BY code,date""").df()
c.register("windows", windows)
fills = c.sql("""SELECT count(*) AS n,max(abs(t.entry_price-b.vwap*1.0005)) AS buy_error,
 max(abs(t.exit_price-e.vwap*.9995)) AS sell_error,
 count(*) FILTER(WHERE b.n<>4 OR e.n<>4 OR b.nu<>4 OR e.nu<>4
 OR t.shares>b.volume*.1 OR t.catalog_sold_shares>e.volume*.1) AS invalid
 FROM trades t JOIN windows b ON t.date=b.date AND t.code=b.code
 JOIN windows e ON t.exit_date=e.date AND t.code=e.code WHERE t.entry_status='filled'""").df().iloc[0]
assert fills.invalid == 0 and max(fills.buy_error, fills.sell_error) < 1e-12
queue = pd.read_parquet(folder / "execution_queue_audit.parquet")
daily = []
for code, group in queue.groupby("code"):
    dates = sorted(set(group.date) | set(group.exit_date.dropna()))
    daily.append(pd.read_parquet(Path("data/baostock/market_2020_2026/daily") / (code.replace(".", "_")+".parquet"),
        filters=[("date", "in", dates)], columns=["date", "code", "preclose", "isST"]))
c.register("daily", pd.concat(daily, ignore_index=True)); c.register("queues", queue)
reproduced = c.sql("""WITH limits AS(SELECT code,date,
 round(preclose::DECIMAL(18,6)*CASE WHEN isST=1 THEN 1.05 ELSE 1.10 END,2) AS upper,
 round(preclose::DECIMAL(18,6)*CASE WHEN isST=1 THEN .95 ELSE .90 END,2) AS lower FROM daily),
 touched AS(SELECT b.code,b.date,bool_or(round(b.high,2)>=d.upper) AS buy_touch,
 bool_or(round(b.low,2)<=d.lower) AS sell_touch FROM bars b JOIN limits d USING(code,date)
 WHERE volume>0 AND label IN ('1452','1453','1454','1455') GROUP BY b.code,b.date)
 SELECT q.date,q.code,q.horizon,b.buy_touch,e.sell_touch,
 q.entry_status='filled' AND (coalesce(b.buy_touch,true) OR coalesce(e.sell_touch,true)) AS unknown
 FROM queues q LEFT JOIN touched b ON q.date=b.date AND q.code=b.code
 LEFT JOIN touched e ON q.exit_date=e.date AND q.code=e.code""").df()
checked = queue.merge(reproduced, on=["date", "code", "horizon"], validate="one_to_one")
for actual, expected in (("entry_limit_touched", "buy_touch"), ("exit_limit_touched", "sell_touch"),
                         ("queue_allocation_unverified", "unknown")):
    assert checked[actual].fillna(True).eq(checked[expected].fillna(True)).all()
rows = rows.merge(queue[["date", "code", "horizon", "queue_allocation_unverified"]],
    on=["date", "code", "horizon"], validate="many_to_one")

# Independently enumerate all four OHLC price corners, including half-cent impact.
bounds = c.sql("""WITH p AS(SELECT *,bps,
 CASE WHEN entry_window_status='valid' THEN entry_price/1.0005 ELSE round(buy_bound,2) END AS rb,
 CASE WHEN exit_window_status='valid' THEN exit_price/.9995 ELSE round(sell_bound,2) END AS rs
 FROM trades CROSS JOIN(VALUES(5),(15))v(bps),
 LATERAL (VALUES(entry_window_low),(entry_window_high))b(buy_bound),
 LATERAL (VALUES(exit_window_low),(exit_window_high))e(sell_bound)),
 val AS(SELECT *,shares*(rb+greatest(rb*bps/10000.,.005)) AS bv,
 catalog_sold_shares*(rs-greatest(rs*bps/10000.,.005)) AS sv FROM p),
 returns AS(SELECT *,CASE WHEN entry_status!='filled' AND entry_window_status='valid' THEN 0.
 WHEN entry_status='filled' AND exit_price IS NOT NULL THEN
 (sv-greatest(5.,sv*.0003)-sv*.00051+catalog_dividend_gross-catalog_dividend_tax)
 /(bv+greatest(5.,bv*.0003)+bv*.00001)-1 ELSE NULL END AS computed FROM val)
 SELECT date,code,horizon,bps,min(computed) AS lo,max(computed) AS hi,
 any_value(CASE WHEN bps=5 THEN tick_lower5 ELSE tick_lower15 END) AS stored_lo,
 any_value(CASE WHEN bps=5 THEN tick_upper5 ELSE tick_upper15 END) AS stored_hi
 FROM returns GROUP BY date,code,horizon,bps""").df()
# Normalize displayed OHLC bounds to cents while preserving every raw VWAP.
assert bounds.lo.isna().eq(bounds.stored_lo.isna()).all()
assert bounds.hi.isna().eq(bounds.stored_hi.isna()).all()
bound_error = max((bounds.lo-bounds.stored_lo).abs().max(), (bounds.hi-bounds.stored_hi).abs().max())
assert bound_error < 1e-12


def period(frame, label):
    if label == "full":
        return frame
    return frame.loc[frame.date.str[:4].eq(label)] if len(label) == 4 else frame.loc[frame.half.eq(label)]


def verify_number(actual, expected):
    assert pd.isna(actual) if expected is None else abs(actual-expected) < 1e-12, (actual, expected)


def bootstrap(values):
    weeks = pd.to_datetime(values.index).to_period("W-SUN")
    blocks = [g.to_numpy() for _, g in values.groupby(weeks)]
    choices = np.random.default_rng(20260926).integers(len(blocks), size=(10000, len(blocks)))
    weights = np.stack([np.bincount(x, minlength=len(blocks)) for x in choices])
    sums, counts = np.array([x.sum() for x in blocks]), np.array([len(x) for x in blocks])
    return np.quantile((weights @ sums)/(weights @ counts), [.025, .975])


for item in comparison["metrics"]:
    part = rows.loc[rows.horizon.eq(item["horizon"]) & rows.bps.eq(item["bps"])]
    if item["metric"] == "same_day_edge":
        part = part.loc[part.arm.eq("high")].merge(part.loc[part.arm.eq("low")],
            on=["date", "pair_id", "horizon", "half"], validate="one_to_one", suffixes=("_a", "_b"))
        part["value"] = part.computed_a-part.computed_b
    else:
        part = part.loc[part.arm.eq("high")].copy(); part["value"] = part.computed
        if item["metric"] == "own_with_queue_unknown_retained":
            part.loc[part.queue_allocation_unverified, "value"] = np.nan
    part = period(part, item["period"])
    daily = part.groupby("date").value.agg(lambda x: np.nan if x.isna().any() else sum(x)/len(x))
    assert len(daily) == item["signal_dates"] and int(daily.isna().sum()) == item["unknown_dates"]
    verify_number(np.nan if daily.isna().any() or daily.empty else daily.mean(), item["daily_mean"])
    if daily.empty or daily.isna().any() or pd.to_datetime(daily.index).to_period("W-SUN").nunique() < 2:
        assert item["weekly_interval"] is None
    else:
        assert np.max(abs(bootstrap(daily)-item["weekly_interval"])) < 1e-12
for item in comparison["trade_distributions"]:
    part = period(rows.loc[rows.arm.eq("high") & rows.horizon.eq(item["horizon"]) & rows.bps.eq(item["bps"])], item["period"])
    bought = part.loc[part.entry_status.eq("filled")]
    assert len(part) == item["orders"] and len(bought) == item["bought"]
    values = np.sort(bought.computed.to_numpy())
    if len(values) == 0 or not np.isfinite(values).all():
        assert item["win_rate"] is None
        continue
    positive, negative = values[values > 0], values[values < 0]
    mean_win = positive.mean() if len(positive) else np.nan
    mean_loss = negative.mean() if len(negative) else np.nan
    expected = {"win_rate": len(positive)/len(values), "mean_win": mean_win, "mean_loss": mean_loss,
        "payoff_ratio": mean_win/-mean_loss, "trade_mean": values.mean(),
        "worst_trade_return": values[0], "worst_five_percent_mean": values[:(len(values)+19)//20].mean()}
    for key, value in expected.items():
        verify_number(value, item[key])
result = {"cost_rows": len(rows), "raw_fills": int(fills.n), "queue_records": len(queue), "price_bounds": len(bounds),
    "economic_max_error": float((rows.computed-rows.stored).abs().max()), "bounds_max_error": float(bound_error),
    "raw_fill_max_error": float(max(fills.buy_error, fills.sell_error)),
    "daily_metrics_and_intervals": len(comparison["metrics"]), "trade_distributions": len(comparison["trade_distributions"]),
    "tick_report_sha256": sha(folder / "tick_report.json"), "comparison_report_sha256": sha(root / "comparison_report.json")}
save_json(root / "independent_economic_checks.json", result)
print(json.dumps(result, indent=2))
