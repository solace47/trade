"""Independent dated-fee, queue-limit, input-identity and raw-minute checks."""
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from trade_research.corporate_cash import save_json, sha

root = Path("data/research/short_horizon_target")
report = json.loads((root / "label_report.json").read_text())
raw_report = json.loads((root / "raw_report.json").read_text())
features = pd.read_parquet(root / "training_signals.parquet")
calendar_raw = pd.read_parquet("data/baostock/market_2020_2026/metadata/calendar.parquet")
calendar = sorted(calendar_raw.loc[calendar_raw.is_trading_day.eq("1") &
    calendar_raw.calendar_date.between("2022-01-01", "2024-12-31"), "calendar_date"])
positions = {date: i for i, date in enumerate(calendar)}
for name, digest in raw_report["output_sha256"].items():
    assert sha(root / name) == digest
c = duckdb.connect(); c.execute("SET threads=4")
daily_dir = Path("data/baostock/market_2020_2026/daily")
c.read_parquet([str(p) for p in sorted([*daily_dir.glob("sh_60*.parquet"), *daily_dir.glob("sz_00*.parquet")])]).create_view("daily")
references = c.sql("SELECT date,code,preclose,isST FROM daily WHERE date BETWEEN '2022-01-01' AND '2024-12-31'").df()
limits = {}
for reference, st in references[["preclose", "isST"]].drop_duplicates().itertuples(index=False, name=None):
    if pd.isna(reference):
        limits[reference, st] = (np.nan, np.nan)
    else:
        value = Decimal(str(reference))
        limits[reference, st] = tuple(float((value*Decimal(multiplier)).quantize(Decimal(".01"), rounding=ROUND_HALF_UP))
            for multiplier in (("1.05", ".95") if st else ("1.10", ".90")))
references["upper"] = [limits.get((p, st), (np.nan, np.nan))[0] for p, st in zip(references.preclose, references.isST)]
references["lower"] = [limits.get((p, st), (np.nan, np.nan))[1] for p, st in zip(references.preclose, references.isST)]
references = references[["date", "code", "upper", "lower"]]
reports, sample_frames, tables = {}, [], {}
for name, horizon in (("t1", 1), ("t5", 5)):
    path = root / (name+"_labels.parquet")
    assert sha(path) == report["models"][name]["labels_sha256"]
    rows = pd.read_parquet(path).sort_values(["date", "code"]).reset_index(drop=True)
    assert len(rows) == len(features) and not rows.duplicated(["date", "code"]).any()
    assert set(rows.horizon) == {horizon}
    pd.testing.assert_frame_equal(rows[["date", "code"]], features[["date", "code"]].sort_values(["date", "code"]).reset_index(drop=True))
    assert rows.target_exit_date.eq([calendar[positions[d]+horizon] for d in rows.date]).all()
    bought = rows.entry_status.eq("filled")
    recorded = bought & rows.exit_date.notna()
    delay = rows.exit_date.map(positions)-rows.target_exit_date.map(positions)
    assert delay[recorded].eq(rows.loc[recorded, "exit_delay_sessions"]).all()
    check = rows[["date", "code", "exit_date", "entry_status", "entry_window_high", "exit_window_low"]].merge(
        references[["date", "code", "upper"]], on=["date", "code"], how="left", validate="many_to_one")
    check = check.merge(references[["date", "code", "lower"]].rename(columns={"date": "exit_date"}),
        on=["exit_date", "code"], how="left", validate="many_to_one")
    expected_queue = bought & (check.upper.isna() | check.lower.isna() | check.entry_window_high.isna()
        | check.exit_window_low.isna() | (np.floor(check.entry_window_high*100+.5)/100).ge(check.upper)
        | (np.floor(check.exit_window_low*100+.5)/100).le(check.lower))
    assert np.array_equal(expected_queue, rows.queue_allocation_unverified)
    known_scenario = rows.score_origin.isin(["recorded_economic_scenario", "adverse_limit_queue_training_penalty"])
    part = rows.loc[known_scenario]
    rb, rs = part.entry_price/1.0005, part.exit_price/.9995
    buy = part.shares*(rb+np.maximum(rb*.0015, .005))
    sell = part.sold_shares*(rs-np.maximum(rs*.0015, .005))
    buy_transfer = np.where(part.date.lt("2022-04-29"), .00002, .00001)
    sell_transfer = np.where(part.exit_date.lt("2022-04-29"), .00002, .00001)
    stamp = np.where(part.exit_date.lt("2023-08-28"), .001, .0005)
    expected = ((sell-np.maximum(5, sell*.0003)-sell*(sell_transfer+stamp)
        +part.dividend_gross-part.dividend_tax)/(buy+np.maximum(5, buy*.0003)+buy*buy_transfer)-1)
    assert np.max(abs(expected-part.economic_scenario15)) < 1e-12
    target = pd.Series(-1., index=rows.index)
    target.loc[rows.score_origin.eq("known_not_bought")] = 0.
    good = rows.score_origin.eq("recorded_economic_scenario")
    target.loc[good] = expected.loc[good]
    assert np.max(abs(target-rows.downside_score)) < 1e-12
    assert rows.loc[rows.score_origin.eq("adverse_limit_queue_training_penalty"), "queue_allocation_unverified"].all()
    tagged = rows.assign(key_hash=[hashlib.sha256((name+d+code).encode()).hexdigest() for d, code in zip(rows.date, rows.code)])
    sample = tagged.sort_values("key_hash").groupby(tagged.date.str[:4], sort=True).head(8)
    sample_frames.append(sample)
    reports[name] = {"rows": len(rows), "economic_scenarios": len(part), "queue_unverified": int(expected_queue.sum()),
        "economic_max_error": float(np.max(abs(expected-part.economic_scenario15))), "raw_samples": len(sample)}
    tables[name] = rows

# Every training-horizon comparison must have the identical modeled buy, including failures.
entry_columns = ["date", "code", "entry_status", "entry_price", "shares"]
pd.testing.assert_frame_equal(tables["t1"][entry_columns], tables["t5"][entry_columns], check_dtype=False, atol=1e-12, rtol=0)
sampled = pd.concat(sample_frames, ignore_index=True)
sample_checks = []
for row in sampled.itertuples():
    exchange, code = row.code.split(".")
    path = Path("data/hf/pilot/data/stock_1m")/exchange.upper()/(code+".parquet")
    dates = [row.date] + ([] if pd.isna(row.exit_date) else [row.exit_date])
    c.read_parquet(str(path)).create_view("source", replace=True)
    minute = c.execute("""SELECT timestamp,high,low,volume,turnover FROM source
      WHERE timestamp>=?::TIMESTAMP AND timestamp<?::DATE+INTERVAL 1 DAY
      AND strftime(timestamp,'%Y-%m-%d') IN (SELECT unnest(?))
      AND strftime(timestamp,'%H%M') IN ('1452','1453','1454','1455') ORDER BY timestamp""", [dates[0], dates[-1], dates]).df()
    minute["date"] = minute.timestamp.dt.strftime("%Y-%m-%d")
    for side, date, stored, multiplier in (("entry", row.date, row.entry_price, 1.0005),
            ("exit", row.exit_date, row.exit_price, .9995)):
        if pd.isna(stored):
            continue
        bars = minute.loc[minute.date.eq(date)]
        assert len(bars) == 4
        vwap = sum(bars.turnover)/sum(bars.volume)
        assert abs(stored-vwap*multiplier) < 1e-12
        assert row.shares <= sum(bars.volume)*.1
        sample_checks.append({"date": row.date, "code": row.code, "horizon": row.horizon, "side": side})
result = {"labels": reports, "same_buys_verified": len(tables["t1"]), "raw_window_checks": len(sample_checks),
    "label_report_sha256": sha(root / "label_report.json"), "new_2026_prices_read": False}
save_json(root / "independent_label_checks.json", result)
print(json.dumps(result, indent=2))
