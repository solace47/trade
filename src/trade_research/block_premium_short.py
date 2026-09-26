"""Study disclosed, volume-weighted premium block trades without chasing limits."""
from __future__ import annotations

import argparse
from decimal import Decimal
import json
from pathlib import Path
import time

import duckdb
import numpy as np
import pandas as pd
import requests

from .block_trade_source import _validate_trade, fetch_sse_day
from .corporate_cash import save_json, sha
from .lhb_institutional_short_eval import execute as execute_short, compare as compare_short
from .quote_precision import quote_cents, fixed_quote_shares
from .reference_gain_eval import DAILY
from .risk_removal_1449 import decision_buyable
from .sector_resilience_1449 import DELIST_NOTICES
from .turnover_reference import CALENDAR

ROOT = Path("data/research/block_premium_short")
PROTOCOL = Path("config/block_premium_short_protocol.json")
RULE_COMMIT = "d1ee74e"
ARCHIVE = Path("data/research/block_trade/daily")


def archive_sse(output: Path = ROOT) -> dict:
    """Persist one venue separately so another venue's missing page is never zero."""
    table = pd.read_parquet(CALENDAR)
    calendar = sorted(table.loc[table.is_trading_day.eq("1") &
        table.calendar_date.between("2024-01-01", "2025-12-31"), "calendar_date"])
    folder = output/"sse_archive"; folder.mkdir(parents=True, exist_ok=True)
    reused, downloaded = 0, 0
    with requests.Session() as session:
        for day in calendar[:-11]:
            path = folder/(day+".json")
            if path.exists():
                record = json.loads(path.read_text())
            else:
                legacy = ARCHIVE/(day+".json")
                if legacy.exists():
                    saved = json.loads(legacy.read_text())
                    if saved["trade_date"] != day:
                        raise ValueError("Archived SSE date mismatch")
                    rows = saved["sse"]; reused += 1
                    origin = {"original_joint_archive_sha256": sha(legacy)}
                else:
                    rows = fetch_sse_day(day, session); downloaded += 1
                    origin = {"transport": "https", "official_pagination_checked": True}
                    time.sleep(.25)
                record = {"schema_version": 1, "exchange": "sse", "trade_date": day, "rows": rows, **origin}
                save_json(path, record)
            if record["exchange"] != "sse" or record["trade_date"] != day or not isinstance(record["rows"], list):
                raise ValueError("Invalid separate SSE source archive")
            for item in record["rows"]:
                _validate_trade(item, day, "sse")
            if downloaded and downloaded % 10 == 0:
                print("SSE source backfill", downloaded, day, flush=True)
    return {"expected_days": len(calendar)-11, "reused_joint_days": reused, "new_sse_days": downloaded}


def integer_unit(value: str, multiplier: int) -> int:
    exact = Decimal(str(value).replace(",", ""))*multiplier
    if not exact.is_finite() or exact <= 0 or exact != exact.to_integral_value():
        raise ValueError("A reported block-trade unit cannot be represented exactly")
    return int(exact)


def source(calendar: list[str], folder: Path = ROOT/"sse_archive") -> tuple[pd.DataFrame, dict[str, str]]:
    rows, sources = [], {}
    for prior, current in zip(calendar[:-11], calendar[1:-10], strict=True):
        path = folder/(prior+".json"); saved = json.loads(path.read_text()); sources[str(path)] = sha(path)
        if saved["exchange"] != "sse" or saved["trade_date"] != prior:
            raise ValueError("Separate SSE source date mismatch")
        for index, item in enumerate(saved["rows"]):
            _validate_trade(item, prior, "sse")
            code = "sh."+item["stockid"]
            if not code.startswith("sh.60"):
                continue
            price, shares, amount = (integer_unit(item[k], n) for k, n in
                zip(("tradeprice", "tradeqty", "tradeamount"), (100, 10000, 1000000)))
            rows.append({"date": current, "trade_date": prior, "code": code, "exchange": "sh",
                "source_row": index, "price_cents": price, "reported_shares": shares,
                "reported_amount_cents": amount, "quoted_price_times_shares": price*shares})
    return pd.DataFrame(rows), sources


def select(events: pd.DataFrame, calendar: list[str]) -> pd.DataFrame:
    pos, last, rows = {d: i for i, d in enumerate(calendar)}, {}, []
    for date, group in events.groupby("date", sort=True):
        rank = 0
        for row in group.sort_values(["premium_fraction", "reported_amount_cents", "code"],
                ascending=[False, False, True]).itertuples():
            if pos[date]-last.get(row.code, -1000) <= 5:
                continue
            rank += 1; last[row.code] = pos[date]
            rows.append({"date": date, "code": row.code, "daily_rank": rank})
            if rank == 5:
                break
    return pd.DataFrame(rows, columns=["date", "code", "daily_rank"])


def match(high: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date, group in high.groupby("date", sort=True):
        available = pool.loc[pool.date.eq(date) & ~pool.has_block].set_index("code", drop=False)
        for row in group.sort_values("daily_rank").itertuples():
            fresh = available.loc[available.exchange.eq(row.exchange)]
            prior = (fresh.return20_prior_adjusted-row.return20_prior_adjusted).abs()
            today = (fresh.return_1450-row.return_1450).abs()
            yesterday = (fresh.prior_day_return-row.prior_day_return).abs()
            amount, price = fresh.amount_1449/row.amount_1449, fresh.price_1449/row.price_1449
            valid = prior.le(.05) & today.le(.02) & yesterday.le(.02) & amount.between(.5,2) & price.between(.5,2)
            choices = fresh.loc[valid].copy()
            if choices.empty:
                continue
            distance = prior/.05+today/.02+yesterday/.02+(np.log(amount).abs()+np.log(price).abs())/np.log(2)
            choices["distance"] = distance.loc[choices.index]
            chosen = choices.reset_index(drop=True).sort_values(["distance", "code"]).iloc[0]
            rows.append({"date": date, "code": chosen.code, "pair_id": row.code,
                "daily_rank": row.daily_rank, "distance": chosen.distance})
            available = available.drop(index=chosen.code)
    return pd.DataFrame(rows, columns=["date", "code", "pair_id", "daily_rank", "distance"])


def freeze(output: Path = ROOT) -> dict:
    if (output / "input_report.json").exists():
        raise ValueError("Do not replace frozen premium block inputs")
    output.mkdir(parents=True, exist_ok=True)
    table = pd.read_parquet(CALENDAR)
    calendar = sorted(table.loc[table.is_trading_day.eq("1") &
        table.calendar_date.between("2024-01-01", "2025-12-31"), "calendar_date"])
    raw, hashes = source(calendar, output/"sse_archive")
    keys = ["date", "trade_date", "code", "exchange"]
    events = raw.groupby(keys, as_index=False).agg(reported_shares=("reported_shares", "sum"),
        reported_amount_cents=("reported_amount_cents", "sum"),
        quoted_price_times_shares=("quoted_price_times_shares", "sum"), source_rows=("code", "size"))
    sources = [*sorted(Path("data/research/minute_prefix_1449").glob("202[45]/*.parquet")),
        *sorted(Path("data/research/market_snapshots_ci").glob("*.parquet")),
        *sorted(DAILY.glob("sh_60*.parquet")), DELIST_NOTICES, CALENDAR, PROTOCOL]
    save_json(output / "source_manifest.json", {"rule_commit": RULE_COMMIT, "archived_days": len(hashes),
        "source_sha256": {**hashes, **{str(p): sha(p) for p in sources}},
        "new_holding_results_read": False, "new_2026_prices_read": False})
    c = duckdb.connect(); c.execute("SET threads=4")
    c.register("schedule", pd.DataFrame({"trade_date": calendar[:-11], "date": calendar[1:-10]}))
    c.read_parquet("data/research/minute_prefix_1449/202[45]/*.parquet").create_view("prefix")
    c.read_parquet("data/research/market_snapshots_ci/*.parquet").create_view("snapshots")
    c.read_parquet([str(p) for p in sources if p.parent == DAILY]).create_view("daily")
    c.read_parquet(str(DELIST_NOTICES)).create_view("notices")
    previous = c.sql("""SELECT d.code,d.date AS trade_date,d.high AS prior_high,
      d.close AS prior_close,d.preclose AS prior_reference FROM daily d JOIN schedule s ON d.date=s.trade_date
      WHERE d.tradestatus=1 AND d.isST=0 AND d.adjustflag=3 AND d.low>0
      AND d.high>=d.close AND d.close>=d.low""").df()
    peers = c.sql("""SELECT p.date,p.code,substr(p.code,1,2) AS exchange,d.trade_date,
      p.price_1449,p.high_1449,p.low_1449,p.amount_1449,p.volume_1449,
      s.preclose,s.isST,s.reference_gap,s.listing_age_sessions,s.return20_prior_adjusted
      FROM prefix p JOIN schedule d USING(date) JOIN snapshots s USING(date,code)
      WHERE p.code LIKE 'sh.60%' AND s.isST=0 AND s.tradestatus=1
      AND s.listing_age_sessions>=20 AND NOT s.reference_gap AND NOT p.quote_outside_traded_range
      AND p.price_1449>=5 AND p.amount_1449>=30000000 AND p.volume_1449>0 AND s.preclose>0
      AND NOT EXISTS(SELECT 1 FROM notices n WHERE n.code=p.code AND n.notice_date<p.date
        AND regexp_matches(n.title,'进入退市整理|退市整理期交易')) ORDER BY p.date,p.code""").df(); c.close()
    events = events.merge(previous, on=["code", "trade_date"], how="left", validate="one_to_one")
    events["prior_high_cents"] = events.prior_high.map(lambda p: quote_cents(p) if pd.notna(p) else np.nan)
    events["premium_fraction"] = events.quoted_price_times_shares/(events.reported_shares*events.prior_high_cents)-1
    events["premium_event"] = (events.prior_high.notna() & events.reported_amount_cents.ge(1_000_000_000)
        & (events.quoted_price_times_shares*100).ge(events.reported_shares*events.prior_high_cents*101))
    events["block_quote_vwap"] = events.quoted_price_times_shares/events.reported_shares/100
    pool = peers.merge(previous, on=["code", "trade_date"], validate="many_to_one")
    pool = pool.loc[[decision_buyable(r.code, r.price_1449, r.preclose) for r in pool.itertuples()]].copy()
    pool["price_1449"] = pool.price_1449.map(lambda p: quote_cents(p)/100)
    pool["return_1450"] = pool.price_1449/pool.preclose-1
    pool["prior_day_return"] = pool.prior_close/pool.prior_reference-1
    fields = [*keys, "reported_shares", "reported_amount_cents", "quoted_price_times_shares",
        "source_rows", "premium_fraction", "premium_event", "block_quote_vwap"]
    pool = pool.merge(events[fields], on=keys, how="left", validate="one_to_one")
    pool["has_block"] = pool.reported_shares.notna()
    pool["candidate"] = (pool.premium_event.eq(True)
        & (np.floor(pool.price_1449*100+.5)*pool.reported_shares).le(pool.quoted_price_times_shares))
    pool = pool.sort_values(["date", "code"]).reset_index(drop=True)
    high = select(pool.loc[pool.candidate], calendar)
    low = match(high.merge(pool, on=["date", "code"], validate="one_to_one"), pool)
    high["arm"], high["pair_id"] = "high", high.code; low["arm"] = "low"
    members = pd.concat([high, low], ignore_index=True); members["pair_id"] = members.date+":"+members.pair_id
    signals = members.merge(pool, on=["date", "code"], validate="one_to_one")
    signals["half"] = signals.date.str[:4]+np.where(signals.date.str[5:7].le("06"), "H1", "H2")
    signals["decision_shares"] = [fixed_quote_shares(r.code, r.price_1449, 20000) for r in signals.itertuples()]
    signals = signals.sort_values(["date", "arm", "daily_rank", "code"]).reset_index(drop=True)
    if len(signals) != len(members) or signals.duplicated(["date", "code"]).any():
        raise ValueError("Frozen premium selection lost an identity")
    for name, frame in (("raw_disclosures", raw), ("source_events", events), ("visible_pool", pool), ("signals", signals)):
        frame.to_parquet(output/(name+".parquet"), index=False, compression="zstd")
    result = {"rule_commit": RULE_COMMIT, "raw_block_rows": len(raw), "source_stock_days": len(events),
        "missing_normal_prior_daily": int(events.prior_high.isna().sum()), "premium_events": int(events.premium_event.sum()),
        "visible_pool_rows": len(pool), "candidate_pool_rows": int(pool.candidate.sum()), "candidates": len(high), "controls": len(low),
        "by_half": signals.groupby(["half", "arm"]).agg(rows=("code", "size"), days=("date", "nunique")).reset_index().to_dict("records"),
        "output_sha256": {name: sha(output/name) for name in ("raw_disclosures.parquet", "source_events.parquet", "visible_pool.parquet")},
        "signals_sha256": sha(output / "signals.parquet"), "source_manifest_sha256": sha(output / "source_manifest.json"),
        "new_holding_results_read": False, "new_2026_prices_read": False}
    save_json(output / "input_report.json", result)
    return result


def execute(output: Path = ROOT) -> dict:
    return execute_short(output, horizons=(1, 3), rule_commit=RULE_COMMIT)


def compare(output: Path = ROOT) -> dict:
    return compare_short(output, horizons=(1, 3), rule_commit=RULE_COMMIT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["archive_sse", "freeze", "execute", "compare"])
    args = parser.parse_args()
    print(json.dumps(globals()[args.stage](), ensure_ascii=False, indent=2))
