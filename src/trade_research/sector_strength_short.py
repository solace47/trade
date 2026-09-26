"""Freeze broad sector strength and active, unsealed stocks for short exits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .corporate_cash import save_json, sha
from .lhb_institutional_short_eval import execute as execute_short, compare as compare_short
from .quote_precision import quote_cents, fixed_quote_shares
from .risk_removal_1449 import decision_buyable
from .sector_resilience_1449 import INDUSTRIES, DELIST_NOTICES
from .turnover_reference import CALENDAR

ROOT = Path("data/research/sector_strength_short")
PROTOCOL = Path("config/sector_strength_short_protocol.json")
RULE_COMMIT = "8a26ac1"


def peer_statistics(peers: pd.DataFrame) -> pd.DataFrame:
    """Every stock has its own leave-one-out industry and market comparison."""
    if peers.duplicated(["date", "code"]).any():
        raise ValueError("Historical industry intervals duplicated a stock-day")
    frame = peers.copy()
    frame["clipped_return"] = frame.return_1450.clip(-.10, .10)
    frame["rising"] = frame.return_1450.gt(0).astype(int)
    groups, markets = frame.groupby(["date", "industry"]), frame.groupby("date")
    frame["peer_count"] = groups.code.transform("size")-1
    frame["market_peer_count"] = markets.code.transform("size")-1
    n, m = frame.peer_count.where(frame.peer_count.gt(0)), frame.market_peer_count.where(frame.market_peer_count.gt(0))
    frame["sector_return"] = (groups.clipped_return.transform("sum")-frame.clipped_return)/n
    frame["sector_rising_fraction"] = (groups.rising.transform("sum")-frame.rising)/n
    frame["market_return"] = (markets.clipped_return.transform("sum")-frame.clipped_return)/m
    frame["sector_excess"] = frame.sector_return-frame.market_return
    frame["strong_sector"] = (frame.peer_count.ge(10) & frame.sector_return.ge(.01)
        & frame.sector_rising_fraction.ge(.60) & frame.sector_excess.ge(.005))
    return frame


def select(pool: pd.DataFrame, calendar: list[str]) -> pd.DataFrame:
    pos, last, records = {d: i for i, d in enumerate(calendar)}, {}, []
    for date, group in pool.loc[pool.strong_sector].groupby("date", sort=True):
        industries, rank = set(), 0
        for row in group.sort_values(["amount_1449", "code"], ascending=[False, True]).itertuples():
            if row.industry in industries or pos[date]-last.get(row.code, -1000) <= 5:
                continue
            rank += 1; last[row.code] = pos[date]; industries.add(row.industry)
            records.append({"date": date, "code": row.code, "daily_rank": rank})
            if rank == 5:
                break
    return pd.DataFrame(records, columns=["date", "code", "daily_rank"])


def match(high: pd.DataFrame, pool: pd.DataFrame, calendar: list[str]) -> pd.DataFrame:
    pos, last, records = {d: i for i, d in enumerate(calendar)}, {}, []
    controls = pool.loc[~pool.strong_sector]
    for date, group in high.groupby("date", sort=True):
        available = controls.loc[controls.date.eq(date)].set_index("code", drop=False)
        for row in group.sort_values("daily_rank").itertuples():
            fresh = available.loc[available.exchange.eq(row.exchange)
                & np.array([pos[date]-last.get(code, -1000) > 5 for code in available.index])]
            prior = (fresh.return20_prior_adjusted-row.return20_prior_adjusted).abs()
            daily = (fresh.return_1450-row.return_1450).abs()
            tail = (fresh.return_last29-row.return_last29).abs()
            amount, price = fresh.amount_1449/row.amount_1449, fresh.price_1449/row.price_1449
            valid = prior.le(.05) & daily.le(.02) & tail.le(.005) & amount.between(.5, 2) & price.between(.5, 2)
            choices = fresh.loc[valid].copy()
            if choices.empty:
                continue
            distance = prior/.05+daily/.02+tail/.005+(np.log(amount).abs()+np.log(price).abs())/np.log(2)
            choices["distance"] = distance.loc[choices.index]
            picked = choices.reset_index(drop=True).sort_values(["distance", "code"]).iloc[0]
            records.append({"date": date, "code": picked.code, "pair_id": row.code,
                "daily_rank": row.daily_rank, "distance": picked.distance})
            last[picked.code] = pos[date]; available = available.drop(index=picked.code)
    return pd.DataFrame(records, columns=["date", "code", "pair_id", "daily_rank", "distance"])


def freeze(output: Path = ROOT) -> dict:
    if (output / "input_report.json").exists():
        raise ValueError("Do not replace a frozen industry-strength list")
    output.mkdir(parents=True, exist_ok=True)
    table = pd.read_parquet(CALENDAR)
    calendar = sorted(table.loc[table.is_trading_day.eq("1")
        & table.calendar_date.between("2024-01-01", "2025-12-31"), "calendar_date"])
    sources = [*sorted(Path("data/research/minute_prefix_1449").glob("202[45]/*.parquet")),
        *sorted(Path("data/research/market_snapshots_ci").glob("*.parquet")), INDUSTRIES, DELIST_NOTICES, CALENDAR, PROTOCOL]
    save_json(output / "source_manifest.json", {"rule_commit": RULE_COMMIT,
        "source_sha256": {str(p): sha(p) for p in sources}, "new_holding_results_read": False})
    c = duckdb.connect(); c.execute("SET threads=4")
    c.register("dates", pd.DataFrame({"date": calendar[:-10]}))
    c.read_parquet("data/research/minute_prefix_1449/202[45]/*.parquet").create_view("prefix")
    c.read_parquet("data/research/market_snapshots_ci/*.parquet").create_view("snapshots")
    c.read_parquet(str(INDUSTRIES)).create_view("industry_history")
    c.read_parquet(str(DELIST_NOTICES)).create_view("notices")
    peers = c.sql("""SELECT p.date,p.code,substr(p.code,1,2) AS exchange,
      h.industry,h.asof,h.updateDate,h.effective_date,h.next_effective_date,
      p.price_1449,p.price_1420,p.amount_1449,p.volume_1449,p.high_1449,p.low_1449,
      s.preclose,s.isST,s.tradestatus,s.listing_age_sessions,s.reference_gap,s.return20_prior_adjusted
      FROM dates d JOIN prefix p USING(date) JOIN snapshots s USING(date,code)
      JOIN industry_history h ON p.code=h.code AND p.date>=h.effective_date
       AND(h.next_effective_date IS NULL OR p.date<h.next_effective_date)
      WHERE(p.code LIKE 'sh.60%' OR p.code LIKE 'sz.00%') AND s.isST=0 AND s.tradestatus=1
      AND s.listing_age_sessions>=20 AND NOT s.reference_gap AND NOT p.quote_outside_traded_range
      AND h.industry<>'' AND h.asof<=p.date AND h.updateDate<p.date
      AND date_diff('day',h.updateDate::DATE,p.date::DATE)<=370
      AND p.amount_1449>=30000000 AND p.volume_1449>0 AND s.preclose>0
      AND p.price_1449>0 AND p.price_1420>0
      AND abs(p.price_1449-round(p.price_1449,2))<=.0001
      AND abs(p.price_1420-round(p.price_1420,2))<=.0001
      AND NOT EXISTS(SELECT 1 FROM notices n WHERE n.code=p.code AND n.notice_date<p.date
        AND regexp_matches(n.title,'进入退市整理|退市整理期交易'))
      ORDER BY p.date,p.code""").df(); c.close()
    peers["price_1449"] = peers.price_1449.map(lambda p: quote_cents(p)/100)
    peers["price_1420"] = peers.price_1420.map(lambda p: quote_cents(p)/100)
    peers["return_1450"] = peers.price_1449/peers.preclose-1  # Legacy name; strictly 14:49.
    peers["return_last29"] = peers.price_1449/peers.price_1420-1
    peers["vwap_1449"] = peers.amount_1449/peers.volume_1449
    peers = peer_statistics(peers)
    pool = peers.loc[peers.peer_count.ge(10) & peers.price_1449.ge(5) & peers.amount_1449.ge(1e8)
        & peers.return_1450.between(.02, .06) & peers.return20_prior_adjusted.ge(0)
        & peers.return_last29.between(0, .01) & peers.price_1449.ge(peers.vwap_1449)].copy()
    pool = pool.loc[[decision_buyable(r.code, r.price_1449, r.preclose) for r in pool.itertuples()]].copy()
    high = select(pool, calendar)
    low = match(high.merge(pool, on=["date", "code"], validate="one_to_one"), pool, calendar)
    high["arm"], high["pair_id"] = "high", high.code; low["arm"] = "low"
    members = pd.concat([high, low], ignore_index=True); members["pair_id"] = members.date+":"+members.pair_id
    signals = members.merge(pool, on=["date", "code"], validate="one_to_one")
    signals["half"] = signals.date.str[:4]+np.where(signals.date.str[5:7].le("06"), "H1", "H2")
    signals["decision_shares"] = [fixed_quote_shares(r.code, r.price_1449, 20000) for r in signals.itertuples()]
    signals = signals.sort_values(["date", "arm", "daily_rank", "code"]).reset_index(drop=True)
    if len(signals) != len(members) or signals.duplicated(["date", "code"]).any():
        raise ValueError("Selection lost or duplicated stock identities")
    for name, frame in (("peers", peers), ("visible_pool", pool), ("signals", signals)):
        frame.to_parquet(output / (name+".parquet"), index=False, compression="zstd")
    result = {"rule_commit": RULE_COMMIT, "peer_rows": len(peers), "pool_rows": len(pool),
        "strong_pool_rows": int(pool.strong_sector.sum()), "candidates": len(high), "controls": len(low),
        "by_half": signals.groupby(["half", "arm"]).agg(rows=("code", "size"), days=("date", "nunique")).reset_index().to_dict("records"),
        "signals_sha256": sha(output / "signals.parquet"), "pool_sha256": sha(output / "visible_pool.parquet"),
        "peers_sha256": sha(output / "peers.parquet"), "source_manifest_sha256": sha(output / "source_manifest.json"),
        "new_holding_results_read": False, "new_2026_prices_read": False}
    save_json(output / "input_report.json", result)
    return result


def execute(output: Path = ROOT) -> dict:
    return execute_short(output, horizons=(1, 2), rule_commit=RULE_COMMIT)


def compare(output: Path = ROOT) -> dict:
    return compare_short(output, horizons=(1, 2), rule_commit=RULE_COMMIT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze", "execute", "compare"])
    args = parser.parse_args()
    print(json.dumps(globals()[args.stage](), ensure_ascii=False, indent=2))
