"""Evaluate all selected late decliners before inspecting sparse sector pairs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .corporate_cash import save_json, sha
from .risk_removal_1449 import decision_buyable
from .quote_precision import fixed_quote_shares
from .turnover_reference import CALENDAR

ROOT = Path("data/research/sector_resilience_1449")
RULE_COMMIT = "2516113"
INDUSTRIES = Path("data/research/industry_intervals.parquet")
DELIST_NOTICES = Path("data/research/risk_removal_1449/notice_index.parquet")


def sector_statistics(peers: pd.DataFrame) -> pd.DataFrame:
    if peers.duplicated(["date", "code"]).any():
        raise ValueError("Overlapping historical industry memberships")
    frame = peers.copy()
    tail = frame.return_last29.clip(-.02, .02)
    prior = frame.return20_prior_adjusted.clip(-.10, .10)
    frame["tail_clip"], frame["prior20_clip"] = tail, prior
    industry = frame.groupby(["date", "industry"])
    market = frame.groupby("date")
    frame["peer_count"] = industry.code.transform("size") - 1
    frame["market_peer_count"] = market.code.transform("size") - 1
    denominator = frame.peer_count.where(frame.peer_count.gt(0))
    frame["sector_tail"] = (industry.tail_clip.transform("sum") - tail) / denominator
    frame["sector_prior20"] = (industry.prior20_clip.transform("sum") - prior) / denominator
    frame["market_tail"] = (market.tail_clip.transform("sum") - tail) / frame.market_peer_count.where(frame.market_peer_count.gt(0))
    frame["sector_residual_tail"] = frame.sector_tail - frame.market_tail
    return frame


def visible_inputs(calendar: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    sources = [*sorted(Path("data/research/minute_prefix_1449").glob("202[45]/*.parquet")),
               *sorted(Path("data/research/market_snapshots_ci").glob("*.parquet")), INDUSTRIES, DELIST_NOTICES]
    c = duckdb.connect()
    c.execute("SET threads=4")
    c.read_parquet("data/research/minute_prefix_1449/202[45]/*.parquet").create_view("prefix")
    c.read_parquet("data/research/market_snapshots_ci/*.parquet").create_view("snapshots")
    c.read_parquet(str(INDUSTRIES)).create_view("industry_history")
    c.read_parquet(str(DELIST_NOTICES)).create_view("risk_notices")
    c.register("allowed_dates", pd.DataFrame({"date": calendar[:-10]}))
    peers = c.execute("""SELECT p.date,p.code,h.industry,h.asof,h.updateDate,h.effective_date,
        h.next_effective_date,round(p.price_1449,2) AS price_1449,round(p.price_1420,2) AS price_1420,
        p.amount_1449,p.volume_1449,round(p.high_1449,2) AS high_1449,round(p.low_1449,2) AS low_1449,
        s.preclose,s.isST,s.listing_age_sessions,s.reference_gap,s.return20_prior_adjusted
        FROM prefix p JOIN allowed_dates d USING(date) JOIN snapshots s USING(date,code)
        JOIN industry_history h ON h.code=p.code AND p.date>=h.effective_date
         AND(h.next_effective_date IS NULL OR p.date<h.next_effective_date)
        WHERE(p.code LIKE 'sh.60%' OR p.code LIKE 'sz.00%') AND s.isST=0
        AND s.listing_age_sessions>=20 AND NOT s.reference_gap AND NOT p.quote_outside_traded_range
        AND h.industry<>'' AND p.amount_1449>=30000000 AND p.volume_1449>0
        AND isfinite(s.return20_prior_adjusted) AND s.preclose>0 AND p.price_1449>0 AND p.price_1420>0
        AND abs(p.price_1449-round(p.price_1449,2))<=.0001
        AND abs(p.price_1420-round(p.price_1420,2))<=.0001
        AND NOT EXISTS(SELECT 1 FROM risk_notices n WHERE n.code=p.code AND n.notice_date<p.date
            AND regexp_matches(n.title,'进入退市整理|退市整理期交易'))
        ORDER BY p.date,p.code""").df()
    c.close()
    if peers["asof"].gt(peers.date).any() or peers.updateDate.ge(peers.date).any():
        raise ValueError("Industry information is not available before the decision")
    peers["return_last29"] = peers.price_1449 / peers.price_1420 - 1
    peers["return_1450"] = peers.price_1449 / peers.preclose - 1  # Legacy name, 14:49 only.
    peers["return_to_1420"] = peers.price_1420 / peers.preclose - 1
    peers["position_1449"] = (peers.price_1449 - peers.low_1449) / (peers.high_1449 - peers.low_1449).where(peers.high_1449.gt(peers.low_1449))
    peers = sector_statistics(peers)
    features = peers.loc[peers.peer_count.ge(10) & peers.price_1449.ge(5)
        & peers.amount_1449.between(1e8, 1e9) & peers.return20_prior_adjusted.between(-.10, .10)
        & peers.return_1450.between(-.03, .03) & peers.return_last29.between(-.01, -.003)
        & peers.position_1449.between(0, 1)].copy()
    features = features.loc[[decision_buyable(r.code, r.price_1449, r.preclose) for r in features.itertuples()]]
    return features, peers, sources


def choose(strong: pd.DataFrame, calendar: list[str]) -> pd.DataFrame:
    positions, last = {d: i for i, d in enumerate(calendar)}, {}
    ranked = strong.copy()
    ranked["selection_order"] = [hashlib.md5(("sector-tail-resilience-v1" + d + code).encode()).hexdigest()
                                  for d, code in zip(ranked.date, ranked.code)]
    selected = []
    for date, group in ranked.groupby("date", sort=True):
        count = 0
        for row in group.sort_values(["selection_order", "code"]).to_dict("records"):
            if positions[date] - last.get(row["code"], -1000) <= 5:
                continue
            count += 1
            row["daily_rank"] = count
            selected.append(row)
            last[row["code"]] = positions[date]
            if count == 5:
                break
    return pd.DataFrame(selected)


def match(chosen: pd.DataFrame, weak: pd.DataFrame, calendar: list[str]) -> pd.DataFrame:
    positions, last = {d: i for i, d in enumerate(calendar)}, {}
    by_date = {date: p.set_index("code", drop=False) for date, p in weak.groupby("date")}
    controls = []
    for date, group in chosen.groupby("date", sort=True):
        available = by_date.get(date, weak.iloc[:0].set_index("code", drop=False)).copy()
        for row in group.sort_values("daily_rank").itertuples():
            fresh = available.loc[[positions[date] - last.get(code, -1000) > 5 for code in available.index]]
            prior = (fresh.return20_prior_adjusted - row.return20_prior_adjusted).abs()
            early = (fresh.return_to_1420 - row.return_to_1420).abs()
            tail = (fresh.return_last29 - row.return_last29).abs()
            sector = (fresh.sector_prior20 - row.sector_prior20).abs()
            amount, price = fresh.amount_1449 / row.amount_1449, fresh.price_1449 / row.price_1449
            position = (fresh.position_1449 - row.position_1449).abs()
            possible = fresh.loc[prior.le(.03) & early.le(.005) & tail.le(.002) & sector.le(.03)
                & amount.between(.5, 2) & price.between(.5, 2) & position.le(.30)].copy()
            if possible.empty:
                continue
            distance = prior / .03 + early / .005 + tail / .002 + sector / .03 + position / .30
            distance += (np.log(amount).abs() + np.log(price).abs()) / np.log(2)
            possible["distance"] = distance.loc[possible.index]
            control = possible.reset_index(drop=True).sort_values(["distance", "code"]).iloc[0].to_dict()
            control["pair_id"], control["daily_rank"] = row.code, row.daily_rank
            controls.append(control)
            available = available.drop(index=control["code"])
            last[control["code"]] = positions[date]
    return pd.DataFrame(controls, columns=[*weak.columns, "distance", "pair_id", "daily_rank"])


def freeze(output: Path = ROOT) -> dict:
    if (output / "repriced.parquet").exists():
        raise ValueError("Do not replace lists after reading their outcomes")
    output.mkdir(parents=True, exist_ok=True)
    table = pd.read_parquet(CALENDAR)
    calendar = sorted(table.loc[table.is_trading_day.eq("1")
        & table.calendar_date.between("2024-01-01", "2025-12-31"), "calendar_date"])
    features, peers, sources = visible_inputs(calendar)
    high = choose(features.loc[features.sector_residual_tail.ge(.001)], calendar)
    low = match(high, features.loc[features.sector_residual_tail.le(-.001)], calendar)
    high["arm"], high["pair_id"] = "high", high.code
    low["arm"] = "low"
    signals = pd.concat([high, low], ignore_index=True).sort_values(["date", "arm", "daily_rank", "code"])
    signals["pair_id"] = signals.date + ":" + signals.pair_id
    signals["half"] = signals.date.str[:4] + np.where(signals.date.str[5:7].le("06"), "H1", "H2")
    signals["decision_shares"] = [fixed_quote_shares(r.code, r.price_1449, 20000) for r in signals.itertuples()]
    if signals.duplicated(["date", "code"]).any():
        raise ValueError("Duplicate event identity")
    for name, frame in (("peers", peers), ("features", features), ("signals", signals)):
        frame.to_parquet(output / f"{name}.parquet", index=False, compression="zstd")
    save_json(output / "feature_sources.json", {str(path): sha(path) for path in sources})
    report = {"rule_commit": RULE_COMMIT, "peers": len(peers), "eligible_declines": len(features),
        "candidates": len(high), "controls": len(low), "unmatched": len(high) - len(low),
        "by_half": signals.groupby(["half", "arm"]).agg(rows=("code", "size"), dates=("date", "nunique"))
            .reset_index().to_dict("records"), "signals_sha256": sha(output / "signals.parquet"),
        "peers_sha256": sha(output / "peers.parquet"), "features_sha256": sha(output / "features.parquet"),
        "feature_sources_sha256": sha(output / "feature_sources.json"), "calendar_sha256": sha(CALENDAR),
        "new_execution_outcomes_read": False, "holdout_prices_read": False}
    save_json(output / "input_report.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(freeze(), ensure_ascii=False, indent=2))
