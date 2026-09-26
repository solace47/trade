"""Complete the economic test of reopened limit-down candidates."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .corporate_cash import save_json, sha
from .hf_outcomes import _limit_price
from .quote_precision import quote_cents, fixed_quote_shares
from .risk_removal_1449 import decision_buyable
from .turnover_reference import CALENDAR

ROOT = Path("data/research/limit_down_recovery_1449")
RULE_COMMIT = "d7481e5"
DELIST_NOTICES = Path("data/research/risk_removal_1449/notice_index.parquet")


def limit_state(price: float, low: float, preclose: float) -> tuple[bool, bool, float]:
    current, minimum = quote_cents(price), quote_cents(low)
    reference = quote_cents(preclose) / 100
    limit = quote_cents(_limit_price(reference, .1, False))
    return minimum == limit and current >= limit + 2, minimum >= limit + 1, limit / 100


def inputs(calendar: list[str]) -> tuple[pd.DataFrame, list[Path]]:
    sources = [*sorted(Path("data/research/minute_prefix_1449").glob("202[45]/*.parquet")),
        *sorted(Path("data/research/market_snapshots_ci").glob("*.parquet")), DELIST_NOTICES]
    c = duckdb.connect()
    c.execute("SET threads=4")
    c.read_parquet("data/research/minute_prefix_1449/202[45]/*.parquet").create_view("prefix")
    c.read_parquet("data/research/market_snapshots_ci/*.parquet").create_view("snapshots")
    c.read_parquet(str(DELIST_NOTICES)).create_view("notices")
    c.register("allowed_dates", pd.DataFrame({"date": calendar[:-10]}))
    frame = c.sql("""SELECT p.date,p.code,substr(p.code,1,2) AS exchange,
        round(p.price_1449,2) AS price_1449,round(p.low_1449,2) AS low_1449,
        round(p.high_1449,2) AS high_1449,round(s.preclose,2) AS preclose,
        p.amount_1449,p.volume_1449,p.return_last29 AS late_return,
        s.return20_prior_adjusted AS prior20_return,s.isST,s.reference_gap,s.listing_age_sessions
        FROM prefix p JOIN snapshots s USING(date,code) JOIN allowed_dates d USING(date)
        WHERE (p.code LIKE 'sh.60%' OR p.code LIKE 'sz.00%') AND s.tradestatus=1 AND s.isST=0
        AND s.listing_age_sessions>=20 AND NOT s.reference_gap AND NOT p.quote_outside_traded_range
        AND s.preclose>0 AND p.price_1449>=5 AND p.amount_1449>=30000000 AND p.volume_1449>0
        AND s.return20_prior_adjusted BETWEEN -.20 AND .20 AND p.return_last29 BETWEEN -.03 AND .03
        AND p.high_1449>p.low_1449
        AND abs(p.price_1449-round(p.price_1449,2))<=.0001
        AND abs(p.low_1449-round(p.low_1449,2))<=.0001
        AND abs(p.high_1449-round(p.high_1449,2))<=.0001
        AND abs(s.preclose-round(s.preclose,2))<=.0001
        AND NOT EXISTS(SELECT 1 FROM notices n WHERE n.code=p.code AND n.notice_date<p.date
            AND regexp_matches(n.title,'进入退市整理|退市整理期交易'))
        ORDER BY p.date,p.code""").df()
    c.close()
    frame["day_return"] = frame.price_1449 / frame.preclose - 1
    frame["position"] = (frame.price_1449 - frame.low_1449) / (frame.high_1449 - frame.low_1449)
    frame = frame.loc[frame.day_return.between(-.098, -.02) & frame.position.between(0, 1)].copy()
    frame = frame.loc[[decision_buyable(r.code, r.price_1449, r.preclose) for r in frame.itertuples()]]
    states = [limit_state(r.price_1449, r.low_1449, r.preclose) for r in frame.itertuples()]
    frame[["reopened", "untouched", "down_limit"]] = pd.DataFrame(states, index=frame.index)
    frame["limit_distance"] = (frame.price_1449 - frame.down_limit) / frame.preclose
    if frame.duplicated(["date", "code"]).any():
        raise ValueError("Duplicated visible stock-day")
    if not np.isfinite(frame[["price_1449", "amount_1449", "late_return", "prior20_return", "day_return", "position"]]).all().all():
        raise ValueError("Nonfinite input cannot become a recovery candidate")
    return frame, sources


def choose(events: pd.DataFrame, calendar: list[str]) -> pd.DataFrame:
    positions, last, selected = {d: i for i, d in enumerate(calendar)}, {}, []
    for date, group in events.groupby("date", sort=True):
        rank = 0
        for row in group.sort_values(["limit_distance", "code"]).to_dict("records"):
            if positions[date] - last.get(row["code"], -1000) <= 5:
                continue
            rank += 1
            row["daily_rank"] = rank
            selected.append(row)
            last[row["code"]] = positions[date]
            if rank == 5:
                break
    return pd.DataFrame(selected, columns=[*events.columns, "daily_rank"])


def match(chosen: pd.DataFrame, untouched: pd.DataFrame, calendar: list[str]) -> pd.DataFrame:
    positions, last, controls = {d: i for i, d in enumerate(calendar)}, {}, []
    by_date = {d: g.set_index("code", drop=False) for d, g in untouched.groupby("date")}
    for date, group in chosen.groupby("date", sort=True):
        available = by_date.get(date, untouched.iloc[:0].set_index("code", drop=False)).copy()
        for row in group.sort_values("daily_rank").itertuples():
            same = available.loc[available.exchange.eq(row.exchange)]
            fresh = same.loc[[positions[date] - last.get(code, -1000) > 5 for code in same.index]]
            day = (fresh.day_return - row.day_return).abs()
            tail = (fresh.late_return - row.late_return).abs()
            prior = (fresh.prior20_return - row.prior20_return).abs()
            location = (fresh.position - row.position).abs()
            amount, price = fresh.amount_1449 / row.amount_1449, fresh.price_1449 / row.price_1449
            possible = fresh.loc[day.le(.01) & tail.le(.015) & prior.le(.10)
                & amount.between(.25, 4) & price.between(.3, 3) & location.le(.25)].copy()
            if possible.empty:
                continue
            distance = day / .01 + tail / .015 + prior / .10 + location / .25
            distance += np.log(amount).abs() / np.log(4) + np.log(price).abs() / np.log(3)
            possible["distance"] = distance.loc[possible.index]
            control = possible.reset_index(drop=True).sort_values(["distance", "code"]).iloc[0].to_dict()
            control["pair_id"], control["daily_rank"] = row.code, row.daily_rank
            controls.append(control)
            available = available.drop(index=control["code"])
            last[control["code"]] = positions[date]
    return pd.DataFrame(controls, columns=[*untouched.columns, "distance", "pair_id", "daily_rank"])


def freeze(output: Path = ROOT) -> dict:
    if (output / "repriced.parquet").exists():
        raise ValueError("Cannot replace a list after reading its holding outcomes")
    output.mkdir(parents=True, exist_ok=True)
    table = pd.read_parquet(CALENDAR)
    calendar = sorted(table.loc[table.is_trading_day.eq("1")
        & table.calendar_date.between("2024-01-01", "2025-12-31"), "calendar_date"])
    pool, sources = inputs(calendar)
    high = choose(pool.loc[pool.reopened], calendar)
    low = match(high, pool.loc[pool.untouched], calendar)
    high["arm"], high["pair_id"] = "high", high.code
    low["arm"] = "low"
    signals = pd.concat([high, low], ignore_index=True).sort_values(["date", "arm", "daily_rank", "code"])
    signals["pair_id"] = signals.date + ":" + signals.pair_id
    signals["half"] = signals.date.str[:4] + np.where(signals.date.str[5:7].le("06"), "H1", "H2")
    signals["decision_shares"] = [fixed_quote_shares(r.code, r.price_1449, 20000) for r in signals.itertuples()]
    if signals.empty or signals.duplicated(["date", "code"]).any():
        raise ValueError("Missing or duplicated fixed signal identities")
    paired = signals.loc[signals.arm.eq("high")].merge(signals.loc[signals.arm.eq("low")],
        on=["date", "pair_id", "half"], suffixes=("_high", "_low"), validate="one_to_one")
    balance = []
    for half, part in paired.groupby("half"):
        balance.append({"half": half, "pairs": len(part), **{
            col + "_median_abs_gap": float((part[col + "_high"] - part[col + "_low"]).abs().median())
            for col in ("day_return", "late_return", "prior20_return", "position")}})
    pool.to_parquet(output / "visible_inputs.parquet", index=False, compression="zstd")
    signals.to_parquet(output / "signals.parquet", index=False, compression="zstd")
    save_json(output / "source_manifest.json", {str(path): sha(path) for path in sources})
    report = {"rule_commit": RULE_COMMIT, "eligible_stock_days": len(pool),
        "reopened_events": int(pool.reopened.sum()), "untouched_controls": int(pool.untouched.sum()),
        "candidates": len(high), "controls": len(low), "unmatched": len(high) - len(low),
        "by_half": signals.groupby(["half", "arm"]).agg(rows=("code", "size"), dates=("date", "nunique"))
            .reset_index().to_dict("records"), "residual_balance": balance,
        "signals_sha256": sha(output / "signals.parquet"), "features_sha256": sha(output / "visible_inputs.parquet"),
        "source_manifest_sha256": sha(output / "source_manifest.json"), "calendar_sha256": sha(CALENDAR),
        "primary": "T1_max_15bps_or_half_cent_per_leg", "new_test_returns_read": False, "holdout_prices_read": False}
    save_json(output / "input_report.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(freeze(), ensure_ascii=False, indent=2))
