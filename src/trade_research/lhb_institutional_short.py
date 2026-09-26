"""Freeze a next-day tail entry after disclosed institution-seat net buying."""
from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .corporate_cash import save_json, sha
from .exchange_public_events import _check_saved_sse, trading_dates
from .lhb_institutional_seats import DAILY_REASONS, _seat_amount
from .limit_down_recovery_1449 import DELIST_NOTICES
from .quote_precision import fixed_quote_shares, quote_cents
from .reference_gain_eval import DAILY
from .risk_removal_1449 import decision_buyable
from .turnover_reference import CALENDAR

ROOT = Path("data/research/lhb_institutional_short")
ARCHIVE = Path("data/research/lhb/sse_daily")
RULE_COMMIT = "9b10b98"


def money_cents(value: str) -> int:
    amount = Decimal(value) * 100
    if not amount.is_finite() or amount < 0 or amount != amount.to_integral_value():
        raise ValueError("A disclosed amount must be nonnegative whole cents")
    return int(amount)


def institution_cents(item: dict, side: str) -> int:
    _seat_amount(item, side)  # Validate the original aligned one-to-five lists.
    return sum(money_cents(value) for name, value in
        zip(item[f"branchName{side}"].split(","), item[f"branchTxAmt{side}"].split(","))
        if name.strip() == "机构专用")


def collapse_daily(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A multi-reason listing is one event only if its monetary facts agree."""
    keys = ["date", "trade_date", "code"]
    money = ["disclosed_turnover_cents", "institution_buy_cents", "institution_sell_cents"]
    records, ambiguous = [], []
    for key, group in rows.groupby(keys, sort=True):
        record = dict(zip(keys, key))
        record["reasons"] = ",".join(sorted(group.ref_type))
        record["source_reason_count"] = len(group)
        if group[money].nunique().gt(1).any():
            ambiguous.append(record)
            continue
        record.update({field: int(group[field].iloc[0]) for field in money})
        record["institution_net_cents"] = record["institution_buy_cents"]-record["institution_sell_cents"]
        record["net_fraction"] = record["institution_net_cents"]/record["disclosed_turnover_cents"]
        records.append(record)
    return pd.DataFrame(records), pd.DataFrame(ambiguous, columns=keys+["reasons", "source_reason_count"])


def source(calendar: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    records = []
    paths = []
    for prior, current in zip(calendar[:-11], calendar[1:-10]):
        path = ARCHIVE / ("sse_"+prior.replace("-", "")+".json")
        day = prior.replace("-", "")
        archived = _check_saved_sse(path, day)
        paths.append(path)
        for item in archived["main"]:
            if (item.get("secType") != "A" or not item["secCode"].startswith("60")
                    or item["refType"] not in DAILY_REASONS):
                continue
            if item["abnormalStart"] != day or item["abnormalEnd"] != day:
                raise ValueError("A purported one-day disclosure spans other dates")
            total = money_cents(item["secTxAmount"])
            buy, sell = institution_cents(item, "B"), institution_cents(item, "S")
            if total <= 0 or max(buy, sell) > total:
                raise ValueError("Institution-side amount exceeds disclosed daily turnover")
            records.append({"date": current, "trade_date": prior,
                "code": "sh."+item["secCode"], "ref_type": item["refType"],
                "disclosed_turnover_cents": total, "institution_buy_cents": buy,
                "institution_sell_cents": sell})
    raw = pd.DataFrame(records)
    if raw.empty or raw.duplicated(["date", "code", "ref_type"]).any():
        raise ValueError("Missing or duplicated source identities")
    events, ambiguous = collapse_daily(raw)
    report = {"archived_trade_days": len(paths), "raw_daily_reason_rows": len(raw),
        "unique_stock_days": len(events), "ambiguous_stock_days": len(ambiguous),
        "identical_multi_reason_stock_days": int(events.source_reason_count.gt(1).sum()),
        "positive_net_stock_days": int(events.institution_net_cents.gt(0).sum()),
        "source_sha256": {str(path): sha(path) for path in paths}}
    return events, ambiguous, report


def select(events: pd.DataFrame, calendar: list[str]) -> pd.DataFrame:
    positions = {date: i for i, date in enumerate(calendar)}
    last, records = {}, []
    for date, group in events.groupby("date", sort=True):
        rank = 0
        for row in group.sort_values(["net_fraction", "code"], ascending=[False, True]).itertuples():
            if positions[date]-last.get(row.code, -1000) <= 5:
                continue
            rank += 1
            records.append({"date": date, "code": row.code, "daily_rank": rank})
            last[row.code] = positions[date]
            if rank == 5:
                break
    return pd.DataFrame(records, columns=["date", "code", "daily_rank"])


def match(high: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    records = []
    controls = pool.loc[~pool.positive_net]
    groups = {date: frame.set_index("code", drop=False) for date, frame in controls.groupby("date")}
    for date, group in high.groupby("date", sort=True):
        available = groups.get(date, pool.iloc[:0]).copy()
        for row in group.sort_values("daily_rank").itertuples():
            if available.empty:
                break
            ar, pr = available.amount_1449/row.amount_1449, available.price_1449/row.price_1449
            prior20 = (available.return20_prior_adjusted-row.return20_prior_adjusted).abs()
            today = (available.return_1450-row.return_1450).abs()
            yesterday = (available.prior_day_return-row.prior_day_return).abs()
            mask = ar.between(.5, 2) & pr.between(.5, 2) & prior20.le(.05) & today.le(.02) & yesterday.le(.02)
            choices = available.loc[mask].copy()
            if choices.empty:
                continue
            choices["distance"] = (prior20/.05+today/.02+yesterday/.02+
                np.abs(np.log(ar))/np.log(2)+np.abs(np.log(pr))/np.log(2)).loc[mask]
            peer = choices.reset_index(drop=True).sort_values(["distance", "code"]).iloc[0]
            records.append({"date": date, "code": peer.code, "daily_rank": row.daily_rank,
                "pair_id": row.code, "distance": peer.distance})
            available = available.drop(index=peer.code)
    return pd.DataFrame(records, columns=["date", "code", "daily_rank", "pair_id", "distance"])


def freeze(output: Path = ROOT) -> dict:
    if (output / "input_report.json").exists():
        raise ValueError("Cannot overwrite frozen institutional short selections")
    output.mkdir(parents=True, exist_ok=True)
    calendar = trading_dates(CALENDAR, "2024-01-01", "2025-12-31")
    events, ambiguous, source_report = source(calendar)
    events.to_parquet(output / "source_events.parquet", index=False, compression="zstd")
    ambiguous.to_parquet(output / "ambiguous_events.parquet", index=False, compression="zstd")
    schedule = pd.DataFrame({"date": calendar[1:-10], "trade_date": calendar[:-11]})
    sources = [*sorted(Path("data/research/minute_prefix_1449").glob("202[45]/*.parquet")),
        *sorted(Path("data/research/market_snapshots_ci").glob("*.parquet")),
        *sorted(DAILY.glob("sh_60*.parquet")), DELIST_NOTICES, CALENDAR]
    save_json(output / "source_manifest.json", {"rule_commit": RULE_COMMIT,
        "source_sha256": {**source_report.pop("source_sha256"), **{str(p): sha(p) for p in sources}},
        "first_signal": calendar[1], "last_signal": calendar[-11],
        "holding_prices_read": False, "new_2026_holding_prices_read": False,
        "incidental_browser_exposure": "SSE default 2026-09-24 daily-LHB first five rows; not used in selection or returns"})
    c = duckdb.connect(); c.execute("SET threads=4")
    c.register("schedule", schedule)
    c.read_parquet("data/research/minute_prefix_1449/202[45]/*.parquet").create_view("prefix")
    c.read_parquet("data/research/market_snapshots_ci/*.parquet").create_view("snapshots")
    c.read_parquet(str(DAILY / "sh_60*.parquet")).create_view("daily")
    c.read_parquet(str(DELIST_NOTICES)).create_view("notices")
    # Read only prior-session daily prices, never current final daily prices.
    past = c.sql("""SELECT code,d.date AS trade_date,close AS prior_close,preclose AS prior_reference
      FROM daily d JOIN schedule s ON d.date=s.trade_date
      WHERE d.tradestatus=1 AND d.isST=0""").df()
    frame = c.sql("""SELECT p.date,p.code,p.price_1449,p.high_1449,p.low_1449,
      p.amount_1449,p.volume_1449,s.preclose,s.isST,s.reference_gap,
      s.listing_age_sessions,s.return20_prior_adjusted,d.trade_date
      FROM prefix p JOIN snapshots s USING(date,code) JOIN schedule d USING(date)
      WHERE p.code LIKE 'sh.60%' AND s.isST=0 AND s.tradestatus=1
      AND s.listing_age_sessions>=20 AND NOT s.reference_gap AND NOT p.quote_outside_traded_range
      AND p.price_1449>=5 AND p.amount_1449>=30000000 AND p.volume_1449>0 AND s.preclose>0
      AND NOT EXISTS(SELECT 1 FROM notices n WHERE n.code=p.code AND n.notice_date<p.date
        AND regexp_matches(n.title,'进入退市整理|退市整理期交易')) ORDER BY p.date,p.code""").df()
    c.close()
    frame = frame.merge(past, on=["code", "trade_date"], validate="many_to_one")
    frame = frame.loc[~frame.set_index(["date", "code"]).index.isin(
        ambiguous.set_index(["date", "code"]).index)].copy()
    initial = len(frame)
    frame = frame.loc[[decision_buyable(r.code, r.price_1449, r.preclose) for r in frame.itertuples()]].copy()
    frame["price_1449"] = frame.price_1449.map(lambda x: quote_cents(x)/100)
    frame["return_1450"] = frame.price_1449/frame.preclose-1
    frame["prior_day_return"] = frame.prior_close/frame.prior_reference-1
    frame = frame.merge(events, on=["date", "trade_date", "code"], how="left", validate="one_to_one")
    frame["positive_net"] = frame.institution_net_cents.gt(0)
    frame["half"] = frame.date.str[:4]+np.where(frame.date.str[5:7].le("06"), "H1", "H2")
    frame = frame.sort_values(["date", "code"]).reset_index(drop=True)
    if frame.duplicated(["date", "code"]).any():
        raise ValueError("Duplicate visible stock-day")
    frame.to_parquet(output / "visible_pool.parquet", index=False, compression="zstd")
    high = select(frame.loc[frame.positive_net], calendar)
    low = match(high.merge(frame, on=["date", "code"], validate="one_to_one"), frame)
    high["arm"], high["pair_id"] = "high", high.code
    low["arm"] = "low"
    members = pd.concat([high, low], ignore_index=True)
    members["pair_id"] = members.date+":"+members.pair_id
    signals = members.merge(frame, on=["date", "code"], validate="one_to_one")
    signals["decision_shares"] = [fixed_quote_shares(r.code, r.price_1449, 20000) for r in signals.itertuples()]
    signals = signals.sort_values(["date", "arm", "daily_rank", "code"]).reset_index(drop=True)
    if len(signals) != len(members):
        raise ValueError("Selection identities changed")
    signals.to_parquet(output / "signals.parquet", index=False, compression="zstd")
    result = {"rule_commit": RULE_COMMIT, "source": source_report, "before_buyable_rows": initial,
        "visible_pool_rows": len(frame), "eligible_positive_net": int(frame.positive_net.sum()),
        "candidates": len(high), "controls": len(low),
        "by_half": signals.groupby(["half", "arm"]).agg(rows=("code", "size"), days=("date", "nunique")).reset_index().to_dict("records"),
        "signals_sha256": sha(output / "signals.parquet"), "pool_sha256": sha(output / "visible_pool.parquet"),
        "source_events_sha256": sha(output / "source_events.parquet"),
        "source_manifest_sha256": sha(output / "source_manifest.json"),
        "new_holding_prices_read": False, "new_2026_holding_prices_read": False}
    save_json(output / "input_report.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(freeze(), ensure_ascii=False, indent=2))
