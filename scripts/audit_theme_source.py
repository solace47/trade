"""Audit cached historical metadata without using any subsequent return."""
import json
from pathlib import Path
import re

import duckdb
import numpy as np
import pandas as pd

from trade_research.corporate_cash import DAILY, save_json, sha

ROOT = Path("data/research/theme_source_probe")


def run():
    out = ROOT / "audit"
    out.mkdir(exist_ok=True)
    if (out / "report.json").exists():
        raise ValueError("Preserve inspected source audit")
    complete = json.loads((ROOT / "complete_50/report.json").read_text())
    zz = json.loads((ROOT / "zzshare/curl_initial/report.json").read_text())
    dates = [item["date"] for item in complete["counts"]]
    raw_files = {}
    limits, members, reasons, relations, events = [], [], [], [], []
    headers = []
    for record in complete["requests"]:
        file = ROOT / "complete_50" / record["body"]
        assert sha(file) == record["body_sha256"]
        raw_files[str(file)] = sha(file)
        receipt = json.loads((ROOT / "complete_50" / record["receipt"]).read_text())
        assert sha(ROOT / "complete_50" / record["receipt"]) == record["receipt_sha256"]
        payload = json.loads(file.read_text())
        day, kind = receipt["date"], receipt["kind"]
        assert payload.get("Day", payload.get("day")) in [day, [day]]
        headers.append(dict(file=file.name, Title=payload.get("Title"), ZB=payload.get("ZB")))
        for i, row in enumerate(payload["list"]):
            common = dict(date=day, stock_code=row[0], source_file=str(file), raw_row=i)
            if kind == "limit_up":
                limits.append(dict(common, timestamp=row[6], source_amount=row[13], source_reason=row[16]))
            elif kind.startswith("members_"):
                members.append(dict(common, plate_code=kind.removeprefix("members_"),
                                    source_price=row[5], source_amount=row[7], raw_tags=row[4]))
    future_columns = set()
    for record in zz["responses"]:
        file = ROOT / "zzshare/curl_initial" / (record["date"] + "_" + record["kind"] + ".body")
        assert sha(file) == record["body_sha256"]
        raw_files[str(file)] = sha(file)
        data = json.loads(file.read_text())["data"]
        s = record["date"]
        day = s[:4] + "-" + s[4:6] + "-" + s[6:]
        if record["kind"] == "reason":
            for i, row in enumerate(data):
                assert row["date"] == day
                reasons.append(dict(date=day, stock_code=row["stock_code"], name=row["stock_name"],
                                    reason=row["reason"], source_file=str(file), raw_row=i))
        else:
            for kind in ["plate_stocks", "plate_stocks_zb", "plate_stocks_bx"]:
                for plate, stocks in data[kind].items():
                    for i, row in enumerate(stocks):
                        future_columns.update(k for k in row if k.startswith("next_"))
                        # Whitelist only identities. Never copy next_* or opaque scores.
                        relations.append(dict(date=day, stock_code=row["stock_code"], plate_code=plate,
                                              source_category=kind, source_file=str(file), raw_row=i))
    for day in dates:
        file = ROOT / "curl_initial" / (day + "_events.body")
        payload = json.loads(file.read_text())
        assert payload["date"] == day
        raw_files[str(file)] = sha(file)
        for i, row in enumerate(payload["List"]):
            stamp = pd.Timestamp(row["TimeMin"], unit="s", tz="UTC").tz_convert("Asia/Shanghai")
            assert stamp.strftime("%Y-%m-%d") == day
            events.append(dict(date=day, event_clock=stamp.strftime("%H:%M:%S"), tag=row["TagName"],
                               detail=row["Detail"], source_file=str(file), raw_row=i))
    frames = {name: pd.DataFrame(value) for name, value in dict(limits=limits, members=members, reasons=reasons,
                                                               relations=relations, events=events).items()}
    for name in ["limits", "members", "reasons", "relations"]:
        frame = frames[name]
        frame["code"] = frame.stock_code.map(lambda s: ("sh." if s.startswith("6") else "sz.") + s
                                               if s.startswith(("0", "3", "6")) else "unsupported." + s)
    keys = pd.concat([frames[n][["date", "code"]] for n in ["limits", "members", "reasons", "relations"]]).drop_duplicates()
    c = duckdb.connect()
    c.register("keys", keys)
    d = c.execute("""select k.*,b.close,b.preclose,b.high,b.amount,b.isST,b.tradestatus,b.adjustflag from keys k
        left join (select * from read_parquet(?) where date in (select distinct date from keys)) b using(date,code)
        order by date,code""", [str(DAILY / "*.parquet")]).df()
    basics = pd.read_parquet("data/baostock/market_2020_2026/metadata/stock_basic.parquet", columns=["code", "ipoDate", "outDate"])
    d = d.merge(basics, on="code", how="left", validate="many_to_one")
    for name in ["close", "preclose", "high", "amount", "isST", "tradestatus", "adjustflag"]:
        d[name] = pd.to_numeric(d[name], errors="coerce").astype(float)
    rate = np.where(d.code.str.startswith(("sz.30", "sh.688")).to_numpy(dtype=bool), 20, np.where(d.isST.eq(1), 5, 10))
    d["ordinary_upper"] = ((np.rint(d.preclose.fillna(0) * 100) * (100 + rate) + 50) // 100) / 100
    d["closed_at_ordinary_limit"] = d.close.sub(d.ordinary_upper).abs().le(.005)
    d["touched_ordinary_limit"] = d.high.ge(d.ordinary_upper - .005)
    # IPO exceptions are not inferred as failures of the ordinary limit formula.
    d["listing_after_observation"] = d.ipoDate.gt(d.date).fillna(False)
    d.to_parquet(out / "daily_inputs.parquet", index=False, compression="zstd")
    per_day = []
    for day in dates:
        l = frames["limits"].query("date == @day").merge(d, on=["date", "code"], validate="one_to_one")
        m = frames["members"].query("date == @day").merge(d, on=["date", "code"], validate="one_to_one")
        r = frames["reasons"].query("date == @day")
        reason_stocks = r[["date", "code"]].drop_duplicates().merge(d, on=["date", "code"], validate="one_to_one")
        tradable = m.tradestatus.eq(1) & m.close.gt(0)
        mismatch = tradable & m.source_price.sub(m.close).abs().gt(.005)
        l["clock"] = pd.to_datetime(l.timestamp, unit="s", utc=True).dt.tz_convert("Asia/Shanghai")
        assert l.clock.dt.strftime("%Y-%m-%d").eq(day).all()
        ratios = (l.source_amount / l.amount - 1).where(l.amount.gt(0))
        per_day.append(dict(date=day, limit_rows=len(l), limit_daily_known=int(l.close.notna().sum()),
            limits_not_closed=int((l.close.notna() & ~l.closed_at_ordinary_limit).sum()),
            max_limit_amount_difference_bps=float(ratios.abs().max() * 10000) if ratios.notna().any() else None,
            reason_rows=len(r), unique_reason_rows=len(r.drop_duplicates(["code", "reason"])), reason_stocks=len(reason_stocks),
            reason_not_closed=reason_stocks.loc[reason_stocks.close.notna() & ~reason_stocks.closed_at_ordinary_limit,
                                               ["code", "touched_ordinary_limit"]].to_dict("records"),
            member_rows=len(m), member_supported_daily=int(m.close.notna().sum()), member_active=int(tradable.sum()),
            member_price_mismatch=int(mismatch.sum()), member_listed_after=int(m.listing_after_observation.sum()),
            unsupported_members=int(m.code.str.startswith("unsupported.").sum()),
            event_rows=int(frames["events"].date.eq(day).sum())))
        if mismatch.any():
            m.loc[mismatch].to_parquet(out / (day + "_member_price_mismatch.parquet"), index=False)
    for name, frame in frames.items():
        frame.to_parquet(out / (name + ".parquet"), index=False, compression="zstd")
    save_json(out / "response_headers.json", headers)
    result = dict(per_day=per_day, excluded_future_columns=sorted(future_columns), raw_sha256=raw_files,
                  source_completion_sha256=sha(ROOT / "complete_50/report.json"),
                  output_sha256={p.name: sha(p) for p in sorted(out.glob("*.parquet"))},
                  precise_first_publication_verified=False, historical_revision_history_verified=False,
                  new_strategy_returns_computed=False, company_identity_inferred_from_flow=False)
    save_json(out / "report.json", result)
    return {k: v for k, v in result.items() if k not in ["raw_sha256", "output_sha256"]}


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
