"""Flag price-limit queue exposure without inventing order-book allocations."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .corporate_cash import save_json, sha
from .hf_outcomes import EXECUTION_LABELS, _board_limit_rate, _limit_price
from .quote_precision import quote_cents
from .reference_gain_eval import DAILY


def window_exposure(bars: pd.DataFrame, limit_price: float, side: str) -> dict:
    if side not in ("buy", "sell"):
        raise ValueError("Unknown order side")
    positive = bars.loc[bars.volume.gt(0)]
    result = {"limit_price":limit_price,"limit_touched":None,
        "all_positive_bars_at_limit":None,"volume_in_wholly_away_minutes":None}
    if positive.empty:
        return result
    try:
        limit = quote_cents(limit_price)
        high, low = positive.high.map(quote_cents), positive.low.map(quote_cents)
    except ValueError:
        return result
    away = high.lt(limit) if side == "buy" else low.gt(limit)
    touched = high.ge(limit) if side == "buy" else low.le(limit)
    result.update({"limit_touched":bool(touched.any()),
        "all_positive_bars_at_limit":bool((high.eq(limit)&low.eq(limit)).all()),
        "volume_in_wholly_away_minutes":float(positive.loc[away,"volume"].sum())})
    return result


def audit(output: Path) -> dict:
    report = json.loads((output / "execution_report.json").read_text())
    for name in ("repriced", "raw_windows"):
        if sha(output / (name+".parquet")) != report["output_sha256"][name]:
            raise ValueError("Execution sources changed before the queue audit")
    rows = pd.read_parquet(output / "repriced.parquet")
    bars = pd.read_parquet(output / "raw_windows.parquet")
    exits = tuple(report.get("exit_labels", EXECUTION_LABELS))
    records = []
    for code, group in rows.groupby("code"):
        dates = sorted(set(group.date) | set(group.exit_date.dropna()))
        daily = pd.read_parquet(DAILY / (code.replace(".","_")+".parquet"),
            filters=[("date","in",dates)], columns=["date","preclose","isST"]).set_index("date")
        minute = bars.loc[bars.code.eq(code)]
        for row in group.itertuples():
            result = {"date":row.date,"code":code,"horizon":int(row.horizon),
                      "entry_status":row.entry_status,"exit_date":row.exit_date}
            for side, date, labels, order_side in (("entry",row.date,EXECUTION_LABELS,"buy"),
                    ("exit",row.exit_date,exits,"sell")):
                if pd.isna(date) or date not in daily.index:
                    fields = {"limit_price":None,"limit_touched":None,
                        "all_positive_bars_at_limit":None,"volume_in_wholly_away_minutes":None}
                else:
                    day = daily.loc[date]
                    rate = _board_limit_rate(code,int(day.isST),str(date))
                    limit = _limit_price(day.preclose,rate,side=="entry")
                    selected = minute.loc[minute.date.eq(date)&minute.label.isin(labels)]
                    fields = window_exposure(selected,limit,order_side)
                result.update({side+"_"+key:value for key,value in fields.items()})
            bought = row.entry_status == "filled"
            result["queue_allocation_unverified"] = bool(bought and (
                result["entry_limit_touched"] is not False or result["exit_limit_touched"] is not False))
            records.append(result)
    frame = pd.DataFrame(records)
    frame.to_parquet(output / "execution_queue_audit.parquet",index=False,compression="zstd")
    bought = frame.entry_status.eq("filled")
    result = {"method":"adverse_limit_touch_is_unknown_allocation_not_automatic_fill_or_zero_return",
        "orderbook_or_queue_data_available":False,"original_ledger_unchanged":True,
        "rows":len(frame),"bought":int(bought.sum()),
        "bought_entry_limit_touch":int((bought&frame.entry_limit_touched.eq(True)).sum()),
        "recorded_exit_limit_touch":int((bought&frame.exit_limit_touched.eq(True)).sum()),
        "simulated_buys_with_all_positive_bars_at_limit":int((bought&frame.entry_all_positive_bars_at_limit.eq(True)).sum()),
        "queue_allocation_unverified":int(frame.queue_allocation_unverified.sum()),
        "execution_report_sha256":sha(output / "execution_report.json"),
        "output_sha256":sha(output / "execution_queue_audit.parquet")}
    save_json(output / "execution_queue_report.json",result)
    return result


def conservative_returns(rows: pd.DataFrame, audit_rows: pd.DataFrame, column: str) -> pd.Series:
    """Unknown fills cannot disappear or silently become uninvested cash."""
    flags = rows[["date","code","horizon"]].merge(audit_rows,
        on=["date","code","horizon"],how="left",validate="one_to_one")
    if flags.queue_allocation_unverified.isna().any():
        raise ValueError("A trade lacks an execution-queue diagnosis")
    return rows[column].where(~flags.queue_allocation_unverified.to_numpy(),np.nan)
