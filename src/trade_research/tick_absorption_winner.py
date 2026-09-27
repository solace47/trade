"""Joint price response and inferred pressure, with explicit wall-clock bounds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .corporate_cash import save_json,sha

ROOT=Path("data/research/tick_absorption_winner")
SOURCE=Path("data/research/tick_flow_winner")
PROTOCOL=Path("config/tick_absorption_winner_protocol.json")
CUTOFF=14*60+48
TAIL_START=14*60+20


def classify(t):
    p=t.loc[t.minute.le(CUTOFF)]
    prior=p.loc[p.minute.lt(TAIL_START)&p.volume_raw.gt(0)]
    positive=p.loc[p.volume_raw.gt(0)]
    tail=p.loc[p.minute.ge(TAIL_START)]
    total=int(tail.volume_raw.sum())
    signed=int(tail.loc[tail.direction_raw.eq(0),"volume_raw"].sum()-tail.loc[tail.direction_raw.eq(1),"volume_raw"].sum())
    result=dict(tail_volume_raw=total,tail_signed_raw=signed,pressure="unknown",response="unknown",group="unknown",
        reference_price_raw=np.nan,last_price_raw=np.nan,reference_minute=np.nan,last_minute=np.nan,tail_price_return=np.nan)
    if total<=0 or not len(prior) or not len(positive):return result
    before=prior.iloc[-1];after=positive.iloc[-1]
    start_price=int(before.price_raw);end_price=int(after.price_raw)
    result.update(reference_price_raw=start_price,last_price_raw=end_price,
        reference_minute=int(before.minute),last_minute=int(after.minute),tail_price_return=end_price/start_price-1)
    pressure="sell" if 5*signed<=-total else "buy" if 5*signed>=total else "balanced"
    result["pressure"]=pressure
    if before.minute<14*60+18 or after.minute<14*60+47:return result
    response="down" if 1000*end_price<995*start_price else "up" if 1000*end_price>1005*start_price else "flat"
    result.update(response=response,group=pressure+":"+response)
    return result


def freeze():
    if (ROOT/"input_report.json").exists():raise ValueError("Do not overwrite joint inputs")
    ROOT.mkdir(exist_ok=True)
    report=json.loads((SOURCE/"input_report.json").read_text())
    check=json.loads((SOURCE/"input_verification.json").read_text())
    assert check["passed"] and check["input_report_sha256"]=="9ac853c6b93c6837572eeaddb87afc0716576d718b85037b96d977dccd744997"
    assert sha(SOURCE/"input_report.json")==check["input_report_sha256"]
    assert sha(SOURCE/"features.parquet")==report["output_sha256"]["features.parquet"]
    source=pd.read_parquet(SOURCE/"features.parquet")
    fields=[]
    for row in source.itertuples(index=False):
        if not row.input_valid:
            values=dict(group="unknown",pressure="unknown",response="unknown")
        else:
            leaf=SOURCE/"sessions"/row.date/row.code.replace(".","_")
            receipt=json.loads((leaf/"receipt.json").read_text())
            assert sha(leaf/"ticks.parquet")==receipt["ticks_sha256"]
            values=classify(pd.read_parquet(leaf/"ticks.parquet"))
            if pd.notna(row.tail_net):assert abs(values["tail_signed_raw"]/values["tail_volume_raw"]-row.tail_net)<2e-12
        fields.append(dict(date=row.date,code=row.code,**values))
    kept=["date","code","half","board","source_valid","input_valid","quality_reason","necessary_tradeable"]
    frame=source[kept].merge(pd.DataFrame(fields),on=["date","code"],validate="one_to_one")
    assert len(frame)==1936
    frame.to_parquet(ROOT/"features.parquet",index=False,compression="zstd")
    result=dict(protocol_sha256=sha(PROTOCOL),source_input_report_sha256=sha(SOURCE/"input_report.json"),
        features_sha256=sha(ROOT/"features.parquet"),rows=len(frame),
        counts=frame.groupby(["half","group"]).size().rename("rows").reset_index().to_dict("records"),
        source_valid=int(frame.source_valid.sum()),input_valid=int(frame.input_valid.sum()),outcomes_read=False,new_2026_prices_read=False)
    save_json(ROOT/"input_report.json",result)
    return result


if __name__=="__main__":print(json.dumps(freeze(),ensure_ascii=False,indent=2))
