"""Independent SQL wall-clock aggregation of every joint-mechanism input."""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from trade_research.corporate_cash import save_json,sha

ROOT=Path("data/research/tick_absorption_winner")
SOURCE=Path("data/research/tick_flow_winner")


def main():
    report=json.loads((ROOT/"input_report.json").read_text())
    assert report["protocol_sha256"]==sha(Path("config/tick_absorption_winner_protocol.json"))
    assert report["source_input_report_sha256"]==sha(SOURCE/"input_report.json")
    assert report["features_sha256"]==sha(ROOT/"features.parquet")
    source=pd.read_parquet(SOURCE/"features.parquet")
    frame=pd.read_parquet(ROOT/"features.parquet").set_index(["date","code"]).sort_index()
    keys=source.loc[source.input_valid,["date","code"]].copy()
    keys["filename"]=[str(SOURCE/"sessions"/r.date/r.code.replace(".","_")/"ticks.parquet") for r in keys.itertuples(index=False)]
    c=duckdb.connect();c.register("keys",keys)
    c.read_parquet(keys.filename.tolist(),filename=True).create_view("all_ticks")
    stats=c.sql('''with clocks as (
       select *,lpad(cast(minute//60 as varchar),2,'0')||':'||lpad(cast(minute%60 as varchar),2,'0') as clock from all_ticks
    ), prefix as (select * from clocks where clock<='14:48'), aggregated as (
       select filename,sum(case when clock>='14:20' then volume_raw else 0 end) as tail_volume_raw,
       sum(case when clock>='14:20' and direction_raw=0 then volume_raw when clock>='14:20' and direction_raw=1 then -volume_raw else 0 end) as tail_signed_raw,
       arg_max(price_raw,tick_seq) filter(where clock<'14:20' and volume_raw>0) as reference_price_raw,
       max(minute) filter(where clock<'14:20' and volume_raw>0) as reference_minute,
       arg_max(price_raw,tick_seq) filter(where volume_raw>0) as last_price_raw,
       max(minute) filter(where volume_raw>0) as last_minute from prefix group by filename
    ) select k.date,k.code,a.* exclude(filename) from keys k join aggregated a using(filename) order by date,code''').df()
    assert len(stats)==int(source.input_valid.sum())
    complete=source[["date","code"]].merge(stats,on=["date","code"],how="left",validate="one_to_one")
    complete["pressure"]="unknown";complete["response"]="unknown";complete["group"]="unknown";complete["tail_price_return"]=np.nan
    for i,row in complete.iterrows():
        usable=pd.notna(row.tail_volume_raw) and row.tail_volume_raw>0 and pd.notna(row.reference_price_raw) and pd.notna(row.last_price_raw)
        if not usable:
            complete.loc[i,["reference_price_raw","reference_minute","last_price_raw","last_minute"]]=np.nan
            continue
        signed=int(row.tail_signed_raw);total=int(row.tail_volume_raw)
        pressure="sell" if signed<=-total/5 else "buy" if signed>=total/5 else "balanced"
        complete.loc[i,"pressure"]=pressure
        complete.loc[i,"tail_price_return"]=row.last_price_raw/row.reference_price_raw-1
        reference=(pd.Timestamp(row.date)+pd.Timedelta(minutes=int(row.reference_minute))).strftime("%H:%M")
        last=(pd.Timestamp(row.date)+pd.Timedelta(minutes=int(row.last_minute))).strftime("%H:%M")
        if reference<"14:18" or last<"14:47":continue
        # Rational thresholds, avoiding floating-point percentage boundary tests.
        from fractions import Fraction
        ratio=Fraction(int(row.last_price_raw),int(row.reference_price_raw))
        response="down" if ratio<Fraction(995,1000) else "up" if ratio>Fraction(1005,1000) else "flat"
        complete.loc[i,["response","group"]]=[response,pressure+":"+response]
    complete=complete.set_index(["date","code"]).sort_index()
    assert complete.index.equals(frame.index)
    for name in ["pressure","response","group"]:assert complete[name].equals(frame[name]),name
    numeric=["tail_volume_raw","tail_signed_raw","reference_price_raw","last_price_raw","reference_minute","last_minute","tail_price_return"]
    for name in numeric:
        assert complete[name].isna().equals(frame[name].isna()),name
        np.testing.assert_allclose(complete[name],frame[name],equal_nan=True,atol=2e-12,rtol=0,err_msg=name)
    original=source.set_index(["date","code"]).sort_index()
    for name in ["source_valid","input_valid","quality_reason","half","necessary_tradeable"]:
        assert original[name].equals(frame[name]),name
    counts=frame.reset_index().groupby(["half","group"]).size().rename("rows").reset_index().to_dict("records")
    assert counts==report["counts"]
    result=dict(passed=True,input_report_sha256=sha(ROOT/"input_report.json"),rows=len(frame),
        raw_sessions=len(keys),numeric_cells=len(frame)*len(numeric),group_cells=len(frame)*3,
        outcomes_read=False,new_2026_prices_read=False)
    save_json(ROOT/"input_verification.json",result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
