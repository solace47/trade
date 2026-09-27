"""Independent sample selection, wire replay, and scalar directional inputs."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct
import sys

import duckdb
import numpy as np
import pandas as pd

from verify_historical_tick_source import independent_decode
from trade_research.corporate_cash import DAILY, save_json, sha

ROOT = Path("data/research/tick_flow_winner")
FEATURES = ["morning_net","afternoon_net","tail_net","tail_acceleration","tail_positive_minutes","tail_flat_net"]


def scalar_features(records):
    def clock(x):
        hour,minute=divmod(x["minute"],60)
        return f"{hour:02d}:{minute:02d}"
    p = [x for x in records if clock(x) <= "14:48"]
    missing = dict.fromkeys(FEATURES,math.nan)
    if not p:
        return missing
    previous = -1
    for x in p:
        t=x["minute"]
        if not (t>=previous and (clock(x)=="09:25" or "09:30"<=clock(x)<="11:30" or "13:00"<=clock(x)<="14:48")
                and x["price_raw"]>0 and x["volume_raw"]>=0 and x["direction_raw"] in (0,1,2)):
            return missing
        previous=t
    sums={"morning":[0,0],"afternoon":[0,0],"before_tail":[0,0],"tail":[0,0]}
    minutes=dict.fromkeys([f"14:{m:02d}" for m in range(20,49)],0)
    flat=0
    prior=None
    for x in p:
        t=x["minute"];v=x["volume_raw"]
        signed=v*({0:1,1:-1,2:0}[x["direction_raw"]])
        for key,lo,hi in [("morning","09:30","11:30"),("afternoon","13:00","14:48"),("before_tail","13:00","14:19"),("tail","14:20","14:48")]:
            if lo<=clock(x)<=hi:
                sums[key][0]+=signed;sums[key][1]+=v
        if "14:20"<=clock(x)<="14:48":
            minutes[clock(x)]+=signed
            if prior and clock(prior)>="13:00" and prior["price_raw"]==x["price_raw"]:
                flat+=signed
        prior=x
    ratios={name:a/b if b else math.nan for name,(a,b) in sums.items()}
    denom=sums["tail"][1]
    return dict(morning_net=ratios["morning"],afternoon_net=ratios["afternoon"],
        tail_net=ratios["tail"],tail_acceleration=ratios["tail"]-ratios["before_tail"],
        tail_positive_minutes=sum(x>0 for x in minutes.values())/29 if denom else math.nan,
        tail_flat_net=flat/denom if denom else math.nan)


def main():
    cfg=json.loads(Path("config/tick_flow_winner_protocol.json").read_text())
    cohort=pd.read_parquet(ROOT/"cohort.parquet")
    report=json.loads((ROOT/"input_report.json").read_text())
    for name,digest in report["output_sha256"].items():assert sha(ROOT/name)==digest
    population=pd.read_parquet("data/research/next_day_winner/cohort.parquet",columns=["date","code","half","board","necessary_tradeable"])
    population=population.loc[population.necessary_tradeable&population.board.eq("main")].copy()
    population["hash"]=[hashlib.sha256(f'{cfg["sampling_seed"]}|{d}|{c}'.encode()).hexdigest() for d,c in population[["date","code"]].itertuples(index=False,name=None)]
    expected=population.sort_values(["date","hash","code"]).groupby("date").head(cfg["symbols_per_day"])
    assert set(map(tuple,expected[["date","code"]].to_numpy()))==set(map(tuple,cohort[["date","code"]].to_numpy()))
    counts=population.groupby("date").size()
    assert np.array_equal(cohort.population_count,cohort.date.map(counts))
    np.testing.assert_allclose(cohort.inclusion_probability,4/cohort.population_count,rtol=0,atol=1e-16)
    frame=pd.read_parquet(ROOT/"features.parquet").set_index(["date","code"])
    assert set(frame.index)==set(map(tuple,cohort[["date","code"]].to_numpy()))
    daily=pd.read_parquet(ROOT/"daily_quality_inputs.parquet")
    c=duckdb.connect();c.register("sampled",cohort[["date","code"]])
    reference=c.execute(f"""select s.date,s.code,d.open,d.high,d.low,d.close,d.volume,d.amount,d.adjustflag,d.tradestatus
        from sampled s left join (select * from read_parquet('{DAILY}/*.parquet')
        where date between '2024-01-01' and '2025-12-31') d on s.date=d.date and s.code=d.code
        order by s.date,s.code""").fetchdf()
    pd.testing.assert_frame_equal(daily,reference)
    daily=daily.set_index(["date","code"])
    pages=records_count=unknown=quality_checked=0
    for pos,row in enumerate(cohort.itertuples(index=False),start=1):
        key=(row.date,row.code);leaf=ROOT/"sessions"/row.date/row.code.replace(".","_")
        receipt=json.loads((leaf/"receipt.json").read_text())
        actual=frame.loc[key]
        if receipt["status"]!="downloaded":
            unknown+=1
            assert not actual.input_valid and not actual.source_valid
            assert actual[FEATURES].isna().all()
            continue
        for path,digest in receipt["request_response_sha256"].items():assert sha(Path(path))==digest
        attempt=[x for x in receipt["attempts"] if x["status"]=="downloaded"][-1]
        assembled=[]
        for page in range(attempt["pages"]):
            folder=Path(attempt["wire_folder"])
            request=(folder/f"history_{page}.request.bin").read_bytes()
            dt,exchange,code,start,count=struct.unpack("<IH6sHH",request[12:])
            assert dt==int(row.date.replace("-","")) and exchange==int(row.code.startswith("sh."))
            assert code.decode()==row.code.split(".")[1] and start==page*cfg["page_size"] and count==cfg["page_size"]
            decoded=independent_decode((folder/f"history_{page}.response.bin").read_bytes())
            assert len(decoded)==count if page<attempt["pages"]-1 else len(decoded)<count
            assembled=decoded+assembled;pages+=1
        ticks=pd.read_parquet(leaf/"ticks.parquet")
        assert ticks.drop(columns="tick_seq").to_dict("records")==assembled
        records_count+=len(assembled)
        features=scalar_features(assembled)
        prefix=[x for x in assembled if f"{x['minute']//60:02d}:{x['minute']%60:02d}"<="14:48"]
        assert int(actual.prefix_rows)==len(prefix)
        if actual.input_valid:
            assert int(actual.last_input_minute)==prefix[-1]["minute"]
            assert f"{int(actual.last_input_minute)//60:02d}:{int(actual.last_input_minute)%60:02d}"<="14:48"
        for name in FEATURES:
            value=features[name];other=actual[name]
            assert (math.isnan(value) and pd.isna(other)) or abs(value-other)<2e-12,(key,name,value,other)
        d=daily.loc[key]
        regular=[x for x in assembled if x["minute"]<=900]
        valid=False
        if len(regular) and pd.notna(d["open"]):
            legal=all(x["minute"]==565 or 570<=x["minute"]<=690 or 780<=x["minute"]<=900 or 905<=x["minute"]<=930 for x in assembled)
            direction_ok=all(x["direction_raw"] in (0,1,2) if x["minute"]<=900 else x["direction_raw"]==5 for x in assembled)
            monotonic=all(a["minute"]<=b["minute"] for a,b in zip(assembled,assembled[1:]))
            quantity=sum(x["volume_raw"]*100 for x in regular)
            valid=(legal and direction_ok and monotonic and all(x["price_raw"]>0 and x["volume_raw"]>=0 for x in assembled)
                and assembled[0]["minute"]==565 and d["adjustflag"]==3 and d["tradestatus"]==1
                and abs(regular[0]["price_raw"]/100-d["open"])<.000001
                and abs(regular[-1]["price_raw"]/100-d["close"])<.000001
                and all(d["low"]-.000001<=x["price_raw"]/100<=d["high"]+.000001 for x in regular)
                and abs(quantity-d["volume"])<=100)
        assert bool(actual.source_valid)==bool(valid),(key,"quality flag")
        quality_checked+=1
        if pos%250==0:print(json.dumps(dict(checked=pos,wire_pages=pages,records=records_count)),flush=True)
    # Recompute the first-half quantiles directly from sorted observations.
    for name in FEATURES:
        x=sorted(frame.loc[frame.half.eq("2024H1"),name].dropna())
        quantiles=[]
        for q in (.2,.8):
            z=(len(x)-1)*q;lo=math.floor(z);hi=math.ceil(z)
            quantiles.append(x[lo]+(x[hi]-x[lo])*(z-lo))
        stored=report["thresholds"][name]
        np.testing.assert_allclose(quantiles,[stored["low"],stored["high"]],rtol=0,atol=2e-12)
        lo,hi=stored["low"],stored["high"]
        groups=["unknown" if pd.isna(v) else "low" if v<lo else "high" if v>hi else "middle" for v in frame[name]]
        assert np.array_equal(groups,frame[name+"_group"])
    result=dict(passed=True,input_report_sha256=sha(ROOT/"input_report.json"),population_rows=len(population),
        sample_rows=len(cohort),wire_pages=pages,decoded_records=records_count,source_unknown=unknown,
        full_day_quality_rows=quality_checked,features_checked=6*len(cohort),thresholds_checked=12,
        outcomes_read=False,new_2026_prices_read=False)
    save_json(ROOT/"input_verification.json",result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
