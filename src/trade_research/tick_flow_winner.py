"""Frozen aggregate-direction inputs; no economic outcomes are read here."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .corporate_cash import DAILY, save_json, sha

ROOT = Path("data/research/tick_flow_winner")
PROTOCOL = Path("config/tick_flow_winner_protocol.json")
FEATURES = ["morning_net", "afternoon_net", "tail_net", "tail_acceleration",
            "tail_positive_minutes", "tail_flat_net"]


def prefix_features(ticks):
    """Use <=14:48 only. Full-day reconciliation is a separate output."""
    p = ticks.loc[ticks.minute.le(868)].copy()
    missing = {name:np.nan for name in FEATURES}
    if not len(p):
        return dict(missing, input_valid=False, input_reason="no_prefix", prefix_rows=0)
    t = p.minute.to_numpy()
    v = p.volume_raw.to_numpy(dtype="float64")
    direction = p.direction_raw.to_numpy()
    price = p.price_raw.to_numpy()
    legal = (t == 565) | ((t >= 570) & (t <= 690)) | ((t >= 780) & (t <= 868))
    if not (legal.all() and (np.diff(t) >= 0).all() and (v >= 0).all()
            and (price > 0).all() and np.isin(direction, [0,1,2]).all()):
        return dict(missing, input_valid=False, input_reason="invalid_prefix_fields", prefix_rows=len(p))
    signed = np.where(direction == 0, v, np.where(direction == 1, -v, 0.))

    def net(lo, hi):
        mask = (t >= lo) & (t <= hi)
        denominator = v[mask].sum()
        return float(signed[mask].sum() / denominator) if denominator > 0 else np.nan

    tail = (t >= 840) & (t <= 868)
    total_tail = v[tail].sum()
    previous_price = np.r_[np.nan, price[:-1]]
    previous_time = np.r_[0, t[:-1]]
    flat = tail & (previous_time >= 780) & (price == previous_price)
    positive_minutes = sum(signed[t == minute].sum() > 0 for minute in range(840,869))
    result = dict(morning_net=net(570,690), afternoon_net=net(780,868),
        tail_net=net(840,868), tail_acceleration=net(840,868)-net(780,839),
        tail_positive_minutes=positive_minutes/29 if total_tail > 0 else np.nan,
        tail_flat_net=float(signed[flat].sum()/total_tail) if total_tail > 0 else np.nan,
        input_valid=True, input_reason="valid_prefix", prefix_rows=len(p),
        last_input_minute=int(t[-1]), tail_volume_raw=float(total_tail),
        prefix_volume_raw=float(v.sum()))
    return result


def full_day_quality(ticks, daily):
    if not len(ticks):
        return dict(source_valid=False, quality_reason="empty_history")
    if daily is None or pd.isna(daily["open"]):
        return dict(source_valid=False, quality_reason="daily_missing")
    t = ticks.minute.to_numpy()
    direction = ticks.direction_raw.to_numpy()
    price = ticks.price_raw.to_numpy()/100
    volume = ticks.volume_raw.to_numpy()*100
    legal = (t == 565) | ((t >= 570)&(t <= 690)) | ((t >= 780)&(t <= 900)) | ((t >= 905)&(t <= 930))
    if not (legal.all() and (np.diff(t) >= 0).all() and (volume >= 0).all()
            and (price > 0).all() and np.isin(direction,[0,1,2,5]).all()
            and (direction[t > 900] == 5).all() and (direction[t <= 900] != 5).all()):
        return dict(source_valid=False, quality_reason="invalid_full_session")
    regular = t <= 900
    if not regular.any():
        return dict(source_valid=False, quality_reason="no_regular_session")
    vdiff = int(volume[regular].sum()-daily["volume"])
    valid = bool(t[0] == 565 and daily["adjustflag"] == 3 and daily["tradestatus"] == 1
        and abs(price[regular][0]-daily["open"]) < .000001
        and abs(price[regular][-1]-daily["close"]) < .000001
        and (price[regular] >= daily["low"]-.000001).all()
        and (price[regular] <= daily["high"]+.000001).all() and abs(vdiff) <= 100)
    return dict(source_valid=valid, quality_reason="daily_reconciled" if valid else "daily_mismatch",
        volume_difference_shares=vdiff, after_hours_rows=int((~regular).sum()),
        approximate_amount_difference_bps=float((np.sum(price[regular]*volume[regular])/daily["amount"]-1)*10000),
        open_difference=float(price[regular][0]-daily["open"]),
        close_difference=float(price[regular][-1]-daily["close"]))


def freeze():
    if (ROOT / "input_report.json").exists():
        raise ValueError("Do not overwrite inspected aggregate-direction inputs")
    cohort = pd.read_parquet(ROOT / "cohort.parquet")
    frozen = json.loads((ROOT / "cohort_report.json").read_text())
    assert sha(PROTOCOL) == frozen["protocol_sha256"]
    assert sha(ROOT / "cohort.parquet") == frozen["cohort_sha256"]
    collected = json.loads((ROOT / "download_report.json").read_text())
    assert collected["cohort_report_sha256"] == sha(ROOT / "cohort_report.json")
    for path, digest in collected["receipt_sha256"].items():
        assert sha(Path(path)) == digest
    con = duckdb.connect()
    con.register("selected", cohort[["date","code"]])
    daily = con.execute(f"""select s.date,s.code,d.open,d.high,d.low,d.close,
        d.volume,d.amount,d.adjustflag,d.tradestatus from selected s
        left join (select * from read_parquet('{DAILY}/*.parquet')
        where date between '2024-01-01' and '2025-12-31') d
        on s.code=d.code and s.date=d.date order by s.date,s.code""").fetchdf()
    assert len(daily) == len(cohort)
    daily.to_parquet(ROOT / "daily_quality_inputs.parquet", index=False, compression="zstd")
    indexed = daily.set_index(["date","code"])
    inputs = []
    for row in cohort.itertuples(index=False):
        leaf = ROOT / "sessions" / row.date / row.code.replace(".","_")
        receipt = json.loads((leaf / "receipt.json").read_text())
        item = dict(date=row.date, code=row.code, downloaded=receipt["status"] == "downloaded")
        if receipt["status"] != "downloaded":
            item.update({name:np.nan for name in FEATURES})
            item.update(input_valid=False, input_reason="source_unavailable", prefix_rows=0,
                        source_valid=False, quality_reason="source_unavailable")
        else:
            assert sha(leaf / "ticks.parquet") == receipt["ticks_sha256"]
            ticks = pd.read_parquet(leaf / "ticks.parquet")
            assert np.array_equal(ticks.tick_seq, np.arange(len(ticks)))
            item.update(prefix_features(ticks))
            item.update(full_day_quality(ticks, indexed.loc[(row.date,row.code)]))
        inputs.append(item)
    frame = cohort.merge(pd.DataFrame(inputs), on=["date","code"], validate="one_to_one")
    assert len(frame) == frozen["rows"]
    thresholds = {}
    for name in FEATURES:
        values = frame.loc[frame.half.eq("2024H1"), name].dropna()
        if not len(values):
            raise ValueError("No first-half input values to define frozen thresholds")
        lo, hi = values.quantile([.2,.8],interpolation="linear")
        thresholds[name] = dict(low=float(lo), high=float(hi), training_input_count=len(values))
        frame[name+"_group"] = np.select([frame[name].isna(), frame[name].lt(lo), frame[name].gt(hi)],
            ["unknown","low","high"], default="middle")
    frame.to_parquet(ROOT / "features.parquet", index=False, compression="zstd")
    save_json(ROOT / "thresholds.json", thresholds)
    correlations = frame[[*FEATURES,"return_1449","return_tail29","tail_signed_amount"]].corr(method="spearman")
    correlations.to_parquet(ROOT / "input_spearman.parquet", compression="zstd")
    report = dict(protocol_sha256=sha(PROTOCOL), cohort_report_sha256=sha(ROOT/"cohort_report.json"),
        download_report_sha256=sha(ROOT/"download_report.json"), rows=len(frame),
        input_status=frame.input_reason.value_counts().to_dict(), quality_status=frame.quality_reason.value_counts().to_dict(),
        group_counts=frame.groupby("half")[[n+"_group" for n in FEATURES]].agg(lambda x:x.value_counts().to_dict()).to_dict("index"),
        thresholds=thresholds, price_2026_read=False, outcomes_read=False,
        output_sha256={name:sha(ROOT/name) for name in ["features.parquet","thresholds.json","daily_quality_inputs.parquet","input_spearman.parquet"]})
    save_json(ROOT / "input_report.json", report)
    return {k:report[k] for k in ["rows","input_status","quality_status","thresholds"]}


if __name__ == "__main__":
    print(json.dumps(freeze(),ensure_ascii=False,indent=2))
