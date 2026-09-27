"""Fixed morning exit on the existing tick sample; keep every old buy order."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .corporate_cash import MINUTES,save_json,sha

ROOT=Path("data/research/tick_morning_exit")
SOURCE=Path("data/research/tick_flow_winner")
JOINT=Path("data/research/tick_absorption_winner")
LABELS=Path("data/research/economic_winner/period_quality")
PROTOCOL=Path("config/tick_morning_exit_protocol.json")


def classify(row):
    # Same equations as the previously verified economic labels. The independent
    # checker uses scalar Decimal cash arithmetic, not this implementation.
    row=row.copy()
    numeric=['next_preclose','next_trade_status','next_isST','next_adjustflag','day_close',
        'entry_volume','entry_vwap','entry_low','entry_high','exit_volume','exit_vwap','exit_low','exit_high']
    for name in numeric:row[name]=row[name].astype('float64')
    for name in ['entry_source_valid','entry_bounds_valid','exit_source_valid','exit_bounds_valid']:
        row[name]=row[name].fillna(False).astype(bool)
    next_ref=row.next_preclose
    valid_reference=np.isfinite(next_ref)&next_ref.gt(0)&(next_ref-next_ref.round(2)).abs().le(.0001)
    valid_day_close=np.isfinite(row.day_close)&row.day_close.gt(0)&(row.day_close-row.day_close.round(2)).abs().le(.0001)
    rates=np.where(row.board.isin(['chinext','star']),20,np.where(row.next_isST.eq(1),5,10))
    pre_cents=np.rint(next_ref.fillna(0)*100).astype('int64')
    row['exit_lower_limit']=((pre_cents*(100-rates)+50)//100)/100
    row['exit_daily_valid']=valid_reference&row.next_trade_status.eq(1)&row.next_isST.isin([0,1])&row.next_adjustflag.eq(3)
    row['reference_gap_or_unknown']=~(valid_reference&valid_day_close)|(next_ref-row.day_close).abs().gt(.005)
    row['corporate_unknown']=~row.catalog_covered|row.action_exposure|row.reference_gap_or_unknown
    shares=row.decision_shares.to_numpy(dtype=float)
    entry_liquid=np.isfinite(row.entry_vwap)&row.entry_vwap.gt(0)&row.entry_volume.gt(0)
    entry_capacity=row.entry_volume.mul(.1).ge(shares)
    entry_at_limit=row.entry_vwap.mul(1.0005).ge(row.upper_limit-.005)
    row['entry_fill_status']=np.select([~row.necessary_tradeable,~entry_liquid,~entry_capacity,entry_at_limit],
        ['not_submitted','no_liquidity','volume_cap','estimated_upper_limit'],default='filled')
    bought=row.entry_fill_status.eq('filled')
    row['entry_recorded']=bought
    row['entry_sealed_limit']=row.entry_bounds_valid&row.entry_low.round(2).eq(row.upper_limit)&row.entry_high.round(2).eq(row.upper_limit)
    row['entry_queue_unknown']=bought&(~row.entry_bounds_valid|row.entry_high.round(2).ge(row.upper_limit))
    exit_liquid=np.isfinite(row.exit_vwap)&row.exit_vwap.gt(0)&row.exit_volume.gt(0)
    exit_capacity=row.exit_volume.mul(.1).ge(shares)
    exit_at_limit=row.exit_vwap.mul(.9995).le(row.exit_lower_limit+.005)
    row['exit_fill_status']=np.select([~bought,~row.exit_daily_valid,~exit_liquid,~exit_capacity,exit_at_limit],
        ['no_recorded_entry','no_trading_bar','no_liquidity','volume_cap','estimated_lower_limit'],default='filled')
    row['exit_queue_unknown']=bought&(~row.exit_bounds_valid|row.exit_low.round(2).le(row.exit_lower_limit))
    no_entry=row.necessary_tradeable&row.entry_source_valid&~bought
    # All diagnostic flags remain present even when an earlier status is primary.
    row['base_status']=np.select([~row.necessary_tradeable,~row.entry_source_valid,no_entry,row.entry_queue_unknown,
        row.corporate_unknown,~row.exit_source_valid,~row.exit_fill_status.eq('filled'),row.exit_queue_unknown],
        ['not_submitted','entry_source_unknown','not_bought','entry_queue_unknown','corporate_action_unknown',
         'exit_source_unknown','unknown_exit','exit_queue_unknown'],default='ordinary_t1')
    for bps in [5,15]:
        buy=row.entry_vwap+np.maximum(row.entry_vwap*bps/10000,.005)
        sell=row.exit_vwap-np.maximum(row.exit_vwap*bps/10000,.005)
        row[f'stress_limit_unknown{bps}']=buy.ge(row.upper_limit-.005)|sell.le(row.exit_lower_limit+.005)
        eligible=row.base_status.eq('ordinary_t1')&~row[f'stress_limit_unknown{bps}']
        buy_value=shares*buy;sell_value=shares*sell
        buy_cash=buy_value+np.maximum(5.,buy_value*.0003)+buy_value*.00001
        sell_cash=sell_value-np.maximum(5.,sell_value*.0003)-sell_value*.00051
        row[f'buy_cash{bps}']=buy_cash.where(eligible)
        row[f'sell_cash{bps}']=sell_cash.where(eligible)
        row[f'net_return{bps}']=(sell_cash/buy_cash-1).where(eligible)
        row[f'known_profit{bps}']=eligible
        row[f'label{bps}']=np.select([~row.necessary_tradeable,no_entry,~eligible,
            row[f'net_return{bps}'].ge(.01),row[f'net_return{bps}'].le(-.01)],
            ['not_submitted','no_trade','unknown','economic_winner','economic_loser'],default='middle')
    entry_bad=row.period_entry_bad_day|row.period_bad_symbol
    exit_bad=row.period_exit_bad_day|row.period_bad_symbol
    # A valid known non-entry has no exit whose provenance could affect P&L.
    source_unknown=row.necessary_tradeable&(entry_bad|(exit_bad&~row.label15.eq('no_trade')))
    row['period_source_unknown']=source_unknown
    row['original_base_status']=row.base_status
    row.loc[source_unknown&row.base_status.isin(['ordinary_t1','not_bought']),'base_status']='period_source_unknown'
    for bps in [5,15]:
        row[f'original_label{bps}']=row[f'label{bps}']
        row.loc[source_unknown,f'label{bps}']='unknown'
        row.loc[source_unknown,f'known_profit{bps}']=False
        row.loc[source_unknown,[f'buy_cash{bps}',f'sell_cash{bps}',f'net_return{bps}']]=np.nan
    return row


def freeze():
    if (ROOT/"input_report.json").exists():raise ValueError("Do not replace fixed morning inputs")
    ROOT.mkdir(parents=True,exist_ok=True)
    for folder in [SOURCE,JOINT]:
        check=json.loads((folder/"input_verification.json").read_text())
        assert check["passed"] and check["input_report_sha256"]==sha(folder/"input_report.json")
        report=json.loads((folder/"input_report.json").read_text())
        digest=report.get("features_sha256",report.get("output_sha256",{}).get("features.parquet"))
        assert sha(folder/"features.parquet")==digest
    label_check=json.loads((LABELS/"analysis_verification.json").read_text())
    assert label_check["passed"] and label_check["analysis_report_sha256"]==sha(LABELS/"analysis_report.json")
    assert sha(LABELS/"labels.parquet")==json.loads((LABELS/"label_report.json").read_text())["labels_sha256"]
    features=pd.read_parquet(SOURCE/"features.parquet")
    joint=pd.read_parquet(JOINT/"features.parquet",columns=["date","code","group"])
    features=features.merge(joint.rename(columns={"group":"pressure_response_group"}),on=["date","code"],validate="one_to_one")
    assert len(features)==1936 and features.date.between("2024-01-01","2025-12-30").all()
    c=duckdb.connect();c.register("keys",features[["date","code"]])
    old=c.execute("select l.* from read_parquet(?) l join keys k using(date,code) order by date,code",[str(LABELS/"labels.parquet")]).fetchdf()
    # All original columns, including cash NaN masks and diagnostic flags, must
    # reproduce before reading any new morning window.
    replay=classify(old)
    pd.testing.assert_frame_equal(old,replay[old.columns],check_dtype=False,atol=1e-10,rtol=0)
    assert (old.next_date>old.date).all() and old.next_date.le("2025-12-31").all()
    old.to_parquet(ROOT/"old_tail_labels.parquet",index=False,compression="zstd")
    features.to_parquet(ROOT/"features.parquet",index=False,compression="zstd")
    keys=old[["next_date","code"]].rename(columns={"next_date":"date"}).drop_duplicates().sort_values(["date","code"])
    assert len(keys)==len(old)
    keys.to_parquet(ROOT/"window_keys.parquet",index=False,compression="zstd")
    outputs={name:sha(ROOT/name) for name in ["features.parquet","old_tail_labels.parquet","window_keys.parquet"]}
    report=dict(protocol_sha256=sha(PROTOCOL),source_input_report_sha256=sha(SOURCE/"input_report.json"),
        joint_input_report_sha256=sha(JOINT/"input_report.json"),old_label_report_sha256=sha(LABELS/"label_report.json"),
        economic_manifest_sha256=sha(Path("data/research/economic_winner/input_manifest.json")),
        output_sha256=outputs,rows=len(features),tail_reproduction_columns=len(old.columns),
        new_2026_prices_read=False,new_morning_outcomes_read=False)
    save_json(ROOT/"input_report.json",report)
    # This gate only verifies copied already-audited inputs and tail reproduction.
    save_json(ROOT/"input_verification.json",dict(passed=True,input_report_sha256=sha(ROOT/"input_report.json"),
        inherited_inputs_unchanged=True,old_tail_all_columns_reproduced=len(old.columns),rows=len(old)))
    return report


def raw():
    if (ROOT/"raw_report.json").exists():raise ValueError("Do not replace morning bars")
    inputs=json.loads((ROOT/"input_report.json").read_text())
    assert inputs["protocol_sha256"]==sha(PROTOCOL)
    for name,digest in inputs["output_sha256"].items():assert sha(ROOT/name)==digest
    source_path=Path("data/research/economic_winner/input_manifest.json")
    assert sha(source_path)==inputs["economic_manifest_sha256"]
    sources=json.loads(source_path.read_text())["source_sha256"]
    keys=pd.read_parquet(ROOT/"window_keys.parquet");codes=sorted(keys.code.unique())
    folder=ROOT/"raw_parts";folder.mkdir(exist_ok=True)
    parts={};total=0
    for offset in range(0,len(codes),64):
        subset=codes[offset:offset+64];dest=folder/f"part_{offset//64:03d}.parquet";meta=dest.with_suffix(".json")
        if dest.exists() and meta.exists():
            saved=json.loads(meta.read_text())
            assert saved["codes"]==subset and saved["input_report_sha256"]==sha(ROOT/"input_report.json")
            assert sha(dest)==saved["sha256"]
        else:
            files=[MINUTES/code[:2].upper()/(code[3:]+".parquet") for code in subset]
            for file in files:assert sha(file)==sources[str(file)]
            c=duckdb.connect();c.execute("SET threads=4");c.execute("SET memory_limit='4GB'")
            c.read_parquet([str(f) for f in files]).create_view("original")
            c.register("keys",keys.loc[keys.code.isin(subset)])
            bars=c.sql("""with s as (select lower(exchange)||'.'||symbol AS code,
                strftime(timestamp,'%Y-%m-%d') AS date,timestamp,
                hour(timestamp)*60+minute(timestamp) AS minute,
                open::double AS open,high::double AS high,low::double AS low,close::double AS close,
                volume::double AS volume,turnover::double AS amount from original
                where timestamp>=TIMESTAMP '2024-01-01' and timestamp<TIMESTAMP '2026-01-01'
                and hour(timestamp)=9 and minute(timestamp) between 35 and 38)
                select s.* from s join keys k using(date,code) order by date,code,minute""").df()
            assert not bars.duplicated(["date","code","timestamp"]).any()
            bars.to_parquet(dest,index=False,compression="zstd");c.close()
            saved=dict(codes=subset,rows=len(bars),sha256=sha(dest),input_report_sha256=sha(ROOT/"input_report.json"),
                sources_sha256={str(f):sources[str(f)] for f in files})
            save_json(meta,saved)
        parts[str(dest)]=saved["sha256"];total+=saved["rows"]
        print(json.dumps(dict(codes=offset+len(subset),total_codes=len(codes),raw_rows=total)),flush=True)
    c=duckdb.connect();c.read_parquet(list(parts)).create_view("raw")
    c.register("keys",keys)
    windows=c.sql("""with bars as (select *,
        coalesce(timestamp=date_trunc('minute',timestamp) and isfinite(open) and isfinite(high) and isfinite(low)
        and isfinite(close) and isfinite(volume) and isfinite(amount) and least(open,high,low,close)>0
        and high+.0001>=greatest(open,close,low) and low-.0001<=least(open,close)
        and volume>=0 and amount>=0 and (volume=0)=(amount=0)
        and (volume=0 or amount/volume between low-.0101 and high+.0101),false) AS valid,
        coalesce(abs(high-round(high,2))<=.0001 and abs(low-round(low,2))<=.0001,false) AS cent_bounds from raw),
        a as (select date,code,count(*) AS bars,count(distinct minute) AS labels,
            count(*) filter(where valid) AS valid_bars,count(*) filter(where volume>0) AS positive_bars,
            count(*) filter(where volume>0 and not cent_bounds) AS invalid_cent_bars,
            sum(volume) AS volume,sum(amount)/nullif(sum(volume),0) AS vwap,
            min(low) filter(where volume>0) AS positive_low,max(high) filter(where volume>0) AS positive_high
            from bars group by date,code)
        select k.date,k.code,a.* exclude(date,code),coalesce(bars=4 and labels=4 and valid_bars=4,false) AS source_valid,
            coalesce(positive_bars>0 and invalid_cent_bars=0,false) AS queue_bounds_valid
        from keys k left join a using(date,code) order by date,code""").df()
    windows.to_parquet(ROOT/"windows.parquet",index=False,compression="zstd")
    report=dict(input_report_sha256=sha(ROOT/"input_report.json"),parts_sha256=parts,raw_rows=total,
        windows_sha256=sha(ROOT/"windows.parquet"),windows=len(windows),new_2026_prices_read=False)
    save_json(ROOT/"raw_report.json",report)
    return {k:v for k,v in report.items() if k!="parts_sha256"}


def labels():
    if (ROOT/"label_report.json").exists():raise ValueError("Do not replace new morning labels")
    check=json.loads((ROOT/"window_verification.json").read_text())
    assert check["passed"] and check["raw_report_sha256"]==sha(ROOT/"raw_report.json")
    assert sha(ROOT/"windows.parquet")==json.loads((ROOT/"raw_report.json").read_text())["windows_sha256"]
    inputs=json.loads((ROOT/"input_report.json").read_text())
    assert sha(ROOT/"old_tail_labels.parquet")==inputs["output_sha256"]["old_tail_labels.parquet"]
    old=pd.read_parquet(ROOT/"old_tail_labels.parquet")
    win=pd.read_parquet(ROOT/"windows.parquet")
    mapping={"bars":"exit_bars","labels":"exit_labels","source_valid":"exit_source_valid",
        "queue_bounds_valid":"exit_bounds_valid","volume":"exit_volume","vwap":"exit_vwap",
        "positive_low":"exit_low","positive_high":"exit_high"}
    new=old.drop(columns=list(mapping.values())).merge(win[["date","code",*mapping]].rename(columns=dict(mapping,date="next_date")),
        on=["next_date","code"],how="left",validate="one_to_one").sort_values(["date","code"]).reset_index(drop=True)
    result=classify(new)
    # Exit uncertainty may hide old/new cash columns, but it cannot alter the buy
    # order, buy-side diagnostics or the unmasked planned cash amount.
    for col in ["decision_shares","price_1449",*[n for n in old if n.startswith("entry_")]]:
        pd.testing.assert_series_equal(old[col],result[col],check_dtype=False)
    for bps in [5,15]:
        buy=old.entry_vwap+np.maximum(old.entry_vwap*bps/10000,.005)
        value=old.decision_shares*buy
        planned=value+np.maximum(5.,.0003*value)+.00001*value
        result[f"intended_buy_cash{bps}"]=planned
        for f in [old,result]:
            mask=f[f"buy_cash{bps}"].notna()
            np.testing.assert_allclose(f.loc[mask,f"buy_cash{bps}"],planned[mask],atol=1e-9,rtol=0)
    result.to_parquet(ROOT/"labels.parquet",index=False,compression="zstd")
    report=dict(labels_sha256=sha(ROOT/"labels.parquet"),window_verification_sha256=sha(ROOT/"window_verification.json"),
        input_report_sha256=sha(ROOT/"input_report.json"),rows=len(result),new_2026_prices_read=False,
        label_counts={str(bps):result[f"label{bps}"].value_counts().to_dict() for bps in [5,15]})
    save_json(ROOT/"label_report.json",report)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("stage",choices=["freeze","raw","labels"])
    print(json.dumps(globals()[p.parse_args().stage](),ensure_ascii=False,indent=2))
