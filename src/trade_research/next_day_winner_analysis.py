"""Full winner portraits and failure denominators; these are exploratory associations."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .corporate_cash import save_json, sha
from .next_day_winner import ROOT, calendar
from .reference_gain_accounting import weekly_interval

FEATURES = {
    "return5_prior_adjusted":"此前5日涨幅", "return20_prior_adjusted":"此前20日涨幅",
    "ma20_distance":"距此前20日均线", "volume_ratio_prior5":"截至14:49相对前5日量比",
    "gap_open":"前日开盘缺口", "return_1449":"前日14:49涨幅",
    "return_open_1000":"开盘至10点涨幅", "return_1000_1130":"10点至午休涨幅",
    "return_1130_1420":"午休至14:20涨幅", "return_tail29":"末29分钟涨幅",
    "return_tail14":"末14分钟涨幅", "position_1449":"日内区间位置",
    "drawdown_1449":"距当日高点", "rebound_1449":"距当日低点",
    "vwap_premium":"相对当日量额均价", "tail_vwap_premium":"相对尾段量额均价",
    "above_vwap_fraction":"在动态均价上方的分钟比例", "afternoon_above_vwap":"午后在动态均价上方比例",
    "tail_amount_fraction":"尾段成交额占比", "tail_volume_intensity":"尾段相对此前每分钟量能",
    "tail_signed_amount":"尾段涨跌分钟方向金额代理", "tail_up_fraction":"尾段上涨分钟比例",
    "tail_path_efficiency":"尾段价格路径效率", "tail_max_jump":"尾段最大单分钟上涨",
    "tail_giveback":"尾段最高收盘价后的回吐", "early_amount_fraction":"开盘半小时成交额占比",
    "industry_return":"同行其他股票平均涨幅", "industry_rising":"同行其他股票上涨比例",
    "industry_excess":"同行相对板块涨幅",
}
EVENTS={"prior_daily_lhb":"上一交易日沪市单日龙虎榜",
    "prior_institution_positive":"上一交易日沪市机构专用净买入",
    "prior_block_trade":"上一交易日沪市大宗交易",
    "prior_premium_block":"上一交易日沪市大额溢价交易"}


def assemble(output: Path=ROOT) -> tuple[pd.DataFrame,dict]:
    report=json.loads((output/"base_report.json").read_text())
    for name,key in (("visible_base","base_sha256"),("labels","labels_sha256")):
        if sha(output/(name+".parquet"))!=report[key]:
            raise ValueError("A frozen input/label changed")
    base=pd.read_parquet(output/"visible_base.parquet")
    labels=pd.read_parquet(output/"labels.parquet")
    raw_report=json.loads((output/"raw_report.json").read_text())
    raw=[]
    for path,digest in raw_report["batch_manifests_sha256"].items():
        meta=Path(path)
        if sha(meta)!=digest or sha(meta.with_suffix(".parquet"))!=json.loads(meta.read_text())["sha256"]:
            raise ValueError("Raw extraction provenance changed")
        raw.append(pd.read_parquet(meta.with_suffix(".parquet")))
    paths=pd.concat(raw,ignore_index=True)
    frame=base.merge(paths,on=["date","code"],how="left",validate="one_to_one")
    if (frame.prefix_bars.ne(230).any() or frame.prefix_labels.ne(230).any()
        or frame.prefix_volume.ne(frame.volume_1449).any()
        or not np.allclose(frame.prefix_amount,frame.amount_1449,rtol=1e-12,atol=.001)
        or not np.allclose(frame.prefix_price_1449,frame.price_1449,rtol=0,atol=1e-10)):
        raise ValueError("New path extraction differs from the frozen 14:49 prefix")
    frame["return_open_1000"]=frame.price_1000/frame.daily_open-1
    frame["return_1000_1130"]=frame.price_1130/frame.price_1000-1
    frame["return_1130_1420"]=frame.price_1420/frame.price_1130-1
    frame["entry_evidence"]=np.select([
        frame.decision_shares.le(0),frame.entry_bars.ne(4)|frame.entry_labels.ne(4),
        frame.entry_valid_bars.ne(4)|frame.entry_amount_bad_bars.gt(0),
        frame.entry_volume.le(0)|frame.entry_vwap.le(0)|frame.entry_vwap.isna(),
        frame.entry_low.ge(frame.upper_limit-.005),frame.decision_shares.gt(frame.entry_volume*.1),
        frame.entry_high.ge(frame.upper_limit-.005),
        (frame.entry_vwap+np.maximum(frame.entry_vwap*.0015,.005)).ge(frame.upper_limit-.005)],
        ["below_lot","missing_window","source_unknown","no_volume","sealed_limit","capacity_rejected","queue_unknown","stress_price_at_limit"],
        default="ordinary_window_candidate")
    # Prior event coverage is explicit: an archive gap or another market is never zero.
    days=calendar(); covered=set(days[1:-10]); sources={}
    inst_path=Path("data/research/lhb_institutional_short/source_events.parquet")
    block_path=Path("data/research/block_premium_short/source_events.parquet")
    rawblock_path=Path("data/research/block_premium_short/raw_disclosures.parquet")
    inst=pd.read_parquet(inst_path);block=pd.read_parquet(block_path);rawblock=pd.read_parquet(rawblock_path)
    inst["prior_daily_lhb"]=1.;inst["prior_institution_positive"]=inst.institution_net_cents.gt(0).astype(float)
    block["prior_premium_block"]=block.premium_event.astype(float)
    allblock=rawblock[["date","code"]].drop_duplicates();allblock["prior_block_trade"]=1.
    for events,cols in ((inst,["prior_daily_lhb","prior_institution_positive"]),
            (block,["prior_premium_block"]),(allblock,["prior_block_trade"])):
        frame=frame.merge(events[["date","code",*cols]],on=["date","code"],how="left",validate="one_to_one")
        coverage=frame.date.isin(covered)&frame.code.str.startswith("sh.60")
        for col in cols:
            frame[col]=frame[col].fillna(0).where(coverage)
    for path in (inst_path,block_path,rawblock_path):sources[str(path)]=sha(path)
    frame=frame.merge(labels[["date","code","next_date","known_label","winner","cohort",
        "next_gain","next_gap","next_peak","next_reference_gap","next_isST"]],on=["date","code"],validate="one_to_one")
    frame["winner_float"]=frame.winner.astype(float)
    frame.to_parquet(output/"cohort.parquet",index=False,compression="zstd")
    return frame,{"event_sources":sources,"cohort_sha256":sha(output/"cohort.parquet")}


def probability_records(frame:pd.DataFrame,groups:list[str]) -> list[dict]:
    records=[]
    for key,part in frame.groupby(groups,observed=True,dropna=False):
        if not isinstance(key,tuple):key=(key,)
        daily=part.groupby("date").winner_float.mean().dropna()
        records.append({**dict(zip(groups,key)),"rows":len(part),"known":int(part.known_label.sum()),
            "unknown":int((~part.known_label).sum()),"winners":int(part.winner_float.sum()),
            "pooled_probability":float(part.winner_float.mean()) if part.known_label.any() else None,
            "daily_probability":float(daily.mean()) if len(daily) else None,
            "probability_week_ci":weekly_interval(daily),"days":len(daily)})
    return records


def summarize(output:Path=ROOT) -> dict:
    if (output/"analysis_report.json").exists():
        raise ValueError("Do not overwrite a completed exploratory portrait")
    frame,provenance=assemble(output)
    overview=frame.groupby(["half","board","cohort"]).agg(rows=("code","size"),
        days=("date","nunique"),necessary_tradeable=("necessary_tradeable","sum"),
        sealed_quote_1449=("sealed_quote_1449","sum"),next_reference_gap=("next_reference_gap","sum")).reset_index()
    entry=frame.groupby(["half","board","cohort","necessary_tradeable","entry_evidence"]).size().rename("rows").reset_index()
    overview.to_parquet(output/"overview.parquet",index=False)
    entry.to_parquet(output/"entry_coverage.parquet",index=False)
    # Market context varies across dates. Ranking a leave-one-out market average
    # within one date mechanically reverses own-stock return, so do not do that.
    context=frame.groupby(["date","half","board"]).agg(market_return=("market_return","mean"),
        market_rising=("market_rising","mean"),winner_probability=("winner_float","mean")).reset_index()
    context.to_parquet(output/"market_context_daily.parquet",index=False)
    # Ranks use the entire point-in-time input pool, including unknown future labels.
    ranks=frame.groupby(["date","board"])[list(FEATURES)].rank(pct=True,method="average")
    feature_rows,band_rows,matched_rows=[],[],[]
    baseline=frame.loc[frame.necessary_tradeable].groupby(["date","board"]).winner_float.mean()
    for feature,title in FEATURES.items():
        rank=ranks[feature]
        for scope,mask in (("all",np.ones(len(frame),dtype=bool)),("necessary",frame.necessary_tradeable)):
            part=frame.loc[mask,["date","half","board","cohort","known_label","winner_float",feature]].copy()
            part["rank"]=rank.loc[part.index]
            desc=part.groupby(["half","board","cohort"])[feature].agg(["count","mean","median"])
            for key,row in desc.iterrows():
                feature_rows.append({"feature":feature,"title":title,"scope":scope,
                    "half":key[0],"board":key[1],"cohort":key[2],**row.to_dict()})
        part=frame.loc[frame.necessary_tradeable,["date","half","board","known_label","winner_float",feature]].copy()
        part["rank"]=rank.loc[part.index]
        part["band"]=np.select([part["rank"].isna(),part["rank"].le(.2),part["rank"].ge(.8)],
            ["missing","low20","high20"],default="middle60")
        # Full denominators include missing future labels; only known labels define an observed rate.
        for (half,board,band),group in part.groupby(["half","board","band"],observed=True):
            daily=group.groupby("date").winner_float.mean().dropna()
            reference=baseline.xs(board,level="board").reindex(daily.index)
            delta=daily-reference
            band_rows.append({"feature":feature,"title":title,"half":half,"board":board,"band":band,
                "rows":len(group),"known":int(group.known_label.sum()),"unknown":int((~group.known_label).sum()),
                "winners":int(group.winner_float.sum()),"probability":float(group.winner_float.mean()) if group.known_label.any() else None,
                "daily_probability":float(daily.mean()) if len(daily) else None,
                "daily_excess_probability":float(delta.mean()) if len(delta) else None,
                "excess_week_ci":weekly_interval(delta),"days":len(daily)})
        # Stratification controls broad prior strength and liquidity, not the path features being studied.
        mask=frame.necessary_tradeable&frame.known_label&frame.return20_prior_adjusted.notna()&rank.notna()
        part=frame.loc[mask,["date","half","board","winner_float","return_1449","return20_prior_adjusted","price_1449","amount_1449"]].copy()
        part["rank"]=rank.loc[part.index]
        for name,val in (("day_bin",part.return_1449/.02),("prior_bin",part.return20_prior_adjusted/.10),
            ("price_bin",np.log2(part.price_1449/5)),("amount_bin",np.log2(part.amount_1449/3e7))):
            part[name]=np.floor(val).astype("int64")
        keys=["date","board","day_bin","prior_bin","price_bin","amount_bin"]
        table=part.groupby(keys+["winner_float"]).agg(mean_rank=("rank","mean"),n=("rank","size")).unstack("winner_float")
        if ("n",1.) not in table or ("n",0.) not in table:
            continue
        table=table.dropna()
        table["weight"]=table["n",1.]
        table["difference"]=(table["mean_rank",1.]-table["mean_rank",0.])*table["weight"]
        clean=pd.DataFrame({"weight":table["weight"],"difference":table["difference"]}).reset_index()
        daily=clean.groupby(["date","board"])[["weight","difference"]].sum()
        daily["rank_difference"]=daily.difference/daily.weight
        daily=daily.reset_index();daily["feature"]=feature
        matched_rows.append(daily)
        print(json.dumps({"feature_completed":feature},ensure_ascii=False),flush=True)
    pd.DataFrame(feature_rows).to_parquet(output/"feature_portraits.parquet",index=False)
    pd.DataFrame(band_rows).to_parquet(output/"feature_probabilities.parquet",index=False)
    matched=pd.concat(matched_rows,ignore_index=True)
    matched.to_parquet(output/"matched_daily.parquet",index=False)
    matched["half"]=matched.date.str[:4]+np.where(matched.date.str[5:7].le("06"),"H1","H2")
    matched_summary=[]
    for (feature,half,board),part in matched.groupby(["feature","half","board"]):
        total=int(frame.loc[frame.necessary_tradeable&frame.half.eq(half)&frame.board.eq(board),"winner_float"].sum())
        matched_summary.append({"feature":feature,"title":FEATURES[feature],"half":half,"board":board,
            "days":len(part),"matched_winners":int(part.weight.sum()),"all_winners":total,
            "coverage":float(part.weight.sum()/total) if total else None,
            "daily_rank_difference":float(part.rank_difference.mean()),
            "rank_difference_week_ci":weekly_interval(part.set_index("date").rank_difference)})
    pd.DataFrame(matched_summary).to_parquet(output/"matched_summary.parquet",index=False)
    events=[]
    for feature,title in EVENTS.items():
        part=frame.loc[frame.necessary_tradeable,["date","half","board","known_label","winner_float",feature]].copy()
        part["event"]=part[feature].fillna(-1)
        for row in probability_records(part,["half","board","event"]):
            events.append({"feature":feature,"title":title,**row})
    pd.DataFrame(events).to_parquet(output/"event_probabilities.parquet",index=False)
    report={**provenance,"rows":len(frame),"feature_count":len(FEATURES),
        "base_probabilities":probability_records(frame.loc[frame.necessary_tradeable],["half","board"]),
        "unknown_label_rows":int((~frame.known_label).sum()),
        "prefix_amount_bad_rows":int(frame.prefix_amount_bad_bars.gt(0).sum()),
        "necessary_tradeable_rows":int(frame.necessary_tradeable.sum()),
        "outputs_sha256":{p.name:sha(p) for p in output.glob("*.parquet")},
        "exploratory_only":True,"strategy_returns_computed":False,"new_2026_prices_read":False}
    save_json(output/"analysis_report.json",report)
    return {k:v for k,v in report.items() if k not in ("base_probabilities","outputs_sha256")}


if __name__=="__main__":
    print(json.dumps(summarize(),ensure_ascii=False,indent=2))
