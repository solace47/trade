"""Check both tails before calling a next-day winner correlate a bullish signal."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .corporate_cash import save_json,sha
from .next_day_winner import ROOT
from .next_day_winner_analysis import FEATURES
from .reference_gain_accounting import weekly_interval

PROTOCOL="config/next_day_winner_tail_balance_protocol.json"


def evaluate() -> dict:
    if (ROOT/"balance_report.json").exists():
        raise ValueError("Cannot overwrite an inspected tail-balance diagnostic")
    from pathlib import Path
    report=json.loads((ROOT/"analysis_report.json").read_text())
    assert sha(ROOT/"cohort.parquet")==report["cohort_sha256"]
    f=pd.read_parquet(ROOT/"cohort.parquet")
    labels=pd.read_parquet(ROOT/"labels.parquet",columns=["date","code","known_label","next_close","next_preclose"])
    close=np.rint(labels.next_close.fillna(0)*100).astype("int64")
    reference=np.rint(labels.next_preclose.fillna(0)*100).astype("int64")
    labels["loser_float"]=(close*100<=reference*95).where(labels.known_label).astype(float)
    f=f.merge(labels[["date","code","loser_float"]],on=["date","code"],validate="one_to_one")
    f["tail_balance"]=f.winner_float-f.loser_float
    f["day_range"]=(f.high_1449-f.low_1449)/f.preclose
    necessary=f.loc[f.necessary_tradeable].copy()
    dailybase=necessary.groupby(["date","board"])[["winner_float","loser_float","next_gain","tail_balance"]].mean()
    ranks=f.groupby(["date","board"])[list(FEATURES)].rank(pct=True,method="average")
    rows,dailyrows,range_rows=[],[],[]
    for feature,title in FEATURES.items():
        p=necessary[["date","half","board","winner_float","loser_float","next_gain","tail_balance","known_label"]].copy()
        p["rank"]=ranks.loc[p.index,feature]
        p["band"]=np.select([p["rank"].isna(),p["rank"].le(.2),p["rank"].ge(.8)],
            ["missing","low20","high20"],default="middle60")
        for (half,board,band),g in p.groupby(["half","board","band"]):
            daily=g.groupby("date")[["winner_float","loser_float","next_gain","tail_balance"]].mean().dropna()
            reference=dailybase.xs(board,level="board").reindex(daily.index)
            delta=daily-reference
            row={"feature":feature,"title":title,"half":half,"board":board,"band":band,
                "rows":len(g),"known":int(g.known_label.sum()),"unknown":int((~g.known_label).sum()),
                "winners":int(g.winner_float.sum()),"losers":int(g.loser_float.sum()),"days":len(daily)}
            for metric,name in (("winner_float","winner_probability"),("loser_float","loser_probability"),
                ("next_gain","reference_day_return"),("tail_balance","tail_balance")):
                row[name]=float(daily[metric].mean()) if len(daily) else None
                row[name+"_difference"]=float(delta[metric].mean()) if len(daily) else None
                row[name+"_difference_week_ci"]=weekly_interval(delta[metric])
            rows.append(row)
            daily=daily.reset_index();daily["feature"]=feature;daily["board"]=board;daily["band"]=band
            dailyrows.append(daily)
        rank=ranks[feature]
        mask=f.necessary_tradeable&f.known_label&f.return20_prior_adjusted.notna()&rank.notna()
        p=f.loc[mask,["date","board","winner_float","return_1449","return20_prior_adjusted","price_1449","amount_1449","day_range"]].copy()
        p["rank"]=rank.loc[p.index]
        for name,val in (("day_bin",p.return_1449/.02),("prior_bin",p.return20_prior_adjusted/.10),
            ("price_bin",np.log2(p.price_1449/5)),("amount_bin",np.log2(p.amount_1449/3e7)),("range_bin",p.day_range/.02)):
            p[name]=np.floor(val).astype("int64")
        keys=["date","board","day_bin","prior_bin","price_bin","amount_bin","range_bin"]
        table=p.groupby(keys+["winner_float"]).agg(mean_rank=("rank","mean"),n=("rank","size")).unstack("winner_float").dropna()
        weighted=(table["mean_rank",1.]-table["mean_rank",0.])*table["n",1.]
        t=pd.DataFrame({"weight":table["n",1.],"weighted_difference":weighted}).reset_index()
        daily=t.groupby(["date","board"])[["weight","weighted_difference"]].sum().reset_index()
        daily["rank_difference"]=daily.weighted_difference/daily.weight
        daily["feature"]=feature
        range_rows.append(daily)
        print(json.dumps({"balance_completed":feature},ensure_ascii=False),flush=True)
    pd.DataFrame(rows).to_parquet(ROOT/"tail_balance.parquet",index=False)
    pd.concat(dailyrows,ignore_index=True).to_parquet(ROOT/"tail_balance_daily.parquet",index=False)
    ranged=pd.concat(range_rows,ignore_index=True)
    ranged.to_parquet(ROOT/"range_matched_daily.parquet",index=False)
    ranged["half"]=ranged.date.str[:4]+np.where(ranged.date.str[5:7].le("06"),"H1","H2")
    summary=[]
    for (feature,half,board),p in ranged.groupby(["feature","half","board"]):
        total=int(necessary.loc[necessary.half.eq(half)&necessary.board.eq(board),"winner_float"].sum())
        summary.append({"feature":feature,"title":FEATURES[feature],"half":half,"board":board,
            "days":len(p),"matched_winners":int(p.weight.sum()),"all_winners":total,"coverage":float(p.weight.sum()/total),
            "daily_rank_difference":float(p.rank_difference.mean()),
            "rank_difference_week_ci":weekly_interval(p.set_index("date").rank_difference)})
    pd.DataFrame(summary).to_parquet(ROOT/"range_matched_summary.parquet",index=False)
    baseline=[]
    for (half,board),p in necessary.groupby(["half","board"]):
        daily=p.groupby("date")[["winner_float","loser_float","next_gain","tail_balance"]].mean().dropna()
        baseline.append({"half":half,"board":board,"known":int(p.known_label.sum()),"winners":int(p.winner_float.sum()),
            "losers":int(p.loser_float.sum()),**daily.mean().to_dict()})
    result={"rule_commit":"82e9e1a","protocol_sha256":sha(Path(PROTOCOL)),
        "base_sha256":sha(ROOT/"cohort.parquet"),"baseline":baseline,
        "output_sha256":{name:sha(ROOT/name) for name in ("tail_balance.parquet","tail_balance_daily.parquet",
            "range_matched_daily.parquet","range_matched_summary.parquet")},
        "negative_labels":int(labels.loser_float.sum()),"exploratory_post_portrait":True,
        "trading_returns_computed":False,"new_2026_prices_read":False}
    save_json(ROOT/"balance_report.json",result)
    return {k:v for k,v in result.items() if k not in ("baseline","output_sha256")}


if __name__=="__main__":
    print(json.dumps(evaluate(),ensure_ascii=False,indent=2))
