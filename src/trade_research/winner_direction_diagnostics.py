"""Post-result explanation of the fixed lists; never a new selection filter."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .corporate_cash import save_json, sha
from .next_day_winner import ROOT as PORTRAIT
from .winner_direction import ROOT, MODELS


def evaluate() -> dict:
    frozen=json.loads((ROOT/"input_report.json").read_text())
    labels_path=PORTRAIT/"labels.parquet"
    if sha(labels_path)!=frozen["training"]["source_labels_sha256"]:
        raise ValueError("The independently checked daily label source changed")
    labels=pd.read_parquet(labels_path,filters=[("date",">=","2025-01-02"),("date","<=","2025-12-17")])
    pool=pd.read_parquet(ROOT/"visible_pool.parquet",filters=[("date",">=","2025-01-02")])
    scores=pd.read_parquet(ROOT/"all_scores.parquet")
    files={"labels":sha(labels_path),"pool":sha(ROOT/"visible_pool.parquet"),"scores":sha(ROOT/"all_scores.parquet")}
    case_rows=[];path_rows=[];raw_paths=[]
    periods=(("2025","2025-01-01","2025-12-31"),("2025H1","2025-01-01","2025-06-30"),("2025H2","2025-07-01","2025-12-31"))
    for name in ("eligible_pool",*MODELS):
        if name=="eligible_pool":frame=pool.copy()
        else:
            selected=pd.read_parquet(ROOT/name/"signals.parquet")
            frame=selected.loc[selected.arm.eq("high")].copy()
        frame=frame.merge(labels,on=["date","code"],how="left",validate="one_to_one")
        frame=frame.merge(scores,on=["date","code"],how="left",validate="one_to_one")
        if frame.known_label.isna().any() or frame.p_up.isna().any():raise ValueError("An explanatory row disappeared")
        close=np.rint(frame.next_close.fillna(0)*100).astype("int64")
        reference=np.rint(frame.next_preclose.fillna(0)*100).astype("int64")
        frame["up_case"]=frame.known_label&(close*100>=reference*105)
        frame["down_case"]=frame.known_label&(close*100<=reference*95)
        frame["unknown_case"]=~frame.known_label
        for period,first,last in periods:
            part=frame.loc[frame.date.between(first,last)]
            for board in ("all","main","chinext","star"):
                p=part if board=="all" else part.loc[part.board.eq(board)]
                if p.empty:continue
                daily=p.groupby("date")[["up_case","down_case","unknown_case","p_up","p_down"]].mean()
                case_rows.append({"model":name,"period":period,"board":board,"rows":len(p),"dates":p.date.nunique(),
                    "up_cases":int(p.up_case.sum()),"down_cases":int(p.down_case.sum()),"unknown_cases":int(p.unknown_case.sum()),
                    "up_rate_daily_lower":float(daily.up_case.mean()),"up_rate_daily_upper":float((daily.up_case+daily.unknown_case).mean()),
                    "down_rate_daily_lower":float(daily.down_case.mean()),"down_rate_daily_upper":float((daily.down_case+daily.unknown_case).mean()),
                    "predicted_up_daily_mean":float(daily.p_up.mean()),"predicted_down_daily_mean":float(daily.p_down.mean()),
                    "visible_gain_median":float(p.return_1449.median()),"prior5_gain_median":float(p.return5_prior_adjusted.median())})
        if name=="eligible_pool":continue
        folder=ROOT/name/"continued";ledger_path=folder/"tick_cost_scenario.parquet"
        files[name+"_ledger"]=sha(ledger_path)
        trades=pd.read_parquet(ledger_path);queue=pd.read_parquet(folder/"execution_queue_audit.parquet")
        t=trades.loc[trades.arm.eq("high")].merge(frame[["date","code","board","next_date","next_open","day_close",
            "next_close","known_label","next_reference_gap"]],on=["date","code"],how="left",validate="one_to_one")
        t=t.merge(queue[["date","code","horizon","queue_allocation_unverified"]],on=["date","code","horizon"],validate="one_to_one")
        # This is an explicitly conditional path description, not sample exclusion from P&L.
        t["path_eligible"]=(t.entry_status.eq("filled")&t.exit_date.eq(t.next_date)&t.known_label
            &t.execution_source_valid&~t.catalog_action_applied&~t.next_reference_gap.fillna(True).astype(bool)
            &~t.queue_allocation_unverified&~t.corporate_action_crossed.fillna(True).astype(bool))
        p=t.loc[t.path_eligible].copy();entry=p.entry_price/1.0005;exit_=p.exit_price/.9995
        p["entry_drift_from_1449"]=entry/p.price_1449-1
        p["after_entry_contribution"]=(p.day_close-entry)/entry
        p["overnight_contribution"]=(p.next_open-p.day_close)/entry
        p["next_session_contribution"]=(exit_-p.next_open)/entry
        p["gross_tail_return"]=exit_/entry-1
        p["after_planned_exit_contribution"]=(p.next_close-exit_)/entry
        parts=["after_entry_contribution","overnight_contribution","next_session_contribution"]
        if (p[parts].sum(axis=1)-p.gross_tail_return).abs().max()>1e-12:
            raise ValueError("The price-path contributions fail to reconcile")
        cols=["entry_drift_from_1449",*parts,"gross_tail_return","after_planned_exit_contribution"]
        p["model"]=name;raw_paths.append(p[["model","date","code","board",*cols]])
        for period,first,last in periods:
            selected=t.loc[t.date.between(first,last)];chosen=p.loc[p.date.between(first,last)]
            for board in ("all","main","chinext","star"):
                s=selected if board=="all" else selected.loc[selected.board.eq(board)]
                g=chosen if board=="all" else chosen.loc[chosen.board.eq(board)]
                if g.empty:continue
                path_rows.append({"model":name,"period":period,"board":board,"selected_rows":len(s),"included_rows":len(g),
                    "dates":g.date.nunique(),**g.groupby("date")[cols].mean().mean().to_dict()})
    path_output=ROOT/"path_diagnostics.parquet";pd.concat(raw_paths,ignore_index=True).to_parquet(path_output,index=False)
    report={"interpretation":"post_result_explanation_on_fixed_lists_no_new_rule_or_claim_of_actual_fund_identity",
        "label_rates_unknowns_bounded":True,"path_diagnostic_exclusions_do_not_change_full_trade_ledger":True,
        "contribution_denominator":"raw_entry_vwap_for_all_three_additive_price_segments",
        "case_rates":case_rows,"path_metrics":path_rows,"source_sha256":files,"paths_sha256":sha(path_output),
        "new_2026_prices_read":False}
    save_json(ROOT/"diagnostic_report.json",report);return report


if __name__=="__main__":print(json.dumps(evaluate(),ensure_ascii=False,indent=2))
