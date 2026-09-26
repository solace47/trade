"""Date and clock evidence in complete, preselected case/control disclosure windows."""
import html
import json
from pathlib import Path
import re

import pandas as pd

from .corporate_cash import save_json,sha
from .winner_event_timing import ROOT,PROTOCOL

PHASES=("any_window","earlier_date","decision_date","after_decision_date",
        "decision_before_1449_clock","decision_after_1449_clock","next_day_after_1500_clock")


def evaluate() -> dict:
    protocol=json.loads(PROTOCOL.read_text());source=json.loads((ROOT/"source_report.json").read_text())
    if not source["all_complete"]:raise ValueError("Incomplete issuer windows must not be treated as no news")
    if sha(ROOT/"announcements.json")!=source["announcements_sha256"] or sha(ROOT/"members.parquet")!=source["members_sha256"]:
        raise ValueError("The fixed disclosure sources changed")
    members=pd.read_parquet(ROOT/"members.parquet");raw=json.loads((ROOT/"announcements.json").read_text())
    categories=protocol["categories"];parsed=[]
    keyed=members.set_index(["date","code"],drop=False)
    for record in raw:
        key=record["case_signal_date"],record["case_code"];member=keyed.loc[key]
        timestamp=pd.to_datetime(record["announcementTime"],unit="ms",utc=True).tz_convert("Asia/Shanghai")
        day=timestamp.strftime("%Y-%m-%d");clock=timestamp.strftime("%H:%M:%S")
        match=re.search(r"finalpage/(\d{4}-\d{2}-\d{2})/",record["adjunctUrl"])
        document_date=match.group(1) if match else None
        title=html.unescape(re.sub(r"<[^>]*>","",record["announcementTitle"]))
        conflict=document_date is None or day!=document_date
        if not member.source_first<=day<=member.source_last:raise ValueError("A disclosure date lies outside its complete query window")
        flags={name:bool(re.search(pattern,title)) for name,pattern in categories.items()}
        row={"date":key[0],"code":key[1],"announcement_id":str(record["announcementId"]),"title":title,
            "source_timestamp":timestamp.isoformat(),"source_date":day,"source_clock":clock,
            "pdf_path_date":document_date,"date_conflict":conflict,"has_nonmidnight_clock":clock!="00:00:00",
            "url":"https://static.cninfo.com.cn/"+record["adjunctUrl"],"any_disclosure":True,
            **flags,"uncategorized":not any(flags.values()),"any_window":True,
            "earlier_date":not conflict and day<member.date,
            "decision_date":not conflict and day==member.date,
            "after_decision_date":not conflict and day>member.date,
            "decision_before_1449_clock":not conflict and day==member.date and "00:00:00"<clock<="14:49:00",
            "decision_after_1449_clock":not conflict and day==member.date and clock>"14:49:00",
            "next_day_after_1500_clock":not conflict and day==member.next_date and clock>"15:00:00"}
        parsed.append(row)
    notices=pd.DataFrame(parsed);notices.to_parquet(ROOT/"parsed_notices.parquet",index=False)
    names=["any_disclosure",*categories,"uncategorized"]
    case_kind=members.loc[members.arm.eq("case"),["pair_id","cohort"]].rename(columns={"cohort":"case_kind"})
    members=members.merge(case_kind,on="pair_id",how="left",validate="many_to_one")
    panel=[]
    for phase in PHASES:
        for name in names:
            present=set(zip(notices.loc[notices[phase]&notices[name],"date"],notices.loc[notices[phase]&notices[name],"code"]))
            p=members[["date","code","pair_id","arm","half","board","case_kind"]].copy()
            p["phase"],p["category"]=phase,name
            p["present"]=[(r.date,r.code) in present for r in p.itertuples()]
            panel.append(p)
    panel=pd.concat(panel,ignore_index=True);panel.to_parquet(ROOT/"presence_panel.parquet",index=False)
    cells=[]
    for (phase,category),p in panel.groupby(["phase","category"],sort=True):
        cases=p.loc[p.arm.eq("case")];controls=p.loc[p.arm.eq("control")]
        paired=cases.merge(controls,on="pair_id",validate="one_to_one",suffixes=("_case","_control"))
        groups=[("all",cases,controls,paired)]
        for half in sorted(cases.half.unique()):groups.append((half,cases.loc[cases.half.eq(half)],controls.loc[controls.half.eq(half)],paired.loc[paired.half_case.eq(half)]))
        for board in ("main","chinext","star"):groups.append((board,cases.loc[cases.board.eq(board)],controls.loc[controls.board.eq(board)],paired.loc[paired.board_case.eq(board)]))
        for kind in ("gap_and_hold","intraday_hold"):groups.append((kind,cases.loc[cases.case_kind.eq(kind)],controls.loc[controls.case_kind.eq(kind)],paired.loc[paired.case_kind_case.eq(kind)]))
        for label,high,low,pairs in groups:
            cells.append({"phase":phase,"category":category,"group":label,"cases":len(high),"controls":len(low),"pairs":len(pairs),
                "case_presence":int(high.present.sum()),"control_presence":int(low.present.sum()),
                "matched_case_presence":int(pairs.present_case.sum()),
                "case_only":int((pairs.present_case&~pairs.present_control).sum()),
                "control_only":int((~pairs.present_case&pairs.present_control).sum()),
                "both":int((pairs.present_case&pairs.present_control).sum()),
                "neither":int((~pairs.present_case&~pairs.present_control).sum()),
                "case_fraction":None if high.empty else float(high.present.mean()),
                "control_fraction":None if low.empty else float(low.present.mean()),
                "paired_fraction_difference":None if pairs.empty else float((pairs.present_case.astype(int)-pairs.present_control.astype(int)).mean())})
    result={"interpretation":"exploratory_case_control_title_presence_not_causation_or_unconditional_win_probability",
        "source_report_sha256":sha(ROOT/"source_report.json"),"protocol_sha256":sha(PROTOCOL),
        "members_sha256":sha(ROOT/"members.parquet"),"notices":len(notices),
        "midnight_placeholders":int(notices.source_clock.eq("00:00:00").sum()),
        "nonmidnight_clocks":int(notices.has_nonmidnight_clock.sum()),"date_conflicts":int(notices.date_conflict.sum()),
        "clock_fields_are_not_independently_verified_first_publication_times":True,
        "rows_by_source_phase":{phase:int(notices[phase].sum()) for phase in PHASES},
        "frequency_cells":cells,"source_windows_complete":source["complete_windows"],
        "outputs_sha256":{name:sha(ROOT/name) for name in ("parsed_notices.parquet","presence_panel.parquet")},
        "new_2026_prices_read":False,"monthly_repurchase_pdfs_read":False}
    save_json(ROOT/"analysis_report.json",result);return result


if __name__=="__main__":print(json.dumps(evaluate(),ensure_ascii=False,indent=2))
