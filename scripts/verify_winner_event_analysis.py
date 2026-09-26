"""Reassemble every source page and independently count the fixed event tables."""
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd

from trade_research.corporate_cash import save_json,sha

root=Path("data/research/winner_event_timing");source=json.loads((root/"source_report.json").read_text())
report=json.loads((root/"analysis_report.json").read_text());protocol=json.loads(Path("config/winner_event_timing_protocol.json").read_text())
members=pd.read_parquet(root/"members.parquet");member=members.set_index(["date","code"])
assert len(source["coverage"])==len(members)==216 and source["all_complete"]
assert sha(root/"source_report.json")==report["source_report_sha256"]
assert sha(root/"announcements.json")==source["announcements_sha256"]
reassembled=[];pages=0
for window in source["coverage"]:
    assert window["complete"] and (window["date"],window["code"]) in member.index
    received=[]
    for item in window["pages"]:
        path=Path(item["path"]);assert sha(path)==item["sha256"]
        payload=json.loads(path.read_text());assert payload["totalAnnouncement"]==window["announcements"]
        rows=payload.get("announcements") or [];received.extend(rows);pages+=1
    assert len(received)==window["announcements"]==len({r["announcementId"] for r in received})
    for record in received:
        assert record["secCode"]==window["code"][3:]
        item=dict(record,case_signal_date=window["date"],case_code=window["code"]);reassembled.append(item)
assert reassembled==json.loads((root/"announcements.json").read_text())
parsed=pd.read_parquet(root/"parsed_notices.parquet").set_index(["date","code","announcement_id"])
for raw in reassembled:
    key=raw["case_signal_date"],raw["case_code"];p=parsed.loc[(*key,str(raw["announcementId"]))]
    dt=datetime.fromtimestamp(raw["announcementTime"]/1000,tz=timezone.utc).astimezone(ZoneInfo("Asia/Shanghai"))
    date,clock=dt.strftime("%Y-%m-%d"),dt.strftime("%H:%M:%S")
    pdf_day=re.fullmatch(r"finalpage/(\d{4}-\d{2}-\d{2})/[^/]+",raw["adjunctUrl"]).group(1)
    assert date==pdf_day==p.source_date==p.pdf_path_date and clock==p.source_clock
    assert member.loc[key].source_first<=date<=member.loc[key].source_last
    title=html.unescape(re.sub("<[^>]*>","",raw["announcementTitle"]));assert title==p.title
    for phase,value in {"earlier_date":date<key[0],"decision_date":date==key[0],"after_decision_date":date>key[0],
        "decision_before_1449_clock":date==key[0] and "00:00:00"<clock<="14:49:00",
        "decision_after_1449_clock":date==key[0] and clock>"14:49:00",
        "next_day_after_1500_clock":date==member.loc[key].next_date and clock>"15:00:00"}.items():assert p[phase]==value
c=duckdb.connect();c.register("notices",parsed.reset_index());c.register("members",members)
for category,pattern in protocol["categories"].items():
    errors=c.execute(f'SELECT count(*) FROM notices WHERE "{category}"<>regexp_matches(title,?)',[pattern]).fetchone()[0]
    assert errors==0
phases=["any_window","earlier_date","decision_date","after_decision_date","decision_before_1449_clock","decision_after_1449_clock","next_day_after_1500_clock"]
categories=["any_disclosure",*protocol["categories"],"uncategorized"]
assert parsed.any_window.all() and parsed.any_disclosure.all()
assert parsed.uncategorized.eq(~parsed[list(protocol["categories"])].any(axis=1)).all()
panel=[]
for phase in phases:
    for category in categories:
        found=c.sql(f'''SELECT m.date,m.code,m.pair_id,m.arm,m.half,m.board,h.cohort AS case_kind,
            bool_or(coalesce(n."{phase}" AND n."{category}",false)) AS present
            FROM members m JOIN members h ON m.pair_id=h.pair_id AND h.arm='case'
            LEFT JOIN notices n ON n.date=m.date AND n.code=m.code
            GROUP BY m.date,m.code,m.pair_id,m.arm,m.half,m.board,h.cohort''').df()
        found["phase"],found["category"]=phase,category;panel.append(found)
checked=pd.concat(panel,ignore_index=True);stored=pd.read_parquet(root/"presence_panel.parquet")
keys=["phase","category","date","code"]
pd.testing.assert_frame_equal(checked[stored.columns].sort_values(keys).reset_index(drop=True),
    stored.sort_values(keys).reset_index(drop=True),check_dtype=False)
for row in report["frequency_cells"]:
    p=checked.loc[checked.phase.eq(row["phase"])&checked.category.eq(row["category"])];group=row["group"]
    if group!="all":p=p.loc[p.half.eq(group)|p.board.eq(group)|p.case_kind.eq(group)]
    high=p.loc[p.arm.eq("case")];low=p.loc[p.arm.eq("control")]
    pairs=high.set_index("pair_id").present.to_frame("case").join(low.set_index("pair_id").present.rename("control"),how="inner")
    counts={"cases":len(high),"controls":len(low),"pairs":len(pairs),"case_presence":int(high.present.sum()),
        "control_presence":int(low.present.sum()),"matched_case_presence":int(pairs.case.sum()),
        "case_only":int((pairs.case&~pairs.control).sum()),"control_only":int((~pairs.case&pairs.control).sum()),
        "both":int((pairs.case&pairs.control).sum()),"neither":int((~pairs.case&~pairs.control).sum())}
    for name,value in counts.items():assert value==row[name]
    for name,values in (("case_fraction",high.present),("control_fraction",low.present),
        ("paired_fraction_difference",pairs.case.astype(int)-pairs.control.astype(int))):
        assert row[name] is None if not len(values) else abs(values.mean()-row[name])<1e-12
result={"complete_windows":len(source["coverage"]),"raw_pages":pages,"raw_announcements":len(reassembled),
    "source_clocks_checked":len(parsed),"presence_rows":len(checked),"frequency_cells":len(report["frequency_cells"]),
    "analysis_report_sha256":sha(root/"analysis_report.json"),"new_2026_prices_read":False}
save_json(root/"independent_source_analysis_checks.json",result);print(json.dumps(result,indent=2))
