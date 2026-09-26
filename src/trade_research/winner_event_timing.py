"""Fixed winner case/control samples and complete issuer disclosure windows."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path
import re

import pandas as pd

from .absolute_ridge import match_controls
from .corporate_cash import API, curl, save_json, sha
from .next_day_winner import ROOT as PORTRAIT, calendar

ROOT=Path("data/research/winner_event_timing")
PROTOCOL=Path("config/winner_event_timing_protocol.json")
RULE_COMMIT="6244021"
STOCK_MAP=Path("data/research/corporate_cash/source/stock_list.json")


def freeze(output:Path=ROOT) -> dict:
    if (output/"input_report.json").exists():raise ValueError("Do not replace fixed event cases")
    protocol=json.loads(PROTOCOL.read_text());output.mkdir(exist_ok=True)
    fields=["date","code","half","board","necessary_tradeable","known_label","winner","cohort","next_date",
        "price_1449","amount_1449","return_1449","return20_prior_adjusted"]
    pool=pd.read_parquet(PORTRAIT/"cohort.parquet",columns=fields)
    expected=json.loads((PORTRAIT/"analysis_report.json").read_text())["cohort_sha256"]
    if sha(PORTRAIT/"cohort.parquet")!=expected:raise ValueError("The checked portrait changed")
    pool=pool.loc[pool.necessary_tradeable&pool.date.between("2024-01-09","2025-12-30")&pool.next_date.le("2025-12-31")].copy()
    cases=pool.loc[pool.known_label&pool.winner].copy()
    cases["identity_hash"]=[hashlib.sha256((protocol["case_hash_prefix"]+r.date+"|"+r.code).encode()).hexdigest() for r in cases.itertuples()]
    chosen=[]
    for _,group in cases.groupby(["half","board","cohort"],sort=True):
        p=group.sort_values(["identity_hash","date","code"]).drop_duplicates("date").head(5)
        if len(p)!=5:raise ValueError("A declared case stratum has fewer than five distinct dates")
        chosen.append(p)
    cases=pd.concat(chosen,ignore_index=True).sort_values(["date","identity_hash","code"])
    if len(cases)!=120 or cases.duplicated(["date","code"]).any():raise ValueError("The case strata differ from the protocol")
    cases["daily_rank"]=cases.groupby("date").cumcount()+1
    controls_pool=pool.loc[pool.known_label&~pool.winner]
    candidates=pd.concat([controls_pool,cases[fields]],ignore_index=True)
    candidates["amount_signal"],candidates["price_signal"]=candidates.amount_1449,candidates.price_1449
    candidates["return_1450"]=candidates.return_1449  # Existing matcher alias only.
    controls=match_controls(cases[["date","code","daily_rank"]],candidates)
    cases["arm"],cases["pair_id"]="case",cases.code
    controls=controls.merge(pool,on=["date","code"],validate="one_to_one")
    controls["arm"]="control"
    members=pd.concat([cases,controls],ignore_index=True).sort_values(["date","arm","daily_rank","code"]).reset_index(drop=True)
    members["pair_id"]=members.date+":"+members.pair_id
    days=calendar();positions={d:i for i,d in enumerate(days)}
    members["source_first"]=[days[positions[d]-5] for d in members.date]
    members["source_last"]=members.next_date
    if members.source_first.min()<"2024-01-01" or members.source_last.max()>"2025-12-31":raise ValueError("The source window is outside the declared years")
    path=output/"members.parquet";members.to_parquet(path,index=False,compression="zstd")
    result={"rule_commit":RULE_COMMIT,"protocol_sha256":sha(PROTOCOL),"source_cohort_sha256":expected,
        "members_sha256":sha(path),"cases":len(cases),"controls":len(controls),"dates":members.date.nunique(),
        "by_case_stratum":cases.groupby(["half","board","cohort"]).size().rename("cases").reset_index().to_dict("records"),
        "source_first":members.source_first.min(),"source_last":members.source_last.max(),
        "stock_mapping_sha256":sha(STOCK_MAP),"source_windows_fetched":False,"new_2026_prices_read":False}
    save_json(output/"input_report.json",result);return result


def fetch(output:Path=ROOT) -> dict:
    inputs=json.loads((output/"input_report.json").read_text())
    checks=json.loads((output/"independent_input_checks.json").read_text())
    if checks["input_report_sha256"]!=sha(output/"input_report.json"):
        raise ValueError("The independently checked event sample changed")
    if sha(output/"members.parquet")!=inputs["members_sha256"] or sha(STOCK_MAP)!=inputs["stock_mapping_sha256"]:
        raise ValueError("The fixed identities or disclosure mapping changed")
    members=pd.read_parquet(output/"members.parquet")
    mapping={r["code"]:r for r in json.loads(STOCK_MAP.read_text())["stockList"]}
    source=output/"source";source.mkdir(exist_ok=True)
    def one(row):
        stem=f"{row.date}_{row.code}";records=[];pages=[]
        if row.code[3:] not in mapping:
            return {"date":row.date,"code":row.code,"complete":False,"reason":"missing_historical_issuer_mapping"},records
        security=mapping[row.code[3:]];expected_total=None
        try:
            for page in range(1,21):
                path=source/f"{stem}_{page:02d}.json"
                if not path.exists():
                    body=curl(API,{"stock":f"{row.code[3:]},{security['orgId']}","tabName":"fulltext",
                        "pageSize":30,"pageNum":page,"column":"sse" if row.code.startswith("sh.") else "szse",
                        "seDate":row.source_first+"~"+row.source_last,"searchkey":""})
                    json.loads(body)
                    path.write_bytes(body)
                payload=json.loads(path.read_text());total=payload.get("totalAnnouncement")
                batch=payload.get("announcements") or []
                if not isinstance(total,int) or total<0 or any(r.get("secCode")!=row.code[3:] for r in batch):
                    raise ValueError("The disclosure response is incomplete or for a different issuer")
                if expected_total is None:expected_total=total
                if total!=expected_total:raise ValueError("Disclosure count changed between pages")
                records.extend(batch);pages.append({"path":str(path),"sha256":sha(path)})
                if len(records)>=total:
                    if len(records)!=total or len({r["announcementId"] for r in records})!=total:
                        raise ValueError("Disclosure pagination has duplicates or missing items")
                    return {"date":row.date,"code":row.code,"complete":True,"announcements":total,"pages":pages},records
                if not batch:raise ValueError("A required page is empty")
            raise ValueError("Unexpectedly long issuer disclosure window")
        except Exception as error:
            return {"date":row.date,"code":row.code,"complete":False,"reason":str(error),"pages":pages},[]
    coverage=[];all_records=[]
    with ThreadPoolExecutor(max_workers=2) as executor:
        for i,(row,(status,records)) in enumerate(zip(members.itertuples(),executor.map(one,members.itertuples())),1):
            coverage.append(status)
            for record in records:
                item=dict(record);item["case_signal_date"]=row.date;item["case_code"]=row.code
                all_records.append(item)
            if i%20==0 or i==len(members):print(f"Issuer windows {i}/{len(members)}, complete {sum(r['complete'] for r in coverage)}",flush=True)
    save_json(output/"announcements.json",all_records)
    result={"members_sha256":sha(output/"members.parquet"),"windows":len(members),
        "complete_windows":sum(r["complete"] for r in coverage),"all_complete":all(r["complete"] for r in coverage),
        "announcement_window_rows":len(all_records),"unique_announcements":len({r["announcementId"] for r in all_records}),
        "coverage":coverage,"announcements_sha256":sha(output/"announcements.json")}
    save_json(output/"source_report.json",result);return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("stage",choices=["freeze","fetch"])
    args=parser.parse_args();print(json.dumps(globals()[args.stage](),ensure_ascii=False,indent=2))
