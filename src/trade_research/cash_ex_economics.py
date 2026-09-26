"""Economic exploration of the unchanged, previously verified ex-date cohort."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path
import shutil

import pandas as pd

from .cash_ex_inputs import ROOT as SOURCE
from .cash_ex_notices import REVIEWS
from .corporate_cash import save_json, sha
from .quote_precision import quote_cents

ROOT = Path("data/research/cash_ex_economics")
RULE_COMMIT = "91a07ea"


def freeze(output: Path = ROOT) -> dict:
    if any((output/str(n)/"repriced.parquet").exists() for n in (20000,100000)):
        raise ValueError("Do not replace the event list after its returns exist")
    matching = json.loads((SOURCE/"matching_report.json").read_text())
    primary = json.loads((SOURCE/"notice_template_report.json").read_text())
    failed = [(r["half"],key) for r in matching["by_half"] for key,passed in r["checks"].items() if not passed]
    if failed != [("2024H2","pairs")] or matching["input_gate_passed"]:
        raise ValueError("This exploratory exception is limited to the documented sample-count failure")
    fingerprints = {**matching["inputs_sha256"],
        str(SOURCE/"input_pairs.parquet"):matching["pairs_sha256"],
        str(SOURCE/"notice_sources/scope_source_index.json"):primary["source_index_sha256"],
        str(SOURCE/"notice_scope.parquet"):primary["scope_sha256"],
        str(SOURCE/"notice_template_audit.json"):primary["audit_sha256"],
        str(REVIEWS):primary["reviews_sha256"]}
    for path,expected in fingerprints.items():
        if sha(Path(path)) != expected:
            raise ValueError("A previously verified source or input changed")
    if not primary["all_decision_terms_verified"] or primary["needs_review"]:
        raise ValueError("Original decision terms must remain verified")
    evidence = json.loads((SOURCE/"notice_template_audit.json").read_text())
    evidence = {(r["code"],r["date"]):r for r in evidence}
    candidates = pd.read_parquet(SOURCE/"verified_candidates.parquet")
    attempts = candidates.loc[candidates.daily_rank.le(5)].copy()
    pairs = pd.read_parquet(SOURCE/"input_pairs.parquet")
    base = pd.read_parquet(SOURCE/"base.parquet")
    verified_sources = []
    for row in attempts.itertuples():
        e = evidence[row.code,row.date]
        if (e["blocking_issues"] or e["positive_share_distribution"] or not e["cash_and_visibility_eligible"]
            or e["record_date"] != row.previous_market_date or e["record_date"] != row.dividRegistDate
            or e["action_date"] != row.date or e["notice_date"] > row.previous_market_date
            or Decimal(e["cash_per_share"]) <= 0
            or Decimal(e["cash_per_share"]) != Decimal(row.cash_per_share)
            or Decimal(e["reference_cash_per_share"]) != Decimal(row.reference_cash_per_share)
            or e["source_sha256"] != row.source_sha256):
            raise ValueError("An entry-reference exception lacks the verified cash-event contract")
        reference = (Decimal(str(row.previous_close))-Decimal(e["reference_cash_per_share"])).quantize(Decimal(".01"),rounding=ROUND_HALF_UP)
        if quote_cents(row.preclose) != int(reference*100):
            raise ValueError("The original cash terms do not reproduce the known ex-date reference")
        for path_key,hash_key in (("source_path","source_sha256"),("text_path","text_sha256"),("pages_path","pages_sha256")):
            if sha(Path(e[path_key])) != e[hash_key]:
                raise ValueError("A certified cached source or extraction changed")
        verified_sources.append({"date":row.date,"code":row.code,"source_sha256":e["source_sha256"],
            "verified_reference":str(reference),"cash_per_share":e["cash_per_share"]})
    columns = list(base.columns)+["daily_rank","cash_per_share","reference_cash_per_share",
        "cash_and_visibility_eligible","source_url","source_sha256","cash_yield","notice_date","dividRegistDate"]
    high = attempts[columns].copy()
    high["arm"],high["pair_id"],high["entry_reference_verified"] = "high",high.code,True
    low = pairs[["date","code","control_code","daily_rank","distance"]].rename(columns={"code":"pair_id","control_code":"code"})
    low = low.merge(base,on=["date","code"],validate="one_to_one")
    low["arm"],low["entry_reference_verified"] = "low",False
    if low.reference_gap.any():
        raise ValueError("Controls must still have an ordinary known reference")
    signals = pd.concat([high,low],ignore_index=True).sort_values(["date","arm","daily_rank","code"])
    signals["pair_id"] = signals.date+":"+signals.pair_id
    signals["raw_price_1449"] = signals.price_1449
    signals["price_1449"] = signals.price_1449.map(lambda p:quote_cents(p)/100)
    signals["isST"],signals["listing_age_sessions"] = 0,60
    if len(high) != 340 or len(low) != 283 or signals.duplicated(["date","code"]).any():
        raise ValueError("The original attempted or paired cohort changed")
    if not high.reference_gap.all() or not high.cash_and_visibility_eligible.all():
        raise ValueError("The ex-date contract does not cover every selected event")
    output.mkdir(parents=True,exist_ok=True)
    signals.to_parquet(output/"signals.parquet",index=False,compression="zstd")
    save_json(output/"verified_entry_sources.json",verified_sources)
    report = {"rule_commit":RULE_COMMIT,"interpretation":"new_exploratory_economics_original_sample_count_failure_preserved",
        "original_failed_checks":failed,"candidates":len(high),"controls":len(low),"unmatched_candidates":len(high)-len(low),
        "by_half":signals.groupby(["half","arm"]).agg(rows=("code","size"),days=("date","nunique")).reset_index().to_dict("records"),
        "signals_sha256":sha(output/"signals.parquet"),"source_fingerprints":fingerprints,
        "notionals":[20000,100000],"horizons":[1,5],"verified_entry_reference_rows":len(high),
        "ex_date_buyer_receives_current_dividend":False,"new_selected_lists_repriced":False,"holdout_read":False}
    report["models"] = {str(n):{key:report[key] for key in ("candidates","controls","signals_sha256","by_half")}
        for n in report["notionals"]}
    save_json(output/"input_report.json",report)
    for notional in report["notionals"]:
        folder = output/str(notional)
        folder.mkdir(exist_ok=True)
        shutil.copyfile(output/"signals.parquet",folder/"signals.parquet")
        save_json(folder/"input_report.json",report)
    return report


if __name__ == "__main__":
    report = freeze()
    print(json.dumps({k:report[k] for k in ("candidates","controls","unmatched_candidates","by_half","signals_sha256")},ensure_ascii=False,indent=2))
