"""Audit the frozen ex-date candidate notices before any strategy outcomes.

Template extraction is provisional: ambiguous fields are exposed for review,
never filled from a future price or silently removed from the candidate list.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path
import re
import unicodedata

import pandas as pd

from .cash_dividend_primary import primary_dates
from .cash_ex_inputs import ROOT
from .corporate_cash import save_json, sha


REVIEWS = Path("config/cash_ex_notice_reviews.json")


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).translate(
        str.maketrans({"−": "-", "－": "-", "﹣": "-", "—": "-", "–": "-"}))


def implementation_section(text: str) -> str:
    sections = re.split(r"(?m)^(?=[一二三四五六七八九][、.．])", text)
    selected = []
    for section in sections:
        head = compact_text(section.splitlines()[0]) if section.splitlines() else ""
        if (head.startswith(("一、", "二、"))
                and ("方案" in head or head == "二、权益分配")
                and not re.search(r"审议|通过|调整", head)):
            selected.append(compact_text(section))
    if not selected:
        # Some notices place the actual scheme after the reviewed proposal in
        # section one. Require the dated actual-scheme label, not the proposal.
        for section in sections:
            compact = compact_text(section)
            match = re.search(r"公司202[45]?\d?年(?:年度|中期|半年度)权益分派方案为:", compact)
            if match and compact.startswith("一、"):
                selected.append(compact[match.start():])
    return selected[0] if len(selected) == 1 else ""


def notice_dates(text: str, code: str) -> tuple[str, str, str]:
    if code.startswith("sh."):
        return primary_dates(text, code)
    table = re.search(r"股权登记日\s+最后交易日\s+除权", text)
    if table:
        return primary_dates("相关日期" + text[table.start():], "sh." + code[3:])
    compact = compact_text(text)
    date = r"(202[45])年(\d{1,2})月(\d{1,2})日"
    prefixes = (r"(?:股权|权益)登记日(?:期)?(?:\(R日\))?(?:为|是)?[:：]?",
                r"(?:除权(?:除)?息日|除息日|除权日)(?:期)?(?:为|是)?[:：]?",
                r"(?:现金(?:红利|股利)|股东红利)[,，]?(?:将)?于")
    dates = []
    for prefix in prefixes:
        match = re.search(prefix + date, compact)
        dates.append("-".join(f"{int(v):02d}" for v in match.groups()) if match else "")
    if not dates[0] or not dates[1]:
        raise ValueError("Primary registration or ex-date is missing")
    return tuple(dates)


def parse_terms(text: str, code: str) -> dict:
    compact = compact_text(text)
    issues = []
    if code[3:] not in compact[:350]:
        issues.append("security_header")
    try:
        record, action, pay = notice_dates(text, code)
    except (ValueError, IndexError):
        record = action = pay = ""
        issues.append("date_template")
    if code.startswith("sh."):
        if "每股分配比例" not in compact or "相关日期" not in compact:
            header = ""
            issues.append("cash_header_template")
        else:
            header = compact.split("每股分配比例", 1)[1].split("相关日期", 1)[0]
        pattern = r"每股(?:派发|分配)?现金(?:红利|股利|股息)(?:人民币)?([\d.]+)元"
        explicit = re.findall(r"A股" + pattern, header)
        values = explicit or re.findall(pattern, header)
        cash_values = {Decimal(v) for v in values}
        share_text = header
        differentiated = bool(re.search(r"差异化分红送转[:：]?是", compact))
        standard_reference = bool(re.search(r"差异化分红送转[:：]?否", compact))
    else:
        section = implementation_section(text)
        pattern = r"每10股(?:派发|派送|分配|派)(?:现金)?(?:红利|股利|股息)?(?:人民币)?(?:约)?([\d.]+)元"
        values = re.findall(pattern, section)
        # Within the actual implementation section the first distribution is
        # gross; later repetitions are tax examples, not alternative cash terms.
        cash_values = {Decimal(values[0]) / 10} if values else set()
        share_text = section
        excluded = re.findall(r"(?:剔除|扣除|扣减).{0,60}?([\d,.]+)股(?:后|的|为|\))", section)
        differentiated = ("差异化分红" in compact
                          or bool(re.search(r"折算.{0,8}每(?:10)?股现金", compact))
                          or any(Decimal(v.replace(",", "")) > 0 for v in excluded))
        standard_reference = bool(section) and not differentiated
    if len(cash_values) != 1:
        cash = None
        issues.append("actual_cash_ambiguous")
    else:
        cash = next(iter(cash_values))
        if cash <= 0:
            issues.append("nonpositive_cash")
    share_values = re.findall(r"(?:每股|每10股)(?:派送红股|送红股|送股|转增(?:股份)?)([\d.]+)股", share_text)
    positive_shares = any(Decimal(v) != 0 for v in share_values)
    if not share_text:
        issues.append("share_terms_template")
    close_label = r"(?:前收盘价格|前收盘价|股权登记日(?:\(202\d年\d+月\d+日\))?(?:股票)?收盘价(?:格)?|前一交易日收盘价|除权除息日的前一日收盘价)"
    references = {Decimal(v) for v in re.findall(close_label + r"-([\d.]+)", compact)}
    references.update(Decimal(v) for v in re.findall(close_label
        + r"-按[^。=]{0,45}?每股现金(?:分红|红利)(?:比例)?\((?:即)?([\d.]+)元", compact))
    if len(references) == 1:
        reference = next(iter(references))
    elif len(references) > 1:
        reference = None
        issues.append("multiple_reference_formulas")
    elif standard_reference:
        reference = cash
    else:
        reference = None
        issues.append("reference_cash_template")
    return {"record_date": record, "action_date": action, "pay_date": pay,
            "cash_per_share": str(cash) if cash is not None else None,
            "reference_cash_per_share": str(reference) if reference is not None else None,
            "positive_share_distribution": positive_shares,
            "share_values": share_values,
            "issues": issues}


def reviewed_terms(parsed: dict, text: str, source: dict, review: dict | None) -> dict:
    if review is None:
        return parsed
    for key in ("code", "expected_action_date", "announcement_id", "notice_date", "source_url", "source_sha256"):
        if review[key] != source[key]:
            raise ValueError("Reviewed notice no longer matches the exact source")
    if any(compact_text(fragment) not in compact_text(text) for fragment in review["evidence"]):
        raise ValueError("Reviewed primary evidence is absent")
    if not set(review["fields"]) <= {"cash_per_share", "reference_cash_per_share"}:
        raise ValueError("Review cannot silently change event dates or share eligibility")
    arithmetic = review.get("arithmetic")
    if arithmetic:
        reference = (Decimal(arithmetic["participating_shares"]) * Decimal(arithmetic["cash_per_share"])
                     / Decimal(arithmetic["total_shares"])).quantize(
                         Decimal(arithmetic["quantum"]), rounding=ROUND_HALF_UP)
        if reference != Decimal(review["fields"]["reference_cash_per_share"]):
            raise ValueError("Reviewed reference cash does not match independent share arithmetic")
    if not set(review["resolves"]) <= set(parsed["issues"]):
        raise ValueError("A review does not match the extraction ambiguity it resolves")
    return {**parsed, **review["fields"], "issues": [issue for issue in parsed["issues"]
             if issue not in review["resolves"]], "manual_review_reason": review["reason"]}


def decision_checks(row: dict, parsed: dict, source: dict) -> tuple[list[str], list[str], bool]:
    issues = parsed["issues"].copy()
    cash = parsed["cash_per_share"]
    positive_shares = parsed["positive_share_distribution"]
    for actual, expected in (("record_date", "dividRegistDate"), ("action_date", "date"),
                             ("pay_date", "dividPayDate")):
        if parsed[actual] != row[expected]:
            issues.append(actual + "_vendor_disagreement")
    if cash is not None and Decimal(cash) != Decimal(row["dividCashPsBeforeTax"]):
        issues.append("cash_vendor_disagreement")
    if positive_shares != (row["action_type"] == "share_distribution"):
        issues.append("share_vendor_disagreement")
    cash_eligible = (not positive_shares and cash is not None
                     and Decimal(cash) / Decimal(str(row["previous_close"])) >= Decimal(".01"))
    visible = ("2024-01-01" <= source["notice_date"] <= row["previous_market_date"]
               and parsed["record_date"] == row["previous_market_date"])
    relevant = cash_eligible and visible
    reference = parsed["reference_cash_per_share"]
    if relevant and reference is not None:
        expected = (Decimal(str(row["previous_close"])) - Decimal(reference)).quantize(
            Decimal(".01"), rounding=ROUND_HALF_UP)
        if expected != Decimal(str(row["preclose"])):
            issues.append("reference_price_disagreement")
    # Cash and payout disagreements remain visible, but the primary notice
    # controls signal terms. Ex-date buyers have no claim on this cash payment.
    informational = {"cash_vendor_disagreement", "record_date_vendor_disagreement",
                     "pay_date_vendor_disagreement"}
    if not relevant:
        informational |= {"reference_cash_template", "multiple_reference_formulas"}
    blocking = sorted(set(issues) - informational)
    return sorted(set(issues)), blocking, relevant


def audit(output: Path = ROOT) -> dict:
    manifest = json.loads((output / "notice_scope_manifest.json").read_text())
    source_report = json.loads((output / "notice_source_report.json").read_text())
    if sha(output / "notice_scope.parquet") != manifest["scope_sha256"]:
        raise ValueError("Pre-outcome notice cohort changed")
    candidates = pd.read_parquet(output / "notice_scope.parquet")
    source_path = output / "notice_sources" / "scope_source_index.json"
    if sha(source_path) != source_report["source_index_sha256"]:
        raise ValueError("Notice source manifest changed")
    sources = json.loads(source_path.read_text())
    keys = {(r["code"], r["expected_action_date"]): r for r in sources}
    if len(keys) != len(sources) or set(keys) != set(zip(candidates.code, candidates.date)):
        raise ValueError("Every potential event needs one unambiguous source notice")
    reviews = json.loads(REVIEWS.read_text())
    review_keys = {(r["code"], r["expected_action_date"]): r for r in reviews}
    if len(review_keys) != len(reviews) or not set(review_keys) <= set(keys):
        raise ValueError("Manual review identities must be unique and in the frozen cohort")
    rows = []
    for row in candidates.to_dict("records"):
        source = keys[row["code"], row["date"]]
        for path_key, hash_key in (("source_path", "source_sha256"), ("text_path", "text_sha256"),
                                   ("pages_path", "pages_sha256")):
            if sha(Path(source[path_key])) != source[hash_key]:
                raise ValueError("An original notice or its extraction changed")
        text = Path(source["text_path"]).read_text()
        parsed = reviewed_terms(parse_terms(text, row["code"]), text, source,
                                review_keys.get((row["code"], row["date"])))
        issues, blocking, relevant = decision_checks(row, parsed, source)
        rows.append({**source, "date": row["date"], **parsed, "issues": issues,
                     "blocking_issues": blocking, "cash_and_visibility_eligible": relevant,
                     "vendor_cash_per_share": row["dividCashPsBeforeTax"],
                     "vendor_pay_date": row["dividPayDate"], "vendor_type": row["action_type"]})
    save_json(output / "notice_template_audit.json", rows)
    complete = all(not r["blocking_issues"] for r in rows)
    report = {"audited_event_states": len(rows), "decision_checks_passed": sum(not r["blocking_issues"] for r in rows),
              "needs_review": [{"code": r["code"], "date": r["date"], "issues": r["issues"]}
                               for r in rows if r["blocking_issues"]],
              "cash_corrections": [{"code": r["code"], "date": r["date"],
                  "vendor": r["vendor_cash_per_share"], "primary": r["cash_per_share"]}
                  for r in rows if "cash_vendor_disagreement" in r["issues"]],
              "payout_date_conflicts": [{"code": r["code"], "date": r["date"],
                  "vendor": r["vendor_pay_date"], "primary": r["pay_date"]}
                  for r in rows if "pay_date_vendor_disagreement" in r["issues"]],
              "source_index_sha256": sha(source_path), "scope_sha256": manifest["scope_sha256"],
              "reviews_sha256": sha(REVIEWS), "audit_sha256": sha(output / "notice_template_audit.json"),
              "all_decision_terms_verified": complete, "strategy_returns_read": False, "holdout_read": False}
    if complete:
        verified = pd.DataFrame(rows)
        selected = candidates.merge(verified[["code", "date", "cash_per_share", "reference_cash_per_share",
            "cash_and_visibility_eligible", "source_url", "source_sha256"]], on=["code", "date"], validate="one_to_one")
        selected = selected.loc[selected.cash_and_visibility_eligible].copy()
        selected["cash_yield_exact"] = [Decimal(cash) / Decimal(str(previous))
            for cash, previous in zip(selected.cash_per_share, selected.previous_close)]
        selected = selected.sort_values(["date", "cash_yield_exact", "day_return", "code"],
                                        ascending=[True, False, True, True])
        selected["cash_yield"] = selected.cash_yield_exact.map(float)
        selected["daily_rank"] = selected.groupby("date", sort=False).cumcount() + 1
        selected = selected.drop(columns="cash_yield_exact")
        selected.to_parquet(output / "verified_candidates.parquet", index=False)
        report["verified_candidates_sha256"] = sha(output / "verified_candidates.parquet")
        report["by_half"] = [{"half": half, "all_candidates": len(group),
            "top5_rows": int(group.daily_rank.le(5).sum()), "candidate_days": int(group.date.nunique())}
            for half, group in selected.groupby("half", sort=True)]
        report["all_candidates"] = len(selected)
    save_json(output / "notice_template_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    report = audit(args.output)
    print({k: v for k, v in report.items() if k != "needs_review"})
    print("Review cases:", report["needs_review"])


if __name__ == "__main__":
    main()
