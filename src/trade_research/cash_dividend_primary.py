"""Apply narrowly reviewed primary-source corrections to the dividend calendar.

Vendor responses remain unchanged. Historical issuer supplements, duplicate
notices and announcements of out-of-period implementations stay distinguishable.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import json
from pathlib import Path
import re

import pandas as pd
import pdfplumber

from .cash_dividend_catalog import ROOT, action_type, record_hash
from .corporate_cash import curl, save_json, sha


REVIEWS = Path("config/cash_dividend_primary_supplements.json")


def primary_dates(text: str, code: str) -> tuple[str, str, str]:
    compact = re.sub(r"\s+", "", text)
    if code.startswith("sh."):
        header = compact.split("相关日期", 1)[1].split("差异化", 1)[0]
        header = re.sub(r"(202\d/)", r" \1", header)
        dates = re.findall(r"202\d/\d{1,2}/\d{1,2}", header)[:3]
        if len(dates) != 3:
            raise ValueError("Primary A-share date table is incomplete")
        return tuple(pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates)
    dates = []
    for prefix in (r"股权登记日(?:为)?[：:]?",
                   r"(?:除权(?:除)?息日|除息日|除权日)(?:为)?[：:]?",
                   r"(?:将于|深圳分公司于)"):
        match = re.search(prefix + r"(202\d)年(\d{1,2})月(\d{1,2})日", compact)
        dates.append("-".join(f"{int(v):02d}" for v in match.groups()) if match else "")
    if not dates[0] or not dates[1]:
        raise ValueError("Primary registration or ex-date is missing")
    return tuple(dates)


def verify_terms(text: str, row: dict) -> None:
    compact = re.sub(r"\s+", "", text)
    if row["code"][3:] not in compact[:300]:
        raise ValueError("Wrong primary issuer")
    dates = primary_dates(text, row["code"])
    if row["disposition"] == "outside_implementation_years":
        if dates[1] != row["action_date"] or "2024-01-01" <= dates[1] <= "2025-12-31":
            raise ValueError("An in-period event cannot be dismissed as a boundary notice")
        return
    if dates != tuple(row[k] for k in ("record_date", "action_date", "pay_date")):
        raise ValueError("Primary dates disagree with the reviewed terms")
    cash = Decimal(row["cash_per_share"])
    if row["code"].startswith("sh."):
        headline = compact.split("每股分配比例", 1)[1].split("相关日期", 1)[0]
        values = re.findall(r"[AＡ]股每股现金红利([\d.]+)元", headline)
        if len(values) != 1 or Decimal(values[0]) != cash:
            raise ValueError("Primary A-share cash amount disagrees")
    elif cash:
        values = re.findall(r"每10股(?:拟)?派(?:发)?(?:现金)?(?:红利)?(?:人民币)?([\d.]+)元", compact)
        if not values or Decimal(values[0]) / 10 != cash:
            raise ValueError("Primary per-ten-share cash amount disagrees")
    elif "不派发现金红利" not in compact:
        raise ValueError("Zero cash needs explicit primary confirmation")
    for name, pattern in (("bonus_per_share", r"每10股送(?:红)?股([\d.]+)股"),
                          ("reserve_per_share", r"每10股转增([\d.]+)股")):
        values = [Decimal(v) / 10 for v in re.findall(pattern, compact)]
        expected = Decimal(row[name])
        if (expected and not values) or any(v != expected for v in values):
            raise ValueError("Primary share-distribution ratio disagrees")
    reference = Decimal(row["reference_cash_per_share"])
    if reference != cash and not re.search(
            r"(?<![\d.])" + re.escape(str(reference)) + r"(?![\d.])", compact):
        raise ValueError("Differentiated cash reference is absent from the primary notice")


def load_reviews() -> list[dict]:
    rows = json.loads(REVIEWS.read_text())
    if len({(r["code"], r["announcement_id"]) for r in rows}) != len(rows):
        raise ValueError("Duplicate primary review identities")
    for row in rows:
        path = Path(row["source_path"])
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(curl(row["source_url"]))
        if sha(path) != row["source_sha256"]:
            raise ValueError("Reviewed primary source changed")
        if f"/finalpage/{row['notice_date']}/" not in row["source_url"]:
            raise ValueError("Primary disclosure URL and notice date disagree")
        with pdfplumber.open(path) as document:
            text = "\n\n".join(page.extract_text() or "" for page in document.pages)
        verify_terms(text, row)
    return rows


def overlay(events: pd.DataFrame, reviews: list[dict], output: Path) -> tuple[pd.DataFrame, dict]:
    source = events.to_dict("records")
    keys = {(r["code"], r["dividOperateDate"]): r for r in source}
    if len(keys) != len(source):
        raise ValueError("Vendor catalog has duplicate event dates")
    counts = {name: 0 for name in ("add_missing_event", "correct_vendor_action",
                                  "existing_event_notice", "outside_implementation_years")}
    fields = {"notice_date": "dividPlanDate", "record_date": "dividRegistDate",
              "action_date": "dividOperateDate", "pay_date": "dividPayDate",
              "cash_per_share": "dividCashPsBeforeTax", "bonus_per_share": "dividStocksPs",
              "reserve_per_share": "dividReserveToStockPs"}
    for row in reviews:
        disposition = row["disposition"]
        counts[disposition] += 1
        if disposition == "outside_implementation_years":
            continue
        key = (row["code"], row["action_date"])
        if not "2024-01-01" <= row["action_date"] <= "2025-12-31":
            raise ValueError("Primary supplements are restricted to 2024–2025")
        current = keys.get(key)
        if disposition == "existing_event_notice":
            if current is None or any(Decimal(current[fields[k]] or "0") != Decimal(row[k])
                    for k in ("cash_per_share", "bonus_per_share", "reserve_per_share")):
                raise ValueError("A supposed repeated notice disagrees with the recorded distribution")
            for k in ("record_date", "action_date", "pay_date"):
                if current[fields[k]] != row[k]:
                    raise ValueError("A supposed repeated notice has different event dates")
            continue  # A later duplicate must not replace the earlier notice date.
        if disposition == "add_missing_event":
            if current is not None:
                raise ValueError("A primary supplement would double-count an existing event")
            current = {k: "" for k in events.columns}
            current["code"] = row["code"]
        elif disposition == "correct_vendor_action":
            if current is None:
                raise ValueError("The reviewed vendor action is missing")
            raw_path = output / "vendor" / f"{row['code']}_{row['action_date'][:4]}.json"
            original = [r for r in json.loads(raw_path.read_text())
                        if r["dividOperateDate"] == row["action_date"]]
            if len(original) != 1 or record_hash(original[0]) != row["vendor_event_sha256"]:
                raise ValueError("Vendor action no longer matches the exact reviewed record")
        else:
            raise ValueError("Unknown primary disposition")
        current.update({vendor: row[review] for review, vendor in fields.items()})
        current.update({"implementation_year": row["action_date"][:4],
            "dividCashStock": "原件核准的分配条款", "primary_terms_verified": True,
            "primary_source_url": row["source_url"],
            "primary_reference_cash": row["reference_cash_per_share"],
            "primary_reference_reserve": row.get("reference_reserve_per_share", row["reserve_per_share"]),
            "source_kind": disposition})
        current["action_type"] = action_type(current)
        keys[key] = current
    for row in keys.values():
        row.setdefault("source_kind", "vendor")
        row.setdefault("primary_terms_verified", False)
        for field in ("primary_source_url", "primary_reference_cash", "primary_reference_reserve"):
            row.setdefault(field, "")
    result = pd.DataFrame(keys.values()).sort_values(["code", "dividOperateDate"]).reset_index(drop=True)
    return result, counts


def assemble(output: Path = ROOT) -> dict:
    catalog = json.loads((output / "catalog_report.json").read_text())
    if sha(output / "events.parquet") != catalog["events_sha256"]:
        raise ValueError("Vendor catalog changed")
    events = pd.read_parquet(output / "events.parquet")
    reviews = load_reviews()
    result, counts = overlay(events, reviews, output)
    result.to_parquet(output / "events_augmented.parquet", index=False)
    report = {"original_events": len(events), "augmented_events": len(result),
              "reviewed_dispositions": counts, "reviews_sha256": sha(REVIEWS),
              "original_events_sha256": catalog["events_sha256"],
              "augmented_events_sha256": sha(output / "events_augmented.parquet"),
              "all_event_terms_verified": False, "strategy_returns_read": False,
              "holdout_prices_read": False}
    save_json(output / "primary_supplement_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    print(assemble(args.output))


if __name__ == "__main__":
    main()
