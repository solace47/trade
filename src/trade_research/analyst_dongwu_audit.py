"""Verify the fixed broker source reviews; this does not create trading events."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path
import re

import pdfplumber

from .analyst_catalog import ROOT
from .analyst_catalog_sources import detail_object
from .corporate_cash import save_json, sha


REVIEWS = Path("config/analyst_dongwu_source_reviews.json")
IDENTITY_ALIAS = Path("config/analyst_source_identity_alias.json")
NUMBER = re.compile(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\)")


def identity_alias(root: Path) -> dict:
    evidence = json.loads(IDENTITY_ALIAS.read_text())
    path = root.parent / "source_docs" / "bse_code_mapping.html"
    if sha(path) != evidence["mapping_sha256"]:
        raise ValueError("Reviewed exchange code mapping changed")
    rows = [row for row in re.findall(r"<tr[^>]*>.*?</tr>", path.read_text(), re.S)
            if evidence["entity_name"] in row]
    if len(rows) != 1 or any(code not in rows[0] for code in (evidence["stock_code"], evidence["printed_legacy_code"])):
        raise ValueError("Exchange mapping does not establish the reviewed entity identity")
    return evidence


def printed_security(page: str, info_code: str, catalog_code: str, alias: dict) -> str:
    head = re.sub(r"\s+", "", page)[:500]
    expected = catalog_code
    if info_code in alias["reviewed_document_ids"]:
        if catalog_code != alias["stock_code"] or alias["entity_name"] not in head:
            raise ValueError("Reviewed source identity alias belongs to another entity")
        expected = alias["printed_legacy_code"]
    if re.findall(r"[（(](\d{6})[）)]", head) != [expected]:
        raise ValueError("Printed security identifier differs from the source evidence")
    return expected


def profit_table(page: str) -> dict[int, Decimal]:
    """Read one explicit fiscal-year table, retaining forecasts from 2024 only."""
    lines = page.splitlines()
    headers = [re.findall(r"(20\d{2})([AE])", line) for line in lines]
    headers = [items for items in headers if len(items) >= 3]
    if len(headers) != 1 or len({year for year, _ in headers[0]}) != len(headers[0]):
        raise ValueError("Forecast-year header is missing or ambiguous")
    pattern = r"(?:归母净利润|归属母公司净利润)[（(]百万元[）)]"
    rows = [line for line in lines if re.search(pattern, line)]
    if len(rows) != 1:
        raise ValueError("Net-profit row or its million-yuan unit is ambiguous")
    match = re.search(pattern, rows[0])
    values = []
    for cell in rows[0][match.end():].split():
        if NUMBER.fullmatch(cell) is None:
            break
        numeric = cell.replace(",", "")
        values.append(-Decimal(numeric[1:-1]) if numeric.startswith("(") else Decimal(numeric))
    if len(values) != len(headers[0]):
        raise ValueError("Fiscal-year and numerical column counts differ")
    return {int(year): value for (year, marker), value in zip(headers[0], values)
            if marker == "E" and int(year) >= 2024}


def reported_direction(review: dict) -> str | None:
    if review["comparison_kind"] != "numeric_prior_levels":
        return None
    before, after = (Decimal(review["reported_prior_profit_cny_100million"]),
                     Decimal(review["reported_current_profit_cny_100million"]))
    return "up" if after > before else "down" if after < before else "unchanged"


def audit(root: Path = ROOT, reviews_path: Path = REVIEWS) -> dict:
    frozen = json.loads(reviews_path.read_text())
    alias = identity_alias(root)
    for name in ("catalog", "sample"):
        if sha(root / f"{name}.json") != frozen[f"{name}_sha256"]:
            raise ValueError("Frozen broker catalog or source sample changed")
    sample = json.loads((root / "sample.json").read_text())
    reviews = frozen["records"]
    if [r["info_code"] for r in sample] != [r["info_code"] for r in reviews]:
        raise ValueError("Reviewed sample order or membership changed")
    manifests = {r["info_code"]: r for r in json.loads((root / "source_manifest.json").read_text())}
    records = []
    for item, review in zip(sample, reviews):
        code = item["info_code"]
        html = root / "html" / f"{code}.html"
        if sha(html) != manifests[code]["html_sha256"]:
            raise ValueError("Source detail snapshot changed")
        detail = detail_object(html.read_text(), code)
        if str(detail["company_code"]) != item["broker_code"]:
            raise ValueError("Report broker differs between catalog and detail")
        if item["stock_code"] not in {s["stock"] for s in detail["security"]}:
            raise ValueError("Report security differs between catalog and detail")
        if detail["notice_date"][:10] != item["display_date"]:
            raise ValueError("Report display dates disagree")
        pdf_path = root / "pdf" / f"{code}.pdf"
        if sha(pdf_path) != review["pdf_sha256"]:
            raise ValueError("Reviewed PDF changed")
        with pdfplumber.open(pdf_path) as pdf:
            first = pdf.pages[0].extract_text() or ""
            body = pdf.pages[review["comparison_page"] - 1].extract_text() or ""
            metadata = {key: pdf.metadata.get(key) for key in ("CreationDate", "ModDate")}
        compact = re.sub(r"\s+", "", first)
        printed_code = printed_security(first, code, item["stock_code"], alias)
        match = re.search(r"(20\d{2})年(\d{1,2})月(\d{1,2})日", compact[:700])
        if match is None:
            raise ValueError("Printed header date is missing")
        printed = "-".join(f"{int(part):0{4 if i == 0 else 2}d}" for i, part in enumerate(match.groups()))
        if printed != review["printed_date"] or review["target_fiscal_year"] != int(printed[:4]) + 1:
            raise ValueError("Reviewed next-full-year mapping changed")
        target = profit_table(first).get(review["target_fiscal_year"])
        if target is None or target != Decimal(review["current_profit_cny_million"]):
            raise ValueError("Reviewed next-full-year profit differs from the original table")
        rounded = Decimal(review["reported_current_profit_cny_100million"])
        if (target / 100).quantize(Decimal(1).scaleb(rounded.as_tuple().exponent), rounding=ROUND_HALF_UP) != rounded:
            raise ValueError("Narrative rounding does not reconcile with the profit table")
        for field in ("reported_current_profit_cny_100million", "reported_prior_profit_cny_100million"):
            token = review[field]
            if token is not None and re.search(r"(?<![\d.])" + re.escape(token) + r"(?![\d.])", body) is None:
                raise ValueError("Reviewed comparison number is absent from its PDF page")
        production = datetime.strptime(detail["eitime"], "%Y-%m-%d %H:%M:%S")
        modified = metadata.get("ModDate") or ""
        pdf_stamp = re.fullmatch(r"D:(\d{14})\+08'00'", modified)
        modification = datetime.strptime(pdf_stamp.group(1), "%Y%m%d%H%M%S") if pdf_stamp else None
        records.append({"info_code": code, "half": item["half"], "stock_code": item["stock_code"],
            "printed_stock_code": printed_code, "catalog_code_is_retrospective_alias": printed_code != item["stock_code"],
            "printed_date": printed, "display_date": item["display_date"],
            "production_time": detail["eitime"], "production_after_display_date": production.date().isoformat() > item["display_date"],
            "production_clock_after_1449": production.strftime("%H:%M:%S") > "14:49:00",
            "production_after_printed_date": production.date().isoformat() > printed,
            "production_before_pdf_modification": None if modification is None else production < modification,
            "target_fiscal_year": review["target_fiscal_year"], "target_profit_cny_million": str(target),
            "comparison_kind": review["comparison_kind"], "reported_direction": reported_direction(review),
            "public_time_verified": review["public_time_verified"],
            "prior_forecast_origin_since_2024_verified": review["prior_forecast_origin_since_2024_verified"]})
    report = {"rule_commit": frozen["rule_commit"], "reviews_sha256": sha(reviews_path),
        "sample_sha256": frozen["sample_sha256"], "records": records,
        "reports": len(records), "explicit_target_forecasts": len(records),
        "comparison_kinds": dict(Counter(r["comparison_kind"] for r in records)),
        "reported_directions_with_numeric_prior": dict(Counter(r["reported_direction"] for r in records if r["reported_direction"])),
        "production_later_than_display_date": sum(r["production_after_display_date"] for r in records),
        "production_later_than_printed_date": sum(r["production_after_printed_date"] for r in records),
        "production_clock_after_1449": sum(r["production_clock_after_1449"] for r in records),
        "production_before_pdf_modification": sum(r["production_before_pdf_modification"] is True for r in records),
        "retrospective_code_aliases_verified": sum(r["catalog_code_is_retrospective_alias"] for r in records),
        "public_times_verified": sum(r["public_time_verified"] for r in records),
        "returns_read": False, "market_data_read": False, "trading_events_created": False}
    save_json(root / "source_audit_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--reviews", type=Path, default=REVIEWS)
    args = parser.parse_args()
    result = audit(args.root, args.reviews)
    print({key: value for key, value in result.items() if key != "records"})


if __name__ == "__main__":
    main()
