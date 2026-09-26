"""Verify fiscal-year mapping and chronology of five fixed public reports.

This reads source documents only, never a market-data file or investment outcomes.
It cannot certify publication time merely from a printed PDF date.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import json
from pathlib import Path
import re

import pdfplumber

from .corporate_cash import save_json, sha


ROOT = Path("data/research/analyst_revision_source")
REVIEWS = Path("config/analyst_source_probe_reviews.json")


def compact(value: str) -> str:
    return re.sub(r"\s+", "", value)


def verify_forecast_table(page: str, label: str, expected: dict[str, str]) -> dict[str, str]:
    """Verify ordered forecast years and a reviewed EPS row, not nearby actuals."""
    headers = re.findall(r"(20\d{2})\s*E", page)
    years = list(expected)
    if not any(headers[i:i+len(years)] == years for i in range(len(headers))):
        raise ValueError("Forecast fiscal-year order differs from the reviewed table")
    rows = [line for line in page.splitlines() if compact(label) in compact(line)]
    if len(rows) != 1:
        raise ValueError("EPS row is missing or ambiguous")
    # Accept extraction spaces inside the label, but preserve spaces between
    # numerical cells. Compacting the whole row can merge adjacent decimals.
    label_pattern = r"\s*".join(re.escape(char) for char in compact(label))
    label_match = re.search(label_pattern, rows[0])
    if label_match is None:
        raise ValueError("EPS label cannot be located without merging cells")
    numeric_part = rows[0][label_match.end():]
    numbers = re.findall(r"(?<![\d.])-?\d+\.\d+(?![\d.])", numeric_part)
    target = list(expected.values())
    if len(numbers) < len(target) or [Decimal(x) for x in numbers[-len(target):]] != [Decimal(x) for x in target]:
        raise ValueError("EPS values differ from the reviewed forecast row")
    return {year: str(Decimal(value)) for year, value in expected.items()}


def compare_record(row: dict, review: dict, current_year: int, pages: list[str]) -> dict:
    if row["infoCode"] != review["info_code"] or row["stockCode"] != review["stock_code"]:
        raise ValueError("Report identity changed")
    printed = review["printed_date"]
    year, month, day = map(int, printed.split("-"))
    if re.search(fr"{year}年0?{month}月0?{day}日", compact(pages[0])[:700]) is None:
        raise ValueError("Printed report date differs from the reviewed header")
    forecasts = verify_forecast_table(pages[review["eps_page"] - 1], review["eps_row_label"], review["forecast_eps"])
    api_value = Decimal(row["predictThisYearEps"])
    matched_years = [year for year, value in forecasts.items() if Decimal(value) == api_value]
    container_year_match = str(current_year) in forecasts and api_value == Decimal(forecasts[str(current_year)])
    conflict = bool(review["chronology_conflict"])
    if conflict:
        referenced = review["referenced_report_date"]
        if referenced not in compact(pages[0]) or referenced <= printed or "2024Q1" not in compact(pages[0]):
            raise ValueError("Reviewed internal date contradiction is not present")
    return {"info_code": row["infoCode"], "stock_code": row["stockCode"],
        "api_publish_date": row["publishDate"], "printed_date": printed,
        "api_and_header_date_equal": row["publishDate"][:10] == printed,
        "current_year": current_year, "api_this_year_eps": str(api_value),
        "reviewed_forecast_eps": forecasts, "numeric_matching_years": matched_years,
        "matches_container_fiscal_year": container_year_match,
        "matches_report_calendar_year": api_value == Decimal(forecasts[str(year)]),
        "chronology_conflict": conflict, "public_time_verified": review["public_time_verified"],
        "source_eligible_as_direct_event": bool(container_year_match and not conflict and review["public_time_verified"])}


def audit(root: Path = ROOT, reviews_path: Path = REVIEWS) -> dict:
    reviews = json.loads(reviews_path.read_text())
    snapshot_path = root / "schema_probe_20240320_20240322.json"
    if sha(snapshot_path) != reviews["snapshot_sha256"]:
        raise ValueError("Frozen schema snapshot changed")
    snapshot = json.loads(snapshot_path.read_text())
    if snapshot["currentYear"] != reviews["snapshot_current_year"]:
        raise ValueError("Query-time fiscal-year mapping changed")
    rows = {r["infoCode"]: r for r in snapshot["data"]}
    if set(rows) != {r["info_code"] for r in reviews["records"]} or len(snapshot["data"]) != 5:
        raise ValueError("Fixed five-report probe changed")
    checked = []
    for review in reviews["records"]:
        path = root / "pdf" / (review["info_code"] + ".pdf")
        if sha(path) != review["pdf_sha256"]:
            raise ValueError("Reviewed report bytes changed")
        with pdfplumber.open(path) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
            metadata = {key: pdf.metadata.get(key) for key in ("CreationDate", "ModDate")}
        record = compare_record(rows[review["info_code"]], review, snapshot["currentYear"], pages)
        record["pdf_sha256"] = review["pdf_sha256"]
        record["pdf_metadata_not_publication_proof"] = metadata
        checked.append(record)
    issuer = root / "pdf" / "issuer_baiya_2024q1.pdf"
    if sha(issuer) != reviews["issuer_pdf_sha256"]:
        raise ValueError("Issuer corroborating source changed")
    with pdfplumber.open(issuer) as pdf:
        first = compact(pdf.pages[0].extract_text() or "")
    if not all(token in first for token in ("2024年第一季度报告", "765,153,323.00", "102,619,879.00")):
        raise ValueError("Issuer source does not contain the referenced quarterly results")
    result = {"rule_commit": reviews["rule_commit"], "reviews_sha256": sha(reviews_path),
        "snapshot_sha256": sha(snapshot_path), "records": checked,
        "container_year_matches": sum(r["matches_container_fiscal_year"] for r in checked),
        "report_calendar_year_matches": sum(r["matches_report_calendar_year"] for r in checked),
        "internal_chronology_conflicts": sum(r["chronology_conflict"] for r in checked),
        "public_times_verified": sum(r["public_time_verified"] for r in checked),
        "direct_event_source_passed": all(r["source_eligible_as_direct_event"] for r in checked),
        "sampling": "five_fixed_schema_probe_records_not_a_population_error_rate",
        "live_2026_report_metadata_accessed_for_frontend_inspection": True,
        "live_metadata_used_for_selection_or_returns": False,
        "historical_forecast_of_2026_is_not_realized_2026_data": True,
        "market_data_files_read": False, "holding_returns_read": False, "holdout_outcomes_read": False}
    save_json(root / "audit_report.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--reviews", type=Path, default=REVIEWS)
    args = parser.parse_args()
    print(audit(args.root, args.reviews))


if __name__ == "__main__":
    main()
