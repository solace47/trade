"""Check the original forecast tables of 22 already-frozen predecessor reports."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path
import re

import pdfplumber
import requests

from .analyst_catalog import ROOT
from .analyst_dongwu_audit import REVIEWS, identity_alias, printed_security, profit_table
from .corporate_cash import save_json, sha


PAIRS_SHA = "b863708ef4cd7958d96c9e07718c18a77d8c224a53846c911dc425c8241d65ed"
TIMES_SHA = "1404a9bca061dc1c3a3447ff053b3a06fddcc47161bb33e0ec1cabc4e816fa3d"


def compare_forecasts(previous: Decimal, current: Decimal, reported_previous: str | None) -> dict:
    agreement = None
    if reported_previous is not None:
        claimed = Decimal(reported_previous)
        agreement = (previous / 100).quantize(Decimal(1).scaleb(claimed.as_tuple().exponent), rounding=ROUND_HALF_UP) == claimed
    old_radius = Decimal(1).scaleb(previous.as_tuple().exponent) / 2
    new_radius = Decimal(1).scaleb(current.as_tuple().exponent) / 2
    precision = ("equal_displayed_values" if current == previous else
                 "up_beyond_display_rounding" if current - new_radius > previous + old_radius else
                 "down_beyond_display_rounding" if current + new_radius < previous - old_radius else
                 "overlapping_display_rounding_intervals")
    return {"previous_profit_cny_million": str(previous), "current_profit_cny_million": str(current),
            "reported_previous_rounding_matches": agreement,
            "display_precision_diagnostic": precision,
            "paired_forecast_direction": "up" if current > previous else "down" if current < previous else "unchanged",
            "paired_forecast_relative_change": None if previous <= 0 else str(current / previous - 1)}


def audit(root: Path = ROOT, fetch: bool = False) -> dict:
    out = root / "predecessors"
    if sha(out / "frozen_pairs.json") != PAIRS_SHA or sha(out / "time_inputs.json") != TIMES_SHA:
        raise ValueError("Predecessor identities or source ordering changed")
    pairs = json.loads((out / "frozen_pairs.json").read_text())
    reviews = {r["info_code"]: r for r in json.loads(REVIEWS.read_text())["records"]}
    alias = identity_alias(root)
    session = requests.Session()
    session.trust_env = False
    session.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://data.eastmoney.com/"})
    records = []
    for index, pair in enumerate(pairs, 1):
        review = reviews[pair["info_code"]]
        record = dict(pair, target_fiscal_year=review["target_fiscal_year"],
                      current_comparison_kind=review["comparison_kind"], public_time_verified=False)
        code = pair["previous_info_code"]
        if code is None:
            record["source_state"] = pair["status"]
            records.append(record)
            continue
        path = root / "pdf" / f"{code}.pdf"
        manifest = path.with_suffix(".pdf.request.json")
        url = f"https://pdf.dfcfw.com/pdf/H3_{code}_1.pdf"
        if not path.exists():
            if not fetch:
                raise FileNotFoundError(path)
            response = session.get(url, timeout=45)
            response.raise_for_status()
            if not response.content.startswith(b"%PDF-"):
                raise ValueError("Source returned a non-PDF document")
            path.write_bytes(response.content)
            save_json(manifest, {"url": url, "sha256": sha(path), "fetched_at_utc": datetime.now(timezone.utc).isoformat()})
        evidence = json.loads(manifest.read_text())
        if evidence["url"] != url or evidence["sha256"] != sha(path):
            raise ValueError("Frozen predecessor source changed")
        with pdfplumber.open(path) as pdf:
            first = pdf.pages[0].extract_text() or ""
            metadata = {key: pdf.metadata.get(key) for key in ("CreationDate", "ModDate")}
            page_count = len(pdf.pages)
        save_json(root / "text" / f"{code}.first_page.json", {"pdf_sha256": sha(path), "first_page": first, "metadata": metadata})
        record.update(pdf_sha256=sha(path), pdf_pages=page_count, metadata_not_publication_proof=metadata)
        compact = re.sub(r"\s+", "", first)
        date = re.search(r"(20\d{2})年(\d{1,2})月(\d{1,2})日", compact[:700])
        record["source_state"] = "unknown_source"
        if date is None:
            record["source_state"] = "printed_date_missing"
        else:
            printed = datetime(*map(int, date.groups())).strftime("%Y-%m-%d")
            record["printed_date"] = printed
            stamp = pair["previous_production_time"]
            modified = re.fullmatch(r"D:(\d{14})\+08'00'", metadata.get("ModDate") or "")
            modification = datetime.strptime(modified.group(1), "%Y%m%d%H%M%S") if modified else None
            record["production_before_pdf_modification"] = (None if modification is None else
                datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S") < modification)
            if printed < "2024-01-01":
                record["source_state"] = "printed_before_2024"
            elif printed > stamp[:10] or record["production_before_pdf_modification"] is True:
                record["source_state"] = "source_date_conflict"
            else:
                try:
                    record["printed_stock_code"] = printed_security(first, code, pair["stock_code"], alias)
                    forecasts = profit_table(first)
                except ValueError as error:
                    record.update(source_state="table_not_parsed", parse_error=str(error))
                else:
                    previous = forecasts.get(review["target_fiscal_year"])
                    if previous is None:
                        record["source_state"] = "target_forecast_year_missing"
                    else:
                        record["source_state"] = "same_year_table_readable_public_time_unverified"
                        record.update(compare_forecasts(previous, Decimal(review["current_profit_cny_million"]),
                                                        review["reported_prior_profit_cny_100million"]))
        records.append(record)
        save_json(out / "value_audit_partial.json", records)
        print({"item": index, "previous": code, "source_state": record["source_state"],
               "prior": record.get("previous_profit_cny_million"),
               "stated_prior_agrees": record.get("reported_previous_rounding_matches")}, flush=True)
    ready = [r for r in records if r["source_state"] == "same_year_table_readable_public_time_unverified"]
    report = {"rule_commit": "cd46958", "frozen_pairs_sha256": PAIRS_SHA, "time_inputs_sha256": TIMES_SHA,
        "records": records, "source_states": dict(Counter(r["source_state"] for r in records)),
        "paired_forecast_directions": dict(Counter(r["paired_forecast_direction"] for r in ready)),
        "display_precision_diagnostics": dict(Counter(r["display_precision_diagnostic"] for r in ready)),
        "display_precision_assumption": "half_last_printed_unit_rounding_scenario_not_a_certified_error_bound",
        "explicit_prior_levels_checked": sum(r["reported_previous_rounding_matches"] is not None for r in ready),
        "explicit_prior_level_disagreements": sum(r["reported_previous_rounding_matches"] is False for r in ready),
        "public_times_verified": 0, "market_data_read": False, "holding_returns_read": False,
        "trading_events_created": False}
    save_json(out / "value_audit_report.json", report)
    print({k: v for k, v in report.items() if k != "records"})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    audit(args.root, args.fetch)


if __name__ == "__main__":
    main()
