"""Freeze same-broker predecessors by source production time, before reading PDFs.

The time is a source-ordering proxy, not a verified first-publication timestamp.
Only identity and date fields from the cached detail pages enter this program.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import requests

from .analyst_catalog import ROOT
from .analyst_catalog_sources import SAMPLE_SHA, detail_object
from .corporate_cash import save_json, sha


RULE_COMMIT = "a85c8b7"
CATALOG_SHA = "e88551396b05a53aab5c5499fc17d54543d397fd4295d4b3d9e98a8af0c68477"


def select_predecessors(sample: list[dict], records: list[dict]) -> list[dict]:
    if len({row["info_code"] for row in records}) != len(records):
        raise ValueError("Duplicate source report identity")
    by_id = {row["info_code"]: row for row in records}
    output = []
    for item in sample:
        current = by_id[item["info_code"]]
        group = [row for row in records if row["stock_code"] == item["stock_code"]
                 and row["broker_code"] == item["broker_code"]]
        status, previous = "ready_for_source_review", None
        if any(row["source_status"] != "valid_time_and_identity" for row in group):
            status = "unknown_order_due_to_source_field"
        else:
            stamp = current["production_time"]
            others = [row for row in group if row["info_code"] != item["info_code"]]
            if any(row["production_time"] == stamp for row in others):
                status = "ambiguous_current_time_tie"
            else:
                before = [row for row in others if "2024-01-01 00:00:00" <= row["production_time"] < stamp]
                if not before:
                    status = "no_predecessor_since_2024_in_catalog"
                else:
                    latest = max(row["production_time"] for row in before)
                    last = [row for row in before if row["production_time"] == latest]
                    if len(last) != 1:
                        status = "ambiguous_predecessor_time_tie"
                    else:
                        previous = last[0]
        output.append({"info_code": item["info_code"], "stock_code": item["stock_code"],
            "broker_code": item["broker_code"], "half": item["half"],
            "current_production_time": current["production_time"], "status": status,
            "previous_info_code": None if previous is None else previous["info_code"],
            "previous_production_time": None if previous is None else previous["production_time"],
            "previous_display_date": None if previous is None else previous["display_date"]})
    return output


def prepare(root: Path = ROOT, fetch: bool = False) -> dict:
    if sha(root / "catalog.json") != CATALOG_SHA or sha(root / "sample.json") != SAMPLE_SHA:
        raise ValueError("Frozen catalog or source sample changed")
    catalog = json.loads((root / "catalog.json").read_text())
    sample = json.loads((root / "sample.json").read_text())
    stocks = {row["stock_code"] for row in sample}
    selected = [row for row in catalog if row["stock_code"] in stocks]
    session = requests.Session()
    session.trust_env = False
    session.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://data.eastmoney.com/"})
    output = root / "predecessors"
    output.mkdir(exist_ok=True)
    records = []
    for index, item in enumerate(selected, 1):
        code = item["info_code"]
        path = root / "html" / f"{code}.html"
        manifest = path.with_suffix(".html.request.json")
        url = f"https://data.eastmoney.com/report/info/{code}.html"
        if not path.exists():
            if not fetch:
                raise FileNotFoundError(path)
            response = session.get(url, timeout=40)
            response.raise_for_status()
            detail_object(response.content.decode("utf-8"), code)
            path.write_bytes(response.content)
            save_json(manifest, {"url": url, "fetched_at_utc": datetime.now(timezone.utc).isoformat(), "sha256": sha(path)})
        evidence = json.loads(manifest.read_text())
        if evidence["url"] != url or evidence["sha256"] != sha(path):
            raise ValueError("Cached source response or identity changed")
        detail = detail_object(path.read_text(), code)
        status = "valid_time_and_identity"
        stamp = detail.get("eitime")
        try:
            parsed = datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S")
            if parsed.strftime("%Y-%m-%d %H:%M:%S") != stamp:
                status = "noncanonical_production_time"
        except (TypeError, ValueError):
            status = "missing_or_invalid_production_time"
        if (str(detail.get("company_code")) != item["broker_code"]
                or item["stock_code"] not in {row["stock"] for row in detail.get("security", [])}
                or detail.get("notice_date", "")[:10] != item["display_date"]):
            status = "identity_or_display_date_conflict"
        records.append({"info_code": code, "stock_code": item["stock_code"],
            "broker_code": item["broker_code"], "display_date": item["display_date"],
            "production_time": stamp, "source_status": status, "html_sha256": sha(path)})
        if index % 20 == 0 or index == len(selected):
            save_json(output / "time_inputs_partial.json", records)
            print({"metadata_checked": index, "total": len(selected)}, flush=True)
    pairs = select_predecessors(sample, records)
    save_json(output / "time_inputs.json", records)
    save_json(output / "frozen_pairs.json", pairs)
    report = {"rule_commit": RULE_COMMIT, "catalog_sha256": CATALOG_SHA, "sample_sha256": SAMPLE_SHA,
        "metadata_records": len(records), "source_states": dict(Counter(row["source_status"] for row in records)),
        "pair_states": dict(Counter(row["status"] for row in pairs)),
        "time_inputs_sha256": sha(output / "time_inputs.json"),
        "frozen_pairs_sha256": sha(output / "frozen_pairs.json"),
        "predecessor_pdfs_read": False, "forecast_values_used_for_ordering": False,
        "production_time_is_verified_publication_time": False, "market_data_read": False, "returns_read": False}
    save_json(output / "freeze_report.json", report)
    print(report, flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    prepare(args.root, args.fetch)


if __name__ == "__main__":
    main()
