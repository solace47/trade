"""Freeze and cache the pre-outcome implementation-notice audit cohort."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re

import pandas as pd
import pdfplumber

from .cash_dividend_catalog import ROOT as CATALOG
from .cash_ex_inputs import ROOT
from .corporate_cash import curl, save_json, sha


def freeze(output: Path = ROOT) -> dict:
    base_report = json.loads((output / "base_report.json").read_text())
    event_report = json.loads((CATALOG / "primary_supplement_report.json").read_text())
    hashes = {str(output / "base.parquet"): base_report["base_sha256"],
              str(CATALOG / "events_augmented.parquet"): event_report["augmented_events_sha256"],
              str(CATALOG / "notices.parquet"): sha(CATALOG / "notices.parquet")}
    for name, digest in hashes.items():
        if sha(Path(name)) != digest:
            raise ValueError("The pre-outcome event input changed")
    base = pd.read_parquet(output / "base.parquet")
    events = pd.read_parquet(CATALOG / "events_augmented.parquet").rename(
        columns={"dividOperateDate": "date"})
    scope = base.merge(events, on=["code", "date"], validate="one_to_one")
    scope = scope.loc[scope.day_return.between(-.03, -.005)].copy()
    notices = pd.read_parquet(CATALOG / "notices.parquet")
    scope = scope.merge(notices, left_on=["code", "dividPlanDate"],
                        right_on=["code", "notice_date"], how="left")
    if scope.duplicated(["code", "date"]).any() or scope.announcement_id.isna().any():
        raise ValueError("Audit cohort needs one unambiguous implementation notice per event")
    if not scope.date.between("2024-01-01", "2025-12-31").all():
        raise ValueError("Audit cohort cannot use holdout event inputs")
    scope = scope.sort_values(["date", "code"]).reset_index(drop=True)
    path = output / "notice_scope.parquet"
    manifest_path = output / "notice_scope_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["inputs_sha256"] != hashes or sha(path) != manifest["scope_sha256"]:
            raise ValueError("Cannot replace the frozen notice scope")
        pd.testing.assert_frame_equal(pd.read_parquet(path), scope)
        return manifest
    scope.to_parquet(path, index=False)
    result = {"rule_commit": "876cc69", "inputs_sha256": hashes, "scope_sha256": sha(path),
              "events": len(scope), "cash_ratio_or_type_filter_applied": False,
              "by_half": scope.groupby("half").size().to_dict(),
              "strategy_returns_read": False, "holdout_prices_read": False}
    save_json(manifest_path, result)
    return result


def extract_pages(pages: list[str]) -> str:
    """Remove only exact page counters and repeated security headers.

    A footer between '5.8' and '元' must never turn a cash amount into 5.81.
    Keep the original per-page extraction as a separate cached artifact.
    """
    cleaned = []
    for number, text in enumerate(pages, 1):
        lines = text.splitlines()
        while lines and not lines[-1].strip():
            lines.pop()
        if lines and re.fullmatch(
                rf"(?:[-－—]?\s*{number}\s*[-－—]?|第\s*{number}\s*页(?:\s*共\s*\d+\s*页)?)",
                lines[-1].strip()):
            lines.pop()
        if number > 1:
            # Repeated running headers otherwise interrupt monetary/date tokens.
            while lines and re.match(r"^(?:证券代码|债券代码)[：:]", lines[0].strip()):
                lines.pop(0)
        cleaned.append("\n".join(lines))
    return "\n\n".join(cleaned)


def fetch(output: Path = ROOT) -> dict:
    manifest = json.loads((output / "notice_scope_manifest.json").read_text())
    if sha(output / "notice_scope.parquet") != manifest["scope_sha256"]:
        raise ValueError("Frozen notice scope changed")
    scope = pd.read_parquet(output / "notice_scope.parquet")
    cache = output / "notice_sources"
    cache.mkdir(parents=True, exist_ok=True)
    index_path = cache / "scope_source_index.json"
    prior = {}
    for name in ("source_index.json", "scope_source_index.json"):
        if (cache / name).exists():
            for row in json.loads((cache / name).read_text()):
                prior[row["announcement_id"]] = row

    def one(row: dict) -> dict:
        url = row["pdf_url"]
        if f"/finalpage/{row['notice_date']}/" not in url:
            raise ValueError("Primary disclosure date differs from its source URL")
        stem = cache / f"{row['code']}_{row['date']}_{row['announcement_id']}"
        pdf = Path(str(stem) + ".pdf")
        if not pdf.exists():
            payload = curl(url)
            if not payload.startswith(b"%PDF"):
                raise ValueError("Implementation notice did not return a PDF")
            pdf.write_bytes(payload)
        digest = sha(pdf)
        known = prior.get(row["announcement_id"])
        if known and digest != known["source_sha256"]:
            raise ValueError("Cached implementation notice changed")
        with pdfplumber.open(pdf) as document:
            pages = [page.extract_text() or "" for page in document.pages]
        text_path = Path(str(stem) + ".clean.txt")
        raw_path = Path(str(stem) + ".pages.json")
        save_json(raw_path, pages)
        text_path.write_text(extract_pages(pages))
        return {"code": row["code"], "expected_action_date": row["date"],
                "notice_date": row["notice_date"], "announcement_id": row["announcement_id"],
                "title": row["title"], "source_url": url, "revision_flag": row["revision_flag"],
                "source_path": str(pdf), "source_sha256": digest, "text_path": str(text_path),
                "text_sha256": sha(text_path), "pages_path": str(raw_path), "pages_sha256": sha(raw_path)}

    results = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(one, scope.to_dict("records")):
            results.append(result)
            if len(results) % 50 == 0:
                print(f"Primary notice sources: {len(results)}/{len(scope)}", flush=True)
    save_json(index_path, results)
    report = {"events": len(results), "scope_sha256": manifest["scope_sha256"],
              "source_index_sha256": sha(index_path), "strategy_returns_read": False,
              "holdout_prices_read": False}
    save_json(output / "notice_source_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "fetch"))
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    print({"freeze": freeze, "fetch": fetch}[args.stage](args.output))


if __name__ == "__main__":
    main()
