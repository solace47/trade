"""Download every midpoint-title original and extract all PDF pages.

Run after the complete CNINFO search index has been screened. The full text
is essential: several mixed-subject progress tables occur after page three.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import pdfplumber

from audit_buyback_pdfs import _download
from trade_research.insider_midpoint_source import ACTOR, MIDPOINT, _clean


def _check(row: dict, pdf_dir: Path) -> dict:
    name = Path(urlparse(row["pdf_url"]).path).name
    path = pdf_dir / name
    try:
        _download(row["pdf_url"], path)
        with pdfplumber.open(path) as document:
            full = "\n".join(page.extract_text() or "" for page in document.pages)
            pages = len(document.pages)
        clean = _clean(full)
        status = ("ok" if row["code"].split(".")[-1] in clean
                  and ACTOR.search(clean) and MIDPOINT.search(clean)
                  else "identity_or_midpoint_unconfirmed")
        return {**row, "status": status, "pages": pages,
                "text_full_pdf": full}
    except Exception as error:
        return {**row, "status": type(error).__name__, "pages": None,
                "text_full_pdf": ""}


def audit(year: int, source_dir: Path, workers: int = 4) -> dict:
    if year not in (2024, 2025) or workers < 1:
        raise ValueError("Only 2024/2025 midpoint originals")
    index = pd.read_parquet(source_dir / f"title_candidates_{year}.parquet")
    names = index.pdf_url.map(lambda url: Path(urlparse(url).path).name)
    if (index.empty or index.pdf_url.duplicated().any()
            or names.duplicated().any()
            or not names.str.lower().str.endswith(".pdf").all()):
        raise ValueError("Malformed original midpoint PDF index")
    pdf_dir = source_dir / "pdfs" / str(year)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(lambda row: _check(row, pdf_dir),
                                index.to_dict("records")))
    records.sort(key=lambda row: (row["notice_date"], row["code"],
                                  row["pdf_url"]))
    output = source_dir / f"pdf_audit_{year}.jsonl"
    temporary = output.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                 for row in records), encoding="utf-8")
    temporary.replace(output)
    return {"year": year, "originals": len(records),
            "all_pages_extracted": sum(row["status"] == "ok" for row in records),
            "status": pd.Series([row["status"] for row in records])
            .value_counts().to_dict()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", required=True, type=int, choices=(2024, 2025))
    parser.add_argument("--source", type=Path, default=Path(
        "data/research/insider_midpoint"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(audit(args.year, args.source, args.workers))


if __name__ == "__main__":
    main()
