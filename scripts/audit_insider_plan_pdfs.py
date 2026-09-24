"""Verify original CNINFO PDFs for prospective insider share-increase plans.

This stage opens no market prices or outcomes. Every candidate gets an audit
status; failed original files are excluded rather than inferred from titles.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import pdfplumber

from audit_buyback_pdfs import _download
from trade_research.insider_plan_source import confirm_pdf_text


def _record(row: dict, pdf_dir: Path) -> dict:
    name = Path(urlparse(row["pdf_url"]).path).name
    if not name.lower().endswith(".pdf") or "/" in name:
        raise ValueError("Malformed original insider-plan PDF URL")
    path = pdf_dir / name
    try:
        _download(row["pdf_url"], path)
        with pdfplumber.open(path) as pdf:
            text = "\n".join((page.extract_text() or "")
                             for page in pdf.pages[:3])
            pages = len(pdf.pages)
        return {**row,
                "status": "ok" if confirm_pdf_text(text, row["code"])
                else "identity_or_plan_unconfirmed",
                "pages": pages, "text_first_three_pages": text}
    except Exception as error:
        return {**row, "status": type(error).__name__, "pages": None,
                "text_first_three_pages": ""}


def audit(year: int, source_dir: Path, output_dir: Path,
          workers: int = 3, limit: int | None = None) -> dict:
    if year not in (2024, 2025) or workers < 1:
        raise ValueError("Invalid PDF audit year or worker count")
    source = pd.read_parquet(source_dir / f"title_candidates_{year}.parquet")
    if source.empty or source.pdf_url.duplicated().any():
        raise ValueError("Malformed insider-plan candidate index")
    if limit is not None:
        source = source.head(limit)
    names = source.pdf_url.map(lambda url: Path(urlparse(url).path).name)
    if names.duplicated().any():
        raise ValueError("Original insider-plan PDFs have colliding filenames")
    pdf_dir = output_dir / "pdfs" / str(year)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_record, row, pdf_dir)
                   for row in source.to_dict("records")]
        for future in as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda row: (row["notice_date"], row["code"],
                                  row["pdf_url"]))
    output = output_dir / f"pdf_audit_{year}.jsonl"
    temporary = output.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                 for row in records), encoding="utf-8")
    temporary.replace(output)
    return {"year": year, "records": len(records),
            "status": pd.Series([row["status"] for row in records])
            .value_counts().to_dict(), "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, choices=(2024, 2025), required=True)
    parser.add_argument("--source", type=Path, default=Path(
        "data/research/insider_buy"))
    parser.add_argument("--output", type=Path, default=Path(
        "data/research/insider_buy"))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(audit(args.year, args.source, args.output, args.workers, args.limit))


if __name__ == "__main__":
    main()
