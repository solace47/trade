"""Archive full text of every broad-search investigation PDF, without returns."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
from urllib.parse import urlparse

import pandas as pd
import pdfplumber

from audit_buyback_pdfs import _download


def _record(row: dict, pdf_dir: Path) -> dict:
    name = Path(urlparse(row["pdf_url"]).path).name
    path = pdf_dir / name
    try:
        _download(row["pdf_url"], path)
        with pdfplumber.open(path) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            pages = len(pdf.pages)
        clean = re.sub(r"\s+", "", text)
        status = ("ok" if row["code"].split(".")[-1] in clean
                  else "identity_unconfirmed")
        return {**row, "status": status, "pages": pages,
                "text_full_pdf": text}
    except Exception as error:
        return {**row, "status": type(error).__name__, "pages": None,
                "text_full_pdf": ""}


def audit(year: int, source_dir: Path, workers: int = 4) -> dict:
    if year not in (2024, 2025) or workers < 1:
        raise ValueError("Only 2024/2025 investigation originals")
    index = pd.read_parquet(source_dir / f"original_candidates_{year}.parquet")
    names = index.pdf_url.map(lambda url: Path(urlparse(url).path).name)
    if (index.empty or index.pdf_url.duplicated().any()
            or names.duplicated().any()
            or not names.str.lower().str.endswith(".pdf").all()):
        raise ValueError("Malformed original investigation PDF index")
    pdf_dir = source_dir / "pdfs" / str(year)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_record, row, pdf_dir)
                   for row in index.to_dict("records")]
        records = [future.result() for future in as_completed(futures)]
    records.sort(key=lambda row: (row["notice_date"], row["code"],
                                  row["pdf_url"]))
    output = source_dir / f"pdf_audit_{year}.jsonl"
    temporary = output.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                 for row in records), encoding="utf-8")
    temporary.replace(output)
    return {"year": year, "originals": len(records),
            "status": pd.Series([row["status"] for row in records])
            .value_counts().to_dict(),
            "pages_extracted": sum(int(row["pages"] or 0) for row in records)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, choices=(2024, 2025), required=True)
    parser.add_argument("--source", type=Path, default=Path(
        "data/research/regulatory_investigation"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(audit(args.year, args.source, args.workers))


if __name__ == "__main__":
    main()
