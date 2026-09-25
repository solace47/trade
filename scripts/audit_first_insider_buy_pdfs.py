"""Check first executed insider purchases against original CNINFO PDFs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import pdfplumber

from audit_buyback_pdfs import _download
from trade_research.first_insider_buy_source import confirm_original


def _check(row: dict, pdf_dir: Path) -> dict:
    name = Path(urlparse(row["pdf_url"]).path).name
    path = pdf_dir / name
    try:
        _download(row["pdf_url"], path)
        with pdfplumber.open(path) as document:
            text = "\n".join((page.extract_text() or "")
                             for page in document.pages[:3])
            pages = len(document.pages)
        return {**row, **confirm_original(text, row["code"], row["notice_date"]),
                "pages": pages, "text_first_three_pages": text}
    except Exception as error:
        return {**row, "status": type(error).__name__, "pages": None,
                "text_first_three_pages": ""}


def audit(year: int, source_dir: Path, output_dir: Path,
          workers: int = 3) -> dict:
    if year not in (2024, 2025) or workers < 1:
        raise ValueError("Invalid original audit year or worker count")
    source = pd.read_parquet(source_dir / f"title_candidates_{year}.parquet")
    if source.empty or source.pdf_url.duplicated().any():
        raise ValueError("Incomplete first-insider-buy candidate index")
    names = source.pdf_url.map(lambda url: Path(urlparse(url).path).name)
    if names.duplicated().any() or not names.str.lower().str.endswith(".pdf").all():
        raise ValueError("Malformed or colliding original PDF names")
    pdf_dir = output_dir / "pdfs" / str(year)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_check, row, pdf_dir)
                   for row in source.to_dict("records")]
        for future in as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda item: (item["notice_date"], item["code"],
                                   item["pdf_url"]))
    output = output_dir / f"pdf_audit_{year}.jsonl"
    temporary = output.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                 for row in records), encoding="utf-8")
    temporary.replace(output)
    return {"year": year, "records": len(records),
            "status": pd.Series([row["status"] for row in records])
            .value_counts().to_dict(),
            "parsed_amounts": sum(item.get("parsed_actual_amount_yuan") is not None
                                  for item in records), "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, choices=(2024, 2025), required=True)
    parser.add_argument("--source", type=Path, default=Path(
        "data/research/first_insider_buy"))
    parser.add_argument("--output", type=Path, default=Path(
        "data/research/first_insider_buy"))
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    print(audit(args.year, args.source, args.output, args.workers))


if __name__ == "__main__":
    main()
