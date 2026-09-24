"""Archive and parse original IPO lockup-release notices without returns."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import pdfplumber

from audit_buyback_pdfs import _download
from trade_research.ipo_unlock_source import extract, strict_title


def _record(row: dict, pdf_dir: Path) -> dict:
    name = Path(urlparse(row["pdf_url"]).path).name
    if not name.lower().endswith(".pdf") or "/" in name:
        raise ValueError("Malformed original unlock PDF URL")
    path = pdf_dir / name
    try:
        _download(row["pdf_url"], path)
        with pdfplumber.open(path) as pdf:
            text = "\n".join((page.extract_text() or "")
                             for page in pdf.pages[:3])
            pages = len(pdf.pages)
        return {**row, **extract(text, row["code"], row["notice_date"]),
                "pages": pages}
    except Exception as error:
        return {**row, "status": type(error).__name__, "pages": None}


def audit(year: int, base: Path, workers: int = 3) -> dict:
    if year not in (2024, 2025) or workers < 1:
        raise ValueError("Invalid IPO unlock audit year or worker count")
    source = pd.read_parquet(base / f"search_{year}.parquet")
    if (source.empty or source.pdf_url.duplicated().any()
            or not source.notice_date.str.startswith(str(year)).all()):
        raise ValueError("Malformed CNINFO IPO unlock index")
    selected = source.loc[source.title.map(strict_title)].copy()
    if selected.empty or selected.pdf_url.duplicated().any():
        raise ValueError("No unique original IPO unlock title candidates")
    pdf_dir = base / "pdfs" / str(year)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    names = selected.pdf_url.map(
        lambda url: Path(urlparse(url).path).name)
    if names.duplicated().any():
        raise ValueError("IPO unlock PDF filenames collide")
    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_record, row, pdf_dir)
                   for row in selected.to_dict("records")]
        for future in as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda row: (row["notice_date"], row["code"],
                                  row["pdf_url"]))
    output = base / f"pdf_audit_{year}.jsonl"
    temporary = output.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                 for row in records), encoding="utf-8")
    temporary.replace(output)
    return {"year": year, "searched_pdf": len(source),
            "title_candidates": len(selected),
            "pdf_status": pd.Series([row["status"] for row in records])
            .value_counts().to_dict()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, choices=(2024, 2025), required=True)
    parser.add_argument("--base", type=Path, default=Path(
        "data/research/unlock"))
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    print(audit(args.year, args.base, args.workers))


if __name__ == "__main__":
    main()
