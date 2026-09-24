"""Archive full original sell-completion PDF text for actor-level review.

This script only checks original identity and extraction. It does not infer that
a qualifying holder sold shares or read any future price or trade outcome.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
from urllib.parse import urlparse

import pandas as pd
import pdfplumber

from audit_buyback_pdfs import _download


ACTOR = re.compile(r"控股股东|实际控制人|董事长|持股\s*5\s*%\s*以上|大股东")


def _record(row: dict, pdf_dir: Path) -> dict:
    name = Path(urlparse(row["pdf_url"]).path).name
    if not name.lower().endswith(".pdf") or "/" in name:
        raise ValueError("Malformed original sell-completion PDF URL")
    try:
        path = pdf_dir / name
        _download(row["pdf_url"], path)
        with pdfplumber.open(path) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            pages = len(pdf.pages)
        code = row["code"].split(".")[1]
        status = "ok" if code in text and "减持" in text else "identity_unconfirmed"
        return {**row, "status": status, "pages": pages, "text_full": text}
    except Exception as error:
        return {**row, "status": type(error).__name__, "pages": None,
                "text_full": ""}


def audit(year: int, source_dir: Path, output_dir: Path,
          workers: int = 3) -> dict:
    if year not in (2024, 2025) or not 1 <= workers <= 6:
        raise ValueError("Invalid sell-completion year or worker count")
    source = pd.read_parquet(source_dir / f"search_{year}.parquet")
    if (source.empty or source.pdf_url.duplicated().any()
            or not source.notice_date.str.startswith(str(year)).all()):
        raise ValueError("Malformed complete sell-completion index")
    source = source.loc[source.title.map(lambda title: bool(ACTOR.search(title)))]
    if source.empty:
        raise ValueError("No actor-title completion candidates")
    names = source.pdf_url.map(lambda url: Path(urlparse(url).path).name)
    if names.duplicated().any():
        raise ValueError("Sell-completion PDFs have colliding filenames")
    pdf_dir = output_dir / "pdfs" / str(year)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(lambda row: _record(row, pdf_dir),
                                source.to_dict("records")))
    records.sort(key=lambda row: (row["notice_date"], row["code"],
                                  row["pdf_url"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"pdf_audit_{year}.jsonl"
    temporary = output.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                 for row in records), encoding="utf-8")
    temporary.replace(output)
    return {"year": year, "actor_title_candidates": len(records),
            "status": pd.Series([row["status"] for row in records])
            .value_counts().to_dict(), "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, choices=(2024, 2025), required=True)
    parser.add_argument("--source", type=Path, default=Path(
        "data/research/insider_sell_complete"))
    parser.add_argument("--output", type=Path, default=Path(
        "data/research/insider_sell_complete"))
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    print(audit(args.year, args.source, args.output, args.workers))


if __name__ == "__main__":
    main()
