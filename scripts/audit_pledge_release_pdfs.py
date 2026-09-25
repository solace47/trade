"""Download original controller pledge-release PDFs and verify issuer identity.

Identity and text extraction are input checks only. A valid PDF has not yet
been shown to contain an executed release or a risk-tier crossing.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import pdfplumber

from scripts.audit_buyback_pdfs import _download
from scripts.audit_sell_completion_pdfs import pdf_code_matches
from scripts.collect_pledge_release_index import candidate_title


def _record(row: dict, pdf_dir: Path) -> dict:
    name = Path(urlparse(row["pdf_url"]).path).name
    if not name.lower().endswith(".pdf") or "/" in name:
        raise ValueError("Malformed pledge-release PDF URL")
    try:
        path = pdf_dir / name
        _download(row["pdf_url"], path)
        with pdfplumber.open(path) as pdf:
            content = "\n".join(page.extract_text() or "" for page in pdf.pages)
            pages = len(pdf.pages)
        status = ("ok" if "解除质押" in content
                  and pdf_code_matches(content, row["code"])
                  else "identity_unconfirmed")
        return {**row, "status": status, "pages": pages,
                "text_full": content}
    except Exception as error:
        return {**row, "status": type(error).__name__, "pages": None,
                "text_full": ""}


def audit(year: int, root: Path, workers: int = 4) -> dict:
    if year not in (2024, 2025) or not 1 <= workers <= 6:
        raise ValueError("Invalid pledge-release audit settings")
    indexed = pd.read_parquet(root / f"search_{year}.parquet")
    source = pd.read_parquet(root / f"title_candidates_{year}.parquet")
    expected = indexed.loc[indexed.title.map(candidate_title)]
    if (source.empty or indexed.pdf_url.duplicated().any()
            or source.pdf_url.duplicated().any()
            or not indexed.notice_date.str.startswith(str(year)).all()
            or not source.notice_date.str.startswith(str(year)).all()
            or not source.sort_values("pdf_url").reset_index(drop=True).equals(
                expected.sort_values("pdf_url").reset_index(drop=True))):
        raise ValueError("Malformed or incomplete pledge-release title index")
    names = source.pdf_url.map(
        lambda url: Path(urlparse(url).path).name)
    if names.duplicated().any():
        raise ValueError("Pledge-release PDF filenames collide")
    pdf_dir = root / "pdfs" / str(year)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n, item in enumerate(pool.map(
                lambda row: _record(row, pdf_dir),
                source.to_dict("records")), start=1):
            records.append(item)
            if n % 100 == 0 or n == len(source):
                print(f"Audited {year}: {n}/{len(source)} originals",
                      flush=True)
    records.sort(key=lambda item: (
        item["notice_date"], item["code"], item["pdf_url"]))
    output = root / f"pdf_audit_{year}.jsonl"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n"
                for item in records), encoding="utf-8")
    temporary.replace(output)
    statuses = pd.Series([item["status"] for item in records])
    report = {"year": year, "title_candidates": len(records),
              "identity_status": statuses.value_counts().to_dict(),
              "outcomes_opened": False}
    (root / f"pdf_identity_audit_{year}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, choices=(2024, 2025), required=True)
    parser.add_argument("--source", type=Path, default=Path(
        "data/research/pledge"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(audit(args.year, args.source, args.workers))


if __name__ == "__main__":
    main()
