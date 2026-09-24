"""Cache and inspect original repurchase-plan PDFs without market outcomes."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
from urllib.parse import urlparse

import pandas as pd
import pdfplumber


def _download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 1000:
        return
    temporary = path.with_suffix(".tmp")
    result = subprocess.run(
        ["curl", "-fLsS", "--max-time", "30", url, "-o", str(temporary)],
        capture_output=True, text=True, check=False)
    if result.returncode != 0 or not temporary.exists() or temporary.stat().st_size <= 1000:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Original repurchase PDF download failed: {url}")
    temporary.replace(path)


def _record(row: dict, pdf_dir: Path) -> dict:
    name = Path(urlparse(row["pdf_url"]).path).name
    if not name.lower().endswith(".pdf") or "/" in name:
        raise ValueError("Malformed original repurchase PDF URL")
    path = pdf_dir / name
    try:
        _download(row["pdf_url"], path)
        with pdfplumber.open(path) as pdf:
            text = "\n".join((page.extract_text() or "")
                             for page in pdf.pages[:3])
            pages = len(pdf.pages)
        code = row["code"].split(".")[1]
        status = ("ok" if code in text and "回购" in text and "股份" in text
                  else "identity_unconfirmed")
        return {**row, "status": status, "pages": pages,
                "text_first_three_pages": text}
    except Exception as error:
        return {**row, "status": type(error).__name__, "pages": None,
                "text_first_three_pages": ""}


def audit(year: int, source_dir: Path, output_dir: Path,
          workers: int = 3, limit: int | None = None) -> dict:
    if year not in (2024, 2025) or workers < 1:
        raise ValueError("Invalid PDF audit year or worker count")
    source = pd.read_parquet(source_dir / f"title_candidates_{year}.parquet")
    if source.empty or source.pdf_url.duplicated().any():
        raise ValueError("Malformed buyback candidate index")
    if limit is not None:
        source = source.head(limit)
    names = source.pdf_url.map(lambda url: Path(urlparse(url).path).name)
    if names.duplicated().any():
        raise ValueError("Original buyback PDFs have colliding filenames")
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
        "data/research/buyback"))
    parser.add_argument("--output", type=Path, default=Path(
        "data/research/buyback"))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(audit(args.year, args.source, args.output, args.workers, args.limit))


if __name__ == "__main__":
    main()
