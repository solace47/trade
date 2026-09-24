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
import unicodedata
from urllib.parse import urlparse

import pandas as pd
import pdfplumber

from scripts.audit_buyback_pdfs import _download


ACTOR = re.compile(r"控股股东|实际控制人|(?<!副)董事长|5\s*%\s*以上|大股东")
COMPLETION = re.compile(r"完成|完毕")
STOCK_CODE = re.compile(
    r"(?:证券代码|股票代码|票代码|公司代码|A股代码)[:：]?([036]\d{5})"
)
# The issuer's 2025-09-18 completion PDF misprints its header as 002268.
# Its body, CNINFO metadata, and the issuer's 2025-05-30 plan PDF all identify
# 保龄宝 as 002286. This exception was checked before reading any outcomes.
# Prior plan: https://static.cninfo.com.cn/finalpage/2025-05-30/1223721665.PDF
VERIFIED_HEADER_TYPOS = {
    "https://static.cninfo.com.cn/finalpage/2025-09-18/1224667235.PDF":
        ("sz.002286", "002268", "保龄宝生物股份有限公司")
}


def candidate_title(title: str) -> bool:
    """Limit full-text search hits to explicit actor/completion titles."""
    title = unicodedata.normalize("NFKC", title)
    return (bool(ACTOR.search(title)) and "减持" in title
            and bool(COMPLETION.search(title)))


def pdf_code_matches(text: str, code: str) -> bool:
    """Check the PDF's header code, not a code repeated elsewhere in its body."""
    header = re.sub(r"\s+", "", text)[:600]
    match = STOCK_CODE.search(header)
    return bool(match and match.group(1) == code.split(".")[1])


def identity_status(text: str, row: dict) -> str:
    if "减持" not in text:
        return "identity_unconfirmed"
    if pdf_code_matches(text, row["code"]):
        return "ok"
    exception = VERIFIED_HEADER_TYPOS.get(row["pdf_url"])
    if exception is None:
        return "identity_unconfirmed"
    code, misprint, issuer = exception
    header = re.sub(r"\s+", "", text)[:600]
    match = STOCK_CODE.search(header)
    if (row["code"] == code and match and match.group(1) == misprint
            and issuer in header):
        return "verified_header_typo"
    return "identity_unconfirmed"


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
        status = identity_status(text, row)
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
    source = source.loc[source.title.map(candidate_title)]
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
