"""Cache the 32 preselected broker source documents without reading market data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import pdfplumber
import requests

from .analyst_catalog import ROOT
from .corporate_cash import save_json, sha


SAMPLE_SHA = "d3ecbeea521add585bdaddf56bf1a9b12aa59f10538533ae488b5f9ae2800001"


def detail_object(html: str, info_code: str) -> dict:
    match = re.search(r"\bvar\s+zwinfo\s*=\s*", html)
    if not match:
        raise ValueError("Historical report metadata is missing")
    value, _ = json.JSONDecoder().raw_decode(html[match.end():])
    if value.get("info_code") != info_code:
        raise ValueError("Detail page changed report identity")
    if not isinstance(value.get("notice_content"), str):
        raise ValueError("Report abstract is missing")
    return value


def collect(root: Path = ROOT) -> list[dict]:
    if sha(root / "sample.json") != SAMPLE_SHA:
        raise ValueError("Preselected 32-report sample changed")
    sample = json.loads((root / "sample.json").read_text())
    session = requests.Session()
    session.trust_env = False
    session.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://data.eastmoney.com/"})
    for folder in ("html", "pdf", "text", "detail"):
        (root / folder).mkdir(exist_ok=True)

    def fetch(url: str, path: Path) -> None:
        manifest = path.with_suffix(path.suffix + ".request.json")
        if path.exists():
            old = json.loads(manifest.read_text())
            if old["url"] != url or old["sha256"] != sha(path):
                raise ValueError("Cached source bytes or URL changed")
            return
        response = session.get(url, timeout=45)
        response.raise_for_status()
        path.write_bytes(response.content)
        save_json(manifest, {"url": url, "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                             "sha256": sha(path)})

    records = []
    for index, item in enumerate(sample, 1):
        code = item["info_code"]
        # A malformed list date must not cause us to open a 2026 document.
        if not re.fullmatch(r"AP202[45]\d+", code):
            raise ValueError("Document identifier requires a separate period review")
        html = root / "html" / f"{code}.html"
        fetch(f"https://data.eastmoney.com/report/info/{code}.html", html)
        detail = detail_object(html.read_text(), code)
        save_json(root / "detail" / f"{code}.json", detail)
        pdf_path = root / "pdf" / f"{code}.pdf"
        fetch(f"https://pdf.dfcfw.com/pdf/H3_{code}_1.pdf", pdf_path)
        text_path = root / "text" / f"{code}.json"
        if text_path.exists():
            extraction = json.loads(text_path.read_text())
            if extraction["pdf_sha256"] != sha(pdf_path):
                raise ValueError("Cached text was extracted from different PDF bytes")
        else:
            with pdfplumber.open(pdf_path) as pdf:
                extraction = {"pdf_sha256": sha(pdf_path), "pages": [p.extract_text() or "" for p in pdf.pages],
                              "metadata": {k: pdf.metadata.get(k) for k in ("CreationDate", "ModDate")}}
            save_json(text_path, extraction)
        record = dict(item, detail_company_code=detail.get("company_code"),
            detail_notice_date=detail.get("notice_date"), production_time=detail.get("eitime"),
            detail_security=detail.get("security"), pdf_pages=len(extraction["pages"]),
            pdf_metadata_not_publication_proof=extraction["metadata"],
            html_sha256=sha(html), pdf_sha256=sha(pdf_path), extraction_sha256=sha(text_path))
        records.append(record)
        save_json(root / "source_manifest_partial.json", records)
        print({"completed": index, "total": len(sample), "info_code": code,
               "pdf_pages": record["pdf_pages"], "display_date": item["display_date"],
               "production_time": record["production_time"]}, flush=True)
    save_json(root / "source_manifest.json", records)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    collect(args.root)


if __name__ == "__main__":
    main()
