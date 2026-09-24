"""Count original sell-completion PDF input risks without reading returns.

This is a feasibility diagnostic, not an eligibility classifier: notices with
multiple actors or positive sales still require a full manual PDF audit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from urllib.parse import urlparse

import pandas as pd
import pdfplumber


ACTOR = re.compile(r"控股股东|实际控制人|董事长|持股\s*5\s*%\s*以上|大股东")
TITLE_CONNECTOR = re.compile(r"及|、|部分")


def diagnose(index: Path, pdf_dir: Path, prior_plan_audit: Path | None = None) -> dict:
    source = pd.read_parquet(index)
    required = {"code", "notice_date", "title", "pdf_url"}
    if not required.issubset(source) or source.pdf_url.duplicated().any():
        raise ValueError("Malformed sell-completion index")
    candidates = source[source.title.map(lambda title: bool(ACTOR.search(title)))]
    if candidates.empty:
        raise ValueError("No actor-title candidates")

    prior = pd.DataFrame(columns=["code", "notice_date"])
    if prior_plan_audit is not None:
        rows = [json.loads(line) for line in prior_plan_audit.read_text(
            encoding="utf-8").splitlines()]
        prior = pd.DataFrame(rows)
        prior = prior.loc[prior.status.eq("ok"), ["code", "notice_date"]]
        prior["notice_date"] = pd.to_datetime(prior.notice_date)

    counts = {"indexed_a_share_pdfs": len(source),
              "actor_title_candidates": len(candidates),
              "originals_opened": 0, "code_in_first_three_pages": 0,
              "title_with_connector": 0, "more_than_three_pages": 0,
              "same_code_prior_plan": 0}
    for row in candidates.itertuples(index=False):
        name = Path(urlparse(row.pdf_url).path).name
        if not name.lower().endswith(".pdf") or "/" in name:
            raise ValueError("Malformed original PDF URL")
        path = pdf_dir / name
        with pdfplumber.open(path) as pdf:
            text = "\n".join((page.extract_text() or "") for page in pdf.pages[:3])
            counts["more_than_three_pages"] += len(pdf.pages) > 3
        counts["originals_opened"] += 1
        counts["code_in_first_three_pages"] += row.code.split(".")[1] in text
        counts["title_with_connector"] += bool(TITLE_CONNECTOR.search(row.title))
        if not prior.empty:
            counts["same_code_prior_plan"] += bool((
                prior.code.eq(row.code)
                & prior.notice_date.lt(pd.Timestamp(row.notice_date))).any())
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--pdf-dir", type=Path, required=True)
    parser.add_argument("--prior-plan-audit", type=Path)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.index, args.pdf_dir, args.prior_plan_audit),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
