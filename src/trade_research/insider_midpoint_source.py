"""Freeze midpoint zero-execution notices from original CNINFO documents.

Only announcement metadata and original PDF text are read here. A small,
versioned review ledger resolves mixed-subject and prior-plan disclosures.
No market outcomes may enter this module.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import pandas as pd


ACTOR = re.compile(r"控股股东|实际控制人|董事长")
MIDPOINT = re.compile(r"时间过半|期限过半|期间过半")
ZERO = re.compile(
    r"(?:尚未|暂未|仍未)(?:通过.{0,16})?增持|"
    r"(?:尚未|暂未|仍未)通过[^。；]{0,60}?(?:进行)?增持|"
    r"尚未实施增持|尚未开展对公司股票的增持"
)
SEARCHES = ("primary", "implementation", "broad")
YEARS = (2024, 2025)


def midpoint_title(title: str) -> bool:
    return bool(ACTOR.search(title) and MIDPOINT.search(title))


def _clean(text: str) -> str:
    return re.sub(r"\s+", "", text)


def zero_evidence(text: str, code: str) -> str | None:
    clean = _clean(text)
    if code.split(".")[-1] not in clean or not MIDPOINT.search(clean):
        return None
    match = ZERO.search(clean)
    if match is None:
        return None
    return clean[max(0, match.start() - 40):match.end() + 40]


def screen(source_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for year in YEARS:
        sources = {}
        for name in SEARCHES:
            frame = pd.read_parquet(source_dir / f"search_{name}_{year}.parquet")
            if (frame.empty or frame.pdf_url.duplicated().any()
                    or not frame.notice_date.str.startswith(str(year)).all()):
                raise ValueError(f"Malformed {name} CNINFO midpoint search")
            sources[name] = frame
        combined = pd.concat(sources.values(), ignore_index=True)
        disagreements = (combined.loc[combined.pdf_url.duplicated(keep=False)]
                         .groupby("pdf_url")[["code", "notice_date", "title"]]
                         .nunique().gt(1).any().any())
        if disagreements:
            raise ValueError("Searches disagree on the same original PDF")
        if not set(sources["primary"].pdf_url).issubset(sources["broad"].pdf_url):
            raise ValueError("Broad search missed primary PDF rows")
        if not set(sources["implementation"].pdf_url).issubset(sources["broad"].pdf_url):
            raise ValueError("Broad search missed implementation PDF rows")
        union = combined.drop_duplicates("pdf_url")
        selected = union.loc[union.title.map(midpoint_title)].copy()
        if (selected.empty or selected.duplicated(["code", "notice_date"]).any()
                or selected.pdf_url.duplicated().any()):
            raise ValueError("Ambiguous midpoint title candidates")
        selected.sort_values(["notice_date", "code"]).to_parquet(
            output_dir / f"title_candidates_{year}.parquet", index=False,
            compression="zstd")
        month = selected.notice_date.str[:7].value_counts()
        report[str(year)] = {
            "search_pdf_rows": {name: len(frame)
                                for name, frame in sources.items()},
            "union_pdf_rows": len(union),
            "midpoint_actor_titles": len(selected),
            "peak_month": month.idxmax(),
            "peak_month_share": float(month.max() / len(selected)),
        }
    (output_dir / "title_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return report


def audit(source_dir: Path, review_path: Path) -> dict:
    review = pd.read_csv(review_path, dtype={"code": str, "notice_date": str})
    required = {"code", "notice_date", "zero_confirmed", "reason"}
    if (set(review.columns) != required
            or review.duplicated(["code", "notice_date"]).any()
            or not review.zero_confirmed.isin([0, 1]).all()
            or review.reason.isna().any()):
        raise ValueError("Malformed manual PDF review ledger")
    marked_keys = set()
    report = {}
    for year in YEARS:
        index = pd.read_parquet(source_dir / f"title_candidates_{year}.parquet")
        originals = pd.read_json(source_dir / f"pdf_audit_{year}.jsonl", lines=True)
        if (index.empty or index.pdf_url.duplicated().any()
                or originals.pdf_url.duplicated().any()
                or not index.title.map(midpoint_title).all()):
            raise ValueError("Incomplete original midpoint audit")
        joined = index.merge(originals[["pdf_url", "status", "text_full_pdf"]],
                             on="pdf_url", how="left", validate="one_to_one")
        if joined.status.isna().any() or joined.text_full_pdf.isna().any():
            raise ValueError("Midpoint title lacks full original PDF review")
        joined["evidence"] = joined.apply(
            lambda row: zero_evidence(row.text_full_pdf, row.code), axis=1)
        marked = joined.loc[joined.evidence.notna()].copy()
        keys = set(zip(marked.code, marked.notice_date))
        marked_keys.update(keys)
        decisions = review.loc[review.notice_date.str.startswith(str(year))]
        if keys != set(zip(decisions.code, decisions.notice_date)):
            raise ValueError("Zero-marker PDFs and reviewed ledger differ")
        marked = marked.merge(decisions, on=["code", "notice_date"],
                              validate="one_to_one")
        if marked.loc[marked.zero_confirmed.eq(1), "status"].ne("ok").any():
            raise ValueError("Accepted zero notice lacks original identity audit")
        confirmed = marked.loc[marked.zero_confirmed.eq(1),
                               ["code", "notice_date", "title", "pdf_url",
                                "announcement_time_ms", "evidence"]].copy()
        confirmed.sort_values(["notice_date", "code"]).to_parquet(
            source_dir / f"confirmed_{year}.parquet", index=False,
            compression="zstd")
        report[str(year)] = {
            "all_title_pdfs_full_text": len(joined),
            "zero_marker_pdfs_reviewed": len(marked),
            "rejected_or_ambiguous_pdfs": int(marked.zero_confirmed.eq(0).sum()),
            "confirmed_zero_pdfs": len(confirmed),
            "confirmed_stocks": int(confirmed.code.nunique()),
        }
    if marked_keys != set(zip(review.code, review.notice_date)):
        raise ValueError("Review ledger has entries outside the search years")
    (source_dir / "source_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(
        "data/research/insider_midpoint"))
    parser.add_argument("--review", type=Path, default=Path(
        "config/insider_midpoint_reviews.csv"))
    args = parser.parse_args()
    print({"search": screen(args.source, args.source),
           "originals": audit(args.source, args.review)})


if __name__ == "__main__":
    main()
