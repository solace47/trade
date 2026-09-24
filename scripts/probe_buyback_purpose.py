"""Input-only, provisional purpose tags for original buyback-plan PDFs.

These tags are a feasibility audit, not a trading signal. First-page summary
tables, mixed uses and fallback cancellation clauses can remain ambiguous.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import pandas as pd


PURPOSE = re.compile(r"(?:拟?回购股份的用途|回购用途|股份用途)[:：]?")
TERMS = {
    "cancel": ("注销并减少注册资本", "注销并减少", "全部用于注销",
               "用于注销", "减少注册资本", "减资注销", "减少公司注册资本"),
    "employee": ("员工持股", "股权激励"),
    "maintain": ("维护公司价值", "股东权益"),
    "convertible": ("可转债", "可转换公司债券"),
}


def provisional_purpose(text: str) -> str:
    clean = re.sub(r"\s+", "", text)
    match = PURPOSE.search(clean)
    if match is None:
        return "unknown"
    snippet = clean[match.end():match.end() + 130]
    # Some Shanghai summaries list unchecked alternatives before the tick.
    if "√" in snippet[:100]:
        snippet = snippet[snippet.index("√"):]
    hits = [(snippet.index(term), label)
            for label, terms in TERMS.items()
            for term in terms if term in snippet]
    return min(hits)[1] if hits else "unknown"


def audit(source_dir: Path, pairs_path: Path, output: Path) -> dict:
    original = pd.concat([
        pd.read_json(source_dir / f"pdf_audit_{year}.jsonl", lines=True)[
            ["pdf_url", "status", "text_first_three_pages"]]
        for year in (2024, 2025)
    ], ignore_index=True)
    events = pd.read_parquet(pairs_path)
    events = events.loc[events.candidate.eq("buyback_plan"),
                        ["date", "code", "pdf_url"]]
    if (events.empty or events.pdf_url.duplicated().any()
            or original.pdf_url.duplicated().any()):
        raise ValueError("Provisional purpose source is duplicated")
    joined = events.merge(original, on="pdf_url", validate="one_to_one")
    if len(joined) != len(events) or not joined.status.eq("ok").all():
        raise ValueError("Frozen event lacks a confirmed original PDF")
    joined["purpose"] = joined.text_first_three_pages.map(provisional_purpose)
    report = {"note": "Input-only provisional tags; no purpose returns read",
              "by_year": {}}
    for year in ("2024", "2025"):
        period = joined.loc[joined.date.str.startswith(year)]
        report["by_year"][year] = {
            label: {"events": len(group),
                    "days": int(group.date.nunique()),
                    "months": int(group.date.str[:7].nunique())}
            for label, group in period.groupby("purpose")
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(
        "data/research/buyback"))
    parser.add_argument("--pairs", type=Path, default=Path(
        "data/research/buyback/pairs.parquet"))
    parser.add_argument("--output", type=Path, default=Path(
        "data/research/buyback/purpose_probe.json"))
    args = parser.parse_args()
    print(audit(args.source, args.pairs, args.output))


if __name__ == "__main__":
    main()
