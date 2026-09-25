"""Propose original-text ratio crossings for manual review, without outcomes.

Only the narrow vertical disclosure layout is parsed. A proposal is never an
eligible signal: a reviewer must verify the direct actor, registration, all
share counts, and absence of new pledge or other excluded transactions.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import json
from pathlib import Path
import re

import pandas as pd


NUMBER = r"((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
SHARES = r"(?:[（(]股[）)])?"
FIELDS = {
    "released": re.compile(
        rf"本次(?:解质|解除质押)(?:[（(]解冻[）)])?股份(?:数量)?{SHARES}[:：]?{NUMBER}(万股|股)?"
    ),
    "held": re.compile(rf"持股数量{SHARES}[:：]?{NUMBER}(万股|股)?"),
    "remaining": re.compile(
        rf"剩余被质押(?:[（(]被冻结[）)])?股份数量{SHARES}[:：]?{NUMBER}(万股|股)?"
    ),
}


def vertical_crossing(text: str) -> dict | None:
    """Return mechanical counts only where all three original fields parse."""
    compact = re.sub(r"\s+", "", text)
    matches = {name: pattern.search(compact)
               for name, pattern in FIELDS.items()}
    if not all(matches.values()):
        return None
    explicit_units = {match.group(2) for match in matches.values()
                      if match.group(2)}
    if len(explicit_units) > 1:
        return None
    context = compact[max(0, matches["released"].start() - 100):
                      matches["released"].start()]
    table_unit = "万股" if re.search(r"单位[:：]万股", context) else "股"
    unit = next(iter(explicit_units), table_unit)
    missing_units = any(match.group(2) is None
                        for match in matches.values())
    if missing_units and unit != table_unit:
        return None
    scale = 10000 if unit == "万股" else 1
    decimal_values = {
        name: Decimal(match.group(1).replace(",", "")) * scale
        for name, match in matches.items()
    }
    if any(value != value.to_integral_value()
           for value in decimal_values.values()):
        return None
    values = {name: int(value) for name, value in decimal_values.items()}
    released, held, remaining = (values[name]
                                 for name in ("released", "held", "remaining"))
    if not (0 < held and 0 < released <= held
            and 0 <= remaining <= held
            and remaining + released <= held):
        return None
    fifty = 2 * (remaining + released) >= held > 2 * remaining
    eighty = 5 * (remaining + released) >= 4 * held > 5 * remaining
    if not (20 * released >= held and (fifty or eighty)):
        return None
    return {**values, "before_pct": round(100 * (remaining + released) / held, 4),
            "after_pct": round(100 * remaining / held, 4),
            "crossed_tier": "80" if eighty else "50"}


def propose(year: int, root: Path) -> dict:
    if year not in (2024, 2025):
        raise ValueError("Only frozen 2024/2025 pledge years")
    indexed = pd.read_parquet(root / f"title_candidates_{year}.parquet")
    rows = [json.loads(line) for line in
            (root / f"pdf_audit_{year}.jsonl").read_text(
                encoding="utf-8").splitlines()]
    if (len(rows) != len(indexed)
            or {row["pdf_url"] for row in rows} != set(indexed.pdf_url)
            or any(row["status"] != "ok" for row in rows)):
        raise ValueError("Incomplete original PDF identity audit")
    proposed = []
    for row in rows:
        counts = vertical_crossing(row["text_full"])
        if counts:
            proposed.append({key: row[key] for key in (
                "code", "notice_date", "title", "pdf_url")}
                            | counts)
    proposed.sort(key=lambda row: (
        row["notice_date"], row["code"], row["pdf_url"]))
    output = root / f"vertical_crossing_proposals_{year}.csv"
    pd.DataFrame(proposed).to_csv(output, index=False)
    return {"year": year, "originals": len(rows),
            "vertical_crossing_proposals": len(proposed),
            "months": len({row["notice_date"][:7] for row in proposed}),
            "outcomes_opened": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, choices=(2024, 2025), required=True)
    parser.add_argument("--root", type=Path, default=Path(
        "data/research/pledge"))
    args = parser.parse_args()
    print(propose(args.year, args.root))


if __name__ == "__main__":
    main()
