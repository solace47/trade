"""Find Shenzhen before/after pledge rows for source-only manual review.

Table arithmetic can locate a possible tier crossing but does not establish
that the row belongs to the direct controller or an executed net release.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re
from urllib.parse import urlparse

import pandas as pd
import pdfplumber


CELL = re.compile(r"((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(%)?")


def _number(cell: str | None) -> tuple[Decimal, bool] | None:
    if cell is None:
        return None
    match = CELL.fullmatch(re.sub(r"\s+", "", cell))
    if not match:
        return None
    try:
        return Decimal(match.group(1).replace(",", "")), bool(match.group(2))
    except InvalidOperation:
        return None


def _table_rows(table: list[list[str | None]]) -> list[dict]:
    header = re.sub(r"\s+", "", "".join(
        cell or "" for row in table[:15] for cell in row))
    if not all(term in header for term in ("持股数量", "质押", "前", "后")):
        return []
    unit = "万股" if "万股" in header else "股"
    scale = 10000 if unit == "万股" else 1
    proposals = []
    for row in table:
        actor = re.sub(r"\s+", "", row[0] or "") if row else ""
        if not actor or actor in ("合计", "股东", "股东名称", "名称"):
            continue
        prior_count = len(proposals)
        numeric = [(index, *parsed) for index, cell in enumerate(row)
                   if (parsed := _number(cell)) is not None]
        if len(numeric) < 4:
            continue
        # Holding, before, after, and printed after-ratio must occur in order.
        # The printed ratio independently checks the columns being inferred.
        for hi, (_, held, held_pct) in enumerate(numeric):
            if held_pct or held <= 0 or held < 1000:
                continue
            for bi in range(hi + 1, min(hi + 3, len(numeric) - 2)):
                _, before, before_pct = numeric[bi]
                _, after, after_pct = numeric[bi + 1]
                if (before_pct or after_pct or before <= after
                        or before > held or after < 0
                        or 20 * (before - after) < held):
                    continue
                fifty = 2 * before >= held > 2 * after
                eighty = 5 * before >= 4 * held > 5 * after
                if not (fifty or eighty):
                    continue
                _, ratio, _ = numeric[bi + 2]
                if abs(100 * after / held - ratio) > Decimal("0.08"):
                    continue
                shares = [amount * scale for amount in (held, before, after)]
                if any(amount != amount.to_integral_value()
                       for amount in shares):
                    continue
                proposals.append({
                    "actor_cell": actor,
                    "held": int(shares[0]),
                    "before_pledged": int(shares[1]),
                    "after_pledged": int(shares[2]),
                    "before_pct": round(100 * before / held, 4),
                    "after_pct": round(100 * after / held, 4),
                    "crossed_tier": "80" if eighty else "50",
                })
            if len(proposals) > prior_count:
                break
    return proposals


def load_tables(year: int, root: Path, rows: list[dict]) -> list[dict]:
    """Cache source PDF tables so additional input screens reuse exact cells."""
    source = [row for row in rows if row["code"].startswith("sz.")]
    cache = root / f"shenzhen_tables_{year}.jsonl"
    if cache.exists():
        extracted = [json.loads(line) for line in cache.read_text(
            encoding="utf-8").splitlines()]
        if (len(extracted) != len(source)
                or len({row["pdf_url"] for row in extracted}) != len(source)
                or {row["pdf_url"] for row in extracted}
                != {row["pdf_url"] for row in source}):
            raise ValueError("Stale Shenzhen source table cache")
        return extracted
    extracted = []
    for count, row in enumerate(source, 1):
        path = root / "pdfs" / str(year) / Path(
            urlparse(row["pdf_url"]).path).name
        with pdfplumber.open(path) as pdf:
            pages = [page.extract_tables() for page in pdf.pages]
        extracted.append({"pdf_url": row["pdf_url"], "pages": pages})
        if count % 100 == 0:
            print(f"Read {year} Shenzhen tables: {count}", flush=True)
    temporary = cache.with_suffix(".tmp")
    temporary.write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in extracted),
        encoding="utf-8")
    temporary.replace(cache)
    return extracted


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
    source = {row["pdf_url"]: row for row in rows
              if row["code"].startswith("sz.")}
    proposed = []
    for extracted in load_tables(year, root, rows):
        row = source[extracted["pdf_url"]]
        for page_no, tables in enumerate(extracted["pages"], 1):
            for table in tables:
                for result in _table_rows(table):
                    proposed.append({key: row[key] for key in (
                        "code", "notice_date", "title", "pdf_url")}
                                    | {"page": page_no} | result)
    proposed.sort(key=lambda row: (
        row["notice_date"], row["code"], row["pdf_url"], row["actor_cell"]))
    output = root / f"table_crossing_proposals_{year}.csv"
    pd.DataFrame(proposed).drop_duplicates().to_csv(output, index=False)
    return {"year": year, "shenzhen_originals": len(source),
            "table_crossing_proposals": len(proposed),
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
