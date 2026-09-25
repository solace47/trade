"""Propose Shenzhen releases whose originals show only post-release pledge stock.

This is a permissive input review queue. A matching arithmetic relationship
does not prove direct-controller identity or completed net release.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import json
from pathlib import Path
import re

import pandas as pd

from scripts.propose_pledge_table_crossings import _number, load_tables


def _header(table: list[list[str | None]]) -> str:
    return re.sub(r"\s+", "", "".join(
        cell or "" for row in table[:15] for cell in row))


def _shares(value: Decimal, unit: str) -> int | None:
    scaled = value * (10000 if unit == "万股" else 1)
    return int(scaled) if scaled == scaled.to_integral_value() else None


def _release_rows(table: list[list[str | None]]) -> list[dict]:
    # The table body may say "until release registration" in an extension
    # schedule. Only the column heading can establish this as a release table.
    header = _header(table[:1])
    if (not ("解除质押" in header or "解质押" in header
             or "解质股份" in header)
            or "持股数量" in header):
        return []
    unit = "万股" if "万股" in header else "股"
    out = []
    current_actor = ""
    for row in table:
        listed_actor = re.sub(r"\s+", "", row[0] or "") if row else ""
        if listed_actor in ("股东", "股东名称", "名称"):
            continue
        if listed_actor:
            current_actor = listed_actor
        actor = current_actor
        if not actor:
            continue
        numbers = [(value, percent) for cell in row
                   if (parsed := _number(cell)) is not None
                   for value, percent in (parsed,)]
        for index, (amount, is_percent) in enumerate(numbers):
            if is_percent or amount <= 0:
                continue
            shares = _shares(amount, unit)
            if shares is None or shares <= 0:
                continue
            next_percent = next((value for value, percent in
                                 numbers[index + 1:index + 4]
                                 if percent and 0 <= value <= 100), None)
            if next_percent is None:
                continue
            out.append({"release_actor_cell": actor,
                        "released": shares,
                        "released_pct_printed": next_percent})
            break
    by_actor: dict[str, list[dict]] = {}
    for row in out:
        if row["release_actor_cell"] != "合计":
            by_actor.setdefault(row["release_actor_cell"], []).append(row)
    for actor, group in by_actor.items():
        if len(group) > 1:
            released = sum(row["released"] for row in group)
            if not any(row["released"] == released for row in group):
                out.append({"release_actor_cell": actor,
                            "released": released,
                            "released_pct_printed": sum(
                                row["released_pct_printed"] for row in group)})
    return out


def _after_rows(table: list[list[str | None]]) -> list[dict]:
    header = _header(table)
    heading_rows = table[:4]
    heading_columns = (
        re.sub(r"\s+", "", "".join(
            row[index] or "" for row in heading_rows if index < len(row)))
        for index in range(max((len(row) for row in heading_rows), default=0)))
    prior_column = any(
        re.search(r"(?:本次|交易|变动|延期|解除|解押|质押)"
                  r".{0,20}前.{0,6}质押|质押前", column)
        for column in heading_columns)
    if ("持股数量" not in header or "质押" not in header
            or prior_column):
        return []
    unit = "万股" if "万股" in header else "股"
    out = []
    for row in table:
        actor = re.sub(r"\s+", "", row[0] or "") if row else ""
        if not actor or actor in ("合计", "股东", "股东名称", "名称"):
            continue
        numbers = [(value, percent) for cell in row
                   if (parsed := _number(cell)) is not None
                   for value, percent in (parsed,)]
        held_items = [(i, value) for i, (value, percent) in enumerate(numbers)
                      if not percent and value >= 1000]
        if not held_items:
            continue
        index, native_held = held_items[0]
        held = _shares(native_held, unit)
        if held is None:
            continue
        for native_after, percent in numbers[index + 1:index + 4]:
            if (percent or (native_after < 100 and native_after != 0)
                    or native_after > native_held):
                continue
            after = _shares(native_after, unit)
            if after is not None:
                out.append({"after_actor_cell": actor,
                            "held": held, "after_pledged": after})
            break
    return out


def after_only_crossings(pages: list[list[list[list[str | None]]]]) -> list[dict]:
    tables = [table for page in pages for table in page]
    releases = [row for table in tables for row in _release_rows(table)]
    afters = [row for table in tables for row in _after_rows(table)]
    proposals = []
    for after in afters:
        held, remaining = after["held"], after["after_pledged"]
        for release in releases:
            released = release["released"]
            if (released <= 0 or remaining + released > held
                    or 20 * released < held
                    or abs(Decimal(100 * released) / held
                           - release["released_pct_printed"]) > Decimal("0.15")):
                continue
            before = remaining + released
            fifty = 2 * before >= held > 2 * remaining
            eighty = 5 * before >= 4 * held > 5 * remaining
            if not (fifty or eighty):
                continue
            proposals.append({"release_actor_cell": release["release_actor_cell"],
                              "after_actor_cell": after["after_actor_cell"],
                              "released": released, "held": held,
                              "after_pledged": remaining,
                              "before_pct": round(100 * before / held, 4),
                              "after_pct": round(100 * remaining / held, 4),
                              "crossed_tier": "80" if eighty else "50"})
    return proposals


def propose(year: int, root: Path) -> dict:
    if year not in (2024, 2025):
        raise ValueError("Only frozen 2024/2025 pledge years")
    index = pd.read_parquet(root / f"title_candidates_{year}.parquet")
    rows = [json.loads(line) for line in (
        root / f"pdf_audit_{year}.jsonl").read_text(
            encoding="utf-8").splitlines()]
    if (len(rows) != len(index)
            or {row["pdf_url"] for row in rows} != set(index.pdf_url)
            or any(row["status"] != "ok" for row in rows)):
        raise ValueError("Incomplete original PDF identity audit")
    source = {row["pdf_url"]: row for row in rows}
    proposals = []
    for extracted in load_tables(year, root, rows):
        row = source[extracted["pdf_url"]]
        for candidate in after_only_crossings(extracted["pages"]):
            proposals.append({key: row[key] for key in (
                "code", "notice_date", "title", "pdf_url")}
                             | candidate)
    output = root / f"after_only_crossing_proposals_{year}.csv"
    frame = pd.DataFrame(proposals).drop_duplicates()
    frame.to_csv(output, index=False)
    return {"year": year, "after_only_proposals": len(frame),
            "originals": frame.pdf_url.nunique() if not frame.empty else 0,
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
