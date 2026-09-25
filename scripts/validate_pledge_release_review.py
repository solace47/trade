"""Gate pledge-release outcome work on a complete original-PDF review ledger."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd


COLUMNS = ("notice_date", "code", "pdf_url", "screens", "decision",
           "actor", "registered_date", "released", "held", "after_pledged",
           "evidence")
SCREENS = ("vertical", "table", "after_only")
DECISIONS = {"pending", "prelim", "reject", "verified"}


def candidate_sources(root: Path) -> pd.DataFrame:
    sources: dict[str, dict] = {}
    for year in (2024, 2025):
        for screen in SCREENS:
            path = root / f"{screen}_crossing_proposals_{year}.csv"
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
            for row in frame.to_dict("records"):
                url = row["pdf_url"]
                identity = (row["notice_date"], row["code"])
                if url in sources and identity != (
                        sources[url]["notice_date"], sources[url]["code"]):
                    raise ValueError("Conflicting original PDF identity")
                item = sources.setdefault(url, {"notice_date": identity[0],
                                                "code": identity[1],
                                                "pdf_url": url,
                                                "screens": set()})
                item["screens"].add(screen)
    if not sources:
        raise ValueError("No pledge-release originals")
    records = [{**item, "screens": "+".join(sorted(item["screens"]))}
               for item in sources.values()]
    return pd.DataFrame(records).sort_values(
        ["notice_date", "code", "pdf_url"]).reset_index(drop=True)


def validate_rows(review: pd.DataFrame, candidates: pd.DataFrame) -> dict:
    if tuple(review.columns) != COLUMNS or review.pdf_url.duplicated().any():
        raise ValueError("Malformed or duplicate review ledger")
    source = candidates.set_index("pdf_url")
    ledger = review.set_index("pdf_url")
    if set(source.index) != set(ledger.index):
        raise ValueError("Review ledger does not cover exact source originals")
    for url, row in ledger.iterrows():
        expected = source.loc[url]
        if (row["notice_date"], row["code"], row["screens"]) != (
                expected["notice_date"], expected["code"],
                expected["screens"]):
            raise ValueError("Review source metadata changed")
        decision = row["decision"]
        if decision not in DECISIONS:
            raise ValueError("Unknown pledge review decision")
        if decision == "reject" and not row["evidence"].strip():
            raise ValueError("Rejected original needs a source-based reason")
        if decision != "verified":
            continue
        if not row["actor"].strip() or not row["evidence"].strip():
            raise ValueError("Verified original needs actor and evidence")
        try:
            registered = date.fromisoformat(row["registered_date"])
            noticed = date.fromisoformat(row["notice_date"])
            released = int(row["released"])
            held = int(row["held"])
            after = int(row["after_pledged"])
        except (ValueError, TypeError) as error:
            raise ValueError("Malformed verified release counts or dates") from error
        before = after + released
        fifty = 2 * before >= held > 2 * after
        eighty = 5 * before >= 4 * held > 5 * after
        if (registered > noticed or held <= 0 or released <= 0
                or after < 0 or before > held or 20 * released < held
                or not (fifty or eighty)):
            raise ValueError("Verified original fails frozen crossing rule")
    counts = ledger["decision"].value_counts().to_dict()
    return {"originals": len(ledger),
            "decisions": {key: int(counts.get(key, 0))
                          for key in sorted(DECISIONS)},
            "ready_for_matching": (counts.get("pending", 0) == 0
                                   and counts.get("prelim", 0) == 0
                                   and counts.get("verified", 0) > 0),
            "outcomes_opened": False}


def validate(path: Path, root: Path) -> dict:
    review = pd.read_csv(path, dtype=str, keep_default_na=False)
    return validate_rows(review, candidate_sources(root))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, default=Path(
        "research/pledge_release_review.csv"))
    parser.add_argument("--root", type=Path, default=Path(
        "data/research/pledge"))
    args = parser.parse_args()
    print(validate(args.review, args.root))


if __name__ == "__main__":
    main()
