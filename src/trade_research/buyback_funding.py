"""Read explicitly disclosed lower funding bounds from original buyback plans.

This is an input audit. An announced lower bound is not an executed buyback.
Ambiguous or upper-bound-only plans deliberately remain unclassified.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from .buyback_inputs import EVENT


ANCHOR = re.compile(
    r"(?:回购(?:股份)?(?:的|拟使用的|所需的|所用的|的)?资金总额|"
    r"用于回购(?:股份)?的资金总额|拟用于回购(?:股份)?的资金总额|"
    r"(?:拟)?回购(?:股份)?(?:的)?(?:总)?金额|预计回购金额|回购规模)"
)
MONEY = (r"(?<![\d,])(?:人民币)?"
         r"(?P<number>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
         r"(?P<unit>万|亿)元")
LOWER = re.compile(r"(?:不低于|不少于|下限(?:为|是|:|：)?)[^。；●□√]{0,18}?" + MONEY)
RANGE = re.compile(
    MONEY.replace("number", "first_number").replace("unit", "first_unit")
    + r"(?:（?含(?:本数)?）?)?[^\d。；●□√]{0,5}?(?:~|～|—|－|-|至|到)"
    + r"(?:人民币)?(?P<second_number>\d[\d,]*(?:\.\d+)?)(?P<second_unit>万|亿)元"
)
END = re.compile(r"[。；●□√]")


def _yuan(number: str, unit: str) -> float:
    return float(number.replace(",", "")) * (1e8 if unit == "亿" else 1e4)


def explicit_funding_floor(text: str) -> tuple[float | None, str, str]:
    """Return (yuan, method, evidence), or an unknown bound.

    Search only the first 130 characters after each funding/amount label. A
    single clear lower amount must recur consistently across the PDF excerpt.
    """
    clean = re.sub(r"\s+", "", text)
    candidates = []
    for anchor in ANCHOR.finditer(clean):
        if re.search(r"(?:若|如)?按(?:照)?$", clean[max(0, anchor.start() - 4):anchor.start()]):
            continue
        after = clean[anchor.end():anchor.end() + 130]
        stop = END.search(after)
        if stop:
            after = after[:stop.start()]
        for pattern, label in ((LOWER, "explicit_lower"), (RANGE, "range_lower")):
            for match in pattern.finditer(after):
                if pattern is LOWER:
                    value = _yuan(match["number"], match["unit"])
                else:
                    value = _yuan(match["first_number"], match["first_unit"])
                    upper = _yuan(match["second_number"], match["second_unit"])
                    if value >= upper:
                        continue
                # The label must refer to funding rather than an unrelated
                # later item such as an offer price or loan amount.
                prefix = after[:match.start()]
                if re.search(r"(?:回购价格|股份数量|回购股数|贷款金额|专项贷款|按照|测算)", prefix):
                    continue
                if pattern is RANGE and (
                        "约为" in prefix
                        or "测算" in clean[max(0, anchor.start() - 40):anchor.start()]):
                    continue
                candidates.append((value, label,
                                   clean[anchor.start():anchor.end() + match.end()]))
    if not candidates:
        return None, "unknown", ""
    values = {round(item[0]) for item in candidates}
    if len(values) != 1:
        return None, "conflicting_bounds", " | ".join(x[2][:100] for x in candidates[:3])
    return candidates[0]


def audit(source_dir: Path, pairs_path: Path, output_dir: Path) -> dict:
    source = pd.concat([
        pd.read_json(source_dir / f"pdf_audit_{year}.jsonl", lines=True)[
            ["pdf_url", "status", "text_first_three_pages"]]
        for year in (2024, 2025)
    ], ignore_index=True)
    events = pd.read_parquet(pairs_path)
    events = events.loc[events.candidate.eq(EVENT),
                        ["date", "code", "float_mv", "pdf_url"]]
    if (events.empty or events.pdf_url.duplicated().any()
            or source.pdf_url.duplicated().any()):
        raise ValueError("Funding source or frozen event is duplicated")
    joined = events.merge(source, on="pdf_url", validate="one_to_one")
    if len(joined) != len(events) or not joined.status.eq("ok").all():
        raise ValueError("Frozen event lacks a confirmed original PDF")
    parsed = joined.text_first_three_pages.map(explicit_funding_floor)
    joined[["floor_yuan", "method", "evidence"]] = pd.DataFrame(
        parsed.tolist(), index=joined.index)
    joined["floor_float_ratio"] = joined.floor_yuan / joined.float_mv
    if ((joined.floor_yuan.dropna() <= 0).any()
            or not joined.float_mv.gt(0).all()):
        raise ValueError("Invalid disclosed funding lower bound or float cap")
    report = {"note": "Input-only PDF funding floor; no funding-subgroup returns read",
              "by_year": {}}
    for year in ("2024", "2025"):
        subset = joined.loc[joined.date.str.startswith(year)]
        known = subset.loc[subset.floor_yuan.notna()]
        high = known.loc[known.floor_float_ratio.ge(.01)]
        report["by_year"][year] = {
            "events": len(subset), "known": len(known),
            "conflicting": int(subset.method.eq("conflicting_bounds").sum()),
            "known_days": int(known.date.nunique()),
            "known_months": int(known.date.str[:7].nunique()),
            "ratio_median": float(known.floor_float_ratio.median()),
            "ratio_p90": float(known.floor_float_ratio.quantile(.9)),
            "high_1pct": len(high), "high_days": int(high.date.nunique()),
            "high_months": int(high.date.str[:7].nunique()),
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    joined.drop(columns="text_first_three_pages").to_parquet(
        output_dir / "funding_inputs.parquet", index=False, compression="zstd")
    (output_dir / "funding_input_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("data/research/buyback"))
    parser.add_argument("--pairs", type=Path,
                        default=Path("data/research/buyback/pairs.parquet"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/research/buyback"))
    args = parser.parse_args()
    print(audit(args.source, args.pairs, args.output))


if __name__ == "__main__":
    main()
