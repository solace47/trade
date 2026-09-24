"""Screen original buyback-plan titles before any price outcome is opened."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


PLAN = re.compile(r"(?:回购.{0,8}股份.{0,12}方案|股份回购.{0,12}方案)")
NOT_NEW_PLAN = re.compile(
    r"实施|进展|首次|完成|完毕|结果|终止|调整|变更|修订|更正|补充|"
    r"注销|提议|提请|建议|问询|回复|董事会决议|法律意见|独立董事|"
    r"员工持股|股权激励|限制性股票|前十大股东|前十名股东|持股情况|"
    r"核查意见|保荐机构|财务顾问|受托管理|事务报告"
)


def strict_plan_title(title: str) -> bool:
    """Conservative new-plan screen; original PDFs still need validation."""
    return bool(PLAN.search(title) and not NOT_NEW_PLAN.search(title))


def screen(source_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = {}
    for year in (2024, 2025):
        source = pd.read_parquet(source_dir / f"search_{year}.parquet")
        if (source.empty or not source.notice_date.str.startswith(str(year)).all()
                or source.duplicated("pdf_url").any()):
            raise ValueError(f"Malformed original buyback search {year}")
        plans = source.loc[source.title.map(strict_plan_title)].copy()
        if plans.empty:
            raise ValueError(f"No conservative new plans in {year}")
        counts = plans.notice_date.value_counts()
        audit[str(year)] = {
            "search_a_share_pdfs": len(source),
            "conservative_title_pdfs": len(plans),
            "unique_codes": int(plans.code.nunique()),
            "notice_days": int(plans.notice_date.nunique()),
            "duplicate_code_days": int(plans.duplicated(
                ["code", "notice_date"], keep=False).sum()),
            "top_five_notice_day_share": float(counts.head(5).sum() / len(plans)),
            "peak_month": plans.notice_date.str[:7].value_counts().idxmax(),
            "peak_month_share": float(
                plans.notice_date.str[:7].value_counts().max() / len(plans)),
        }
        plans.to_parquet(output_dir / f"title_candidates_{year}.parquet",
                         index=False, compression="zstd")
    (output_dir / "title_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(
        "data/research/buyback"))
    parser.add_argument("--output", type=Path, default=Path(
        "data/research/buyback"))
    args = parser.parse_args()
    print(screen(args.source, args.output))


if __name__ == "__main__":
    main()
