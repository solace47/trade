"""Screen original new controlling-holder/chairman share-increase plans.

Only notice metadata is used here. Search results are candidates, and the
original PDF still needs an identity and prospective-plan check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import pandas as pd


ACTOR = re.compile(r"控股股东|实际控制人|董事长|大股东|第一大股东")
PLAN = re.compile(r"增持.{0,20}计划|计划.{0,15}增持")
NOT_NEW_PLAN = re.compile(
    r"进展|完成|完毕|结果|终止|届满|延期|延长|变更|更正|补充|修订|"
    r"调整|增加增持主体|实施情况|时间过半|期限过半|法律意见|律师|"
    r"财务顾问|核查意见|B股|H股|境内上市外资股|免于|回购|"
    r"及后续|暨后续|触及|达到1%|取得贷款|及增持计划|暨增持计划"
)


def strict_plan_title(title: str) -> bool:
    return bool(ACTOR.search(title) and PLAN.search(title)
                and not NOT_NEW_PLAN.search(title))


FUTURE_TEXT = re.compile(
    r"拟增持|计划增持|拟通过.{0,45}增持|"
    r"计划通过.{0,45}增持|拟自.{0,70}增持|"
    r"计划自.{0,70}增持|拟在.{0,70}增持|"
    r"计划在.{0,70}增持|拟以.{0,70}增持"
)


def confirm_pdf_text(text: str, code: str) -> bool:
    clean = re.sub(r"\s+", "", text)
    return bool(code.split(".")[-1] in clean
                and ACTOR.search(clean)
                and "增持" in clean and "股份" in clean
                and ("增持计划" in clean or FUTURE_TEXT.search(clean)))


def screen(source_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for year in (2024, 2025):
        sources = [pd.read_parquet(source_dir / name)
                   for name in (f"search_{year}.parquet",
                                f"search_plan_{year}.parquet")]
        for source in sources:
            if (source.empty or source.pdf_url.duplicated().any()
                    or not source.notice_date.str.startswith(str(year)).all()):
                raise ValueError("Malformed CNINFO insider-plan search index")
        combined = pd.concat(sources, ignore_index=True)
        if (combined.loc[combined.pdf_url.duplicated(keep=False)]
                .groupby("pdf_url")[["code", "notice_date", "title"]]
                .nunique().gt(1).any().any()):
            raise ValueError("CNINFO searches disagree on the same PDF")
        combined = combined.drop_duplicates("pdf_url")
        picked = combined.loc[combined.title.map(strict_plan_title)].copy()
        if picked.empty or picked.pdf_url.duplicated().any():
            raise ValueError("No unique original insider-plan candidates")
        month = picked.notice_date.str[:7].value_counts()
        report[str(year)] = {
            "first_search_pdf_rows": len(sources[0]),
            "second_search_pdf_rows": len(sources[1]),
            "union_pdf_rows": len(combined),
            "conservative_titles": len(picked),
            "duplicate_code_notice_rows": int(picked.duplicated(
                ["code", "notice_date"], keep=False).sum()),
            "stocks": int(picked.code.nunique()),
            "notice_days": int(picked.notice_date.nunique()),
            "peak_month": month.idxmax(),
            "peak_month_share": float(month.max() / len(picked)),
        }
        picked.sort_values(["notice_date", "code", "pdf_url"]).to_parquet(
            output_dir / f"title_candidates_{year}.parquet", index=False,
            compression="zstd")
    (output_dir / "title_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(
        "data/research/insider_buy"))
    parser.add_argument("--output", type=Path, default=Path(
        "data/research/insider_buy"))
    args = parser.parse_args()
    print(screen(args.source, args.output))


if __name__ == "__main__":
    main()
